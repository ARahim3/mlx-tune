"""
Speech-to-Text (STT) Fine-Tuning Support for MLX-Tune

Provides Unsloth-compatible API for STT models on Apple Silicon using mlx-audio:
- Whisper / Distil-Whisper (OpenAI) - Encoder-decoder speech recognition
- Moonshine (Useful Sensors) - Lightweight STT with conv frontend
- Qwen3-ASR (Alibaba) - Audio-LLM for multilingual ASR
- Canary (NVIDIA) - Conformer encoder-decoder with multilingual support
- Voxtral (Mistral) - Audio encoder + Llama LM for speech recognition

Usage (matches Unsloth patterns):
    from mlx_tune import FastSTTModel, STTSFTTrainer, STTSFTConfig, STTDataCollator

    model, processor = FastSTTModel.from_pretrained(
        "mlx-community/whisper-large-v3-turbo",
    )
    model = FastSTTModel.get_peft_model(model, r=16, lora_alpha=16)
"""

from typing import Optional, Any, List, Dict, Union, Tuple
from pathlib import Path
import warnings
import json
import numpy as np

import mlx.core as mx
import mlx.nn as nn

from mlx_tune.audio_profiles import (
    STTModelProfile,
    STT_PROFILES,
    detect_stt_model_type,
)

# Try to import mlx-audio for Whisper
HAS_MLX_AUDIO = False
_whisper_audio = None
_stt_load = None

try:
    from mlx_audio.stt import load as _stt_load_fn
    from mlx_audio.stt.models.whisper import audio as _whisper_audio_module
    _stt_load = _stt_load_fn
    _whisper_audio = _whisper_audio_module
    HAS_MLX_AUDIO = True
except ImportError:
    pass


def _try_load_whisper_processor(model_name: str):
    """
    Try to load a WhisperProcessor from the model repo or fall back to
    the matching openai/whisper-* model.
    """
    try:
        from transformers import WhisperProcessor
    except ImportError:
        return None

    # First try the model name directly
    try:
        return WhisperProcessor.from_pretrained(model_name)
    except Exception:
        pass

    # Try to guess the openai model from the name
    # e.g. "mlx-community/whisper-tiny" -> "openai/whisper-tiny"
    name_lower = model_name.lower()
    for size in ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3", "large-v2", "large"]:
        if size in name_lower:
            try:
                return WhisperProcessor.from_pretrained(f"openai/whisper-{size}")
            except Exception:
                pass
            break

    return None


def _push_to_hub(local_path: str, repo_id: str, **kwargs):
    """Upload a model directory to HuggingFace Hub."""
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, **{k: v for k, v in kwargs.items() if k in ("private", "token")})
    api.upload_folder(
        repo_id=repo_id,
        folder_path=local_path,
        commit_message=kwargs.get("commit_message", "Upload fine-tuned model via mlx-tune"),
        token=kwargs.get("token"),
    )
    print(f"Uploaded to: https://huggingface.co/{repo_id}")


def _require_mlx_audio():
    """Check that mlx-audio is available."""
    if not HAS_MLX_AUDIO:
        raise ImportError(
            "mlx-audio is required for STT model support. "
            "Install with: uv pip install 'mlx-tune[audio]'"
        )


# Whisper constants
WHISPER_SAMPLE_RATE = 16000
WHISPER_N_SAMPLES = 480000  # 30 seconds at 16kHz
WHISPER_N_MELS = 80
WHISPER_HOP_LENGTH = 160
WHISPER_N_FFT = 400


class FastSTTModel:
    """
    Unsloth-compatible API for STT (Whisper) models on Apple Silicon.

    Provides the same API patterns as FastLanguageModel / FastVisionModel
    but specialized for speech-to-text encoder-decoder models.

    Example:
        >>> from mlx_tune import FastSTTModel
        >>> model, processor = FastSTTModel.from_pretrained(
        ...     "mlx-community/whisper-large-v3-turbo",
        ... )
        >>> model = FastSTTModel.get_peft_model(model, r=16, lora_alpha=16)
    """

    @staticmethod
    def from_pretrained(
        model_name: str,
        max_seq_length: int = 448,
        dtype: Optional[Any] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        token: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> Tuple["STTModelWrapper", Any]:
        """
        Load a pretrained Whisper STT model.

        Args:
            model_name: HuggingFace model ID (e.g., "mlx-community/whisper-large-v3-turbo")
            max_seq_length: Maximum decoder sequence length (default 448 for Whisper)
            dtype: Data type (default: float16)
            load_in_4bit: Not supported for encoder-decoder models
            load_in_8bit: Not supported for encoder-decoder models
            token: HuggingFace API token
            trust_remote_code: Trust remote code
            **kwargs: Additional arguments

        Returns:
            Tuple of (STTModelWrapper, processor)
        """
        _require_mlx_audio()

        if load_in_4bit or load_in_8bit:
            warnings.warn(
                "Quantization is not currently supported for STT encoder-decoder models. "
                "Loading in default precision.",
                UserWarning,
            )

        model_dtype = mx.float16
        if dtype is not None:
            if dtype == "float32" or dtype == mx.float32:
                model_dtype = mx.float32
            elif dtype == "bfloat16" or dtype == mx.bfloat16:
                model_dtype = mx.bfloat16

        # Auto-detect model profile
        profile_key = detect_stt_model_type(model_name, {})
        profile = STT_PROFILES.get(profile_key) if profile_key else None

        print(f"Loading STT model: {model_name}")

        # Canary workaround: mlx-audio's canary sanitize() always transposes
        # 4D conv weights (NeMo→MLX), but MLX-converted repos already have
        # correct layout. Patch sanitize to skip the conv transpose.
        _canary_sanitize_patched = False
        if profile and profile.name == "canary":
            try:
                from mlx_audio.stt.models.canary.canary import Model as _CanaryModel
                _original_sanitize = _CanaryModel.sanitize

                def _safe_canary_sanitize(self_model, weights):
                    sanitized = {}
                    for key, value in weights.items():
                        new_key = key
                        # Apply the same key remapping as the original
                        if key.startswith("encoder.") and not key.startswith("encoder_decoder"):
                            new_key = "encoder.conformer." + key[len("encoder."):]
                        # (skip other remappings — they only apply to NeMo keys)
                        # Do NOT transpose conv weights (they're already in MLX format)
                        if "attn_dropout" in key or "layer_dropout" in key:
                            continue
                        if key == "log_softmax.mlp.log_softmax":
                            continue
                        if "num_batches_tracked" in key:
                            continue
                        sanitized[new_key] = value
                    return sanitized

                _CanaryModel.sanitize = _safe_canary_sanitize
                _canary_sanitize_patched = True
            except ImportError:
                pass

        # Voxtral workaround: AutoProcessor.from_pretrained crashes in
        # transformers 5.x because deepcopy of Mistral's tekken tokenizer
        # fails during logger.info(f"Processor {processor}") __repr__.
        # Patch post_load_hook to load tokenizer + feature_extractor separately.
        _voxtral_hook_patched = False
        if profile and profile.name == "voxtral":
            try:
                from mlx_audio.stt.models.voxtral.voxtral import Model as _VoxtralModel
                _original_hook = _VoxtralModel.post_load_hook

                @classmethod
                def _safe_voxtral_hook(cls, model_obj, model_path):
                    from transformers import AutoTokenizer, AutoFeatureExtractor
                    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                    feature_extractor = AutoFeatureExtractor.from_pretrained(str(model_path))

                    class _Proc:
                        pass
                    proc = _Proc()
                    proc.tokenizer = tokenizer
                    proc.feature_extractor = feature_extractor
                    proc.tokenizer.eos_token_ids = getattr(
                        tokenizer, "eos_token_ids", [2, 4, 32000]
                    )
                    model_obj._processor = proc
                    if not hasattr(model_obj.config, "model_repo") or model_obj.config.model_repo is None:
                        try:
                            index = model_path.parts.index("hub")
                            model_obj.config.model_repo = (
                                model_path.parts[index + 1]
                                .replace("models--", "")
                                .replace("--", "/")
                            )
                        except (ValueError, IndexError):
                            model_obj.config.model_repo = str(model_path)
                    return model_obj

                _VoxtralModel.post_load_hook = _safe_voxtral_hook
                _voxtral_hook_patched = True
            except ImportError:
                pass

        model = _stt_load(model_name)

        # Restore patched methods
        if _canary_sanitize_patched:
            _CanaryModel.sanitize = _original_sanitize
        if _voxtral_hook_patched:
            _VoxtralModel.post_load_hook = _original_hook

        # Build processor based on model type
        hf_processor = None
        tokenizer = None

        if profile and profile.architecture == "audio_llm":
            # Audio-LLM models (Qwen3-ASR, Voxtral) use their own tokenizers
            _inner = getattr(model, "_model", model)
            # Try various tokenizer attribute names
            tokenizer = getattr(_inner, "tokenizer", None)
            if tokenizer is None:
                tokenizer = getattr(_inner, "_tokenizer", None)
            if tokenizer is None:
                tokenizer = getattr(model, "tokenizer", None)
            if tokenizer is None:
                tokenizer = getattr(model, "_tokenizer", None)
            hf_processor = getattr(model, "_processor", None)
        elif profile and profile.architecture == "voxtral_realtime":
            # Voxtral Realtime: mlx-audio attaches a decode-only TekkenTokenizer
            # at model._tokenizer (set in post_load_hook). For training we ALSO
            # need the encode side, which lives in `mistral_common`. Load it
            # alongside (using the same tekken.json) and stash both on the
            # model so the data collator can call .encode() during training.
            #
            # mistral_common is required for fine-tuning Voxtral Realtime —
            # it's optional for inference but mandatory for training.
            tokenizer = getattr(model, "_tokenizer", None)
            hf_processor = None
            try:
                from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
                # Find the resolved tekken.json path. mlx-audio's
                # TekkenTokenizer.from_model_path stores it implicitly via
                # `vocab` etc., but it's easier to re-find via HF cache.
                from huggingface_hub import try_to_load_from_cache
                tekken_cached = try_to_load_from_cache(
                    repo_id=model_name, filename="tekken.json"
                )
                if tekken_cached:
                    mc_tok = MistralTokenizer.from_file(str(tekken_cached))
                    # The inner Tekkenizer has .encode(text, bos, eos)
                    inner = mc_tok.instruct_tokenizer.tokenizer
                    model._mistral_tokenizer = mc_tok
                    model._tekkenizer_encoder = inner
            except ImportError:
                # mistral_common not installed — inference still works,
                # training will raise a clear error in the data collator.
                pass
        else:
            # Encoder-decoder models (Whisper, Canary) — try WhisperProcessor
            hf_processor = getattr(model, "_processor", None)
            if hf_processor is None:
                hf_processor = _try_load_whisper_processor(model_name)
                if hf_processor is not None:
                    model._processor = hf_processor

            # Canary has its own tokenizer (stored as _tokenizer)
            if hf_processor is None and profile and profile.name == "canary":
                tokenizer = getattr(model, "_tokenizer", None)
                if tokenizer is None:
                    tokenizer = getattr(model, "tokenizer", None)
            elif hf_processor is None and profile and profile.architecture == "voxtral_realtime":
                # already handled above
                pass
            elif hf_processor is None:
                raise ValueError(
                    f"Could not load processor for '{model_name}'. "
                    "Try a model like 'mlx-community/whisper-tiny-asr-fp16'."
                )

        # Extract tokenizer from processor if available
        if tokenizer is None and hf_processor is not None:
            tokenizer = getattr(hf_processor, "tokenizer", hf_processor)

        processor = STTProcessor(
            tokenizer=tokenizer, model=model, hf_processor=hf_processor,
            profile=profile,
        )

        wrapper = STTModelWrapper(
            model=model,
            processor=processor,
            model_name=model_name,
            max_seq_length=max_seq_length,
            profile=profile,
        )

        print(f"STT model loaded: {model_name}")
        arch = profile.architecture if profile else "encoder_decoder"
        print(f"  Architecture: {arch}, Profile: {profile.name if profile else 'whisper'}")
        print(f"  Encoder layers: {wrapper.n_audio_layer}, Decoder layers: {wrapper.n_text_layer}")
        print(f"  Sample rate: {wrapper.profile.sample_rate}Hz, Max decode length: {max_seq_length}")

        return wrapper, processor

    @staticmethod
    def get_peft_model(
        model: "STTModelWrapper",
        r: int = 16,
        target_modules: Optional[List[str]] = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        finetune_encoder: bool = True,
        finetune_decoder: bool = True,
        bias: str = "none",
        use_gradient_checkpointing: Union[bool, str] = "unsloth",
        random_state: int = 3407,
        use_dora: bool = False,
        **kwargs,
    ) -> "STTModelWrapper":
        """
        Add LoRA adapters to Whisper model.

        Args:
            model: STTModelWrapper from from_pretrained()
            r: LoRA rank
            target_modules: Target modules (defaults to attention projections)
            lora_alpha: LoRA alpha scaling
            lora_dropout: LoRA dropout rate
            finetune_encoder: Whether to add LoRA to encoder layers
            finetune_decoder: Whether to add LoRA to decoder layers
            bias: Bias configuration
            use_gradient_checkpointing: Gradient checkpointing mode
            random_state: Random seed
            use_dora: Use weight-decomposed LoRA
            **kwargs: Additional configuration

        Returns:
            STTModelWrapper with LoRA configured
        """
        if not isinstance(model, STTModelWrapper):
            raise TypeError(
                f"Expected STTModelWrapper, got {type(model)}. "
                "Use FastSTTModel.from_pretrained() first."
            )

        if not finetune_encoder and not finetune_decoder:
            raise ValueError("At least one of finetune_encoder or finetune_decoder must be True")

        # Default target modules from profile
        if target_modules is None:
            if hasattr(model, 'profile') and model.profile:
                target_modules = list(model.profile.lora_target_modules)
            else:
                target_modules = ["query", "key", "value", "out"]

        model.configure_lora(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            finetune_encoder=finetune_encoder,
            finetune_decoder=finetune_decoder,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
            use_dora=use_dora,
        )

        return model

    @staticmethod
    def for_training(model: "STTModelWrapper") -> "STTModelWrapper":
        """Enable training mode."""
        if isinstance(model, STTModelWrapper):
            model.inference_mode = False
            model.model.train()
        return model

    @staticmethod
    def for_inference(model: "STTModelWrapper") -> "STTModelWrapper":
        """Enable inference mode."""
        if isinstance(model, STTModelWrapper):
            model.inference_mode = True
            model.model.eval()
        return model

    @staticmethod
    def convert(
        hf_model: str,
        output_dir: str = "mlx_model",
        quantize: bool = False,
        q_bits: int = 4,
        dtype: Optional[str] = None,
        upload_repo: Optional[str] = None,
    ):
        """
        Convert a HuggingFace Whisper model to MLX format.

        Uses mlx-audio's conversion pipeline which handles Whisper's
        encoder-decoder architecture properly.

        Args:
            hf_model: HuggingFace model ID (e.g., "openai/whisper-large-v3")
            output_dir: Output directory for MLX model
            quantize: Whether to quantize the model
            q_bits: Quantization bits (4, 8)
            dtype: Data type ("float16", "bfloat16", "float32")
            upload_repo: Optional HF repo to upload converted model
        """
        _require_mlx_audio()

        try:
            from mlx_audio.convert import convert
        except ImportError:
            raise ImportError("mlx-audio conversion requires: uv pip install 'mlx-tune[audio]'")

        print(f"Converting STT model: {hf_model} -> {output_dir}")
        convert(
            hf_path=hf_model,
            mlx_path=output_dir,
            quantize=quantize,
            model_domain="stt",
            upload_repo=upload_repo,
        )
        print(f"Conversion complete: {output_dir}")


class STTProcessor:
    """
    Combined processor for Whisper models.

    Wraps the tokenizer and audio processing functionality
    into a single processor object (similar to HF WhisperProcessor).
    """

    def __init__(self, tokenizer: Any = None, model: Any = None, hf_processor: Any = None,
                 max_audio_samples: int = WHISPER_N_SAMPLES,
                 profile: Optional["STTModelProfile"] = None):
        self._raw_tokenizer = tokenizer
        self._model = model
        self._hf_processor = hf_processor
        self._whisper_tokenizer = None
        self._max_audio_samples = max_audio_samples
        self._profile = profile

        # Try to get the Whisper tokenizer wrapper (has sot_sequence, eot, etc.)
        # Only for Whisper-family models
        if model is not None and hf_processor is not None:
            try:
                self._whisper_tokenizer = model.get_tokenizer(language="en", task="transcribe")
            except Exception:
                pass

        # Use the whisper tokenizer if available (has encode/decode + sot_sequence)
        self.tokenizer = self._whisper_tokenizer or self._raw_tokenizer

    def get_tokenizer(self, language: str = "en", task: str = "transcribe"):
        """Get a language/task-specific tokenizer wrapper."""
        if self._model is not None and self._hf_processor is not None:
            try:
                return self._model.get_tokenizer(language=language, task=task)
            except Exception:
                pass
        return self.tokenizer

    def encode(self, text: str, **kwargs) -> List[int]:
        """Tokenize text."""
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)
        return []

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs to text."""
        if self.tokenizer is not None:
            skip = kwargs.get("skip_special_tokens", False)
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip)
        return ""

    def compute_mel(self, audio: Union[np.ndarray, mx.array], n_mels: int = 80) -> mx.array:
        """
        Compute log-mel spectrogram from audio.

        Args:
            audio: Audio waveform at 16kHz
            n_mels: Number of mel bands (default 80)

        Returns:
            Log-mel spectrogram as mx.array
        """
        if _whisper_audio is None:
            raise ImportError("mlx-audio whisper audio module not available")

        # pad_or_trim and log_mel_spectrogram expect mx.array
        if isinstance(audio, np.ndarray):
            audio = mx.array(audio)

        # Pad or trim to max audio length (default 30 seconds)
        audio = _whisper_audio.pad_or_trim(audio, self._max_audio_samples)

        # Compute mel spectrogram
        mel = _whisper_audio.log_mel_spectrogram(audio, n_mels=n_mels)

        if isinstance(mel, np.ndarray):
            mel = mx.array(mel)

        return mel

    def preprocess_raw_audio(self, audio: Union[np.ndarray, mx.array]) -> mx.array:
        """
        Preprocess raw audio for models with conv frontends (e.g., Moonshine).

        No mel spectrogram — just normalize and convert to mx.array.
        The model's conv encoder handles feature extraction.

        Args:
            audio: Audio waveform at model sample rate

        Returns:
            Preprocessed audio as mx.array
        """
        if isinstance(audio, np.ndarray):
            audio = mx.array(audio, dtype=mx.float32)

        return audio

    @property
    def sot_sequence(self) -> Tuple[int, ...]:
        """Start of transcript token sequence."""
        if self._whisper_tokenizer and hasattr(self._whisper_tokenizer, "sot_sequence"):
            return self._whisper_tokenizer.sot_sequence
        if self.tokenizer and hasattr(self.tokenizer, "sot_sequence"):
            return self.tokenizer.sot_sequence
        # Fallback: SOT token only
        return (50258,)  # Default Whisper SOT


class STTModelWrapper:
    """
    Wraps a Whisper encoder-decoder model for STT fine-tuning.

    Handles LoRA application to encoder and/or decoder,
    mel spectrogram computation, and transcription.
    """

    def __init__(
        self,
        model: Any,
        processor: STTProcessor,
        model_name: str,
        max_seq_length: int = 448,
        config: Optional[Dict] = None,
        profile: Optional[STTModelProfile] = None,
    ):
        # Resolve profile: use provided, or fall back to Whisper default
        if profile is None:
            profile = STT_PROFILES["whisper"]
        self.profile = profile

        self.model = model
        self.processor = processor
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.config = config or {}

        # LoRA state
        self.lora_config = None
        self.lora_enabled = False
        self._lora_applied = False
        self._adapter_path: Optional[Path] = None

        # Mode
        self.inference_mode = False

        # Model dimensions — extract from various model structures
        if hasattr(model, "dims"):
            # Whisper-style dims dataclass
            self.n_audio_layer = model.dims.n_audio_layer
            self.n_text_layer = model.dims.n_text_layer
            self.n_mels = model.dims.n_mels
            self.n_vocab = model.dims.n_vocab
        else:
            # Count layers by traversing block paths
            self.n_mels = self.profile.n_mels
            self.n_vocab = 51865  # default, overridden below if possible
            self.n_audio_layer = self._count_blocks(self.profile.encoder_block_path)
            self.n_text_layer = self._count_blocks(self.profile.decoder_block_path)

            # Try to get vocab size from config
            _inner = getattr(model, "_model", model)
            _cfg = getattr(_inner, "config", getattr(model, "config", None))
            if _cfg is not None:
                # Various config attribute names for vocab size
                for attr in ("vocab_size", "n_vocab"):
                    v = getattr(_cfg, attr, None)
                    if v is None and hasattr(_cfg, "text_config"):
                        v = getattr(_cfg.text_config, attr, None)
                    if v is not None:
                        self.n_vocab = v
                        break

    def _count_blocks(self, block_path: str) -> int:
        """Count the number of blocks at the given path."""
        obj = self.model
        for part in block_path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return 0
        try:
            return len(obj)
        except TypeError:
            return 0

    def configure_lora(
        self,
        r: int = 16,
        target_modules: Optional[List[str]] = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        finetune_encoder: bool = True,
        finetune_decoder: bool = True,
        bias: str = "none",
        use_gradient_checkpointing: Union[bool, str] = "unsloth",
        random_state: int = 3407,
        use_dora: bool = False,
        **kwargs,
    ):
        """Configure LoRA parameters for encoder-decoder model."""
        self.lora_config = {
            "r": r,
            "target_modules": target_modules or ["query", "key", "value", "out"],
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "bias": bias,
            "finetune_encoder": finetune_encoder,
            "finetune_decoder": finetune_decoder,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "random_state": random_state,
            "use_dora": use_dora,
            **kwargs,
        }
        self.lora_enabled = True
        self._lora_applied = False

        parts = []
        if finetune_encoder:
            parts.append("encoder")
        if finetune_decoder:
            parts.append("decoder")
        print(
            f"LoRA configuration set: rank={r}, alpha={lora_alpha}, "
            f"targets={'+'.join(parts)}, modules={target_modules}"
        )

    def _apply_lora(self) -> bool:
        """
        Apply LoRA adapters to Whisper encoder and/or decoder.

        Unlike decoder-only models, Whisper requires custom LoRA application
        that traverses both encoder.blocks and decoder.blocks.

        Returns:
            True if LoRA was applied successfully.
        """
        if not self.lora_enabled:
            print("LoRA not configured. Call configure_lora() first.")
            return False

        if self._lora_applied:
            return False

        r = self.lora_config["r"]
        lora_alpha = self.lora_config["lora_alpha"]
        scale = lora_alpha / r
        dropout = self.lora_config.get("lora_dropout", 0.0)
        target_modules = self.lora_config.get("target_modules",
                                                list(self.profile.lora_target_modules))
        finetune_encoder = self.lora_config.get("finetune_encoder", True)
        finetune_decoder = self.lora_config.get("finetune_decoder", True)

        # Freeze entire model first
        self.model.freeze()

        total_replaced = 0

        # Navigate to encoder/decoder blocks using profile paths
        enc_parts = self.profile.encoder_block_path.split(".")  # e.g. ["encoder", "blocks"]
        dec_parts = self.profile.decoder_block_path.split(".")  # e.g. ["decoder", "blocks"]

        # Determine encoder-specific targets (may differ from decoder)
        enc_targets = list(self.profile.encoder_lora_targets) if self.profile.encoder_lora_targets else target_modules

        # For audio-LLM models, decoder has no cross-attention
        has_decoder_cross_attn = self.profile.architecture == "encoder_decoder"

        # Apply LoRA to encoder blocks
        if finetune_encoder:
            enc_obj = self.model
            for part in enc_parts:
                enc_obj = getattr(enc_obj, part, None)
                if enc_obj is None:
                    break
            if enc_obj is not None:
                for block in enc_obj:
                    total_replaced += self._apply_lora_to_block(
                        block, enc_targets, r, scale, dropout,
                        has_cross_attn=False,
                    )
            print(f"  Encoder: LoRA applied to {self.n_audio_layer} blocks")

        # Apply LoRA to decoder blocks
        if finetune_decoder:
            dec_obj = self.model
            for part in dec_parts:
                dec_obj = getattr(dec_obj, part, None)
                if dec_obj is None:
                    break
            if dec_obj is not None:
                for block in dec_obj:
                    total_replaced += self._apply_lora_to_block(
                        block, target_modules, r, scale, dropout,
                        has_cross_attn=has_decoder_cross_attn,
                    )
            print(f"  Decoder: LoRA applied to {self.n_text_layer} blocks")

        self._lora_applied = True

        # Count trainable parameters
        from mlx.utils import tree_flatten
        trainable = tree_flatten(self.model.trainable_parameters())
        lora_params = [k for k, _ in trainable if "lora" in k.lower()]
        print(f"LoRA applied: {len(lora_params)} trainable parameter groups ({total_replaced} layers replaced)")

        return True

    def _apply_lora_to_block(
        self,
        block: Any,
        target_modules: List[str],
        r: int,
        scale: float,
        dropout: float,
        has_cross_attn: bool = False,
    ) -> int:
        """
        Apply LoRA to a single ResidualAttentionBlock.

        Whisper attention blocks have:
        - block.attn.{query, key, value, out} (self-attention)
        - block.cross_attn.{query, key, value, out} (cross-attention, decoder only)
        - block.mlp1, block.mlp2 (FFN)

        Args:
            block: ResidualAttentionBlock
            target_modules: Which modules to target
            r: LoRA rank
            scale: LoRA scale (alpha/r)
            dropout: LoRA dropout
            has_cross_attn: Whether block has cross-attention

        Returns:
            Number of layers replaced
        """
        replaced = 0

        # Self-attention - use profile's attn_names to find the attribute
        self_attn_attr = self.profile.attn_names.get("self_attn", "attn")
        if hasattr(block, self_attn_attr):
            attn_module = getattr(block, self_attn_attr)
            for module_name in target_modules:
                if hasattr(attn_module, module_name):
                    original = getattr(attn_module, module_name)
                    if _is_linear_like(original):
                        lora_layer = _create_lora_linear(original, r, scale, dropout)
                        setattr(attn_module, module_name, lora_layer)
                        replaced += 1

        # Cross-attention (decoder only) - use profile's cross_attn_attr
        cross_attn_attr = self.profile.cross_attn_attr
        if has_cross_attn and hasattr(block, cross_attn_attr):
            cross_attn_module = getattr(block, cross_attn_attr)
            if cross_attn_module is not None:
                for module_name in target_modules:
                    if hasattr(cross_attn_module, module_name):
                        original = getattr(cross_attn_module, module_name)
                        if _is_linear_like(original):
                            lora_layer = _create_lora_linear(original, r, scale, dropout)
                            setattr(cross_attn_module, module_name, lora_layer)
                            replaced += 1

        # MLP layers (if "mlp1" or "mlp2" in target_modules)
        for mlp_name in ["mlp1", "mlp2"]:
            if mlp_name in target_modules and hasattr(block, mlp_name):
                original = getattr(block, mlp_name)
                if _is_linear_like(original):
                    lora_layer = _create_lora_linear(original, r, scale, dropout)
                    setattr(block, mlp_name, lora_layer)
                    replaced += 1

        # Decoder FFN targets at block-level (e.g. Voxtral Realtime's
        # feed_forward_w1/w2/w3 — direct children of DecoderLayer, not nested
        # under .attention or .mlp).
        ffn_targets = self.profile.decoder_ffn_targets
        if ffn_targets:
            for ffn_name in ffn_targets:
                if hasattr(block, ffn_name):
                    original = getattr(block, ffn_name)
                    if _is_linear_like(original):
                        lora_layer = _create_lora_linear(original, r, scale, dropout)
                        setattr(block, ffn_name, lora_layer)
                        replaced += 1

        return replaced

    def transcribe(
        self,
        audio: Union[str, np.ndarray, mx.array],
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio file path, numpy array, or mx.array (at 16kHz)
            language: Language code (e.g., "en", "fr")
            task: "transcribe" or "translate"
            **kwargs: Additional arguments for Whisper generate

        Returns:
            Transcribed text
        """
        if hasattr(self.model, "generate"):
            # Voxtral Realtime's generate() doesn't accept language/task; it
            # detects language internally. Filter unsupported kwargs.
            if self.profile and self.profile.architecture == "voxtral_realtime":
                result = self.model.generate(audio, **kwargs)
            else:
                result = self.model.generate(
                    audio,
                    language=language,
                    task=task,
                    **kwargs,
                )
            # Handle different return types from mlx-audio
            if isinstance(result, dict):
                return result.get("text", "")
            # STTOutput dataclass has .text attribute
            if hasattr(result, "text"):
                return result.text
            return str(result)
        else:
            warnings.warn("Model does not have generate() method")
            return ""

    def save_pretrained(self, output_dir: str, **kwargs):
        """Save LoRA adapters and configuration."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self._lora_applied:
            from mlx.utils import tree_flatten
            import safetensors.numpy

            # Save adapter weights
            trainable = dict(tree_flatten(self.model.trainable_parameters()))
            weights_np = {k: np.array(v) for k, v in trainable.items()}
            safetensors.numpy.save_file(
                weights_np, str(output_path / "adapters.safetensors")
            )

            # Save adapter config
            adapter_config = {
                "model_name": self.model_name,
                "model_type": "stt",
                "fine_tune_type": "lora",
                "lora_parameters": {
                    "rank": self.lora_config["r"],
                    "alpha": self.lora_config["lora_alpha"],
                    "dropout": self.lora_config.get("lora_dropout", 0.0),
                    "scale": self.lora_config["lora_alpha"] / self.lora_config["r"],
                    "target_modules": self.lora_config.get("target_modules", []),
                    "finetune_encoder": self.lora_config.get("finetune_encoder", True),
                    "finetune_decoder": self.lora_config.get("finetune_decoder", True),
                },
                "whisper_config": {
                    "n_audio_layer": self.n_audio_layer,
                    "n_text_layer": self.n_text_layer,
                    "n_mels": self.n_mels,
                    "n_vocab": self.n_vocab,
                    "sample_rate": self.profile.sample_rate,
                    "max_seq_length": self.max_seq_length,
                },
            }
            with open(output_path / "adapter_config.json", "w") as f:
                json.dump(adapter_config, f, indent=2)

            print(f"STT adapters saved to: {output_path}")
        else:
            print("No LoRA adapters to save (LoRA not applied)")

    def save_pretrained_merged(
        self,
        output_dir: str,
        processor: Any = None,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Fuse LoRA weights into base Whisper model and save the merged model.

        Args:
            output_dir: Directory to save merged model
            processor: Processor to save (uses self.processor if None)
            push_to_hub: Whether to upload to HuggingFace Hub
            repo_id: HuggingFace repo ID (required if push_to_hub=True)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        processor = processor or self.processor

        if not self._lora_applied:
            warnings.warn("LoRA not applied — saving base model as-is.")

        # Fuse LoRA layers into base weights
        fused_count = 0
        for name, module in self.model.named_modules():
            if hasattr(module, "fuse"):
                fused = module.fuse(dequantize=kwargs.get("dequantize", False))
                # Navigate to parent and replace
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    if p.isdigit():
                        parent = parent[int(p)]
                    else:
                        parent = getattr(parent, p)
                last = parts[-1]
                if last.isdigit():
                    parent[int(last)] = fused
                else:
                    setattr(parent, last, fused)
                fused_count += 1

        if fused_count > 0:
            print(f"Fused {fused_count} LoRA layers into base Whisper model")

        # Save merged model weights. Use safetensors (no kwarg limit) for large
        # models — mx.savez() chokes at >1024 keys (nanobind vectorcall limit),
        # which trips for any decoder-heavy model like Voxtral Realtime.
        from mlx.utils import tree_flatten
        weights = dict(tree_flatten(self.model.parameters()))
        try:
            mx.save_safetensors(str(output_path / "weights.safetensors"), weights)
        except Exception:
            # Fallback to .npz for tiny models / older mlx versions
            mx.savez(str(output_path / "weights.npz"), **weights)

        # Save model config
        if hasattr(self.model, "dims"):
            import dataclasses
            dims_dict = dataclasses.asdict(self.model.dims)
            with open(output_path / "config.json", "w") as f:
                json.dump(dims_dict, f, indent=2)
        elif hasattr(self.model, "config"):
            # Voxtral Realtime: dataclass-based ModelConfig with nested
            # encoder_args / decoder / audio_encoding_args.
            try:
                import dataclasses
                cfg_dict = dataclasses.asdict(self.model.config)
                with open(output_path / "config.json", "w") as f:
                    json.dump(cfg_dict, f, indent=2)
            except (TypeError, ValueError):
                # Fallback: model config is not a dataclass; skip silently
                pass

        # Save processor/tokenizer
        if processor and hasattr(processor, "_hf_processor") and processor._hf_processor is not None:
            processor._hf_processor.save_pretrained(str(output_path))

        print(f"Merged STT model saved to: {output_path}")

        if push_to_hub and repo_id:
            _push_to_hub(str(output_path), repo_id, **kwargs)

    def push_to_hub(self, repo_id: str, **kwargs):
        """
        Push saved adapters or merged model to HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo ID (e.g., "username/my-whisper-model")
        """
        if self._adapter_path and self._adapter_path.exists():
            _push_to_hub(str(self._adapter_path), repo_id, **kwargs)
        else:
            raise ValueError(
                "No saved model to push. Call save_pretrained() or "
                "save_pretrained_merged() first."
            )

    def load_adapter(self, adapter_path: str, **kwargs):
        """Load LoRA adapters from a saved checkpoint."""
        adapter_dir = Path(adapter_path)

        # Load adapter config
        config_path = adapter_dir / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                adapter_config = json.load(f)

            # Restore LoRA config for re-application
            lora_params = adapter_config.get("lora_parameters", {})
            if lora_params and not self.lora_enabled:
                self.configure_lora(
                    r=lora_params.get("rank", 16),
                    lora_alpha=lora_params.get("alpha", 16),
                    lora_dropout=lora_params.get("dropout", 0.0),
                    target_modules=lora_params.get("target_modules", []),
                    finetune_encoder=lora_params.get("finetune_encoder", True),
                    finetune_decoder=lora_params.get("finetune_decoder", True),
                )

        # Load weights
        weights_path = adapter_dir / "adapters.safetensors"
        if weights_path.exists():
            from mlx.utils import tree_unflatten
            import safetensors.numpy

            weights_np = safetensors.numpy.load_file(str(weights_path))
            weights_mx = {k: mx.array(v) for k, v in weights_np.items()}

            # Apply LoRA structure first if not already done
            if not self._lora_applied and self.lora_enabled:
                self._apply_lora()

            # Load weights into model (strict=False: adapter has only LoRA params, not full model)
            self.model.load_weights(list(weights_mx.items()), strict=False)
            mx.eval(self.model)

            self._adapter_path = adapter_dir
            print(f"STT adapters loaded from: {adapter_dir}")
        else:
            raise FileNotFoundError(f"No adapters found at {weights_path}")


def _is_linear_like(module: Any) -> bool:
    """True if `module` is an nn.Linear or nn.QuantizedLinear (or subclass).

    QuantizedLinear is NOT a subclass of Linear in MLX, so a plain isinstance
    check on Linear misses 4-bit/6-bit/8-bit quantized models. mlx-lm's
    LoRALinear.from_base() supports both, so we only need to recognize them.
    """
    if isinstance(module, nn.Linear):
        return True
    QL = getattr(nn, "QuantizedLinear", None)
    if QL is not None and isinstance(module, QL):
        return True
    return False


def _create_lora_linear(
    original: Any,
    r: int,
    scale: float,
    dropout: float,
) -> Any:
    """
    Replace an nn.Linear (or nn.QuantizedLinear) with a LoRALinear layer.

    Uses mlx-lm's LoRALinear.from_base() which supports both dense and
    quantized base layers — this is what makes 4-bit / 6-bit LoRA work.
    """
    try:
        from mlx_lm.tuner.lora import LoRALinear
    except ImportError:
        raise ImportError(
            "mlx_lm.tuner.lora.LoRALinear not available. "
            "Please install mlx-lm: uv pip install mlx-lm"
        )

    lora_layer = LoRALinear.from_base(
        original,
        r=r,
        scale=scale,
        dropout=dropout,
    )

    return lora_layer


class STTSFTConfig:
    """
    Training configuration for STT (Whisper) fine-tuning.

    Mirrors SFTConfig / VLMSFTConfig for API compatibility.
    """

    def __init__(
        self,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 5,
        max_steps: Optional[int] = 100,
        num_train_epochs: int = 1,
        learning_rate: float = 1e-5,
        logging_steps: int = 1,
        output_dir: str = "./stt_outputs",
        lr_scheduler_type: str = "linear",
        weight_decay: float = 0.01,
        seed: int = 3407,
        sample_rate: int = 16000,
        language: str = "en",
        task: str = "transcribe",
        max_audio_length: float = 30.0,
        n_mels: int = 80,
        **kwargs,
    ):
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.logging_steps = logging_steps
        self.output_dir = output_dir
        self.lr_scheduler_type = lr_scheduler_type
        self.weight_decay = weight_decay
        self.seed = seed
        self.sample_rate = sample_rate
        self.language = language
        self.task = task
        self.max_audio_length = max_audio_length
        self.n_mels = n_mels

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class STTDataCollator:
    """
    Data collator for Whisper STT fine-tuning.

    Processes audio+transcript pairs into training tensors:
    1. Load audio, resample to 16kHz
    2. Compute 80-bin log-mel spectrogram
    3. Tokenize transcript with Whisper tokenizer
    4. Build decoder input/output for teacher forcing
    """

    def __init__(
        self,
        model: STTModelWrapper,
        processor: STTProcessor,
        language: str = "en",
        task: str = "transcribe",
        audio_column: str = "audio",
        text_column: Optional[str] = None,
        max_audio_length: float = 30.0,
    ):
        self.model = model
        self.processor = processor
        self.language = language
        self.task = task
        self.audio_column = audio_column
        self.text_column = text_column  # Auto-detect if None
        self.max_audio_length = max_audio_length

        # Possible text column names
        self._text_candidates = ["text", "transcription", "sentence", "transcript"]

        # Get a language/task-specific tokenizer so SOT sequence has correct
        # language and task tokens (e.g. <|fr|> instead of hardcoded <|en|>)
        self._lang_tokenizer = None
        if language != "en" or task != "transcribe":
            try:
                self._lang_tokenizer = processor.get_tokenizer(
                    language=language, task=task
                )
            except Exception:
                pass

    def _find_text_column(self, sample: Dict) -> str:
        """Auto-detect the text column name."""
        if self.text_column:
            return self.text_column

        for candidate in self._text_candidates:
            if candidate in sample:
                return candidate

        raise ValueError(
            f"Could not find text column. Available: {list(sample.keys())}. "
            f"Expected one of: {self._text_candidates}. "
            "Set text_column explicitly."
        )

    def __call__(self, samples: Union[List[Dict], Dict]) -> Dict:
        """
        Collate a batch of audio+transcript samples.

        Args:
            samples: List of dicts with audio and text columns

        Returns:
            Dict with model-appropriate keys as mx.arrays
        """
        if isinstance(samples, dict):
            samples = [samples]

        profile = self.model.profile if self.model else None
        arch = profile.architecture if profile else "encoder_decoder"

        if arch == "voxtral_realtime":
            return self._collate_voxtral_realtime(samples)
        if arch == "audio_llm":
            return self._collate_audio_llm(samples)
        return self._collate_encoder_decoder(samples)

    def _collate_encoder_decoder(self, samples: List[Dict]) -> Dict:
        """Collate for encoder-decoder models (Whisper, Canary, Moonshine)."""
        all_mels = []
        all_labels = []
        all_decoder_inputs = []

        for sample in samples:
            mel, decoder_input_ids, labels = self._process_sample(sample)
            all_mels.append(mel)
            all_labels.append(labels)
            all_decoder_inputs.append(decoder_input_ids)

        # Pad decoder sequences to same length
        max_dec_len = max(len(ids) for ids in all_decoder_inputs)
        tok = self.processor.tokenizer
        if hasattr(tok, "eot"):
            eot = tok.eot
        elif hasattr(tok, "eos_id"):
            eot = tok.eos_id
        else:
            eot = 50257

        padded_decoder_inputs = []
        padded_labels = []
        for dec_ids, labs in zip(all_decoder_inputs, all_labels):
            pad_len = max_dec_len - len(dec_ids)
            padded_decoder_inputs.append(dec_ids + [eot] * pad_len)
            padded_labels.append(labs + [-100] * pad_len)

        # Stack mel spectrograms (all are same size after pad_or_trim)
        # Canary mel is already batched (1, T, features) — squeeze batch dim before stacking
        if all_mels and all_mels[0].ndim == 3 and all_mels[0].shape[0] == 1:
            all_mels = [m.squeeze(0) for m in all_mels]
        mel_batch = mx.stack(all_mels)

        return {
            "input_features": mel_batch,
            "decoder_input_ids": mx.array(padded_decoder_inputs),
            "labels": mx.array(padded_labels),
        }

    def _collate_audio_llm(self, samples: List[Dict]) -> Dict:
        """Collate for audio-LLM models (Qwen3-ASR, Voxtral)."""
        processed = [self._process_audio_llm_sample(s) for s in samples]

        all_input_ids = [p["input_ids"] for p in processed]
        all_labels = [p["labels"] for p in processed]

        # Pad sequences
        max_len = max(len(ids) for ids in all_input_ids)
        pad_id = 0

        padded_ids = []
        padded_labels = []
        for ids, labs in zip(all_input_ids, all_labels):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            padded_labels.append(labs + [-100] * pad_len)

        result = {
            "input_ids": mx.array(padded_ids),
            "labels": mx.array(padded_labels),
        }

        # Pass through audio feature keys (batch_size=1, use first sample's features)
        for key in ("input_features", "feature_attention_mask"):
            if key in processed[0]:
                result[key] = processed[0][key]

        return result

    def _process_sample(self, sample: Dict) -> Tuple[mx.array, List[int], List[int]]:
        """Process a single sample into audio features, decoder_input_ids, and labels."""
        # Get audio
        audio_data = sample.get(self.audio_column)
        if audio_data is None:
            raise ValueError(f"Sample missing '{self.audio_column}' column")

        # Get target sample rate from profile
        target_sr = self.model.profile.sample_rate

        # Handle HuggingFace Audio format
        if isinstance(audio_data, dict):
            audio_array = np.array(audio_data.get("array", audio_data.get("data")), dtype=np.float32)
            sr = audio_data.get("sampling_rate", target_sr)
        elif isinstance(audio_data, np.ndarray):
            audio_array = audio_data.astype(np.float32)
            sr = target_sr
        else:
            audio_array = np.array(audio_data, dtype=np.float32)
            sr = target_sr

        # Resample if needed
        if sr != target_sr:
            try:
                from mlx_audio.stt.utils import resample_audio
                audio_array = resample_audio(audio_array, sr, target_sr)
            except ImportError:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)

        # Compute audio features based on preprocessor type
        preprocessor = self.model.profile.preprocessor
        if preprocessor == "raw_conv":
            # Raw waveform for conv-frontend models (Moonshine)
            features = self.processor.preprocess_raw_audio(audio_array)
        elif preprocessor == "canary_mel":
            # Canary: use model's own _preprocess_audio (parakeet mel spectrogram)
            actual_model = self.model.model if hasattr(self.model, "model") else self.model
            features = actual_model._preprocess_audio(audio_array)
            # Returns (1, T, 128) — already batched, squeeze for consistency
            # We'll re-stack later in _collate_encoder_decoder
        else:
            # Default: log-mel spectrogram (Whisper)
            features = self.processor.compute_mel(audio_array, n_mels=self.model.n_mels)

        # Get text
        text_col = self._find_text_column(sample)
        text = sample[text_col]

        # Use language-specific tokenizer if available, otherwise default
        tokenizer = self._lang_tokenizer or self.processor.tokenizer

        # Canary-specific tokenization
        _profile_name = self.model.profile.name if self.model.profile else ""
        if _profile_name == "canary":
            # Canary uses build_prompt_tokens for SOT, encode for text
            transcript_tokens = tokenizer.encode(text)
            sot_seq = tokenizer.build_prompt_tokens(
                source_lang=self.language,
                target_lang=self.language,
                use_pnc=True,
            )
            decoder_input_ids = sot_seq + transcript_tokens
            eot = tokenizer.eos_id
            labels = transcript_tokens + [eot]
            while len(labels) < len(decoder_input_ids):
                labels = [-100] + labels
            return features, decoder_input_ids, labels

        if tokenizer is not None:
            transcript_tokens = tokenizer.encode(text)
        else:
            transcript_tokens = []

        # Build decoder input: SOT sequence + transcript tokens
        # Use language-specific SOT sequence (contains correct <|lang|> and <|task|> tokens)
        if self._lang_tokenizer and hasattr(self._lang_tokenizer, "sot_sequence"):
            sot_seq = list(self._lang_tokenizer.sot_sequence)
        else:
            sot_seq = list(self.processor.sot_sequence)

        # Decoder input: [SOT, lang, task, ...transcript_tokens]
        decoder_input_ids = sot_seq + transcript_tokens

        # Labels: shifted by 1 (teacher forcing)
        # Labels are the transcript tokens + EOT
        eot = tokenizer.eot if hasattr(tokenizer, "eot") else 50257
        labels = transcript_tokens + [eot]

        # Pad labels to match decoder_input_ids length
        while len(labels) < len(decoder_input_ids):
            labels = [-100] + labels  # Pad front with ignore

        return features, decoder_input_ids, labels

    def _process_audio_llm_sample(self, sample: Dict) -> Dict:
        """Process a sample for audio-LLM models (Qwen3-ASR, Voxtral).

        Returns a dict with 'input_ids', 'labels', and model-specific audio keys.
        """
        profile = self.model.profile
        audio_token_id = profile.audio_token_id

        # Get audio
        audio_data = sample.get(self.audio_column)
        if audio_data is None:
            raise ValueError(f"Sample missing '{self.audio_column}' column")

        target_sr = profile.sample_rate

        if isinstance(audio_data, dict):
            audio_array = np.array(audio_data.get("array", audio_data.get("data")), dtype=np.float32)
            sr = audio_data.get("sampling_rate", target_sr)
        elif isinstance(audio_data, np.ndarray):
            audio_array = audio_data.astype(np.float32)
            sr = target_sr
        else:
            audio_array = np.array(audio_data, dtype=np.float32)
            sr = target_sr

        if sr != target_sr:
            try:
                from mlx_audio.stt.utils import resample_audio
                audio_array = resample_audio(audio_array, sr, target_sr)
            except ImportError:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)

        # Preprocess audio using model's own method
        _inner = getattr(self.model.model, "_model", self.model.model)
        extra_keys = {}
        n_audio_tokens = 10  # fallback

        if profile.name == "voxtral":
            # Voxtral: use feature_extractor → transpose → get_audio_embeds to count tokens
            feat_ext = getattr(self.model.model._processor, "feature_extractor", None)
            if feat_ext is not None:
                features = feat_ext(audio_array, sampling_rate=target_sr, return_tensors="np")
                input_feats = mx.array(features["input_features"])  # (1, 128, T)
                input_feats = input_feats.transpose(0, 2, 1)  # (1, T, 128)
                extra_keys["input_features"] = input_feats
                # Compute exact audio token count via audio tower
                audio_embeds = self.model.model.get_audio_embeds(input_feats)
                mx.eval(audio_embeds)
                n_audio_tokens = audio_embeds.shape[0]
            else:
                raise ValueError("Voxtral model missing feature_extractor in processor")
        elif hasattr(_inner, "_preprocess_audio"):
            # Qwen3-ASR: model has _preprocess_audio
            result = _inner._preprocess_audio(audio_array)
            if isinstance(result, tuple) and len(result) == 3:
                # Qwen3-ASR: (input_features, attention_mask, num_audio_tokens)
                extra_keys["input_features"] = result[0]
                extra_keys["feature_attention_mask"] = result[1]
                n_audio_tokens = int(result[2])
            elif isinstance(result, tuple) and len(result) == 2:
                extra_keys["input_features"] = result[0]
                n_audio_tokens = result[0].shape[1] if result[0].ndim >= 2 else 10
            else:
                extra_keys["input_features"] = result
                n_audio_tokens = result.shape[1] if hasattr(result, "shape") and result.ndim >= 2 else 10
        else:
            # Fallback: compute mel spectrogram
            audio_mx = mx.array(audio_array, dtype=mx.float32)
            try:
                from mlx_audio.stt.models.whisper.audio import log_mel_spectrogram, pad_or_trim
                audio_mx = pad_or_trim(audio_mx)
                features = log_mel_spectrogram(audio_mx, n_mels=profile.n_mels)
                extra_keys["input_features"] = features[None]  # add batch dim
                n_audio_tokens = features.shape[0] // 2  # rough estimate
            except ImportError:
                extra_keys["input_features"] = audio_mx.reshape(1, 1, -1)

        # Get text
        text_col = self._find_text_column(sample)
        text = sample[text_col]

        # Tokenize transcription
        tokenizer = self.processor.tokenizer
        if hasattr(tokenizer, "encode"):
            transcript_tokens = tokenizer.encode(text)
        elif callable(tokenizer):
            transcript_tokens = list(tokenizer(text)["input_ids"])
        else:
            transcript_tokens = []

        # Build input sequence: [audio_placeholder × N, transcription_tokens]
        audio_placeholder = [audio_token_id] * n_audio_tokens
        input_ids = audio_placeholder + transcript_tokens

        # Labels: -100 for audio positions, transcript tokens (shifted by 1)
        eos_id = getattr(tokenizer, "eos_token_id", 0)
        if eos_id is None:
            eos_id = 0
        labels = [-100] * len(audio_placeholder) + transcript_tokens[1:] + [eos_id]

        # Ensure same length
        while len(labels) < len(input_ids):
            labels.append(-100)
        labels = labels[:len(input_ids)]

        return {"input_ids": input_ids, "labels": labels, **extra_keys}

    def _collate_voxtral_realtime(self, samples: List[Dict]) -> Dict:
        """Collate for Voxtral Realtime: build inputs_embeds + labels.

        Forces batch_size=1 (variable embedding sequence length per sample).
        """
        if len(samples) > 1:
            warnings.warn(
                "Voxtral Realtime training only supports batch_size=1. "
                "Using only the first sample in the batch."
            )
        processed = self._process_voxtral_realtime_sample(samples[0])
        return processed

    def _process_voxtral_realtime_sample(self, sample: Dict) -> Dict:
        """Build inputs_embeds + labels for one Voxtral Realtime sample.

        Mirrors the inference path in mlx_audio's voxtral_realtime.Model:
        1. Resample audio to 16kHz
        2. Compute mel via model._prepare_mel(...)
        3. Run conv stem + encoder.encode_full() (no_grad) to get adapter_out
        4. Tokenize transcript via model._tokenizer
        5. Build prefix tokens [BOS] + [STREAMING_PAD] * (n_left + n_delay)
        6. Build per-position embeddings:
           - prefix:    adapter_out[:prompt_len] + tok_emb(prefix_tokens)
           - audio span (pos < adapter_len): adapter_out[pos] + tok_emb(text_token)
           - text only (pos >= adapter_len): tok_emb(text_token)
        7. Labels: -100 for prefix, then text token IDs (next-token convention)
        """
        rt_model = self.model.model  # the mlx-audio voxtral_realtime.Model
        config = rt_model.config

        # ---- 1. Get audio array
        audio_data = sample.get(self.audio_column)
        if audio_data is None:
            raise ValueError(f"Sample missing '{self.audio_column}' column")

        target_sr = self.model.profile.sample_rate  # 16000
        if isinstance(audio_data, dict):
            audio_array = np.array(
                audio_data.get("array", audio_data.get("data")), dtype=np.float32
            )
            sr = audio_data.get("sampling_rate", target_sr)
        elif isinstance(audio_data, np.ndarray):
            audio_array = audio_data.astype(np.float32)
            sr = target_sr
        else:
            audio_array = np.array(audio_data, dtype=np.float32)
            sr = target_sr

        if sr != target_sr:
            try:
                from mlx_audio.stt.utils import resample_audio
                audio_array = resample_audio(audio_array, sr, target_sr)
            except ImportError:
                import librosa
                audio_array = librosa.resample(
                    audio_array, orig_sr=sr, target_sr=target_sr
                )

        # ---- 2. Compute mel + n_delay (uses model's own helper for exact match)
        mel, n_delay = rt_model._prepare_mel(
            audio_array, transcription_delay_ms=config.transcription_delay_ms
        )

        # ---- 3. Run conv stem + encoder once (no LoRA gradients on encoder
        #         since it's frozen by default; even when finetune_encoder is on
        #         we still compute the adapter outputs without recording grads
        #         on this branch — gradients flow only through the decoder).
        conv_out = rt_model.encoder.conv_stem(mel)
        sliding_window = rt_model.encoder.config.sliding_window
        if conv_out.shape[0] <= sliding_window:
            adapter_out = rt_model.encoder.encode_full(conv_out)
        else:
            # Long audio: collect all chunks, skip empty trailing chunks
            # (downsample_and_project returns (0, encoder_dim) if the last
            # chunk has fewer than 4 frames — shape won't concat with the
            # projected chunks which are at decoder_dim).
            chunks = []
            for chunk_out in rt_model.encoder.encode_chunks(conv_out):
                chunk_adapter = rt_model.encoder.downsample_and_project(chunk_out)
                if chunk_adapter.shape[0] > 0:
                    chunks.append(chunk_adapter)
            adapter_out = mx.concatenate(chunks, axis=0) if chunks else None
            if adapter_out is None:
                raise ValueError("Audio encoding produced no output tokens.")
        mx.eval(adapter_out)
        # Stop gradient flow into the encoder branch
        adapter_out = mx.stop_gradient(adapter_out)

        n_audio_total = adapter_out.shape[0]

        # ---- 4. Tokenize transcript via mistral_common's Tekkenizer.
        # mlx-audio's TekkenTokenizer is decode-only; the encode side lives
        # in `mistral_common`, which we load in FastSTTModel.from_pretrained
        # and stash on the model as `_tekkenizer_encoder`.
        text_col = self._find_text_column(sample)
        text = sample[text_col]
        encoder = getattr(rt_model, "_tekkenizer_encoder", None)
        if encoder is None:
            raise RuntimeError(
                "Voxtral Realtime training requires `mistral_common` for text "
                "encoding (mlx-audio's TekkenTokenizer is decode-only). "
                "Install with: uv pip install mistral-common"
            )
        # Tekkenizer.encode signature: encode(text, bos, eos)
        text_tokens = list(encoder.encode(text, bos=False, eos=False))

        if not text_tokens:
            raise ValueError(f"Empty tokenization for transcript: {text!r}")

        # Append EOS so the model learns when to stop
        eos = config.eos_token_id
        text_tokens = text_tokens + [eos]

        # ---- 5. Build prefix
        bos = config.bos_token_id
        pad_id = config.streaming_pad_token_id
        n_left = config.n_left_pad_tokens
        prompt_len = 1 + n_left + n_delay
        prefix_token_ids = [bos] + [pad_id] * (n_left + n_delay)
        prefix_token_ids_mx = mx.array(prefix_token_ids, dtype=mx.int32)

        # Sanity: adapter_out must cover at least prompt_len positions
        if n_audio_total < prompt_len:
            raise ValueError(
                f"Audio too short: adapter_out has {n_audio_total} tokens but "
                f"prompt_len is {prompt_len}. Use a longer audio sample."
            )

        # ---- 6. Build full embedding sequence
        # Prefix embeddings (audio + prefix tokens)
        prefix_text_embeds = rt_model.decoder.embed_tokens(prefix_token_ids_mx)
        prefix_embeds = adapter_out[:prompt_len] + prefix_text_embeds  # [prompt_len, dim]

        # Per-position text embeddings: each text token sits at position
        # prompt_len + i. If that position is still inside the audio span,
        # we ADD adapter_out[pos]; otherwise we use the token embedding alone.
        text_tokens_mx = mx.array(text_tokens, dtype=mx.int32)
        text_token_embeds = rt_model.decoder.embed_tokens(text_tokens_mx)  # [T_text, dim]

        n_text = len(text_tokens)
        # How many text positions overlap with the audio span?
        n_overlap = max(0, min(n_text, n_audio_total - prompt_len))
        if n_overlap > 0:
            audio_overlap = adapter_out[prompt_len : prompt_len + n_overlap]
            overlap_embeds = audio_overlap + text_token_embeds[:n_overlap]
        else:
            overlap_embeds = None

        if n_text > n_overlap:
            tail_embeds = text_token_embeds[n_overlap:]
        else:
            tail_embeds = None

        parts = [prefix_embeds]
        if overlap_embeds is not None:
            parts.append(overlap_embeds)
        if tail_embeds is not None:
            parts.append(tail_embeds)
        inputs_embeds = mx.concatenate(parts, axis=0)  # [seq, dim]

        # ---- 7. Build labels (causal-LM next-token convention)
        # Position p contributes a logit for what comes at position p+1.
        # Prefix positions: -100 (no supervision)
        # The first text token sits at index prompt_len. The label for the
        # logit at index prompt_len-1 should be text_tokens[0]. So we set:
        #   labels[i] = -100              for i < prompt_len - 1
        #   labels[prompt_len - 1] = text_tokens[0]
        #   labels[prompt_len + j] = text_tokens[j + 1]   for j in [0, n_text-2]
        #   labels[last] = -100  (no next-token to predict, will be sliced off)
        seq_len = inputs_embeds.shape[0]
        labels = [-100] * seq_len
        # First text label goes one position before the first text input embed
        if prompt_len - 1 >= 0:
            labels[prompt_len - 1] = int(text_tokens[0])
        for j in range(1, n_text):
            pos = prompt_len + j - 1
            if pos < seq_len:
                labels[pos] = int(text_tokens[j])

        return {
            "inputs_embeds": inputs_embeds,
            "labels": mx.array(labels, dtype=mx.int32),
        }


class STTSFTTrainer:
    """
    Supervised Fine-Tuning Trainer for STT (Whisper) models.

    Uses a custom seq2seq training loop:
    1. Compute mel spectrogram from audio
    2. Encode mel through Whisper encoder
    3. Decode with teacher forcing
    4. Cross-entropy loss on decoder output
    """

    def __init__(
        self,
        model: Any,
        processor: Any = None,
        data_collator: Any = None,
        train_dataset: Any = None,
        args: Any = None,
        **kwargs,
    ):
        _require_mlx_audio()

        self.wrapper = model if isinstance(model, STTModelWrapper) else None
        self.actual_model = model.model if self.wrapper else model
        self.processor = processor or (model.processor if self.wrapper else None)
        self.data_collator = data_collator
        self.train_dataset = train_dataset

        # Parse training args
        if args is not None:
            self.learning_rate = getattr(args, "learning_rate", 1e-5)
            self.max_steps = getattr(args, "max_steps", None)
            self.num_train_epochs = getattr(args, "num_train_epochs", 1)
            self.batch_size = getattr(args, "per_device_train_batch_size", 1)
            self.gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 4)
            self.warmup_steps = getattr(args, "warmup_steps", 5)
            self.logging_steps = getattr(args, "logging_steps", 1)
            self.output_dir = getattr(args, "output_dir", "./stt_outputs")
            self.lr_scheduler_type = getattr(args, "lr_scheduler_type", "linear")
            self.weight_decay = getattr(args, "weight_decay", 0.01)
            self.seed = getattr(args, "seed", 3407)
        else:
            self.learning_rate = kwargs.get("learning_rate", 1e-5)
            self.max_steps = kwargs.get("max_steps", None)
            self.num_train_epochs = kwargs.get("num_train_epochs", 1)
            self.batch_size = kwargs.get("batch_size", 1)
            self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 4)
            self.warmup_steps = kwargs.get("warmup_steps", 5)
            self.logging_steps = kwargs.get("logging_steps", 1)
            self.output_dir = kwargs.get("output_dir", "./stt_outputs")
            self.lr_scheduler_type = kwargs.get("lr_scheduler_type", "linear")
            self.weight_decay = kwargs.get("weight_decay", 0.01)
            self.seed = kwargs.get("seed", 3407)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def train(self):
        """
        Train the Whisper model using seq2seq training loop.

        Returns:
            _TrainerStats with training metrics
        """
        import mlx.optimizers as optim
        from mlx.utils import tree_map
        from tqdm import tqdm

        print("=" * 70)
        print("Starting STT Fine-Tuning (Whisper)")
        print("=" * 70)

        # Ensure LoRA is applied
        if self.wrapper and self.wrapper.lora_enabled and not self.wrapper._lora_applied:
            self.wrapper._apply_lora()

        # For audio-LLM wrapper models (Qwen3-ASR, Voxtral), use inner model
        # directly so MLX's tree_map works correctly with optimizer
        _profile = self.wrapper.profile if self.wrapper else None
        _arch = _profile.architecture if _profile else "encoder_decoder"
        train_model = self.actual_model
        if _arch == "audio_llm":
            _inner = getattr(self.actual_model, "_model", None)
            if _inner is not None:
                train_model = _inner

        # Set training mode
        train_model.train()

        # Optimizer
        optimizer = optim.Adam(learning_rate=self.learning_rate)

        # Determine total steps
        dataset_len = len(self.train_dataset) if hasattr(self.train_dataset, "__len__") else 0
        if self.max_steps:
            total_steps = self.max_steps
        elif dataset_len > 0:
            total_steps = (dataset_len // self.batch_size) * self.num_train_epochs
        else:
            total_steps = 100

        print(f"  Model: {self.wrapper.model_name if self.wrapper else 'unknown'}")
        print(f"  Dataset: {dataset_len} samples")
        print(f"  Total steps: {total_steps}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  Learning rate: {self.learning_rate}")

        # Detect model architecture for forward pass dispatch
        _model_name = _profile.name if _profile else "whisper"

        # Seq2seq / audio-LLM / voxtral_realtime loss function
        def loss_fn(model, batch):
            labels = batch["labels"]

            if _arch == "voxtral_realtime":
                # Voxtral Realtime: forward pre-built embeddings through the
                # decoder (single sequence, no KV cache for training).
                embeds = batch["inputs_embeds"]
                h, _ = model.decoder.forward(embeds, start_pos=0, cache=None)
                logits = model.decoder.logits(h)  # [seq, vocab]
                # Compute cross entropy directly: prefix labels are -100 so the
                # masked-mean below handles ignoring them. No reshape needed.
                vocab_size = logits.shape[-1]
                ce = nn.losses.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    labels.reshape(-1),
                    reduction="none",
                ).reshape(labels.shape)
                mask = (labels != -100).astype(ce.dtype)
                return (ce * mask).sum() / mx.maximum(mask.sum(), 1)

            if _arch == "audio_llm":
                # Audio-LLM models (Qwen3-ASR, Voxtral): model(input_ids, input_features=audio)
                input_ids = batch["input_ids"]
                input_features = batch.get("input_features")
                fwd_kwargs = {}
                if "feature_attention_mask" in batch:
                    fwd_kwargs["feature_attention_mask"] = batch["feature_attention_mask"]
                logits = model(input_ids, input_features=input_features, **fwd_kwargs)
            elif _model_name == "moonshine":
                # Moonshine: encoder(audio) -> decoder(tokens, encoder_out)
                mel = batch["input_features"]
                decoder_input_ids = batch["decoder_input_ids"]
                encoder_out = model.encoder(mel)
                decoder_out, _ = model.decoder(decoder_input_ids, encoder_out)
                if hasattr(model, "proj_out"):
                    logits = model.proj_out(decoder_out)
                else:
                    logits = model.decoder.embed_tokens.as_linear(decoder_out)
            elif _model_name == "canary":
                # Canary: encoder(mel, lengths) -> decoder(tokens, enc_out)
                mel = batch["input_features"]
                decoder_input_ids = batch["decoder_input_ids"]
                enc_lengths = mx.array([mel.shape[1]])
                enc_out, enc_lengths_out = model.encoder(mel, enc_lengths)
                enc_mask = mx.ones((1, enc_out.shape[1]), dtype=mx.bool_)
                logits, _ = model.decoder(decoder_input_ids, enc_out, encoder_mask=enc_mask)
            else:
                # Whisper: model(mel, decoder_input_ids) -> logits
                mel = batch["input_features"]
                decoder_input_ids = batch["decoder_input_ids"]
                logits = model(mel, decoder_input_ids)

            # Handle models that return objects
            if hasattr(logits, "logits"):
                logits = logits.logits

            # Cross-entropy loss
            vocab_size = logits.shape[-1]
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, vocab_size),
                labels.reshape(-1),
                reduction="none",
            ).reshape(labels.shape)

            # Mask: only compute loss where labels != -100
            mask = (labels != -100).astype(loss.dtype)
            loss = (loss * mask).sum() / mx.maximum(mask.sum(), 1)
            return loss

        loss_and_grad_fn = nn.value_and_grad(train_model, loss_fn)

        grad_accum = self.gradient_accumulation_steps
        progress = tqdm(range(total_steps), desc="Training")
        total_loss = 0.0
        step = 0
        micro_step = 0
        accum_loss = 0.0
        accumulated_grads = None
        epoch = 0

        while step < total_steps:
            epoch += 1
            for i in range(0, max(dataset_len, 1), self.batch_size):
                if step >= total_steps:
                    break

                # Get batch
                if self.data_collator is not None:
                    batch_samples = self.train_dataset[i: i + self.batch_size]
                    if isinstance(batch_samples, dict):
                        batch_list = []
                        keys = list(batch_samples.keys())
                        num_items = len(batch_samples[keys[0]]) if keys else 0
                        for j in range(num_items):
                            batch_list.append({k: batch_samples[k][j] for k in keys})
                        batch = self.data_collator(batch_list)
                    else:
                        batch = self.data_collator(batch_samples)
                else:
                    batch = self.train_dataset[i]

                loss, grads = loss_and_grad_fn(train_model, batch)
                mx.eval(loss)

                # Accumulate gradients
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = tree_map(
                        lambda a, g: a + g, accumulated_grads, grads
                    )
                accum_loss += loss.item()
                micro_step += 1

                # Update weights every grad_accum micro-steps
                if micro_step >= grad_accum:
                    averaged_grads = tree_map(
                        lambda g: g / grad_accum, accumulated_grads
                    )
                    optimizer.update(train_model, averaged_grads)
                    mx.eval(train_model, optimizer.state)

                    loss_val = accum_loss / grad_accum
                    total_loss += loss_val
                    step += 1
                    micro_step = 0
                    accum_loss = 0.0
                    accumulated_grads = None

                    progress.update(1)
                    if step % self.logging_steps == 0:
                        avg_loss = total_loss / step
                        progress.set_postfix(
                            {"loss": f"{loss_val:.4f}", "avg_loss": f"{avg_loss:.4f}"}
                        )

        progress.close()

        # Save adapters
        if self.wrapper:
            adapter_dir = Path(self.output_dir) / "adapters"
            self.wrapper.save_pretrained(str(adapter_dir))
            self.wrapper._adapter_path = adapter_dir

        avg_loss = total_loss / max(step, 1)
        print(f"\nTraining complete! Average loss: {avg_loss:.4f}")

        return _TrainerStats({"train_loss": avg_loss, "train_runtime": 0})


class _TrainerStats:
    """Simple container for training metrics."""

    def __init__(self, metrics: Dict):
        self.metrics = metrics
