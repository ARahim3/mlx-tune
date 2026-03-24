"""
Speech-to-Text (STT) Fine-Tuning Support for MLX-Tune

Provides Unsloth-compatible API for STT models on Apple Silicon using mlx-audio:
- Whisper (OpenAI) - Encoder-decoder model for speech recognition
- And other encoder-decoder STT models

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
                "Quantization is not currently supported for Whisper encoder-decoder models. "
                "Loading in default precision.",
                UserWarning,
            )

        model_dtype = mx.float16
        if dtype is not None:
            if dtype == "float32" or dtype == mx.float32:
                model_dtype = mx.float32
            elif dtype == "bfloat16" or dtype == mx.bfloat16:
                model_dtype = mx.bfloat16

        print(f"Loading STT model: {model_name}")
        model = _stt_load(model_name)

        # mlx_audio.stt.load() attaches a WhisperProcessor via post_load_hook
        hf_processor = getattr(model, "_processor", None)

        # If processor not found in model repo, try loading from openai/whisper-*
        if hf_processor is None:
            hf_processor = _try_load_whisper_processor(model_name)
            if hf_processor is not None:
                model._processor = hf_processor

        if hf_processor is None:
            raise ValueError(
                f"Could not load WhisperProcessor for '{model_name}'. "
                "The model repo must include processor files (preprocessor_config.json, tokenizer.json). "
                "Try a model like 'mlx-community/whisper-tiny-asr-fp16' or "
                "'mlx-community/whisper-large-v3-turbo' instead."
            )

        # Build our processor wrapper
        tokenizer = hf_processor.tokenizer

        processor = STTProcessor(tokenizer=tokenizer, model=model, hf_processor=hf_processor)

        # Auto-detect model profile
        profile_key = detect_stt_model_type(model_name, {})
        profile = STT_PROFILES.get(profile_key) if profile_key else None

        wrapper = STTModelWrapper(
            model=model,
            processor=processor,
            model_name=model_name,
            max_seq_length=max_seq_length,
            profile=profile,
        )

        n_enc = model.dims.n_audio_layer if hasattr(model, "dims") else "?"
        n_dec = model.dims.n_text_layer if hasattr(model, "dims") else "?"
        print(f"STT model loaded: {model_name}")
        print(f"  Encoder layers: {n_enc}, Decoder layers: {n_dec}")
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

        # Default target modules for Whisper attention layers
        if target_modules is None:
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
                 max_audio_samples: int = WHISPER_N_SAMPLES):
        self._raw_tokenizer = tokenizer  # HF WhisperTokenizer
        self._model = model
        self._hf_processor = hf_processor  # transformers.WhisperProcessor
        self._whisper_tokenizer = None  # HFTokenizerWrapper from model.get_tokenizer()
        self._max_audio_samples = max_audio_samples

        # Try to get the Whisper tokenizer wrapper (has sot_sequence, eot, etc.)
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

        # Model dimensions (from Whisper)
        if hasattr(model, "dims"):
            self.n_audio_layer = model.dims.n_audio_layer
            self.n_text_layer = model.dims.n_text_layer
            self.n_mels = model.dims.n_mels
            self.n_vocab = model.dims.n_vocab
        else:
            self.n_audio_layer = 0
            self.n_text_layer = 0
            self.n_mels = self.profile.n_mels
            self.n_vocab = 51865

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
                        block, target_modules, r, scale, dropout,
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
                        has_cross_attn=True,
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
                    if isinstance(original, nn.Linear):
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
                        if isinstance(original, nn.Linear):
                            lora_layer = _create_lora_linear(original, r, scale, dropout)
                            setattr(cross_attn_module, module_name, lora_layer)
                            replaced += 1

        # MLP layers (if "mlp1" or "mlp2" in target_modules)
        for mlp_name in ["mlp1", "mlp2"]:
            if mlp_name in target_modules and hasattr(block, mlp_name):
                original = getattr(block, mlp_name)
                if isinstance(original, nn.Linear):
                    lora_layer = _create_lora_linear(original, r, scale, dropout)
                    setattr(block, mlp_name, lora_layer)
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
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], fused)
                fused_count += 1

        if fused_count > 0:
            print(f"Fused {fused_count} LoRA layers into base Whisper model")

        # Save merged model weights
        from mlx.utils import tree_flatten
        weights = dict(tree_flatten(self.model.parameters()))
        mx.savez(str(output_path / "weights.npz"), **weights)

        # Save model config
        if hasattr(self.model, "dims"):
            import dataclasses
            dims_dict = dataclasses.asdict(self.model.dims)
            with open(output_path / "config.json", "w") as f:
                json.dump(dims_dict, f, indent=2)

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


def _create_lora_linear(
    original: nn.Linear,
    r: int,
    scale: float,
    dropout: float,
) -> nn.Linear:
    """
    Replace an nn.Linear with a LoRALinear layer.

    Uses MLX's built-in LoRALinear if available, otherwise creates
    a custom implementation.
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
            Dict with 'input_features', 'labels', 'decoder_input_ids' as mx.arrays
        """
        if isinstance(samples, dict):
            samples = [samples]

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
        eot = self.processor.tokenizer.eot if hasattr(self.processor.tokenizer, "eot") else 50257

        padded_decoder_inputs = []
        padded_labels = []
        for dec_ids, labs in zip(all_decoder_inputs, all_labels):
            pad_len = max_dec_len - len(dec_ids)
            padded_decoder_inputs.append(dec_ids + [eot] * pad_len)
            padded_labels.append(labs + [-100] * pad_len)

        # Stack mel spectrograms (all are same size after pad_or_trim)
        mel_batch = mx.stack(all_mels)

        return {
            "input_features": mel_batch,
            "decoder_input_ids": mx.array(padded_decoder_inputs),
            "labels": mx.array(padded_labels),
        }

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
        else:
            # Default: log-mel spectrogram (Whisper)
            features = self.processor.compute_mel(audio_array, n_mels=self.model.n_mels)

        # Get text
        text_col = self._find_text_column(sample)
        text = sample[text_col]

        # Use language-specific tokenizer if available, otherwise default
        tokenizer = self._lang_tokenizer or self.processor.tokenizer
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

        # Set training mode
        self.actual_model.train()

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
        _is_moonshine = (self.wrapper and self.wrapper.profile
                         and self.wrapper.profile.name == "moonshine")

        # Seq2seq loss function
        def loss_fn(model, batch):
            mel = batch["input_features"]
            decoder_input_ids = batch["decoder_input_ids"]
            labels = batch["labels"]

            # Forward pass: dispatch based on model architecture
            if _is_moonshine:
                # Moonshine: encoder(audio) -> decoder(tokens, encoder_out)
                encoder_out = model.encoder(mel)
                decoder_out, _ = model.decoder(decoder_input_ids, encoder_out)
                # Project to vocab
                if hasattr(model, "proj_out"):
                    logits = model.proj_out(decoder_out)
                else:
                    logits = model.decoder.embed_tokens.as_linear(decoder_out)
            else:
                # Whisper: model(mel, decoder_input_ids) -> logits
                logits = model(mel, decoder_input_ids)

            # Handle models that return objects
            if hasattr(logits, "logits"):
                logits = logits.logits

            # Cross-entropy loss (no shift needed - decoder already shifted)
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

        loss_and_grad_fn = nn.value_and_grad(self.actual_model, loss_fn)

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

                loss, grads = loss_and_grad_fn(self.actual_model, batch)
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
                    optimizer.update(self.actual_model, averaged_grads)
                    mx.eval(self.actual_model, optimizer.state)

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
