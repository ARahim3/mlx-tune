"""
Text-to-Speech (TTS) Fine-Tuning Support for MLX-Tune

Provides Unsloth-compatible API for TTS models on Apple Silicon using mlx-audio:
- Orpheus-TTS (3B) - Llama-based, decoder-only, uses SNAC audio codec
- And other decoder-only TTS models that tokenize audio to discrete codes

Usage (matches Unsloth patterns):
    from mlx_tune import FastTTSModel, TTSSFTTrainer, TTSSFTConfig, TTSDataCollator

    model, tokenizer = FastTTSModel.from_pretrained(
        "mlx-community/orpheus-3b-0.1-ft-bf16",
    )
    model = FastTTSModel.get_peft_model(model, r=16, lora_alpha=16)
"""

from typing import Optional, Any, List, Dict, Union, Tuple
from pathlib import Path
import warnings
import json
import numpy as np

import mlx.core as mx

from mlx_tune.audio_profiles import (
    TTSModelProfile,
    TTS_PROFILES,
    detect_tts_model_type,
)
from mlx_tune.audio_codecs import CodecAdapter, SNACCodecAdapter, create_codec

# Try to import mlx_lm for model loading and LoRA
try:
    from mlx_lm import load as mlx_load
    from mlx_lm import generate as mlx_generate
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False

try:
    from mlx_lm.tuner.utils import linear_to_lora_layers
    HAS_MLX_LM_TUNER = True
except ImportError:
    HAS_MLX_LM_TUNER = False

# Try to import mlx-audio for SNAC codec
HAS_MLX_AUDIO = False
try:
    from mlx_audio.codec.models.snac import SNAC
    HAS_MLX_AUDIO = True
except ImportError:
    pass

# Try to import mlx-audio TTS loader (for OuteTTS, Spark, Sesame)
HAS_MLX_AUDIO_TTS = False
_tts_load_fn = None
try:
    from mlx_audio.tts import load as _tts_load_fn
    HAS_MLX_AUDIO_TTS = True
except ImportError:
    pass


def _require_mlx_audio():
    """Check that mlx-audio is available."""
    if not HAS_MLX_AUDIO:
        raise ImportError(
            "mlx-audio is required for TTS model support. "
            "Install with: uv pip install 'mlx-tune[audio]'"
        )


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


# Orpheus-specific constants
# These are configurable but default to Orpheus-3B values
ORPHEUS_START_TOKEN = 128259  # <custom_token_3>
ORPHEUS_END_TOKENS = [128009, 128260]  # <|eot_id|>, <custom_token_4>
ORPHEUS_AUDIO_TOKEN_OFFSET = 128266  # Audio codes start after this offset
ORPHEUS_CODEBOOK_SIZE = 4096


class FastTTSModel:
    """
    Unsloth-compatible API for TTS models on Apple Silicon.

    Provides the same API patterns as FastLanguageModel / FastVisionModel
    but specialized for text-to-speech models that use audio codecs.

    Example:
        >>> from mlx_tune import FastTTSModel
        >>> model, tokenizer = FastTTSModel.from_pretrained(
        ...     "mlx-community/orpheus-3b-0.1-ft-bf16",
        ... )
        >>> model = FastTTSModel.get_peft_model(model, r=16, lora_alpha=16)
    """

    @staticmethod
    def from_pretrained(
        model_name: str,
        max_seq_length: Optional[int] = 2048,
        dtype: Optional[Any] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        codec_model: str = "mlx-community/snac_24khz",
        token: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> Tuple["TTSModelWrapper", Any]:
        """
        Load a pretrained TTS model with SNAC audio codec.

        Args:
            model_name: HuggingFace model ID (e.g., "mlx-community/orpheus-3b-0.1-ft-bf16")
            max_seq_length: Maximum sequence length
            dtype: Data type override
            load_in_4bit: Use 4-bit quantization (not recommended for TTS)
            load_in_8bit: Use 8-bit quantization
            codec_model: SNAC codec model ID (default: 24kHz for Orpheus)
            token: HuggingFace API token
            trust_remote_code: Trust remote code
            **kwargs: Additional arguments

        Returns:
            Tuple of (TTSModelWrapper, tokenizer)
        """
        _require_mlx_audio()

        if load_in_4bit:
            warnings.warn(
                "4-bit quantization is not recommended for TTS models. "
                "Audio quality may be significantly degraded.",
                UserWarning,
            )

        # Auto-detect model profile from name
        profile_key = detect_tts_model_type(model_name, {})
        profile = TTS_PROFILES.get(profile_key) if profile_key else None

        # Determine codec model from profile if not explicitly provided
        if codec_model == "mlx-community/snac_24khz" and profile and profile.codec_repo:
            codec_model = profile.codec_repo

        print(f"Loading TTS model: {model_name}")

        # Dispatch on loader type
        if profile and profile.loader == "mlx_audio_tts":
            # Load via mlx-audio's TTS loader (OuteTTS, Spark, Sesame)
            if not HAS_MLX_AUDIO_TTS:
                raise ImportError(
                    "mlx-audio TTS loader is required for this model. "
                    "Install with: uv pip install 'mlx-tune[audio]'"
                )

            full_model = _tts_load_fn(model_name)

            # Get inner LM for LoRA (e.g., model.model for OuteTTS/Spark)
            inner_model = full_model
            if profile.inner_model_attr:
                inner_model = getattr(full_model, profile.inner_model_attr, full_model)

            # Get tokenizer from inner model or full model
            tokenizer = getattr(inner_model, "tokenizer", None)
            if tokenizer is None:
                tokenizer = getattr(full_model, "tokenizer", None)
            if tokenizer is None and HAS_MLX_LM:
                # Fallback: try loading tokenizer via mlx_lm
                from transformers import AutoTokenizer
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
                except Exception:
                    pass

            # The codec may be bundled with the model under different attributes
            codec = getattr(full_model, "codec", None)
            if codec is None:
                codec = getattr(full_model, "_audio_tokenizer", None)
            if codec is None:
                codec = getattr(full_model, "audio_tokenizer", None)
            if codec is None:
                # Qwen3-TTS: speech_tokenizer is the built-in codec
                codec = getattr(full_model, "speech_tokenizer", None)
            if codec is None:
                # Try to get codec from model-specific audio processors
                if profile.codec_type == "dac":
                    try:
                        from mlx_audio.tts.models.outetts.audio_processor import AudioProcessor
                        ap = AudioProcessor()
                        codec = ap.audio_codec
                    except ImportError:
                        codec = None
                elif profile.codec_repo:
                    try:
                        codec = SNAC.from_pretrained(profile.codec_repo)
                    except Exception:
                        codec = None

            # Qwen3-TTS: validate speech tokenizer encoder is available for training
            if profile.codec_type == "qwen3_speech" and codec is not None:
                has_encoder = getattr(codec, "has_encoder", False) or getattr(codec, "encoder_model", None) is not None
                if not has_encoder:
                    warnings.warn(
                        "Qwen3-TTS speech tokenizer encoder not found. "
                        "Audio encoding for training may not work. "
                        "Use a model variant that includes the encoder (e.g., base model).",
                        UserWarning,
                    )

            config = {}
            if hasattr(inner_model, "config"):
                cfg = inner_model.config
                config = cfg if isinstance(cfg, dict) else {}

            model = inner_model

        else:
            # Default: load via mlx_lm (Orpheus and other decoder-only LLMs)
            if not HAS_MLX_LM:
                raise ImportError(
                    "mlx-lm is required for TTS model loading. "
                    "Install with: uv pip install mlx-lm"
                )

            model, tokenizer = mlx_load(model_name)

            # Load SNAC audio codec
            print(f"Loading audio codec: {codec_model}")
            codec = SNAC.from_pretrained(codec_model)

            config = {}
            if hasattr(model, "config"):
                config = model.config if isinstance(model.config, dict) else {}

        wrapper = TTSModelWrapper(
            model=model,
            tokenizer=tokenizer,
            codec=codec,
            model_name=model_name,
            codec_model_name=codec_model,
            max_seq_length=max_seq_length,
            config=config,
            profile=profile,
            full_model=full_model if (profile and profile.loader == "mlx_audio_tts") else None,
        )

        sr = profile.sample_rate if profile else getattr(codec, "sampling_rate", 24000)
        codec_name = profile.codec_type if profile else codec_model
        print(f"TTS model loaded: {model_name}")
        print(f"Audio codec: {codec_name} (sample rate: {sr}Hz)")

        return wrapper, tokenizer

    @staticmethod
    def get_peft_model(
        model: "TTSModelWrapper",
        r: int = 16,
        target_modules: Optional[List[str]] = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        bias: str = "none",
        use_gradient_checkpointing: Union[bool, str] = False,
        random_state: int = 3407,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> "TTSModelWrapper":
        """
        Add LoRA adapters to TTS model.

        Args:
            model: TTSModelWrapper from from_pretrained()
            r: LoRA rank
            target_modules: Target modules (defaults to attention + MLP)
            lora_alpha: LoRA alpha scaling
            lora_dropout: LoRA dropout rate
            bias: Bias configuration
            use_gradient_checkpointing: Gradient checkpointing mode
            random_state: Random seed
            use_rslora: Use rank-stabilized LoRA
            use_dora: Use weight-decomposed LoRA
            **kwargs: Additional configuration

        Returns:
            TTSModelWrapper with LoRA configured
        """
        if not isinstance(model, TTSModelWrapper):
            raise TypeError(
                f"Expected TTSModelWrapper, got {type(model)}. "
                "Use FastTTSModel.from_pretrained() first."
            )

        # Default target modules for Llama-based TTS (Orpheus)
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

        model.configure_lora(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
            use_dora=use_dora,
        )

        return model

    @staticmethod
    def for_training(model: "TTSModelWrapper") -> "TTSModelWrapper":
        """Enable training mode."""
        if isinstance(model, TTSModelWrapper):
            model.inference_mode = False
            model.model.train()
        return model

    @staticmethod
    def for_inference(model: "TTSModelWrapper") -> "TTSModelWrapper":
        """Enable inference mode."""
        if isinstance(model, TTSModelWrapper):
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
        Convert a HuggingFace TTS model to MLX format.

        TTS models like Orpheus are decoder-only LLMs, so this uses
        mlx_lm.convert() under the hood.

        Args:
            hf_model: HuggingFace model ID (e.g., "canopylabs/orpheus-3b-0.1-ft")
            output_dir: Output directory for MLX model
            quantize: Whether to quantize the model
            q_bits: Quantization bits (4, 8)
            dtype: Data type ("float16", "bfloat16", "float32")
            upload_repo: Optional HF repo to upload converted model
        """
        try:
            from mlx_lm import convert
        except ImportError:
            raise ImportError("mlx-lm is required for model conversion: uv pip install mlx-lm")

        print(f"Converting TTS model: {hf_model} -> {output_dir}")
        convert(
            hf_path=hf_model,
            mlx_path=output_dir,
            quantize=quantize,
            q_bits=q_bits,
            dtype=dtype,
            upload_repo=upload_repo,
        )
        print(f"Conversion complete: {output_dir}")


class TTSModelWrapper:
    """
    Wraps a decoder-only TTS model (e.g., Orpheus) + SNAC audio codec.

    Handles audio tokenization (encode/decode via SNAC) and provides
    LoRA management, training, and inference methods.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        codec: Any,
        model_name: str,
        codec_model_name: str = "mlx-community/snac_24khz",
        max_seq_length: int = 2048,
        config: Optional[Dict] = None,
        # Audio token configuration (defaults for Orpheus)
        start_token: int = ORPHEUS_START_TOKEN,
        end_tokens: Optional[List[int]] = None,
        audio_token_offset: int = ORPHEUS_AUDIO_TOKEN_OFFSET,
        codebook_size: int = ORPHEUS_CODEBOOK_SIZE,
        # Profile-based configuration (optional, overrides above if provided)
        profile: Optional[TTSModelProfile] = None,
        # Full model reference for models that need it (Qwen3-TTS)
        full_model: Any = None,
    ):
        # Resolve profile: use provided, or fall back to Orpheus default
        if profile is None:
            profile = TTS_PROFILES["orpheus"]
        self.profile = profile

        self.model = model
        self.tokenizer = tokenizer
        self.codec = codec
        self.full_model = full_model  # For Qwen3-TTS: the complete Model with speech_tokenizer etc.
        self.model_name = model_name
        self.codec_model_name = codec_model_name
        self.max_seq_length = max_seq_length
        self.config = config or {}

        # Audio token configuration - profile values as defaults,
        # explicit params override (for backward compat)
        self.start_token = start_token
        self.end_tokens = end_tokens or list(self.profile.end_tokens)
        self.audio_token_offset = audio_token_offset
        self.codebook_size = codebook_size

        # Create codec adapter for encode/decode delegation
        self.codec_adapter: CodecAdapter = create_codec(self.profile, codec)

        # LoRA state
        self.lora_config = None
        self.lora_enabled = False
        self._lora_applied = False
        self._adapter_path: Optional[Path] = None

        # Mode
        self.inference_mode = False

    @property
    def sample_rate(self) -> int:
        """Audio sample rate from codec."""
        if self.profile.codec_type == "qwen3_speech":
            return self.profile.sample_rate
        return getattr(self.codec, "sampling_rate", 24000)

    def configure_lora(
        self,
        r: int = 16,
        target_modules: Optional[List[str]] = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        bias: str = "none",
        use_gradient_checkpointing: Union[bool, str] = False,
        random_state: int = 3407,
        use_dora: bool = False,
        **kwargs,
    ):
        """Configure LoRA parameters."""
        self.lora_config = {
            "r": r,
            "target_modules": target_modules or [],
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "bias": bias,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "random_state": random_state,
            "use_dora": use_dora,
            **kwargs,
        }
        self.lora_enabled = True
        self._lora_applied = False

        print(
            f"LoRA configuration set: rank={r}, alpha={lora_alpha}, "
            f"modules={target_modules}, dropout={lora_dropout}"
        )

    def _apply_lora(self, num_layers: Optional[int] = None) -> bool:
        """
        Apply LoRA adapters to model layers using mlx_lm's native API.

        Args:
            num_layers: Number of transformer layers to apply LoRA to.

        Returns:
            True if LoRA was applied successfully.
        """
        if not self.lora_enabled:
            print("LoRA not configured. Call configure_lora() first.")
            return False

        if self._lora_applied:
            return False

        if not HAS_MLX_LM_TUNER:
            raise RuntimeError(
                "mlx_lm.tuner is not available. Install with: uv pip install mlx-lm"
            )

        # Detect number of layers
        if num_layers is None:
            if hasattr(self.model, "layers"):
                num_layers = len(self.model.layers)
            elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                num_layers = len(self.model.model.layers)
            else:
                raise ValueError(
                    "Could not detect number of layers. Specify num_layers explicitly."
                )

        r = self.lora_config["r"]
        lora_alpha = self.lora_config["lora_alpha"]
        scale = lora_alpha / r

        mlx_lora_config = {
            "rank": r,
            "scale": scale,
            "dropout": self.lora_config.get("lora_dropout", 0.0),
        }

        # Map short module names to full paths using profile mapping
        target_modules = self.lora_config.get("target_modules", [])
        if target_modules:
            short_to_full = dict(self.profile.lora_module_mapping)
            full_paths = []
            for module in target_modules:
                full_paths.append(short_to_full.get(module, module))
            mlx_lora_config["keys"] = full_paths

        use_dora = self.lora_config.get("use_dora", False)

        print(f"Applying LoRA to {num_layers} layers: {mlx_lora_config}")

        # Freeze base model, then apply LoRA
        self.model.freeze()
        # Qwen3-TTS: layers are at talker.model.layers, not talker.layers
        lora_target = self.model
        if self.profile and self.profile.codec_type == "qwen3_speech":
            lora_target = self.model.model
        linear_to_lora_layers(
            model=lora_target,
            num_layers=num_layers,
            config=mlx_lora_config,
            use_dora=use_dora,
        )
        self._lora_applied = True

        # Verify trainable parameters
        from mlx.utils import tree_flatten

        trainable = tree_flatten(self.model.trainable_parameters())
        lora_params = [k for k, _ in trainable if "lora" in k]
        print(f"LoRA applied: {len(lora_params)} trainable parameter groups across {num_layers} layers")

        return True

    def encode_audio(self, audio: Union[np.ndarray, mx.array], sr: int = 24000) -> List[int]:
        """
        Encode audio waveform to discrete token IDs using SNAC codec.

        The encoding produces hierarchical VQ codes at 3 temporal resolutions,
        which are interleaved into a flat token sequence for the language model.

        Args:
            audio: Audio waveform as numpy array or mx.array
            sr: Sample rate of input audio (must match codec)

        Returns:
            List of audio token IDs (with codec offset applied)
        """
        return self.codec_adapter.encode(audio, sr=sr)

    def _interleave_codes(self, codes: List[mx.array]) -> List[int]:
        """
        Interleave hierarchical VQ codes into a flat token sequence.
        Delegates to codec adapter.
        """
        return self.codec_adapter.interleave(codes)

    def _deinterleave_codes(self, token_ids: List[int]) -> List[np.ndarray]:
        """
        De-interleave flat token sequence back to hierarchical VQ codes.
        Delegates to codec adapter.
        """
        return self.codec_adapter.deinterleave(token_ids)

    def decode_audio(self, token_ids: List[int]) -> np.ndarray:
        """
        Decode discrete token IDs back to audio waveform.

        Args:
            token_ids: List of audio token IDs (with codec offset)

        Returns:
            Audio waveform as numpy array
        """
        return self.codec_adapter.decode(token_ids)

    def generate(
        self,
        text: str,
        speaker: Optional[str] = None,
        max_tokens: int = 1250,
        temperature: float = 0.6,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate speech from text.

        Args:
            text: Input text to synthesize
            speaker: Speaker name/ID (if model supports multiple speakers)
            max_tokens: Maximum audio tokens to generate (~10s at 1250)
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling
            repetition_penalty: Repetition penalty
            **kwargs: Additional generation arguments

        Returns:
            Audio waveform as numpy array at codec sample rate
        """
        # Build prompt
        prompt = self._build_tts_prompt(text, speaker)

        # Generate tokens
        response = mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Extract audio tokens from response
        audio_tokens = self._extract_audio_tokens(response)

        if not audio_tokens:
            warnings.warn("No audio tokens generated. Try increasing max_tokens.")
            return np.zeros(0)

        # Decode to waveform
        return self.decode_audio(audio_tokens)

    def _build_tts_prompt(self, text: str, speaker: Optional[str] = None) -> str:
        """Build the TTS prompt in Orpheus format."""
        speaker_str = speaker or self.profile.default_speaker
        # Use profile's prompt template
        prompt = self.profile.prompt_template.format(speaker=speaker_str, text=text)
        return prompt

    def _extract_audio_tokens(self, generated_text: str) -> List[int]:
        """Extract audio token IDs from generated text/tokens."""
        # Re-tokenize the generated text to get token IDs
        token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)

        # Filter to audio tokens (those >= audio_token_offset)
        audio_tokens = [t for t in token_ids if t >= self.audio_token_offset]
        return audio_tokens

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
                "codec_model": self.codec_model_name,
                "model_type": "tts",
                "fine_tune_type": "lora",
                "lora_parameters": {
                    "rank": self.lora_config["r"],
                    "alpha": self.lora_config["lora_alpha"],
                    "dropout": self.lora_config.get("lora_dropout", 0.0),
                    "scale": self.lora_config["lora_alpha"] / self.lora_config["r"],
                    "keys": self.lora_config.get("target_modules", []),
                },
                "audio_config": {
                    "start_token": self.start_token,
                    "end_tokens": self.end_tokens,
                    "audio_token_offset": self.audio_token_offset,
                    "codebook_size": self.codebook_size,
                    "sample_rate": self.sample_rate,
                },
            }
            with open(output_path / "adapter_config.json", "w") as f:
                json.dump(adapter_config, f, indent=2)

            print(f"TTS adapters saved to: {output_path}")
        else:
            print("No LoRA adapters to save (LoRA not applied)")

    def save_pretrained_merged(
        self,
        output_dir: str,
        tokenizer: Any = None,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Fuse LoRA weights into base model and save the merged model.

        Args:
            output_dir: Directory to save merged model
            tokenizer: Tokenizer to save alongside model (uses self.tokenizer if None)
            push_to_hub: Whether to upload to HuggingFace Hub
            repo_id: HuggingFace repo ID (required if push_to_hub=True)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        tokenizer = tokenizer or self.tokenizer

        if not self._lora_applied:
            warnings.warn("LoRA not applied — saving base model as-is.")

        # Fuse LoRA layers into base weights
        fused_count = 0
        for name, module in self.model.named_modules():
            if hasattr(module, "fuse"):
                fused = module.fuse(dequantize=kwargs.get("dequantize", False))
                # Replace in model — handle both attribute and index access
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
            print(f"Fused {fused_count} LoRA layers into base model")

        # Save merged model
        from mlx_lm.utils import save_model
        save_model(str(output_path), self.model)

        # Save tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(str(output_path))

        # Save model config
        if self.config:
            with open(output_path / "config.json", "w") as f:
                json.dump(self.config, f, indent=2)

        print(f"Merged TTS model saved to: {output_path}")

        if push_to_hub and repo_id:
            _push_to_hub(str(output_path), repo_id)

    def push_to_hub(self, repo_id: str, **kwargs):
        """
        Push saved adapters or merged model to HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo ID (e.g., "username/my-tts-model")
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

            # Restore audio config if present
            audio_cfg = adapter_config.get("audio_config", {})
            if audio_cfg:
                self.start_token = audio_cfg.get("start_token", self.start_token)
                self.end_tokens = audio_cfg.get("end_tokens", self.end_tokens)
                self.audio_token_offset = audio_cfg.get("audio_token_offset", self.audio_token_offset)
                self.codebook_size = audio_cfg.get("codebook_size", self.codebook_size)

        # Load weights
        weights_path = adapter_dir / "adapters.safetensors"
        if weights_path.exists():
            from mlx.utils import tree_unflatten
            import safetensors.numpy

            weights_np = safetensors.numpy.load_file(str(weights_path))
            weights_mx = {k: mx.array(v) for k, v in weights_np.items()}

            # Apply LoRA first if not already done
            if not self._lora_applied and self.lora_enabled:
                self._apply_lora()

            # Load weights into model (strict=False: adapter has only LoRA params, not full model)
            self.model.load_weights(list(weights_mx.items()), strict=False)
            mx.eval(self.model)

            self._adapter_path = adapter_dir
            print(f"TTS adapters loaded from: {adapter_dir}")
        else:
            raise FileNotFoundError(f"No adapters found at {weights_path}")


class TTSSFTConfig:
    """
    Training configuration for TTS fine-tuning.

    Mirrors SFTConfig / VLMSFTConfig for API compatibility.
    """

    def __init__(
        self,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 5,
        max_steps: Optional[int] = 60,
        num_train_epochs: int = 1,
        learning_rate: float = 2e-4,
        logging_steps: int = 1,
        output_dir: str = "./tts_outputs",
        lr_scheduler_type: str = "linear",
        weight_decay: float = 0.01,
        seed: int = 3407,
        max_seq_length: int = 2048,
        sample_rate: int = 24000,
        train_on_completions: bool = True,
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
        self.max_seq_length = max_seq_length
        self.sample_rate = sample_rate
        self.train_on_completions = train_on_completions

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class TTSDataCollator:
    """
    Data collator for TTS fine-tuning.

    Processes audio+text pairs into training sequences:
    1. Tokenize text prompt
    2. Encode audio through SNAC codec to discrete tokens
    3. Build sequence: [text_tokens, START_AUDIO, audio_tokens, END_AUDIO]
    4. Create labels with prompt masking (only train on audio tokens)
    """

    def __init__(
        self,
        model: TTSModelWrapper,
        tokenizer: Any,
        max_seq_length: int = 2048,
        text_column: str = "text",
        audio_column: str = "audio",
        speaker_column: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.text_column = text_column
        self.audio_column = audio_column
        self.speaker_column = speaker_column

    def __call__(self, samples: Union[List[Dict], Dict]) -> Dict:
        """
        Collate a batch of audio+text samples into training tensors.

        Args:
            samples: List of dicts with 'text' and 'audio' keys,
                    or a single dict (for indexing from HF dataset)

        Returns:
            Dict with 'input_ids'/'inputs_embeds' and 'labels' as mx.arrays
        """
        if isinstance(samples, dict):
            samples = [samples]

        profile = self.model.profile

        # Qwen3-TTS: returns inputs_embeds (pre-computed embeddings) instead of input_ids
        if profile.codec_type == "qwen3_speech":
            return self._collate_qwen3_tts(samples)

        from mlx_tune._perf import bucket_length

        all_input_ids = []
        all_labels = []

        for sample in samples:
            input_ids, labels = self._process_sample(sample)
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # Pad to same length within batch, rounded up to the next bucket so
        # `mx.compile`'s per-shape cache hits across batches.
        max_len = min(
            bucket_length(max(len(ids) for ids in all_input_ids)),
            self.max_seq_length,
        )

        padded_input_ids = []
        padded_labels = []
        for ids, labs in zip(all_input_ids, all_labels):
            ids = ids[:max_len]
            labs = labs[:max_len]
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.tokenizer.pad_token_id or 0] * pad_len
                labs = labs + [-100] * pad_len
            padded_input_ids.append(ids)
            padded_labels.append(labs)

        return {
            "input_ids": mx.array(padded_input_ids),
            "labels": mx.array(padded_labels),
        }

    def _collate_qwen3_tts(self, samples: List[Dict]) -> Dict:
        """Collate Qwen3-TTS samples into inputs_embeds + labels."""
        from mlx_tune._perf import bucket_length

        all_embeds = []
        all_labels = []

        for sample in samples:
            embeds, labels = self._process_sample_qwen3(sample)
            all_embeds.append(embeds)
            all_labels.append(labels)

        # Pad to same length within batch, rounded up to the next bucket.
        max_len = min(
            bucket_length(max(e.shape[0] for e in all_embeds)),
            self.max_seq_length,
        )
        hidden_size = all_embeds[0].shape[-1]

        padded_embeds = []
        padded_labels = []
        for emb, labs in zip(all_embeds, all_labels):
            emb = emb[:max_len]
            labs = labs[:max_len]
            pad_len = max_len - emb.shape[0]
            if pad_len > 0:
                pad = mx.zeros((pad_len, hidden_size), dtype=emb.dtype)
                emb = mx.concatenate([emb, pad], axis=0)
                labs = labs + [-100] * pad_len
            padded_embeds.append(emb)
            padded_labels.append(labs)

        return {
            "inputs_embeds": mx.stack(padded_embeds, axis=0),
            "labels": mx.array(padded_labels),
        }

    def _process_sample(self, sample: Dict) -> Tuple[List[int], List[int]]:
        """Process a single sample into input_ids and labels."""
        profile = self.model.profile

        # Get text
        text = sample.get(self.text_column, "")
        default_speaker = profile.default_speaker
        speaker = sample.get(self.speaker_column, default_speaker) if self.speaker_column else default_speaker

        # Get audio
        audio_data = sample.get(self.audio_column)
        if audio_data is None:
            raise ValueError(f"Sample missing '{self.audio_column}' column")

        # Handle HuggingFace Audio format
        if isinstance(audio_data, dict):
            audio_array = audio_data.get("array", audio_data.get("data"))
            sr = audio_data.get("sampling_rate", self.model.sample_rate)
        elif isinstance(audio_data, np.ndarray):
            audio_array = audio_data
            sr = self.model.sample_rate
        else:
            audio_array = np.array(audio_data)
            sr = self.model.sample_rate

        if audio_array is None:
            raise ValueError("Could not extract audio array from sample")

        audio_array = np.array(audio_array, dtype=np.float32)

        # Encode audio to codec indices
        audio_codes = self.model.encode_audio(audio_array, sr=sr)

        if profile.token_format == "text" and profile.audio_token_formats:
            # Text-token format: format codes as text tokens, tokenize everything together
            n_cb = profile.num_codebooks
            audio_text_parts = []

            if profile.codec_type == "bicodec":
                # BiCodec: sequential layout — first N are global, rest are semantic
                # The encoder returns [global_0..global_N, semantic_0..semantic_M]
                n_global = 32  # BiCodec always produces 32 global tokens
                for i, code in enumerate(audio_codes):
                    if i < n_global:
                        fmt = profile.audio_token_formats[0]
                    else:
                        fmt = profile.audio_token_formats[1]
                    audio_text_parts.append(fmt.format(code=code))
                # Insert structural markers for Spark prompt format
                audio_text = (
                    "<|start_global_token|>"
                    + "".join(audio_text_parts[:n_global])
                    + "<|end_global_token|>"
                    + "<|start_semantic_token|>"
                    + "".join(audio_text_parts[n_global:])
                )
            else:
                # Interleaved format (OuteTTS/DAC): cycle through codebooks
                for i, code in enumerate(audio_codes):
                    cb_idx = i % n_cb
                    fmt = profile.audio_token_formats[cb_idx]
                    audio_text_parts.append(fmt.format(code=code))
                audio_text = "".join(audio_text_parts)

            # Build full prompt using profile template
            prompt_text = profile.prompt_template.format(speaker=speaker, text=text)
            full_text = prompt_text + audio_text

            # Tokenize the full text
            input_ids = self.tokenizer.encode(full_text, add_special_tokens=True)

            # Labels: mask prompt tokens, train on audio portion
            prompt_only_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
            num_masked = len(prompt_only_ids)
            labels = [-100] * num_masked + input_ids[num_masked:]

        else:
            # Numeric-token format (Orpheus, Sesame): offset-based token IDs
            prompt_text = profile.prompt_template.format(speaker=speaker, text=text)
            text_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)

            # Build full sequence: [text_tokens, START_TOKEN, audio_tokens, END_TOKENS]
            start_tokens = [self.model.start_token] if self.model.start_token else []
            input_ids = text_tokens + start_tokens + audio_codes + self.model.end_tokens

            # Labels: mask text tokens, only train on audio + end tokens
            num_masked = len(text_tokens) + len(start_tokens)
            labels = [-100] * num_masked + audio_codes + self.model.end_tokens

        return input_ids, labels

    def _process_sample_qwen3(self, sample: Dict) -> Tuple["mx.array", List[int]]:
        """
        Process a Qwen3-TTS sample into inputs_embeds and labels.

        Builds combined text+codec embeddings for teacher-forcing training:
        - Text conditioning via text_embedding + text_projection
        - Audio codes via codec_embedding (summed across all 16 codebooks)
        - Labels: code_0 tokens only (talker predicts first codebook)

        Returns:
            (inputs_embeds, labels) where inputs_embeds is [seq_len, hidden_size]
        """
        import mlx.nn as nn

        profile = self.model.profile
        talker = self.model.model  # Qwen3TTSTalkerForConditionalGeneration

        # Get text
        text = sample.get(self.text_column, "")

        # Get audio
        audio_data = sample.get(self.audio_column)
        if audio_data is None:
            raise ValueError(f"Sample missing '{self.audio_column}' column")

        if isinstance(audio_data, dict):
            audio_array = audio_data.get("array", audio_data.get("data"))
            sr = audio_data.get("sampling_rate", self.model.sample_rate)
        elif isinstance(audio_data, np.ndarray):
            audio_array = audio_data
            sr = self.model.sample_rate
        else:
            audio_array = np.array(audio_data)
            sr = self.model.sample_rate

        if audio_array is None:
            raise ValueError("Could not extract audio array from sample")
        audio_array = np.array(audio_array, dtype=np.float32)

        # --- Encode audio to all 16 codebooks ---
        from mlx_tune.audio_codecs import Qwen3SpeechCodecAdapter
        adapter = self.model.codec_adapter
        if not isinstance(adapter, Qwen3SpeechCodecAdapter):
            raise TypeError("Expected Qwen3SpeechCodecAdapter for qwen3_speech codec type")
        all_codes = adapter.encode_all_codebooks(audio_array, sr)  # mx.array [16, T]
        T = all_codes.shape[1]

        # --- Get model config for special token IDs ---
        full_model = self.model.full_model
        if full_model is not None and hasattr(full_model, "config"):
            model_config = full_model.config
        else:
            model_config = None

        # Special token IDs from ModelConfig
        tts_bos_id = getattr(model_config, "tts_bos_token_id", 151672)
        tts_eos_id = getattr(model_config, "tts_eos_token_id", 151673)
        tts_pad_id = getattr(model_config, "tts_pad_token_id", 151671)

        talker_config = getattr(model_config, "talker_config", None) if model_config else None
        codec_bos_id = getattr(talker_config, "codec_bos_id", 2149)
        codec_eos_id = getattr(talker_config, "codec_eos_token_id", 2150)
        codec_pad_id = getattr(talker_config, "codec_pad_id", 2148)
        codec_nothink_id = getattr(talker_config, "codec_nothink_id", 2155)
        codec_think_bos_id = getattr(talker_config, "codec_think_bos_id", 2156)
        codec_think_eos_id = getattr(talker_config, "codec_think_eos_id", 2157)

        # --- Build text embeddings ---
        chat_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        text_ids = mx.array(self.tokenizer.encode(chat_text))[None, :]
        text_embed = talker.text_projection(
            talker.get_text_embeddings()(text_ids)
        )  # [1, text_len, hidden]

        # TTS special token embeddings
        tts_tokens = mx.array([[tts_bos_id, tts_eos_id, tts_pad_id]])
        tts_embeds = talker.text_projection(
            talker.get_text_embeddings()(tts_tokens)
        )
        tts_bos_embed = tts_embeds[:, 0:1, :]
        tts_eos_embed = tts_embeds[:, 1:2, :]
        tts_pad_embed = tts_embeds[:, 2:3, :]

        # --- Build codec prefix embeddings ---
        codec_prefill = [codec_nothink_id, codec_think_bos_id, codec_think_eos_id]
        codec_prefix_embed = talker.get_input_embeddings()(mx.array([codec_prefill]))
        codec_suffix_embed = talker.get_input_embeddings()(
            mx.array([[codec_pad_id, codec_bos_id]])
        )
        codec_prefix_embed = mx.concatenate([codec_prefix_embed, codec_suffix_embed], axis=1)
        # codec_prefix_embed: [1, 5, hidden]

        # --- Build prefix (replicating _prepare_generation_inputs) ---
        role_embed = text_embed[:, :3, :]  # <|im_start|>assistant\n
        hidden_size = role_embed.shape[-1]

        pad_count = codec_prefix_embed.shape[1] - 2  # 3
        pad_embeds = mx.broadcast_to(tts_pad_embed, (1, pad_count, hidden_size))
        combined_prefix = mx.concatenate([pad_embeds, tts_bos_embed], axis=1)  # [1, 4, h]
        combined_prefix = combined_prefix + codec_prefix_embed[:, :-1, :]

        # First text token added to last codec prefix token
        first_text_embed = text_embed[:, 3:4, :] + codec_prefix_embed[:, -1:, :]

        prefix_embeds = mx.concatenate([role_embed, combined_prefix, first_text_embed], axis=1)
        # prefix_embeds: [1, prefix_len, hidden]
        prefix_len = prefix_embeds.shape[1]

        # --- Build audio portion (teacher forcing, vectorized) ---
        # Sum all 16 codebook embeddings at each timestep
        codec_combined = talker.get_input_embeddings()(all_codes[0:1, :])  # code_0: [1, T, hidden]
        for cb_idx in range(1, 16):
            codec_combined = codec_combined + talker.code_predictor.codec_embedding[cb_idx - 1](
                all_codes[cb_idx:cb_idx + 1, :]
            )

        # Align text embeddings to audio length T
        trailing_text = mx.concatenate(
            [text_embed[:, 4:-5, :], tts_eos_embed], axis=1
        )  # text tokens 4..end-5, then tts_eos
        trail_len = trailing_text.shape[1]

        if trail_len < T:
            pad_fill = mx.broadcast_to(tts_pad_embed, (1, T - trail_len, hidden_size))
            aligned_text = mx.concatenate([trailing_text, pad_fill], axis=1)
        elif trail_len > T:
            aligned_text = trailing_text[:, :T, :]
        else:
            aligned_text = trailing_text

        audio_embeds = aligned_text + codec_combined  # [1, T, hidden]

        # EOS position (model learns to predict EOS after last audio token)
        eos_codec_embed = talker.get_input_embeddings()(mx.array([[codec_eos_id]]))
        eos_embed = tts_pad_embed + eos_codec_embed  # [1, 1, hidden]

        # --- Full inputs_embeds ---
        inputs_embeds = mx.concatenate([prefix_embeds, audio_embeds, eos_embed], axis=1)
        # [1, prefix_len + T + 1, hidden]

        mx.eval(inputs_embeds)

        # --- Labels ---
        code_0_list = np.array(all_codes[0]).flatten().tolist()
        labels = [-100] * prefix_len + code_0_list + [codec_eos_id]

        return inputs_embeds[0], labels  # Remove batch dim: [seq_len, hidden]


class TTSSFTTrainer:
    """
    Supervised Fine-Tuning Trainer for TTS models.

    Provides native MLX training loop with gradient accumulation,
    following the same patterns as VLMSFTTrainer.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
        data_collator: Any = None,
        train_dataset: Any = None,
        args: Any = None,
        **kwargs,
    ):
        _require_mlx_audio()

        self.wrapper = model if isinstance(model, TTSModelWrapper) else None
        self.actual_model = model.model if self.wrapper else model
        self.tokenizer = tokenizer or (model.tokenizer if self.wrapper else None)
        self.data_collator = data_collator
        self.train_dataset = train_dataset

        # Parse training args
        if args is not None:
            self.learning_rate = getattr(args, "learning_rate", 2e-4)
            self.max_steps = getattr(args, "max_steps", None)
            self.num_train_epochs = getattr(args, "num_train_epochs", 1)
            self.batch_size = getattr(args, "per_device_train_batch_size", 1)
            self.gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 4)
            self.warmup_steps = getattr(args, "warmup_steps", 5)
            self.logging_steps = getattr(args, "logging_steps", 1)
            self.output_dir = getattr(args, "output_dir", "./tts_outputs")
            self.lr_scheduler_type = getattr(args, "lr_scheduler_type", "linear")
            self.weight_decay = getattr(args, "weight_decay", 0.01)
            self.seed = getattr(args, "seed", 3407)
            self.train_on_completions = getattr(args, "train_on_completions", True)
        else:
            self.learning_rate = kwargs.get("learning_rate", 2e-4)
            self.max_steps = kwargs.get("max_steps", None)
            self.num_train_epochs = kwargs.get("num_train_epochs", 1)
            self.batch_size = kwargs.get("batch_size", 1)
            self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 4)
            self.warmup_steps = kwargs.get("warmup_steps", 5)
            self.logging_steps = kwargs.get("logging_steps", 1)
            self.output_dir = kwargs.get("output_dir", "./tts_outputs")
            self.lr_scheduler_type = kwargs.get("lr_scheduler_type", "linear")
            self.weight_decay = kwargs.get("weight_decay", 0.01)
            self.seed = kwargs.get("seed", 3407)
            self.train_on_completions = kwargs.get("train_on_completions", True)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def train(self):
        """
        Train the TTS model using native MLX training loop.

        Returns:
            _TrainerStats with training metrics
        """
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_map
        from tqdm import tqdm

        from mlx_tune._perf import (
            compiled_step, configure_wired_limit,
            enable_grad_checkpoint, make_step_state,
        )

        print("=" * 70)
        print("Starting TTS Fine-Tuning")
        print("=" * 70)

        configure_wired_limit()

        # Ensure LoRA is applied
        if self.wrapper and self.wrapper.lora_enabled and not self.wrapper._lora_applied:
            self.wrapper._apply_lora()
        if self.wrapper:
            enable_grad_checkpoint(self.wrapper)

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
            print(f"Warning: Could not determine dataset size, using {total_steps} steps")

        print(f"  Model: {self.wrapper.model_name if self.wrapper else 'unknown'}")
        print(f"  Dataset: {dataset_len} samples")
        print(f"  Total steps: {total_steps}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  Learning rate: {self.learning_rate}")

        # Loss function: standard cross-entropy (TTS is next-token prediction)
        is_qwen3_tts = (self.wrapper and self.wrapper.profile.codec_type == "qwen3_speech")

        def loss_fn(model, batch):
            labels = batch["labels"]

            if "inputs_embeds" in batch:
                # Qwen3-TTS: forward with pre-computed embeddings
                result = model(batch["inputs_embeds"])
                logits = result[0] if isinstance(result, tuple) else result
            else:
                # Standard path: forward with input_ids
                logits = model(batch["input_ids"])

            # Handle models that return objects instead of raw tensors
            if hasattr(logits, "logits"):
                logits = logits.logits

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            # Cross-entropy loss
            vocab_size = shift_logits.shape[-1]

            # Compute per-token loss
            token_loss = nn.losses.cross_entropy(
                shift_logits.reshape(-1, vocab_size),
                shift_labels.reshape(-1),
                reduction="none",
            ).reshape(shift_labels.shape)

            # Mask: only compute loss where labels != -100
            mask = (shift_labels != -100).astype(token_loss.dtype)
            loss = (token_loss * mask).sum() / mx.maximum(mask.sum(), 1)
            return loss

        loss_and_grad_fn = nn.value_and_grad(self.actual_model, loss_fn)

        # Compiled step (mlx-lm pattern). `shapeless=True` because TTS token
        # sequences vary in length per batch.
        grad_accum = self.gradient_accumulation_steps
        actual_model = self.actual_model
        state = make_step_state(actual_model, optimizer)

        def step_fn(batch, prev_grad, do_update):
            loss, grad = loss_and_grad_fn(actual_model, batch)
            if prev_grad is not None:
                grad = tree_map(lambda a, g: a + g, prev_grad, grad)
            if do_update:
                if grad_accum > 1:
                    grad = tree_map(lambda g: g / grad_accum, grad)
                optimizer.update(actual_model, grad)
                return loss, None
            return loss, grad

        step_fn = compiled_step(step_fn, state)

        progress = tqdm(range(total_steps), desc="Training")
        total_loss = 0.0
        step_counter = 0
        micro_step = 0
        accum_loss_arr = mx.zeros((), dtype=mx.float32)  # on-device accumulator
        accumulated_grads = None
        epoch = 0

        while step_counter < total_steps:
            epoch += 1
            for i in range(0, max(dataset_len, 1), self.batch_size):
                if step_counter >= total_steps:
                    break

                # Get batch
                if self.data_collator is not None:
                    batch_samples = self.train_dataset[i: i + self.batch_size]
                    if isinstance(batch_samples, dict):
                        # HF dataset returns dict of lists
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
                    if not isinstance(batch, dict):
                        batch = {"input_ids": batch}

                do_update = (micro_step + 1) >= grad_accum
                loss, accumulated_grads = step_fn(batch, accumulated_grads, do_update)

                accum_loss_arr = accum_loss_arr + loss.astype(mx.float32)
                mx.eval(state, accum_loss_arr, accumulated_grads)
                micro_step += 1

                if do_update:
                    loss_val = float(accum_loss_arr.item()) / grad_accum
                    total_loss += loss_val
                    step_counter += 1
                    micro_step = 0
                    accum_loss_arr = mx.zeros((), dtype=mx.float32)
                    accumulated_grads = None

                    progress.update(1)
                    if step_counter % self.logging_steps == 0:
                        avg_loss = total_loss / step_counter
                        progress.set_postfix(
                            {"loss": f"{loss_val:.4f}", "avg_loss": f"{avg_loss:.4f}"}
                        )

        progress.close()

        # Save adapters
        if self.wrapper:
            adapter_dir = Path(self.output_dir) / "adapters"
            self.wrapper.save_pretrained(str(adapter_dir))
            self.wrapper._adapter_path = adapter_dir

        avg_loss = total_loss / max(step_counter, 1)
        print(f"\nTraining complete! Average loss: {avg_loss:.4f}")

        return _TrainerStats({"train_loss": avg_loss, "train_runtime": 0})


class _TrainerStats:
    """Simple container for training metrics."""

    def __init__(self, metrics: Dict):
        self.metrics = metrics
