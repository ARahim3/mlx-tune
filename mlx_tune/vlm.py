"""
Vision Language Model (VLM) Support for MLX-Tune

Provides Unsloth-compatible API for Vision-Language models using mlx-vlm:
- Qwen3.5 (natively multimodal)
- Qwen2-VL / Qwen3-VL
- LLaVA 1.5 / 1.6
- Pixtral
- PaliGemma
- Gemma 3
- And other VLMs supported by mlx-vlm

Usage (matches Unsloth exactly):
    from mlx_tune import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        "mlx-community/Qwen3.5-0.8B-bf16",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        r=16, lora_alpha=16,
    )
"""

from typing import Optional, Any, List, Dict, Union, Tuple
from pathlib import Path
import warnings
import json


# Check for mlx-vlm availability
# Supports both mlx-vlm >=0.4.0 (new API) and 0.3.x (legacy API)
_MLX_VLM_LEGACY = False  # True if using 0.3.x API
try:
    from mlx_vlm import load as vlm_load
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm import stream_generate as vlm_stream_generate
    from mlx_vlm.utils import prepare_inputs, load_image_processor
    from mlx_vlm.trainer.utils import (
        get_peft_model as vlm_get_peft_model,
        find_all_linear_names,
        apply_lora_layers,
        freeze_model,
    )

    # mlx-vlm >=0.4.0: trainer.trainer removed, split into sft_trainer + datasets
    try:
        from mlx_vlm.trainer.sft_trainer import save_adapter
        from mlx_vlm.trainer.datasets import VisionDataset as VLMDataset
        # Trainer class replaced by train() function — we handle this in VLMSFTTrainer
        VLMTrainerInternal = None
    except ImportError:
        # mlx-vlm <0.4.0: legacy imports
        from mlx_vlm.trainer.trainer import (
            Trainer as VLMTrainerInternal,
            Dataset as VLMDataset,
            save_adapter,
        )
        _MLX_VLM_LEGACY = True

    HAS_MLX_VLM = True
except ImportError:
    HAS_MLX_VLM = False


def _require_mlx_vlm():
    if not HAS_MLX_VLM:
        raise ImportError(
            "mlx-vlm is required for vision model support. "
            "Install with: uv pip install mlx-vlm"
        )


def _config_to_dict(obj):
    """Recursively convert a config object (dataclass or dict) to a plain dict."""
    if isinstance(obj, dict):
        return {k: _config_to_dict(v) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _config_to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        # Generic object with __dict__ but not a dataclass — check if it's a config-like object
        d = obj.__dict__
        if d:
            return {k: _config_to_dict(v) for k, v in d.items()}
    if isinstance(obj, (list, tuple)):
        return [_config_to_dict(item) for item in obj]
    return obj


class FastVisionModel:
    """
    Unsloth-compatible API for Vision Language Models on Apple Silicon.

    Provides the same API patterns as Unsloth's FastVisionModel but uses
    mlx-vlm under the hood for Apple Silicon optimization.

    Example:
        >>> from mlx_tune import FastVisionModel
        >>> model, processor = FastVisionModel.from_pretrained(
        ...     "mlx-community/Qwen3.5-0.8B-bf16",
        ... )
        >>> model = FastVisionModel.get_peft_model(model, r=16)
        >>> FastVisionModel.for_training(model)
        >>> # ... train ...
        >>> FastVisionModel.for_inference(model)
    """

    @staticmethod
    def from_pretrained(
        model_name: str,
        max_seq_length: Optional[int] = None,
        dtype: Optional[Any] = None,
        load_in_4bit: bool = False,
        use_gradient_checkpointing: Union[bool, str] = False,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """
        Load a pretrained Vision Language Model.

        Args:
            model_name: HuggingFace model ID (e.g., "mlx-community/Qwen3.5-0.8B-bf16")
            max_seq_length: Maximum sequence length
            dtype: Data type (auto-selected by MLX)
            load_in_4bit: Whether to use 4-bit quantized model
            use_gradient_checkpointing: Enable gradient checkpointing ("unsloth" or True)
            **kwargs: Additional arguments passed to mlx_vlm.load()

        Returns:
            Tuple of (VLMModelWrapper, processor)
        """
        _require_mlx_vlm()

        print(f"Loading VLM: {model_name}")

        model, processor = vlm_load(model_name, **kwargs)

        # Load image processor if available
        image_processor = None
        try:
            image_processor = load_image_processor(model_name)
        except Exception:
            pass

        wrapped = VLMModelWrapper(
            model=model,
            processor=processor,
            image_processor=image_processor,
            max_seq_length=max_seq_length or 2048,
            model_name=model_name,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        return wrapped, processor

    @staticmethod
    def get_peft_model(
        model: "VLMModelWrapper",
        finetune_vision_layers: bool = True,
        finetune_language_layers: bool = True,
        finetune_attention_modules: bool = True,
        finetune_mlp_modules: bool = True,
        r: int = 16,
        target_modules: Optional[List[str]] = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        bias: str = "none",
        random_state: int = 3407,
        use_rslora: bool = False,
        loftq_config: Optional[Any] = None,
        **kwargs,
    ) -> "VLMModelWrapper":
        """
        Add LoRA adapters to a VLM for fine-tuning (Unsloth-compatible API).

        Args:
            model: VLMModelWrapper from from_pretrained()
            finetune_vision_layers: Whether to fine-tune vision encoder layers
            finetune_language_layers: Whether to fine-tune language model layers
            finetune_attention_modules: Whether to fine-tune attention layers
            finetune_mlp_modules: Whether to fine-tune MLP layers
            r: LoRA rank
            target_modules: Specific target modules (auto-detected if None)
            lora_alpha: LoRA scaling parameter
            lora_dropout: LoRA dropout rate
            bias: Bias training mode ("none", "all", "lora_only")
            random_state: Random seed
            use_rslora: Use rank-stabilized LoRA
            loftq_config: LoftQ configuration

        Returns:
            Model with LoRA adapters configured (applied on first train call)
        """
        _require_mlx_vlm()

        # Store LoRA config for deferred application
        model.lora_config = {
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "finetune_vision_layers": finetune_vision_layers,
            "finetune_language_layers": finetune_language_layers,
            "finetune_attention_modules": finetune_attention_modules,
            "finetune_mlp_modules": finetune_mlp_modules,
            "target_modules": target_modules,
        }
        model.lora_enabled = True

        # Determine which linear layers to target
        if target_modules is not None:
            linear_layers = target_modules
        else:
            linear_layers = _get_target_modules(
                model.model,
                finetune_attention_modules=finetune_attention_modules,
                finetune_mlp_modules=finetune_mlp_modules,
            )

        # Compute alpha as scale factor (mlx-vlm uses alpha as scale = alpha/rank)
        alpha_scale = lora_alpha / r if r > 0 else 1.0

        # Actually apply LoRA now using mlx-vlm's get_peft_model
        model.model = vlm_get_peft_model(
            model.model,
            linear_layers,
            rank=r,
            alpha=alpha_scale,
            dropout=lora_dropout,
            freeze=True,
            verbose=True,
        )
        model._lora_applied = True
        model._linear_layers = linear_layers

        print(f"LoRA configured: r={r}, alpha={lora_alpha}, "
              f"modules={linear_layers}")

        return model

    @staticmethod
    def for_training(model: "VLMModelWrapper") -> "VLMModelWrapper":
        """Enable training mode for the VLM."""
        model.inference_mode = False
        model.model.train()
        return model

    @staticmethod
    def for_inference(model: "VLMModelWrapper") -> "VLMModelWrapper":
        """Enable inference mode for the VLM."""
        model.inference_mode = True
        model.model.eval()
        return model


def _get_target_modules(
    model,
    finetune_attention_modules: bool = True,
    finetune_mlp_modules: bool = True,
) -> List[str]:
    """Auto-detect target modules based on finetune flags."""
    attention_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_modules = ["gate_proj", "up_proj", "down_proj"]

    target = []
    if finetune_attention_modules:
        target.extend(attention_modules)
    if finetune_mlp_modules:
        target.extend(mlp_modules)

    if not target:
        # Fallback: find all linear layers
        try:
            target = find_all_linear_names(model.language_model)
        except Exception:
            target = attention_modules + mlp_modules

    return target


class VLMModelWrapper:
    """
    Wrapper around mlx-vlm models providing Unsloth-compatible interface.

    Handles LoRA application, training, inference, and model saving.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        image_processor: Any = None,
        max_seq_length: int = 2048,
        model_name: Optional[str] = None,
        use_gradient_checkpointing: Union[bool, str] = False,
    ):
        self.model = model
        self.processor = processor
        self.image_processor = image_processor
        self.max_seq_length = max_seq_length
        self.model_name = model_name
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # LoRA state
        self.lora_config = None
        self.lora_enabled = False
        self._lora_applied = False
        self._linear_layers = None
        self._adapter_path: Optional[Path] = None

        # Mode
        self.inference_mode = False

    @property
    def config(self):
        """Access the underlying model config."""
        return self.model.config

    def generate(
        self,
        prompt: Optional[str] = None,
        image: Optional[Any] = None,
        image_path: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        min_p: float = 0.0,
        verbose: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate response for image+text input.

        Args:
            prompt: Text prompt
            image: Image path(s), PIL Image, or list of images
            image_path: Alternative path to image file
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            min_p: Minimum probability threshold for sampling (recommended: 0.1)
            verbose: Print generation details

        Returns:
            Generated text response
        """
        _require_mlx_vlm()
        import tempfile
        import os
        import re
        from PIL import Image as PILImage

        # Handle image_path as alias
        if image_path and image is None:
            image = image_path

        gen_kwargs = dict(max_tokens=max_tokens, temp=temperature, **kwargs)
        if min_p > 0:
            gen_kwargs["min_p"] = min_p

        # If image is provided, we need to format prompt with vision tokens
        # and pre-compute inputs (mlx-vlm's generate doesn't do this for all models)
        if image is not None:
            # Build messages with image marker for chat template
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt or "Describe this image."},
                ]},
            ]
            formatted_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

            # Convert PIL images to temp file paths (mlx-vlm expects file paths)
            temp_files = []
            image_paths = []
            images_list = image if isinstance(image, list) else [image]
            for img in images_list:
                if isinstance(img, PILImage.Image):
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    img.save(tmp.name)
                    image_paths.append(tmp.name)
                    temp_files.append(tmp.name)
                else:
                    image_paths.append(img)

            try:
                config = self.model.config.__dict__ if hasattr(self.model.config, "__dict__") else self.model.config
                image_token_index = config.get("image_token_index", config.get("image_token_id"))

                inputs = prepare_inputs(
                    processor=self.processor,
                    images=image_paths,
                    prompts=formatted_prompt,
                    image_token_index=image_token_index,
                )

                # Generate using pre-computed inputs
                extra = {k: v for k, v in inputs.items()
                         if k not in ("input_ids", "pixel_values", "attention_mask")}

                text = ""
                for response in vlm_stream_generate(
                    self.model, self.processor, prompt=formatted_prompt,
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    mask=inputs.get("attention_mask"),
                    **extra, **gen_kwargs,
                ):
                    text += response.text
            finally:
                for f in temp_files:
                    os.unlink(f)
        else:
            # Text-only generation — apply chat template so the model
            # sees proper <|im_start|>user/assistant tokens
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt or ""},
                ]},
            ]
            formatted_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            result = vlm_generate(
                self.model, self.processor,
                prompt=formatted_prompt, verbose=verbose,
                **gen_kwargs,
            )
            text = result.text if hasattr(result, "text") else str(result)

        # Strip thinking tags from output (Qwen3.5 inserts <think>...</think>)
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        text = re.sub(r"</think>\s*", "", text)
        text = re.sub(r"<think>\s*", "", text)
        return text.strip()

    def stream_generate(self, prompt: str, image: Optional[Any] = None, **kwargs):
        """Stream-generate response for image+text input."""
        _require_mlx_vlm()
        return vlm_stream_generate(
            self.model, self.processor,
            prompt=prompt, image=image, **kwargs,
        )

    def save_pretrained(self, output_dir: str, **kwargs):
        """Save LoRA adapters with mlx-lm compatible config."""
        _require_mlx_vlm()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._lora_applied:
            adapter_file = output_dir / "adapters.safetensors"
            save_adapter(self.model, str(adapter_file))

            # Write mlx-lm compatible adapter_config.json
            # (mlx-vlm's save_adapter writes a minimal format that mlx-lm can't load)
            self._save_adapter_config(output_dir)

            # Save model config.json (needed for loading and GGUF export)
            self._save_model_config(output_dir)

            print(f"Adapters saved to {output_dir}")
        else:
            print("No LoRA adapters to save. Train the model first.")

    def _save_adapter_config(self, output_dir: Path):
        """Write adapter_config.json in mlx-lm format."""
        lora_config = getattr(self, 'lora_config', {}) or {}
        r = lora_config.get('r', 16)
        alpha = lora_config.get('lora_alpha', r)
        dropout = lora_config.get('lora_dropout', 0.0)

        # Detect number of layers
        num_layers = None
        actual = self.model
        for attr_path in [
            ('language_model', 'model', 'layers'),
            ('model', 'layers'),
            ('layers',),
        ]:
            obj = actual
            for a in attr_path:
                obj = getattr(obj, a, None)
                if obj is None:
                    break
            if obj is not None:
                num_layers = len(obj)
                break

        # Build target module keys in mlx-lm full-path format
        target_modules = lora_config.get('target_modules') or [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ]
        short_to_full = {
            'q_proj': 'self_attn.q_proj', 'k_proj': 'self_attn.k_proj',
            'v_proj': 'self_attn.v_proj', 'o_proj': 'self_attn.o_proj',
            'gate_proj': 'mlp.gate_proj', 'up_proj': 'mlp.up_proj',
            'down_proj': 'mlp.down_proj',
        }
        keys = [short_to_full.get(m, m) for m in target_modules]

        adapter_config = {
            "fine_tune_type": "lora",
            "num_layers": num_layers,
            "lora_parameters": {
                "rank": r,
                "scale": alpha / r if r else 1.0,
                "dropout": dropout,
                "keys": keys,
            },
        }

        with open(output_dir / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2)

    def _save_model_config(self, output_dir: Path):
        """Save config.json, properly serializing nested dataclasses."""
        actual = self.model
        if not hasattr(actual, "config"):
            return

        config_dict = _config_to_dict(actual.config)

        with open(output_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    def load_adapter(self, adapter_path: str, **kwargs):
        """Load LoRA adapters from a saved directory."""
        _require_mlx_vlm()
        from mlx_vlm.trainer.utils import get_peft_model

        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

        adapter_file = adapter_path / "adapters.safetensors"
        config_file = adapter_path / "adapter_config.json"
        if not adapter_file.exists():
            raise FileNotFoundError(f"adapters.safetensors not found in {adapter_path}")
        if not config_file.exists():
            raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

        with open(config_file) as f:
            config = json.load(f)

        # Extract LoRA params from either mlx-lm or mlx-vlm format
        if "lora_parameters" in config:
            lp = config["lora_parameters"]
            rank = lp["rank"]
            # mlx-vlm uses alpha directly as scale factor in forward pass
            # (NOT the raw lora_alpha value). Our config stores scale = alpha/r.
            alpha = lp.get("scale", 1.0)
            dropout = lp.get("dropout", 0.0)
            # Convert mlx-lm full paths back to short names for get_peft_model
            full_to_short = {
                'self_attn.q_proj': 'q_proj', 'self_attn.k_proj': 'k_proj',
                'self_attn.v_proj': 'v_proj', 'self_attn.o_proj': 'o_proj',
                'mlp.gate_proj': 'gate_proj', 'mlp.up_proj': 'up_proj',
                'mlp.down_proj': 'down_proj',
            }
            keys = lp.get("keys", [])
            modules = [full_to_short.get(k, k) for k in keys]
        else:
            rank = config.get("rank", 16)
            alpha = config.get("alpha", 1.0)
            dropout = config.get("dropout", 0.0)
            modules = None

        # If no specific modules, use all linear layers (mlx-vlm default)
        if not modules:
            from mlx_vlm.trainer.utils import find_all_linear_names
            modules = find_all_linear_names(self.model.language_model)

        # Apply LoRA to only the correct modules, then load trained weights
        self.model = get_peft_model(
            self.model, modules,
            rank=rank, alpha=alpha, dropout=dropout,
            freeze=True, verbose=False,
        )
        self.model.load_weights(str(adapter_file), strict=False)

        self._lora_applied = True
        self._adapter_path = adapter_path
        print(f"Adapters loaded from {adapter_path}")

    def save_pretrained_merged(
        self,
        output_dir: str,
        tokenizer_or_processor: Any = None,
        save_method: str = "merged_16bit",
        **kwargs,
    ):
        """Save merged model (base + LoRA fused)."""
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        actual_model = self.model

        # Fuse LoRA layers if applied
        if self._lora_applied:
            from mlx_vlm.trainer.lora import LoRaLayer
            fused_count = 0
            for name, module in actual_model.named_modules():
                if isinstance(module, LoRaLayer):
                    # LoRaLayer has original_layer, A, B, alpha
                    original = module.original_layer
                    # Compute fused weight: W + alpha * B^T @ A^T
                    lora_weight = (module.B.T @ module.A.T) * module.alpha
                    if isinstance(original, nn.QuantizedLinear):
                        warnings.warn(
                            f"Cannot fuse LoRA into quantized layer {name}. "
                            "Use a non-quantized base model for merged saving."
                        )
                        continue
                    original.weight = original.weight + lora_weight
                    # Replace LoRA layer with original
                    parts = name.split(".")
                    parent = actual_model
                    for p in parts[:-1]:
                        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                    setattr(parent, parts[-1], original)
                    fused_count += 1
            if fused_count:
                print(f"Fused {fused_count} LoRA layers into base model")

        # Save weights
        weights = dict(tree_flatten(actual_model.parameters()))
        from mlx_lm.utils import save_model
        save_model(str(output_dir), actual_model)

        # Save processor/tokenizer
        if tokenizer_or_processor is not None:
            tokenizer_or_processor.save_pretrained(str(output_dir))

        # Save config (properly serialize nested dataclasses)
        if hasattr(actual_model, "config"):
            config_dict = _config_to_dict(actual_model.config)
            with open(output_dir / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)

        print(f"Merged model saved to {output_dir}")

    def save_pretrained_gguf(
        self,
        output_dir: str,
        tokenizer_or_processor: Any = None,
        quantization_method: str = "q8_0",
        **kwargs,
    ):
        """
        Save model in GGUF format.

        Note: VLM GGUF export has limited support. The language model
        portion can be exported, but vision components may not be included.
        """
        warnings.warn(
            "GGUF export for VLMs is experimental. Only the language model "
            "portion will be exported. For full VLM deployment, use "
            "save_pretrained() or save_pretrained_merged() instead.",
            UserWarning,
        )
        # Save merged first, then attempt GGUF conversion
        merged_dir = Path(output_dir) / "_merged_tmp"
        self.save_pretrained_merged(str(merged_dir), tokenizer_or_processor)

        from mlx_tune.trainer import export_to_gguf
        export_to_gguf(
            str(merged_dir),
            output_path=str(Path(output_dir) / "model.gguf"),
            quantization=quantization_method,
        )

    def set_adapter_path(self, path: str):
        """Set the adapter save/load path."""
        self._adapter_path = Path(path)

    def get_adapter_path(self) -> Optional[Path]:
        """Get the current adapter path."""
        return self._adapter_path

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        # Avoid infinite recursion for attributes accessed during __init__
        if name in ("model", "processor", "lora_config", "lora_enabled",
                     "_lora_applied", "_linear_layers", "_adapter_path",
                     "inference_mode", "image_processor", "max_seq_length",
                     "model_name", "use_gradient_checkpointing"):
            raise AttributeError(name)
        return getattr(self.model, name)


class UnslothVisionDataCollator:
    """
    Data collator for vision fine-tuning (Unsloth-compatible API).

    Handles batching of image+text data for VLM training. This is the
    equivalent of Unsloth's UnslothVisionDataCollator.

    Usage:
        >>> from mlx_tune import UnslothVisionDataCollator
        >>> collator = UnslothVisionDataCollator(model, processor)
        >>> batch = collator(samples)
    """

    def __init__(self, model: Any, processor: Any):
        """
        Args:
            model: VLMModelWrapper or raw mlx-vlm model
            processor: The model's processor/tokenizer
        """
        self.model = model.model if isinstance(model, VLMModelWrapper) else model
        self.processor = processor
        self._config = self.model.config.__dict__ if hasattr(self.model.config, "__dict__") else self.model.config

    def __call__(self, samples: List[Dict]) -> Dict:
        """
        Collate a list of vision samples into a training batch.

        Each sample should have a 'messages' key with the conversation format:
        [
            {"role": "user", "content": [{"type": "text", ...}, {"type": "image", ...}]},
            {"role": "assistant", "content": [{"type": "text", ...}]},
        ]
        """
        import mlx.core as mx

        all_input_ids = []
        all_pixel_values = []
        all_masks = []
        extra_kwargs = {}

        for sample in samples:
            messages = sample.get("messages", sample.get("conversations", []))
            images = []

            # Extract images from messages and build clean messages for template
            clean_messages = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    final_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                final_content.append({"type": "text", "text": item.get("text", "")})
                            elif item.get("type") == "image":
                                img = item.get("image")
                                if img is not None:
                                    images.append(img)
                                # Keep image marker for processor's chat template
                                final_content.append({"type": "image"})
                    clean_messages.append({"role": msg["role"], "content": final_content})
                else:
                    clean_messages.append(msg)

            # Use processor's own apply_chat_template (NOT mlx-vlm's prompt_utils)
            # The processor correctly inserts vision tokens like <|vision_start|><|image_pad|><|vision_end|>
            prompt = self._apply_chat_template(clean_messages)

            # Get image token index from config
            image_token_index = self._config.get(
                "image_token_index",
                self._config.get("image_token_id"),
            )

            # Prepare inputs using mlx-vlm (handles image tokenization)
            inputs = prepare_inputs(
                processor=self.processor,
                images=images if images else None,
                audio=None,
                prompts=[prompt],
                image_token_index=image_token_index,
            )

            # Ensure input_ids/mask are 2D (batch, seq) — prepare_inputs
            # can return 3D when there are no images
            input_ids = inputs["input_ids"]
            if input_ids.ndim == 3:
                input_ids = input_ids.reshape(-1, input_ids.shape[-1])
            all_input_ids.append(input_ids)

            if inputs.get("pixel_values") is not None:
                all_pixel_values.append(inputs["pixel_values"])

            mask = inputs.get("attention_mask", mx.ones_like(input_ids))
            if mask.ndim == 3:
                mask = mask.reshape(-1, mask.shape[-1])
            all_masks.append(mask)

            # Collect extra kwargs (model-specific like image_grid_thw)
            for k, v in inputs.items():
                if k not in ("input_ids", "pixel_values", "attention_mask"):
                    if k not in extra_kwargs:
                        extra_kwargs[k] = v

        # Pad sequences to same length and stack into batch
        if len(all_input_ids) > 1:
            max_len = max(ids.shape[-1] for ids in all_input_ids)
            padded_ids = []
            padded_masks = []
            for ids, mask in zip(all_input_ids, all_masks):
                pad_len = max_len - ids.shape[-1]
                if pad_len > 0:
                    ids = mx.concatenate([ids, mx.zeros((1, pad_len), dtype=ids.dtype)], axis=1)
                    mask = mx.concatenate([mask, mx.zeros((1, pad_len), dtype=mask.dtype)], axis=1)
                padded_ids.append(ids)
                padded_masks.append(mask)
            all_input_ids = padded_ids
            all_masks = padded_masks

        batch = {
            "input_ids": mx.concatenate(all_input_ids, axis=0) if all_input_ids else None,
            "attention_mask": mx.concatenate(all_masks, axis=0) if all_masks else None,
        }
        if all_pixel_values:
            batch["pixel_values"] = mx.concatenate(all_pixel_values, axis=0)
        else:
            batch["pixel_values"] = None

        batch.update(extra_kwargs)
        return batch

    def _apply_chat_template(self, messages: List[Dict]) -> str:
        """Apply chat template using the processor (inserts proper vision tokens)."""
        if hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        elif hasattr(self.processor, "tokenizer"):
            return self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        else:
            return "\n".join(
                f"{m['role']}: {m.get('content', '')}" for m in messages
            )


class _VLMTrainerShim:
    """
    Compatibility shim for mlx-vlm >=0.4.0 where the Trainer class was removed.

    Replicates the subset of the old Trainer API that VLMSFTTrainer relies on:
    model, optimizer, loss_fn(), train_step().
    """

    def __init__(self, model, optimizer, train_on_completions=False, assistant_id=None):
        self.model = model
        self.optimizer = optimizer
        self.train_on_completions = train_on_completions
        self.assistant_id = assistant_id

    def loss_fn(self, model, batch):
        """Compute cross-entropy loss, optionally masking prompt tokens."""
        import mlx.core as mx
        import mlx.nn as nn

        input_ids = batch["input_ids"]
        pixel_values = batch.get("pixel_values")
        mask = batch.get("attention_mask")

        # Forward pass — build kwargs for model
        fwd_kwargs = {"input_ids": input_ids}
        if pixel_values is not None:
            fwd_kwargs["pixel_values"] = pixel_values
        if mask is not None:
            fwd_kwargs["attention_mask"] = mask
        # Pass through any extra model-specific keys (e.g. image_grid_thw)
        for k, v in batch.items():
            if k not in ("input_ids", "pixel_values", "attention_mask"):
                fwd_kwargs[k] = v

        logits = model(**fwd_kwargs)
        # mlx-vlm 0.4.0+ returns LanguageModelOutput dataclass
        if hasattr(logits, "logits"):
            logits = logits.logits
        elif isinstance(logits, tuple):
            logits = logits[0]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # Cross-entropy loss
        loss = nn.losses.cross_entropy(
            shift_logits, shift_labels, reduction="none"
        )

        # Mask padding tokens
        if mask is not None:
            loss_mask = mask[:, 1:].astype(loss.dtype)
        else:
            loss_mask = mx.ones_like(loss)

        # Mask prompt tokens (train on completions only)
        if self.train_on_completions and self.assistant_id is not None:
            mask_rows = []
            for i in range(shift_labels.shape[0]):
                labels_list = shift_labels[i].tolist()
                in_completion = False
                mask_row = []
                for tok in labels_list:
                    if tok == self.assistant_id:
                        in_completion = True
                    mask_row.append(1.0 if in_completion else 0.0)
                mask_rows.append(mask_row)
            completion_mask = mx.array(mask_rows, dtype=loss.dtype)
            loss_mask = loss_mask * completion_mask

        loss = (loss * loss_mask).sum() / mx.maximum(loss_mask.sum(), 1)
        return loss

    def train_step(self, batch):
        """Single training step: forward + backward + update."""
        import mlx.core as mx
        import mlx.nn as nn

        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
        loss, grads = loss_and_grad_fn(self.model, batch)
        self.optimizer.update(self.model, grads)
        mx.eval(self.model, self.optimizer.state)
        return loss


class VLMSFTTrainer:
    """
    Supervised Fine-Tuning Trainer for Vision Language Models.

    Uses mlx-vlm's native Trainer and Dataset classes under the hood.
    Provides Unsloth/TRL-compatible API.

    Example:
        >>> from mlx_tune import FastVisionModel, VLMSFTTrainer, UnslothVisionDataCollator
        >>>
        >>> model, processor = FastVisionModel.from_pretrained(
        ...     "mlx-community/Qwen3.5-0.8B-bf16",
        ... )
        >>> model = FastVisionModel.get_peft_model(model, r=16, lora_alpha=16)
        >>> FastVisionModel.for_training(model)
        >>>
        >>> trainer = VLMSFTTrainer(
        ...     model=model,
        ...     tokenizer=processor,
        ...     data_collator=UnslothVisionDataCollator(model, processor),
        ...     train_dataset=converted_dataset,
        ...     args=VLMSFTConfig(max_steps=30, learning_rate=2e-4),
        ... )
        >>> trainer.train()
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
        _require_mlx_vlm()

        self.wrapper = model if isinstance(model, VLMModelWrapper) else None
        self.actual_model = model.model if self.wrapper else model
        self.processor = tokenizer or (model.processor if self.wrapper else None)
        self.data_collator = data_collator
        self.train_dataset = train_dataset

        # Parse training args
        if args is not None:
            self.learning_rate = getattr(args, "learning_rate", 2e-4)
            self.max_steps = getattr(args, "max_steps", None)
            self.num_train_epochs = getattr(args, "num_train_epochs", 1)
            self.batch_size = getattr(args, "per_device_train_batch_size", 1)
            if self.batch_size > 1:
                warnings.warn(
                    f"VLM training with batch_size={self.batch_size} is not recommended. "
                    "Images of different sizes produce different vision token counts, "
                    "which causes shape mismatches. Using batch_size=1 instead.",
                    UserWarning,
                )
                self.batch_size = 1
            self.gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 1)
            self.warmup_steps = getattr(args, "warmup_steps", 0)
            self.logging_steps = getattr(args, "logging_steps", 10)
            self.output_dir = getattr(args, "output_dir", "./vlm_outputs")
            self.lr_scheduler_type = getattr(args, "lr_scheduler_type", "linear")
            self.weight_decay = getattr(args, "weight_decay", 0.0)
            self.seed = getattr(args, "seed", 3407)
            self.max_length = getattr(args, "max_length", 2048)
            self.train_on_completions = getattr(args, "train_on_completions", False)
        else:
            self.learning_rate = kwargs.get("learning_rate", 2e-4)
            self.max_steps = kwargs.get("max_steps", None)
            self.num_train_epochs = kwargs.get("num_train_epochs", 1)
            self.batch_size = kwargs.get("batch_size", 1)
            self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
            self.warmup_steps = kwargs.get("warmup_steps", 0)
            self.logging_steps = kwargs.get("logging_steps", 10)
            self.output_dir = kwargs.get("output_dir", "./vlm_outputs")
            self.lr_scheduler_type = kwargs.get("lr_scheduler_type", "linear")
            self.weight_decay = kwargs.get("weight_decay", 0.0)
            self.seed = kwargs.get("seed", 3407)
            self.max_length = kwargs.get("max_length", 2048)
            self.train_on_completions = kwargs.get("train_on_completions", False)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def train(self):
        """
        Train the VLM using mlx-vlm's native training loop.

        Returns:
            Training metrics dict
        """
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from tqdm import tqdm

        print("=" * 70)
        print("Starting VLM Fine-Tuning")
        print("=" * 70)

        # Ensure model is in training mode
        self.actual_model.train()

        # Set up optimizer
        optimizer = optim.Adam(learning_rate=self.learning_rate)

        # Auto-detect assistant token ID for response-only training
        assistant_id = 77091  # default fallback
        if self.processor is not None:
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            if hasattr(tokenizer, "encode"):
                ids = tokenizer.encode("assistant", add_special_tokens=False)
                if ids:
                    assistant_id = ids[0]

        # Set up training internals
        # train_on_completions=True trains only on assistant response tokens
        # (matches Unsloth/TRL SFTTrainer behavior)
        if _MLX_VLM_LEGACY and VLMTrainerInternal is not None:
            # mlx-vlm <0.4.0: use Trainer class
            trainer = VLMTrainerInternal(
                self.actual_model,
                optimizer,
                train_on_completions=True,
                assistant_id=assistant_id,
            )
        else:
            # mlx-vlm >=0.4.0: Trainer class removed, use _VLMTrainerShim
            trainer = _VLMTrainerShim(
                self.actual_model,
                optimizer,
                train_on_completions=True,
                assistant_id=assistant_id,
            )

        # Determine total steps
        dataset_len = len(self.train_dataset) if hasattr(self.train_dataset, "__len__") else 0
        if self.max_steps:
            total_steps = self.max_steps
        elif dataset_len > 0:
            total_steps = (dataset_len // self.batch_size) * self.num_train_epochs
        else:
            total_steps = 100
            print(f"Warning: Could not determine dataset size, using {total_steps} steps")

        # Prepare dataset
        # If train_dataset is a list of dicts with 'messages' key (Unsloth format),
        # use our collator; otherwise try mlx-vlm's Dataset directly
        use_collator = (
            self.data_collator is not None
            and isinstance(self.train_dataset, list)
        )

        if use_collator:
            # Use data collator for Unsloth-format data
            print(f"Training with data collator, {len(self.train_dataset)} samples")
            return self._train_with_collator(trainer, total_steps)
        else:
            # Try mlx-vlm's Dataset
            print("Training with mlx-vlm Dataset")
            return self._train_with_vlm_dataset(trainer, total_steps)

    def _train_with_collator(self, trainer, total_steps):
        """Train using our UnslothVisionDataCollator."""
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_map
        from tqdm import tqdm

        grad_accum = self.gradient_accumulation_steps
        progress = tqdm(range(total_steps), desc="Training")
        total_loss = 0.0
        step = 0
        epoch = 0
        micro_step = 0
        accum_loss = 0.0

        # For gradient accumulation, we compute loss+grad manually
        loss_and_grad_fn = nn.value_and_grad(trainer.model, trainer.loss_fn)

        accumulated_grads = None

        while step < total_steps:
            epoch += 1
            for i in range(0, len(self.train_dataset), self.batch_size):
                if step >= total_steps:
                    break

                batch_samples = self.train_dataset[i : i + self.batch_size]
                batch = self.data_collator(batch_samples)

                loss, grads = loss_and_grad_fn(trainer.model, batch)
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
                    # Average gradients
                    averaged_grads = tree_map(
                        lambda g: g / grad_accum, accumulated_grads
                    )
                    trainer.optimizer.update(trainer.model, averaged_grads)
                    mx.eval(trainer.model, trainer.optimizer.state)

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
        adapter_dir = Path(self.output_dir) / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        save_adapter(trainer.model, str(adapter_dir / "adapters.safetensors"))

        if self.wrapper:
            self.wrapper._adapter_path = adapter_dir

        avg_loss = total_loss / max(step, 1)
        print(f"\nTraining complete! Average loss: {avg_loss:.4f}")
        print(f"Adapters saved to: {adapter_dir}")

        return _TrainerStats({"train_loss": avg_loss, "train_runtime": 0})

    def _train_with_vlm_dataset(self, trainer, total_steps):
        """Train using mlx-vlm's native Dataset."""
        import mlx.core as mx
        from tqdm import tqdm

        config = self.actual_model.config.__dict__ if hasattr(self.actual_model.config, "__dict__") else self.actual_model.config

        # Wrap dataset
        if _MLX_VLM_LEGACY:
            # mlx-vlm <0.4.0: Dataset accepts image_processor kwarg
            vlm_dataset = VLMDataset(
                self.train_dataset,
                config,
                self.processor,
                image_processor=self.wrapper.image_processor if self.wrapper else None,
            )
        else:
            # mlx-vlm >=0.4.0: VisionDataset has simplified constructor
            vlm_dataset = VLMDataset(
                self.train_dataset,
                config,
                self.processor,
            )

        progress = tqdm(range(total_steps), desc="Training")
        total_loss = 0.0

        for step in range(total_steps):
            idx = step % len(vlm_dataset)
            batch = vlm_dataset[idx * self.batch_size : (idx + 1) * self.batch_size]

            loss = trainer.train_step(batch)
            mx.eval(trainer.model, trainer.optimizer.state)

            loss_val = loss.item()
            total_loss += loss_val
            progress.update(1)

            if (step + 1) % self.logging_steps == 0:
                avg = total_loss / (step + 1)
                progress.set_postfix({"loss": f"{loss_val:.4f}", "avg": f"{avg:.4f}"})

        progress.close()

        # Save adapters
        adapter_dir = Path(self.output_dir) / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        save_adapter(trainer.model, str(adapter_dir / "adapters.safetensors"))

        if self.wrapper:
            self.wrapper._adapter_path = adapter_dir

        avg_loss = total_loss / max(total_steps, 1)
        print(f"\nTraining complete! Average loss: {avg_loss:.4f}")
        print(f"Adapters saved to: {adapter_dir}")

        return _TrainerStats({"train_loss": avg_loss, "train_runtime": 0})


class _TrainerStats:
    """Simple container for training metrics (matches HF Trainer output)."""
    def __init__(self, metrics: Dict):
        self.metrics = metrics


class VLMSFTConfig:
    """
    Training configuration for VLM fine-tuning.

    Mirrors TRL's SFTConfig for API compatibility.
    """
    def __init__(
        self,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 5,
        max_steps: Optional[int] = None,
        num_train_epochs: int = 1,
        learning_rate: float = 2e-4,
        logging_steps: int = 1,
        optim: str = "adam",
        weight_decay: float = 0.01,
        lr_scheduler_type: str = "linear",
        seed: int = 3407,
        output_dir: str = "outputs",
        report_to: str = "none",
        remove_unused_columns: bool = False,
        dataset_text_field: str = "",
        dataset_kwargs: Optional[Dict] = None,
        max_length: int = 2048,
        train_on_completions: bool = False,
        **kwargs,
    ):
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.logging_steps = logging_steps
        self.optim = optim
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler_type
        self.seed = seed
        self.output_dir = output_dir
        self.report_to = report_to
        self.remove_unused_columns = remove_unused_columns
        self.dataset_text_field = dataset_text_field
        self.dataset_kwargs = dataset_kwargs or {}
        self.max_length = max_length
        self.train_on_completions = train_on_completions
        # Store any extra kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


def load_vlm_dataset(
    dataset_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
    split: str = "train",
    image_column: str = "image",
    text_column: str = "text",
) -> Any:
    """
    Load a VLM dataset from HuggingFace Hub or local path.

    Args:
        dataset_name: HuggingFace dataset name
        dataset_path: Local dataset path (JSONL)
        split: Dataset split to load
        image_column: Column name for images
        text_column: Column name for text

    Returns:
        Loaded dataset
    """
    if dataset_name:
        from datasets import load_dataset
        return load_dataset(dataset_name, split=split)
    elif dataset_path:
        import json
        with open(dataset_path) as f:
            return [json.loads(line) for line in f]
    else:
        raise ValueError("Provide dataset_name or dataset_path")
