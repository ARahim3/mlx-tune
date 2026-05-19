"""
FastLanguageModel - Main API entry point for MLX-Tune

This module provides Unsloth-compatible API for loading and configuring language models
using Apple's MLX framework under the hood.
"""

from typing import Optional, Tuple, Union, List, Any, Dict, Set
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_load
import warnings

# Try to import mlx_lm tuner utilities for native LoRA support
try:
    from mlx_lm.tuner.utils import linear_to_lora_layers
    HAS_MLX_LM_TUNER = True
except ImportError:
    HAS_MLX_LM_TUNER = False
    warnings.warn(
        "mlx_lm.tuner not available. Install with: pip install 'mlx-lm[train]'. "
        "Native LoRA application will not work.",
        ImportWarning
    )

# Try to import MoE switch layers for detection
try:
    from mlx_lm.models.switch_layers import SwitchLinear, QuantizedSwitchLinear
    _HAS_SWITCH_LAYERS = True
except ImportError:
    _HAS_SWITCH_LAYERS = False

# Try to import LoRA layer types (needed for post-LoRA path resolution)
try:
    from mlx_lm.tuner.lora import LoRALinear, LoRASwitchLinear
    _HAS_LORA_TYPES = True
except ImportError:
    _HAS_LORA_TYPES = False


def _resolve_target_modules(model: Any, target_modules: List[str]) -> List[str]:
    """
    Dynamically resolve short module names to full layer-relative paths
    by inspecting the actual model structure.

    For dense models: gate_proj → ["mlp.gate_proj"]
    For MoE models:   gate_proj → ["mlp.switch_mlp.gate_proj", "mlp.shared_expert.gate_proj", ...]
    For mixed models:  both dense and MoE paths are returned.

    Routers (mlp.gate) are naturally excluded because their name doesn't
    end with any standard target like "gate_proj".

    Args:
        model: The MLX model (unwrapped, with .layers attribute).
        target_modules: Short module names like ["q_proj", "gate_proj"].

    Returns:
        Deduplicated list of full layer-relative paths.
    """
    # Determine convertible types (include LoRA variants for post-application resolution)
    convertible_types = (nn.Linear, nn.QuantizedLinear)
    if _HAS_SWITCH_LAYERS:
        convertible_types = convertible_types + (SwitchLinear, QuantizedSwitchLinear)
    if _HAS_LORA_TYPES:
        convertible_types = convertible_types + (LoRALinear, LoRASwitchLinear)

    # Get model layers
    layers = None
    if hasattr(model, 'layers'):
        layers = model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers

    if not layers:
        # Fallback to static mapping
        static = {
            'q_proj': 'self_attn.q_proj', 'k_proj': 'self_attn.k_proj',
            'v_proj': 'self_attn.v_proj', 'o_proj': 'self_attn.o_proj',
            'gate_proj': 'mlp.gate_proj', 'up_proj': 'mlp.up_proj',
            'down_proj': 'mlp.down_proj',
        }
        return [static.get(m, m) for m in target_modules]

    # Scan ALL layers to collect convertible module paths (handles mixed architectures)
    all_paths: Set[str] = set()
    moe_detected = False

    for layer in layers:
        for path, module in layer.named_modules():
            if isinstance(module, convertible_types):
                all_paths.add(path)
                if _HAS_SWITCH_LAYERS and isinstance(module, (SwitchLinear, QuantizedSwitchLinear)):
                    moe_detected = True
                elif _HAS_LORA_TYPES and isinstance(module, LoRASwitchLinear):
                    moe_detected = True

    if moe_detected:
        print("MoE architecture detected — LoRA will target expert layers (SwitchLinear)")

    # Resolve each target: find all paths ending with the target name
    resolved: Set[str] = set()
    for module_name in target_modules:
        if module_name in all_paths:
            # Exact match (already a full path)
            resolved.add(module_name)
        else:
            matches = [p for p in all_paths if p.endswith('.' + module_name)]
            if matches:
                resolved.update(matches)
            else:
                # Keep as-is (user may have specified a custom path)
                resolved.add(module_name)

    return sorted(resolved)


class FastLanguageModel:
    """
    Unsloth-compatible wrapper around MLX language models.

    This class provides the same API as Unsloth's FastLanguageModel but uses
    MLX for Apple Silicon optimization instead of CUDA/Triton kernels.

    Example:
        >>> from mlx_tune import FastLanguageModel
        >>> model, tokenizer = FastLanguageModel.from_pretrained(
        ...     model_name="mlx-community/Llama-3.2-3B-Instruct-4bit",
        ...     max_seq_length=2048,
        ...     load_in_4bit=True,
        ... )
    """

    @staticmethod
    def from_pretrained(
        model_name: str,
        max_seq_length: Optional[int] = None,
        dtype: Optional[Any] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        token: Optional[str] = None,
        device_map: Optional[str] = None,
        rope_scaling: Optional[Any] = None,
        fix_tokenizer: bool = True,
        trust_remote_code: bool = False,
        use_gradient_checkpointing: Optional[Union[bool, str]] = None,
        resize_model_vocab: Optional[int] = None,
        revision: Optional[str] = None,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Load a pretrained language model with Unsloth-compatible parameters.

        This method loads models from HuggingFace Hub or local paths. MLX will
        automatically convert any HuggingFace model to MLX format on first load.

        Args:
            model_name: Model identifier from HuggingFace Hub (e.g., "meta-llama/Llama-3.2-3B")
                       or local path. Supports ANY HuggingFace model.
            max_seq_length: Maximum sequence length for training/inference
            dtype: Data type (MLX uses its own dtype system, usually auto-selected)
            load_in_4bit: Whether to use 4-bit quantization (recommended for memory)
            load_in_8bit: Whether to use 8-bit quantization
            token: HuggingFace API token for gated/private models
            device_map: Device mapping (not used in MLX - unified memory architecture)
            rope_scaling: RoPE scaling configuration (passed to MLX if supported)
            fix_tokenizer: Whether to fix tokenizer issues (MLX handles this)
            trust_remote_code: Whether to trust remote code in model/tokenizer
            use_gradient_checkpointing: Gradient checkpointing mode
            resize_model_vocab: Resize model vocabulary to this size
            revision: Model revision/branch to load
            **kwargs: Additional arguments passed to MLX load function

        Returns:
            Tuple of (model, tokenizer) compatible with Unsloth API

        Note:
            - MLX automatically converts HuggingFace models to MLX format
            - Converted models are cached locally for faster subsequent loads
            - For pre-quantized models, check mlx-community on HuggingFace
            - Unified memory means device_map is ignored
            - Any model that works with transformers works with MLX

        Examples:
            >>> # Load any HuggingFace model
            >>> model, tokenizer = FastLanguageModel.from_pretrained(
            ...     "meta-llama/Llama-3.2-3B-Instruct"
            ... )
            >>>
            >>> # Load pre-quantized model (faster)
            >>> model, tokenizer = FastLanguageModel.from_pretrained(
            ...     "mlx-community/Llama-3.2-3B-Instruct-4bit",
            ...     load_in_4bit=True
            ... )
        """

        # Warn about unused parameters (for compatibility)
        if device_map is not None:
            print("Note: device_map is not used with MLX (unified memory architecture)")

        # Build tokenizer config
        tokenizer_config = {}
        if trust_remote_code:
            tokenizer_config["trust_remote_code"] = True
        if token:
            tokenizer_config["token"] = token

        # Prepare MLX load arguments
        mlx_kwargs = {
            "tokenizer_config": tokenizer_config if tokenizer_config else {},
        }

        # Add revision if specified
        if revision:
            mlx_kwargs["revision"] = revision

        # Merge additional kwargs
        mlx_kwargs.update(kwargs)

        try:
            # Load model using MLX (with config for saving later)
            model, tokenizer, config = mlx_load(model_name, return_config=True, **mlx_kwargs)

            # Wrap model with our compatibility layer
            wrapped_model = MLXModelWrapper(
                model=model,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                model_name=model_name,
                config=config,
            )

            return wrapped_model, tokenizer

        except Exception as e:
            model_lower = model_name.lower()
            if 'gemma-4' in model_lower or 'gemma4' in model_lower:
                raise RuntimeError(
                    f"Gemma 4 models are multimodal (VLM). Use FastVisionModel instead:\n\n"
                    f"  from mlx_tune import FastVisionModel\n"
                    f"  model, processor = FastVisionModel.from_pretrained(\"{model_name}\")\n"
                ) from e
            raise RuntimeError(
                f"Failed to load model '{model_name}'. "
                f"Error: {str(e)}\n\n"
                f"Tips:\n"
                f"- Ensure model exists on HuggingFace Hub\n"
                f"- For gated models (Llama, etc.), provide your HF token\n"
                f"- For faster loading, use pre-converted mlx-community models\n"
                f"- MLX will auto-convert HF models on first load (may take time)"
            ) from e

    @staticmethod
    def get_peft_model(
        model: Any,
        r: int = 16,
        target_modules: Optional[List[str]] = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        bias: str = "none",
        use_gradient_checkpointing: Union[bool, str] = False,
        random_state: int = 3407,
        use_rslora: bool = False,
        loftq_config: Optional[Any] = None,
        max_seq_length: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Add LoRA (Low-Rank Adaptation) adapters to the model.

        This method configures the model for parameter-efficient fine-tuning using
        LoRA, compatible with Unsloth's API but using MLX's LoRA implementation.

        Args:
            model: The model to add LoRA adapters to
            r: LoRA rank (dimension of low-rank matrices)
            target_modules: List of module names to apply LoRA to
                           (e.g., ["q_proj", "k_proj", "v_proj", "o_proj"])
            lora_alpha: LoRA scaling parameter
            lora_dropout: Dropout probability for LoRA layers
            bias: Bias configuration ("none", "all", or "lora_only")
            use_gradient_checkpointing: Gradient checkpointing mode. Defaults to
                ``False`` (faster training, more memory). Pass ``True`` or
                ``"unsloth"`` to wrap each transformer block's forward with
                ``mx.checkpoint`` — roughly halves activation memory at the cost
                of ~2× wall-time per step (recompute in backward).

                When to opt in:

                * Long context (ctx≥4096) on M-series Macs — without GC, ORPO/DPO
                  with both chosen+rejected forwards held live in
                  ``value_and_grad`` will exceed the Metal wired-memory limit
                  (~75% of unified RAM) and OOM.
                * Large models on tighter-RAM machines (e.g., 7B+ on a 24GB Mac).
                * Otherwise leave at ``False`` for ~2× faster training.

                ``True`` and ``"unsloth"`` behave identically — mlx-tune doesn't
                ship Unsloth's custom GC kernel; both map to mlx-lm's
                ``mx.checkpoint``-based path.
            random_state: Random seed for initialization
            use_rslora: Use Rank-Stabilized LoRA
            loftq_config: LoftQ configuration (for quantization-aware init)
            max_seq_length: Maximum sequence length
            **kwargs: Additional LoRA configuration parameters

        Returns:
            Model with LoRA adapters configured

        Note:
            - LoRA configuration is stored in the model wrapper
            - Actual LoRA application happens during training
            - MLX handles LoRA differently than PEFT library
        """

        # Validate target modules
        if target_modules is None:
            # Default target modules for common architectures
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        # Warn about unsupported features
        if use_rslora:
            warnings.warn(
                "RSLoRA is not yet implemented in MLX. Using standard LoRA.",
                UserWarning
            )

        if loftq_config is not None:
            warnings.warn(
                "LoftQ is not yet implemented in MLX. Using standard LoRA initialization.",
                UserWarning
            )

        if lora_dropout > 0:
            warnings.warn(
                "LoRA dropout may have limited support in MLX. Dropout value will be set but "
                "behavior may differ from PyTorch PEFT.",
                UserWarning
            )

        # Configure LoRA settings on the model wrapper
        if hasattr(model, 'configure_lora'):
            model.configure_lora(
                r=r,
                target_modules=target_modules,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=bias,
                use_gradient_checkpointing=use_gradient_checkpointing,
                random_state=random_state,
                **kwargs
            )
        else:
            raise TypeError(
                f"Model does not support LoRA configuration. "
                f"Expected MLXModelWrapper, got {type(model)}"
            )

        return model

    @staticmethod
    def for_inference(
        model: Any,
        use_cache: bool = True,
    ) -> Any:
        """
        Prepare model for optimized inference.

        This method configures the model for inference by disabling dropout,
        enabling caching, and applying MLX-specific optimizations.

        Args:
            model: The model to prepare for inference
            use_cache: Whether to use KV caching for faster generation

        Returns:
            Model configured for inference

        Note:
            - Disables dropout and training-specific features
            - Enables key-value caching for autoregressive generation
            - Applies MLX memory optimizations
        """

        if hasattr(model, 'enable_inference_mode'):
            model.enable_inference_mode(use_cache=use_cache)
        else:
            warnings.warn(
                f"Model does not support inference mode configuration. "
                f"Expected MLXModelWrapper, got {type(model)}"
            )

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
        Convert a HuggingFace model to MLX format.

        Args:
            hf_model: HuggingFace model ID (e.g., "meta-llama/Llama-3-8B")
            output_dir: Output directory for MLX model
            quantize: Whether to quantize the model
            q_bits: Quantization bits (4, 8)
            dtype: Data type ("float16", "bfloat16", "float32")
            upload_repo: Optional HF repo to upload converted model

        Example:
            >>> FastLanguageModel.convert(
            ...     "meta-llama/Llama-3-8B",
            ...     output_dir="./llama3-mlx",
            ...     quantize=True,
            ...     q_bits=4,
            ... )
        """
        try:
            from mlx_lm import convert
        except ImportError:
            raise ImportError("mlx-lm is required for model conversion: uv pip install mlx-lm")

        print(f"Converting model: {hf_model} -> {output_dir}")
        convert(
            hf_path=hf_model,
            mlx_path=output_dir,
            quantize=quantize,
            q_bits=q_bits,
            dtype=dtype,
            upload_repo=upload_repo,
        )
        print(f"Conversion complete: {output_dir}")


class MLXModelWrapper:
    """
    Wrapper around MLX models to provide Unsloth-compatible interface.

    This class wraps MLX models and provides methods compatible with Unsloth's
    expected API, including LoRA configuration and inference optimization.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_seq_length: Optional[int] = None,
        model_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the MLX model wrapper.

        Args:
            model: The MLX model instance
            tokenizer: The tokenizer instance
            max_seq_length: Maximum sequence length
            model_name: Name/path of the model
            config: Model configuration dict (for saving)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.model_name = model_name
        self.config = config  # Store for saving

        # LoRA configuration
        self.lora_config = None
        self.lora_enabled = False
        self._lora_applied = False  # Track if LoRA has been applied to model layers

        # Adapter path tracking
        self._adapter_path: Optional[Path] = None

        # Inference mode flag
        self.inference_mode = False
        self.use_cache = True

    def configure_lora(
        self,
        r: int = 16,
        target_modules: Optional[List[str]] = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        bias: str = "none",
        use_gradient_checkpointing: Union[bool, str] = False,
        random_state: int = 3407,
        **kwargs
    ):
        """
        Configure LoRA parameters for this model.

        Args:
            r: LoRA rank
            target_modules: Target modules for LoRA
            lora_alpha: LoRA alpha scaling
            lora_dropout: LoRA dropout rate
            bias: Bias configuration
            use_gradient_checkpointing: Gradient checkpointing mode. Defaults to
                ``False`` for fast training. Pass ``True`` or ``"unsloth"`` to
                halve activation memory at ~2× step-time cost — needed for
                long-context (ctx≥4096) preference/RL training and for large
                models on tight-RAM Macs. ``True`` and ``"unsloth"`` behave
                identically; mlx-tune doesn't ship Unsloth's custom GC kernel.
            random_state: Random seed
            **kwargs: Additional configuration
        """
        self.lora_config = {
            "r": r,
            "target_modules": target_modules or [],
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "bias": bias,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "random_state": random_state,
            **kwargs
        }
        self.lora_enabled = True
        self._lora_applied = False  # Reset - needs to be applied again

        # Store for later use in training
        print(f"LoRA configuration set: rank={r}, alpha={lora_alpha}, "
              f"modules={target_modules}, dropout={lora_dropout}")

    def _apply_lora(self, num_layers: Optional[int] = None) -> bool:
        """
        Apply LoRA adapters to model layers using mlx_lm's native API.

        This method actually modifies the model's layers to include LoRA adapters.
        It should be called before training starts.

        Args:
            num_layers: Number of transformer layers to apply LoRA to.
                       If None, applies to all layers.

        Returns:
            True if LoRA was applied, False if already applied or not enabled.

        Raises:
            RuntimeError: If mlx_lm.tuner is not available.
        """
        if not self.lora_enabled:
            print("LoRA not configured. Call configure_lora() first.")
            return False

        if self._lora_applied:
            print("LoRA already applied to model layers.")
            return False

        if not HAS_MLX_LM_TUNER:
            raise RuntimeError(
                "mlx_lm.tuner is not available. Install with: pip install 'mlx-lm[train]'"
            )

        # Determine number of layers - must be detected, no silent fallback
        if num_layers is None:
            # Try to detect from model structure
            if hasattr(self.model, 'layers'):
                num_layers = len(self.model.layers)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                num_layers = len(self.model.model.layers)
            else:
                raise ValueError(
                    "Could not detect number of layers in model. "
                    "Please specify num_layers explicitly when calling _apply_lora() or in SFTConfig."
                )

        # Convert lora_alpha to scale: scale = alpha / r
        r = self.lora_config['r']
        lora_alpha = self.lora_config['lora_alpha']
        scale = lora_alpha / r

        # Build mlx_lm LoRA config
        mlx_lora_config = {
            "rank": r,
            "scale": scale,
            "dropout": self.lora_config.get('lora_dropout', 0.0),
        }

        # Resolve target module short names to full layer-relative paths.
        # Uses dynamic inspection to handle both dense and MoE architectures.
        # IMPORTANT: Must resolve BEFORE linear_to_lora_layers() changes module types
        # (nn.Linear → LoRALinear, SwitchLinear → LoRASwitchLinear).
        target_modules = self.lora_config.get('target_modules', [])
        if target_modules:
            full_paths = _resolve_target_modules(self.model, target_modules)
            mlx_lora_config["keys"] = full_paths
            # Cache resolved keys for later use by _save_adapter_config()
            self._resolved_lora_keys = full_paths

        # Check for DoRA
        use_dora = self.lora_config.get('use_dora', False)

        print(f"Applying LoRA to {num_layers} layers: {mlx_lora_config}")

        # CRITICAL: Freeze base model first, then apply LoRA
        # This ensures only LoRA parameters are trainable
        self.model.freeze()

        # Apply LoRA using mlx_lm utility
        # This creates LoRALinear layers which are unfrozen by default
        linear_to_lora_layers(
            model=self.model,
            num_layers=num_layers,
            config=mlx_lora_config,
            use_dora=use_dora,
        )

        self._lora_applied = True

        # Verify trainable parameters
        from mlx.utils import tree_flatten
        trainable = tree_flatten(self.model.trainable_parameters())
        lora_params = [k for k, _ in trainable if 'lora' in k]
        print(f"✓ LoRA applied successfully to {num_layers} layers")
        print(f"  Trainable LoRA parameters: {len(lora_params)}")

        return True

    def set_adapter_path(self, path: str) -> None:
        """
        Set the path where adapters will be saved/loaded.

        Args:
            path: Path to adapter directory or file.
        """
        self._adapter_path = Path(path)

    def get_adapter_path(self) -> Optional[Path]:
        """
        Get the current adapter path.

        Returns:
            Path to adapters, or None if not set.
        """
        return self._adapter_path

    def enable_inference_mode(self, use_cache: bool = True):
        """
        Enable inference mode optimizations.

        Args:
            use_cache: Whether to enable KV caching
        """
        self.inference_mode = True
        self.use_cache = use_cache
        print("Inference mode enabled with KV caching")

    def generate(self, *args, **kwargs):
        """
        Generate text using the model.

        This method provides a compatible interface for text generation,
        delegating to MLX's generation utilities.

        Args:
            *args: Positional arguments passed to generate
            **kwargs: Keyword arguments including:
                - prompt: Text prompt for generation
                - max_tokens: Maximum number of tokens to generate
                - temp: Temperature for sampling (default: 0.0)
                - input_ids: Alternative to prompt (will be decoded)

        Returns:
            Generated text string
        """
        from mlx_lm import generate

        # If input_ids is provided, we need to decode it first for MLX
        if "input_ids" in kwargs:
            input_ids = kwargs.pop("input_ids")
            # MLX generate expects a prompt string
            prompt = self.tokenizer.decode(input_ids[0])
            return generate(self.model, self.tokenizer, prompt=prompt, **kwargs)

        return generate(self.model, self.tokenizer, *args, **kwargs)

    def stream_generate(self, prompt: str, **kwargs):
        """
        Generate text with streaming output.

        This method yields tokens as they are generated, useful for
        real-time applications and chat interfaces.

        Args:
            prompt: Text prompt for generation
            **kwargs: Keyword arguments including:
                - max_tokens: Maximum number of tokens to generate
                - temp: Temperature for sampling (default: 0.0)

        Yields:
            Generated text chunks as they become available

        Example:
            >>> for chunk in model.stream_generate("Tell me about AI"):
            ...     print(chunk, end="", flush=True)
        """
        from mlx_lm import stream_generate

        for chunk in stream_generate(self.model, self.tokenizer, prompt=prompt, **kwargs):
            yield chunk

    def save_pretrained(self, output_dir: str, **kwargs):
        """
        Save LoRA adapters (Unsloth-compatible API).

        Args:
            output_dir: Directory to save adapters
            **kwargs: Additional save options

        Example:
            >>> model.save_pretrained("lora_model")
        """
        import shutil

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving LoRA adapters to {output_dir}...")

        # Check for adapter file in tracked path first, then fallback locations
        adapter_locations = []

        # 1. Tracked adapter path (set by trainer)
        if self._adapter_path:
            if self._adapter_path.is_file():
                adapter_locations.append(self._adapter_path)
            else:
                adapter_locations.append(self._adapter_path / "adapters.safetensors")

        # 2. Fallback: common locations
        adapter_locations.extend([
            Path("./adapters/adapters.safetensors"),
            Path("./lora_finetuned/adapters/adapters.safetensors"),
            Path("./outputs/adapters/adapters.safetensors"),
        ])

        # Find first existing adapter file
        adapter_file = None
        for loc in adapter_locations:
            if loc.exists():
                adapter_file = loc
                break

        if adapter_file and adapter_file.exists():
            shutil.copy(adapter_file, output_dir / "adapters.safetensors")
            print(f"✓ Adapters saved to {output_dir}")

            # Also copy adapter config if it exists
            config_file = adapter_file.parent / "adapter_config.json"
            if config_file.exists():
                shutil.copy(config_file, output_dir / "adapter_config.json")
        else:
            searched = [str(loc) for loc in adapter_locations[:3]]
            print(f"⚠️  No adapters found. Searched: {searched}")
            print("   Train the model first with SFTTrainer")

    def load_adapter(self, adapter_path: str, **kwargs):
        """
        Load LoRA adapters from a saved adapter directory.

        This allows loading fine-tuned adapters into a base model for inference.

        Args:
            adapter_path: Path to directory containing adapters.safetensors and adapter_config.json
            **kwargs: Additional options

        Example:
            >>> model, tokenizer = FastLanguageModel.from_pretrained("base-model")
            >>> model.load_adapter("lora_model")  # Load saved adapters
            >>> # Now model has the fine-tuned weights loaded
        """
        from mlx_lm.tuner.utils import load_adapters

        adapter_path = Path(adapter_path)

        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

        # Check for required files
        adapter_file = adapter_path / "adapters.safetensors"
        config_file = adapter_path / "adapter_config.json"

        if not adapter_file.exists():
            raise FileNotFoundError(
                f"adapters.safetensors not found in {adapter_path}. "
                "Make sure you saved the adapters with model.save_pretrained()"
            )

        if not config_file.exists():
            raise FileNotFoundError(
                f"adapter_config.json not found in {adapter_path}. "
                "This file is required by mlx_lm to load adapters. "
                "Re-train with mlx-tune >= 0.3.4 which saves this file."
            )

        print(f"Loading adapters from {adapter_path}...")

        # Get the actual model
        actual_model = self.model if hasattr(self, 'model') else self

        # Load adapters using mlx_lm
        self.model = load_adapters(actual_model, str(adapter_path))

        # Mark that LoRA is now applied
        self._lora_applied = True
        self._adapter_path = adapter_path

        print(f"✓ Adapters loaded successfully")

    def save_pretrained_merged(
        self,
        output_dir: str,
        tokenizer: Any,
        save_method: str = "merged_16bit",
        **kwargs
    ):
        """
        Save merged model (base + adapters) in HuggingFace format.

        Args:
            output_dir: Directory to save merged model
            tokenizer: Tokenizer to save
            save_method: Save method ("merged_16bit", "merged_4bit", etc.)
            **kwargs: Additional options

        Example:
            >>> model.save_pretrained_merged("merged_model", tokenizer)
        """
        from mlx_tune.trainer import save_model_hf_format

        print(f"Saving merged model to {output_dir}...")
        save_model_hf_format(self, tokenizer, output_dir, **kwargs)

    def save_pretrained_gguf(
        self,
        output_dir: str,
        tokenizer: Any,
        quantization_method: str = "q4_k_m",
        **kwargs
    ):
        """
        Save model in GGUF format for llama.cpp, Ollama, LM Studio, etc.

        This method exports the model (optionally with fused LoRA adapters) to GGUF format
        for use with llama.cpp, Ollama, LM Studio, and other GGUF-compatible tools.

        Args:
            output_dir: Directory/filename for GGUF file
            tokenizer: Tokenizer
            quantization_method: GGUF quantization type (for documentation only,
                               mlx_lm exports in fp16)
            **kwargs: Additional options including:
                - dequantize: Whether to dequantize the model before export

        Example:
            >>> # With non-quantized model (recommended)
            >>> model.save_pretrained_gguf("model", tokenizer)

            >>> # With quantized model (requires dequantize)
            >>> model.save_pretrained_gguf("model", tokenizer, dequantize=True)

        Important - Quantized Model Limitation:
            GGUF export from quantized (4-bit) base models is NOT supported by mlx_lm.
            This is an upstream limitation, not an mlx-tune bug.
            See: https://github.com/ml-explore/mlx-lm/issues/353

            Workarounds:
            1. Use a non-quantized base model (e.g., "Llama-3.2-1B-Instruct" not "-4bit")
            2. Use dequantize=True (creates large fp16 file, re-quantize with llama.cpp)
            3. Skip GGUF and use save_pretrained_merged() for MLX-only inference

        Note:
            - Supported architectures: Llama, Mistral, Mixtral
            - Output is fp16 precision (use llama.cpp to quantize further)
        """
        from mlx_tune.trainer import export_to_gguf
        from pathlib import Path

        output_path = Path(output_dir)
        if not output_path.suffix:
            output_path = output_path / "model.gguf"

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get the original model path/name - this is what mlx_lm.fuse needs
        model_path = self.model_name
        if model_path is None:
            raise ValueError(
                "Cannot export to GGUF: model_name is not set. "
                "The model must be loaded with FastLanguageModel.from_pretrained() "
                "to track the original model path."
            )

        # Check for adapter path if LoRA was applied
        adapter_path = None
        if self._lora_applied:
            if self._adapter_path:
                adapter_path = str(self._adapter_path)
            else:
                # Check common adapter locations
                common_paths = [
                    Path("./adapters"),
                    Path("./lora_finetuned/adapters"),
                    Path("./outputs/adapters"),
                ]
                for path in common_paths:
                    if (path / "adapters.safetensors").exists():
                        adapter_path = str(path)
                        break

            if adapter_path:
                print(f"  LoRA adapters will be fused from: {adapter_path}")
            else:
                print("  Warning: LoRA was applied but no adapter path found.")
                print("  Export will use base model only. Train and save adapters first.")

        print(f"Exporting to GGUF format...")
        export_to_gguf(
            model_path,  # Use original model path, not output directory
            output_path=str(output_path),
            quantization=quantization_method,
            adapter_path=adapter_path,
            **kwargs
        )

    def __call__(self, *args, **kwargs):
        """
        Forward pass through the model.

        Note: This is a simplified interface. For training, use MLX's
        training utilities directly.
        """
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying MLX model.
        """
        return getattr(self.model, name)
