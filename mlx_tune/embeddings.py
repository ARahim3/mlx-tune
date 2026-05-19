"""
Embedding Model Fine-Tuning for MLX-Tune.

Provides Unsloth-compatible API for fine-tuning sentence embedding models
(BERT, ModernBERT, Qwen3-Embedding, Harrier, etc.) on Apple Silicon using
contrastive learning with LoRA.

Uses mlx-embeddings for model loading and MLX for native training.
"""

from typing import Optional, Tuple, Union, List, Any, Dict
from pathlib import Path
import json
import warnings

import mlx.core as mx
import mlx.nn as nn

from mlx_embeddings.utils import load as mlx_emb_load


def _load_with_relaxed_weights(model_name: str, tokenizer_config: dict = {}) -> Tuple:
    """Load embedding model with strict=False to handle missing optional weights.

    Some models (e.g., Harrier) don't include dense projection layers that
    mlx-embeddings model classes expect. This loader tolerates missing weights.
    """
    import glob
    from mlx_embeddings.utils import (
        get_model_path,
        load_config,
        load_tokenizer,
    )
    from mlx_embeddings.utils import _get_classes

    model_path = get_model_path(model_name)
    config = load_config(model_path)

    # Find weight files (same logic as mlx_embeddings.utils.load_model)
    weight_files = glob.glob(str(model_path / "**/model*.safetensors"), recursive=True)
    if not weight_files:
        weight_files = glob.glob(str(model_path / "weight*.safetensors"))

    weights = {}
    for wf in weight_files:
        loaded = mx.load(wf)
        if Path(wf).parent != model_path:
            folder_name = Path(wf).parent.name
            weights.update({f"{folder_name}.{k}": v for k, v in loaded.items()})
        else:
            weights.update(loaded)

    # Create model instance
    model_class, model_args_class, text_config, vision_config = _get_classes(config)
    model_args = model_args_class.from_dict(config)
    if text_config is not None:
        model_args.text_config = text_config(**model_args.text_config)
    if vision_config is not None:
        model_args.vision_config = vision_config(**model_args.vision_config)

    model = model_class(model_args)

    # Sanitize weights
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # Load with strict=False — tolerates missing dense projection weights
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    # Load tokenizer
    tokenizer = load_tokenizer(model_path, tokenizer_config)
    return model, tokenizer


def _create_lora_linear(
    original: nn.Linear,
    r: int,
    scale: float,
    dropout: float,
) -> nn.Linear:
    """Replace an nn.Linear with a LoRALinear layer."""
    try:
        from mlx_lm.tuner.lora import LoRALinear
    except ImportError:
        raise ImportError(
            "mlx_lm.tuner.lora.LoRALinear not available. "
            "Please install mlx-lm: uv pip install mlx-lm"
        )

    return LoRALinear.from_base(original, r=r, scale=scale, dropout=dropout)


# ──────────────────────────────────────────────────────────────────────────────
# Architecture detection and LoRA target defaults
# ──────────────────────────────────────────────────────────────────────────────

# Maps architecture name → (block_path_parts, default_lora_targets)
_ARCH_CONFIG = {
    "bert": {
        "block_path": ["encoder", "layer"],
        "targets": ["query", "key", "value"],
        "attn_attr": "attention.self",
    },
    "xlm-roberta": {
        "block_path": ["encoder", "layer"],
        "targets": ["query", "key", "value"],
        "attn_attr": "attention.self",
    },
    "modernbert": {
        "block_path": ["model", "layers"],
        "targets": ["Wqkv"],
        "attn_attr": "attn",
    },
    "qwen3": {
        "block_path": ["model", "layers"],
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "attn_attr": "self_attn",
    },
    "gemma": {
        "block_path": ["model", "layers"],
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "attn_attr": "self_attn",
    },
}


def _detect_architecture(model: Any, config: Optional[Dict] = None) -> str:
    """Detect embedding model architecture from model structure or config."""
    # Check config model_type first
    if config:
        model_type = config.get("model_type", "").lower()
        if "modernbert" in model_type:
            return "modernbert"
        if "qwen" in model_type:
            return "qwen3"
        if "gemma" in model_type:
            return "gemma"
        if "roberta" in model_type or "xlm" in model_type:
            return "xlm-roberta"
        if "bert" in model_type:
            return "bert"

    # Fallback: inspect model structure
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return "bert"
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # Check for specific attributes to distinguish
        layers = model.model.layers
        if len(layers) > 0:
            first = layers[0]
            if hasattr(first, "attn") and hasattr(first.attn, "Wqkv"):
                return "modernbert"
            if hasattr(first, "self_attn") and hasattr(first.self_attn, "q_proj"):
                return "qwen3"
        return "qwen3"  # Default decoder-style

    return "bert"  # Default


def _get_encoder_blocks(model: Any, arch: str) -> Optional[list]:
    """Navigate to the list of encoder/transformer blocks."""
    arch_cfg = _ARCH_CONFIG.get(arch, _ARCH_CONFIG["bert"])
    obj = model
    for part in arch_cfg["block_path"]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    # Convert to list if needed
    if hasattr(obj, "__iter__"):
        return list(obj)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# EmbeddingModelWrapper
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingModelWrapper:
    """
    Wrapper around an MLX embedding model with LoRA management.

    Provides encoding, pooling, and adapter save/load functionality.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_name: str,
        max_seq_length: int = 512,
        pooling_strategy: str = "mean",
        config: Optional[Dict] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.pooling_strategy = pooling_strategy
        self.config = config or {}

        # LoRA state
        self.lora_config = None
        self.lora_enabled = False
        self._lora_applied = False
        self._adapter_path = None

        # Detect architecture
        self.architecture = _detect_architecture(model, config)
        self._arch_cfg = _ARCH_CONFIG.get(self.architecture, _ARCH_CONFIG["bert"])

    def configure_lora(
        self,
        r: int = 16,
        target_modules: Optional[List[str]] = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        bias: str = "none",
        use_gradient_checkpointing: Union[bool, str] = False,
        random_state: int = 3407,
        **kwargs,
    ):
        """Configure LoRA parameters (applied later during training)."""
        if target_modules is None:
            target_modules = list(self._arch_cfg["targets"])

        self.lora_config = {
            "r": r,
            "target_modules": target_modules,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "bias": bias,
            "random_state": random_state,
        }
        self.lora_enabled = True

    def _apply_lora(self) -> bool:
        """Apply LoRA adapters to the model's encoder blocks."""
        if self._lora_applied:
            return True
        if not self.lora_enabled or self.lora_config is None:
            return False

        r = self.lora_config["r"]
        lora_alpha = self.lora_config["lora_alpha"]
        scale = lora_alpha / r
        dropout = self.lora_config["lora_dropout"]
        target_modules = self.lora_config["target_modules"]

        # Freeze all parameters first
        self.model.freeze()

        # Get encoder blocks
        blocks = _get_encoder_blocks(self.model, self.architecture)
        if blocks is None:
            warnings.warn(
                f"Could not find encoder blocks for architecture '{self.architecture}'. "
                f"LoRA was not applied."
            )
            return False

        total_replaced = 0
        attn_attr = self._arch_cfg["attn_attr"]

        for block in blocks:
            total_replaced += self._apply_lora_to_block(
                block, target_modules, r, scale, dropout, attn_attr
            )

        self._lora_applied = True

        # Count trainable parameters
        from mlx.utils import tree_flatten
        trainable = tree_flatten(self.model.trainable_parameters())
        lora_params = [k for k, _ in trainable if "lora" in k.lower()]
        print(f"LoRA applied: {len(lora_params)} trainable parameter groups ({total_replaced} layers replaced)")
        print(f"  Architecture: {self.architecture}")
        print(f"  Blocks: {len(blocks)}")
        print(f"  Targets: {target_modules}")

        return True

    def _apply_lora_to_block(
        self,
        block: Any,
        target_modules: List[str],
        r: int,
        scale: float,
        dropout: float,
        attn_attr: str,
    ) -> int:
        """Apply LoRA to a single encoder block."""
        replaced = 0

        # Navigate to the attention module
        # attn_attr can be dotted like "attention.self"
        parts = attn_attr.split(".")
        attn_module = block
        for part in parts:
            attn_module = getattr(attn_module, part, None)
            if attn_module is None:
                break

        # Types eligible for LoRA replacement
        _linear_types = (nn.Linear, nn.QuantizedLinear)

        if attn_module is not None:
            for module_name in target_modules:
                if hasattr(attn_module, module_name):
                    original = getattr(attn_module, module_name)
                    if isinstance(original, _linear_types):
                        lora_layer = _create_lora_linear(original, r, scale, dropout)
                        setattr(attn_module, module_name, lora_layer)
                        replaced += 1

        # Also check MLP modules if target_modules includes them
        # (e.g., ModernBERT has Wi, Wf in mlp)
        mlp_modules = ["mlp", "intermediate", "output"]
        for mlp_attr in mlp_modules:
            mlp_module = getattr(block, mlp_attr, None)
            if mlp_module is not None:
                for module_name in target_modules:
                    if hasattr(mlp_module, module_name):
                        original = getattr(mlp_module, module_name)
                        if isinstance(original, _linear_types):
                            lora_layer = _create_lora_linear(original, r, scale, dropout)
                            setattr(mlp_module, module_name, lora_layer)
                            replaced += 1

        return replaced

    def _pool(self, last_hidden_state: mx.array, attention_mask: mx.array) -> mx.array:
        """Apply pooling strategy to hidden states."""
        if self.pooling_strategy == "cls":
            return last_hidden_state[:, 0, :]
        elif self.pooling_strategy == "last_token":
            # For decoder-based models (Qwen3-Embedding, Harrier, etc.).
            # Vectorized gather — no per-sample .item() sync.
            seq_lengths = attention_mask.sum(axis=1).astype(mx.int32) - 1  # (B,)
            B, _, D = last_hidden_state.shape
            idx = mx.broadcast_to(seq_lengths[:, None, None], (B, 1, D))
            gathered = mx.take_along_axis(last_hidden_state, idx, axis=1)
            return gathered.squeeze(1)
        else:
            # Mean pooling (default)
            mask_expanded = attention_mask[:, :, None].astype(last_hidden_state.dtype)
            sum_embeddings = (last_hidden_state * mask_expanded).sum(axis=1)
            sum_mask = mask_expanded.sum(axis=1)
            return sum_embeddings / mx.maximum(sum_mask, 1e-9)

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> mx.array:
        """
        Encode texts to embedding vectors.

        Args:
            texts: Single text or list of texts to encode.
            normalize: Whether to L2-normalize embeddings.
            batch_size: Batch size for encoding.

        Returns:
            Embeddings of shape [num_texts, embed_dim].
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="np",
            )

            input_ids = mx.array(inputs["input_ids"])
            attention_mask = mx.array(inputs["attention_mask"])

            # Forward pass
            outputs = self.model(input_ids, attention_mask=attention_mask)

            # Extract hidden states
            if hasattr(outputs, "last_hidden_state"):
                hidden = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                hidden = outputs[0]
            else:
                hidden = outputs

            # Pool
            pooled = self._pool(hidden, attention_mask)

            # Normalize
            if normalize:
                norms = mx.linalg.norm(pooled, axis=-1, keepdims=True)
                pooled = pooled / mx.maximum(norms, 1e-9)

            mx.eval(pooled)
            all_embeddings.append(pooled)

        return mx.concatenate(all_embeddings, axis=0)

    def save_pretrained(self, output_dir: str, **kwargs):
        """Save LoRA adapter weights."""
        from mlx.utils import tree_flatten

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save adapter weights
        trainable = dict(tree_flatten(self.model.trainable_parameters()))
        if trainable:
            mx.savez(str(output_path / "adapters.npz"), **trainable)

        # Save adapter config
        adapter_config = {
            "model_name": self.model_name,
            "architecture": self.architecture,
            "pooling_strategy": self.pooling_strategy,
            "max_seq_length": self.max_seq_length,
            "lora_config": self.lora_config,
        }
        with open(output_path / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2)

        print(f"Adapter saved to {output_path}")

    def load_adapter(self, adapter_path: str, **kwargs):
        """
        Load saved LoRA adapter weights.

        The model must already have LoRA applied (call get_peft_model + _apply_lora first).

        Args:
            adapter_path: Path to directory containing adapters.npz and adapter_config.json.
        """
        adapter_dir = Path(adapter_path)
        npz_path = adapter_dir / "adapters.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"No adapters.npz found at {adapter_dir}")

        weights = mx.load(str(npz_path))
        self.model.load_weights(list(weights.items()), strict=False)
        mx.eval(self.model.parameters())

        # Load config if available
        config_path = adapter_dir / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                adapter_config = json.load(f)
            if "pooling_strategy" in adapter_config:
                self.pooling_strategy = adapter_config["pooling_strategy"]

        self._adapter_path = adapter_path
        print(f"Adapter loaded from {adapter_dir} ({len(weights)} weight arrays)")

    def enable_inference_mode(self, **kwargs):
        """Set model to evaluation mode."""
        self.model.eval()


# ──────────────────────────────────────────────────────────────────────────────
# FastEmbeddingModel
# ──────────────────────────────────────────────────────────────────────────────

class FastEmbeddingModel:
    """
    Unsloth-compatible API for loading and fine-tuning embedding models on MLX.

    Supports BERT, XLM-RoBERTa, ModernBERT, Qwen3-Embedding, Harrier, and other
    sentence-transformers compatible models.

    Example:
        >>> from mlx_tune import FastEmbeddingModel
        >>> model, tokenizer = FastEmbeddingModel.from_pretrained(
        ...     "mlx-community/all-MiniLM-L6-v2",
        ...     max_seq_length=256,
        ... )
        >>> model = FastEmbeddingModel.get_peft_model(model, r=16)
    """

    @staticmethod
    def from_pretrained(
        model_name: str,
        max_seq_length: int = 512,
        dtype: Optional[Any] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        token: Optional[str] = None,
        trust_remote_code: bool = False,
        pooling_strategy: str = "mean",
        **kwargs,
    ) -> Tuple[EmbeddingModelWrapper, Any]:
        """
        Load a pretrained embedding model via mlx-embeddings.

        Args:
            model_name: HuggingFace model ID (e.g., "mlx-community/all-MiniLM-L6-v2").
            max_seq_length: Maximum sequence length for tokenization.
            dtype: Data type (auto-selected by MLX).
            load_in_4bit: Use 4-bit quantized model (use mlx-community 4bit variants).
            load_in_8bit: Use 8-bit quantized model.
            token: HuggingFace API token.
            trust_remote_code: Whether to trust remote code.
            pooling_strategy: Pooling strategy - "mean", "cls", or "last_token".
            **kwargs: Additional arguments passed to mlx_embeddings.load().

        Returns:
            Tuple of (EmbeddingModelWrapper, tokenizer).
        """
        tokenizer_config = {}
        if trust_remote_code:
            tokenizer_config["trust_remote_code"] = True
        if token:
            tokenizer_config["token"] = token

        try:
            try:
                model, tokenizer = mlx_emb_load(
                    model_name,
                    tokenizer_config=tokenizer_config,
                    **kwargs,
                )
            except (ValueError, RuntimeError) as load_err:
                # Some models (e.g., Harrier) lack optional dense projection
                # weights that mlx-embeddings model classes define.
                # Retry with strict=False to tolerate missing weights.
                if "Missing" in str(load_err) and "dense" in str(load_err):
                    model, tokenizer = _load_with_relaxed_weights(
                        model_name, tokenizer_config=tokenizer_config,
                    )
                else:
                    raise

            # Try to load config for architecture detection
            config = None
            try:
                from huggingface_hub import hf_hub_download
                config_path = hf_hub_download(model_name, "config.json")
                with open(config_path) as f:
                    config = json.load(f)
            except Exception:
                pass

            wrapper = EmbeddingModelWrapper(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                max_seq_length=max_seq_length,
                pooling_strategy=pooling_strategy,
                config=config,
            )

            return wrapper, tokenizer

        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'. "
                f"Error: {str(e)}\n\n"
                f"Tips:\n"
                f"- Ensure mlx-embeddings is installed: uv pip install mlx-embeddings\n"
                f"- For pre-converted models, check mlx-community on HuggingFace\n"
                f"- Supported: BERT, XLM-RoBERTa, ModernBERT, Qwen3-Embedding, Harrier"
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
        **kwargs,
    ) -> Any:
        """
        Add LoRA adapters to the embedding model.

        Args:
            model: EmbeddingModelWrapper from from_pretrained().
            r: LoRA rank.
            target_modules: Modules to apply LoRA to. Auto-detected if None.
            lora_alpha: LoRA scaling parameter.
            lora_dropout: Dropout probability for LoRA layers.
            bias: Bias configuration.
            use_gradient_checkpointing: Enable gradient checkpointing.
            random_state: Random seed.
            **kwargs: Additional LoRA configuration.

        Returns:
            Model with LoRA configured.
        """
        if not isinstance(model, EmbeddingModelWrapper):
            raise TypeError(
                f"Expected EmbeddingModelWrapper, got {type(model)}. "
                f"Use FastEmbeddingModel.from_pretrained() first."
            )

        model.configure_lora(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
            **kwargs,
        )

        return model

    @staticmethod
    def for_inference(model: Any) -> Any:
        """Prepare model for inference."""
        if hasattr(model, "enable_inference_mode"):
            model.enable_inference_mode()
        return model

    @staticmethod
    def for_training(model: Any) -> Any:
        """Prepare model for training."""
        if hasattr(model, "model") and hasattr(model.model, "train"):
            model.model.train()
        return model


# ──────────────────────────────────────────────────────────────────────────────
# EmbeddingSFTConfig
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingSFTConfig:
    """
    Training configuration for embedding model fine-tuning.

    Example:
        >>> config = EmbeddingSFTConfig(
        ...     per_device_train_batch_size=32,
        ...     learning_rate=2e-5,
        ...     max_steps=100,
        ...     loss_type="infonce",
        ...     temperature=0.05,
        ... )
    """

    def __init__(
        self,
        per_device_train_batch_size: int = 32,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 10,
        max_steps: Optional[int] = 100,
        num_train_epochs: int = 1,
        learning_rate: float = 2e-5,
        logging_steps: int = 1,
        output_dir: str = "./embedding_outputs",
        lr_scheduler_type: str = "cosine",
        weight_decay: float = 0.01,
        seed: int = 3407,
        save_steps: Optional[int] = None,
        # Embedding-specific
        loss_type: str = "infonce",
        temperature: float = 0.05,
        margin: float = 1.0,
        normalize_embeddings: bool = True,
        anchor_column: str = "anchor",
        positive_column: str = "positive",
        negative_column: Optional[str] = None,
        max_seq_length: int = 512,
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
        self.save_steps = save_steps
        # Embedding-specific
        self.loss_type = loss_type
        self.temperature = temperature
        self.margin = margin
        self.normalize_embeddings = normalize_embeddings
        self.anchor_column = anchor_column
        self.positive_column = positive_column
        self.negative_column = negative_column
        self.max_seq_length = max_seq_length

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# ──────────────────────────────────────────────────────────────────────────────
# EmbeddingDataCollator
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingDataCollator:
    """
    Data collator for embedding fine-tuning.

    Tokenizes anchor/positive (and optional negative) text pairs into batches.

    Example:
        >>> collator = EmbeddingDataCollator(
        ...     model=model, tokenizer=tokenizer,
        ...     anchor_column="anchor", positive_column="positive",
        ... )
    """

    # Common column name patterns
    _ANCHOR_PATTERNS = ["anchor", "query", "question", "sentence1", "text1", "sent1"]
    _POSITIVE_PATTERNS = ["positive", "passage", "answer", "sentence2", "text2", "sent2", "pos"]
    _NEGATIVE_PATTERNS = ["negative", "hard_negative", "neg", "sentence3", "text3"]

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        anchor_column: str = "anchor",
        positive_column: str = "positive",
        negative_column: Optional[str] = None,
        max_seq_length: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.anchor_column = anchor_column
        self.positive_column = positive_column
        self.negative_column = negative_column
        self.max_seq_length = max_seq_length

    def _tokenize_texts(self, texts: List[str]) -> Dict[str, mx.array]:
        """Tokenize a list of texts and pad to the next bucket length.

        Bucketing keeps `mx.compile`'s per-shape cache small — without it,
        every batch with a unique sequence length triggers a recompile.
        """
        from mlx_tune._perf import bucket_length
        import numpy as np

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="np",
        )
        ids = encoded["input_ids"]
        mask = encoded["attention_mask"]
        cur_len = ids.shape[1]
        target_len = min(bucket_length(cur_len), self.max_seq_length)
        if target_len > cur_len:
            try:
                pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
            except (TypeError, ValueError):
                pad_id = 0
            pad_w = target_len - cur_len
            ids = np.pad(ids, ((0, 0), (0, pad_w)), constant_values=pad_id)
            mask = np.pad(mask, ((0, 0), (0, pad_w)), constant_values=0)
        return {
            "input_ids": mx.array(ids),
            "attention_mask": mx.array(mask),
        }

    def __call__(self, samples: Union[List[Dict], Dict]) -> Dict[str, mx.array]:
        """
        Collate samples into a training batch.

        Args:
            samples: List of dicts with anchor/positive/negative text columns.

        Returns:
            Dict with tokenized anchor, positive, and optional negative tensors.
        """
        # Handle list of dicts vs dict of lists
        if isinstance(samples, dict):
            anchors = samples.get(self.anchor_column, [])
            positives = samples.get(self.positive_column, [])
            negatives = samples.get(self.negative_column, []) if self.negative_column else []
        else:
            anchors = [s[self.anchor_column] for s in samples]
            positives = [s[self.positive_column] for s in samples]
            negatives = (
                [s[self.negative_column] for s in samples]
                if self.negative_column and self.negative_column in samples[0]
                else []
            )

        batch = {}

        # Tokenize anchors
        anchor_tok = self._tokenize_texts(anchors)
        batch["anchor_input_ids"] = anchor_tok["input_ids"]
        batch["anchor_attention_mask"] = anchor_tok["attention_mask"]

        # Tokenize positives
        pos_tok = self._tokenize_texts(positives)
        batch["positive_input_ids"] = pos_tok["input_ids"]
        batch["positive_attention_mask"] = pos_tok["attention_mask"]

        # Tokenize negatives (optional)
        if negatives:
            neg_tok = self._tokenize_texts(negatives)
            batch["negative_input_ids"] = neg_tok["input_ids"]
            batch["negative_attention_mask"] = neg_tok["attention_mask"]

        return batch


# ──────────────────────────────────────────────────────────────────────────────
# EmbeddingSFTTrainer
# ──────────────────────────────────────────────────────────────────────────────

class _TrainerStats:
    """Simple container for training metrics."""

    def __init__(self, metrics: Dict):
        self.metrics = metrics


class EmbeddingSFTTrainer:
    """
    Trainer for embedding model fine-tuning with contrastive loss.

    Supports InfoNCE (MultipleNegativesRankingLoss), cosine embedding loss,
    and triplet loss.

    Example:
        >>> trainer = EmbeddingSFTTrainer(
        ...     model=model, tokenizer=tokenizer,
        ...     data_collator=collator, train_dataset=dataset,
        ...     args=EmbeddingSFTConfig(loss_type="infonce", temperature=0.05),
        ... )
        >>> result = trainer.train()
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
        self.wrapper = model if isinstance(model, EmbeddingModelWrapper) else None
        self.actual_model = model.model if isinstance(model, EmbeddingModelWrapper) else model
        self.tokenizer = tokenizer or (model.tokenizer if isinstance(model, EmbeddingModelWrapper) else None)
        self.data_collator = data_collator
        self.train_dataset = train_dataset

        # Parse training args
        if args is not None:
            self.learning_rate = getattr(args, "learning_rate", 2e-5)
            self.max_steps = getattr(args, "max_steps", None)
            self.num_train_epochs = getattr(args, "num_train_epochs", 1)
            self.batch_size = getattr(args, "per_device_train_batch_size", 32)
            self.gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 1)
            self.warmup_steps = getattr(args, "warmup_steps", 10)
            self.logging_steps = getattr(args, "logging_steps", 1)
            self.output_dir = getattr(args, "output_dir", "./embedding_outputs")
            self.lr_scheduler_type = getattr(args, "lr_scheduler_type", "cosine")
            self.weight_decay = getattr(args, "weight_decay", 0.01)
            self.seed = getattr(args, "seed", 3407)
            self.loss_type = getattr(args, "loss_type", "infonce")
            self.temperature = getattr(args, "temperature", 0.05)
            self.margin = getattr(args, "margin", 1.0)
            self.normalize_embeddings = getattr(args, "normalize_embeddings", True)
            self.save_steps = getattr(args, "save_steps", None)
        else:
            self.learning_rate = kwargs.get("learning_rate", 2e-5)
            self.max_steps = kwargs.get("max_steps", None)
            self.num_train_epochs = kwargs.get("num_train_epochs", 1)
            self.batch_size = kwargs.get("batch_size", 32)
            self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
            self.warmup_steps = kwargs.get("warmup_steps", 10)
            self.logging_steps = kwargs.get("logging_steps", 1)
            self.output_dir = kwargs.get("output_dir", "./embedding_outputs")
            self.lr_scheduler_type = kwargs.get("lr_scheduler_type", "cosine")
            self.weight_decay = kwargs.get("weight_decay", 0.01)
            self.seed = kwargs.get("seed", 3407)
            self.loss_type = kwargs.get("loss_type", "infonce")
            self.temperature = kwargs.get("temperature", 0.05)
            self.margin = kwargs.get("margin", 1.0)
            self.normalize_embeddings = kwargs.get("normalize_embeddings", True)
            self.save_steps = kwargs.get("save_steps", None)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def train(self) -> _TrainerStats:
        """
        Train the embedding model using contrastive learning.

        Returns:
            _TrainerStats with training metrics.
        """
        import mlx.optimizers as optim
        from mlx.utils import tree_map
        from mlx_tune.losses import infonce_loss, cosine_embedding_loss, triplet_loss
        from tqdm import tqdm

        from mlx_tune._perf import (
            compiled_step, configure_wired_limit,
            enable_grad_checkpoint, make_step_state,
        )

        print("=" * 70)
        print("Starting Embedding Fine-Tuning")
        print("=" * 70)

        configure_wired_limit()

        # Ensure LoRA is applied
        if self.wrapper and self.wrapper.lora_enabled and not self.wrapper._lora_applied:
            self.wrapper._apply_lora()
        if self.wrapper:
            enable_grad_checkpoint(self.wrapper)

        train_model = self.actual_model
        train_model.train()

        # Pooling helper
        pooling_strategy = self.wrapper.pooling_strategy if self.wrapper else "mean"

        def pool(hidden: mx.array, mask: mx.array) -> mx.array:
            if pooling_strategy == "cls":
                return hidden[:, 0, :]
            elif pooling_strategy == "last_token":
                # Vectorized gather of the last valid token per row — no
                # per-sample .item() syncs inside value_and_grad.
                seq_lengths = mask.sum(axis=1).astype(mx.int32) - 1  # (B,)
                B, _, D = hidden.shape
                idx = mx.broadcast_to(seq_lengths[:, None, None], (B, 1, D))
                gathered = mx.take_along_axis(hidden, idx, axis=1)
                return gathered.squeeze(1)
            else:
                mask_exp = mask[:, :, None].astype(hidden.dtype)
                return (hidden * mask_exp).sum(axis=1) / mx.maximum(mask_exp.sum(axis=1), 1e-9)

        # Loss function
        temperature = self.temperature
        margin = self.margin
        loss_type = self.loss_type
        normalize = self.normalize_embeddings

        def loss_fn(model, batch):
            # Encode anchors
            anchor_out = model(
                batch["anchor_input_ids"],
                attention_mask=batch["anchor_attention_mask"],
            )
            if hasattr(anchor_out, "last_hidden_state"):
                anchor_hidden = anchor_out.last_hidden_state
            elif isinstance(anchor_out, tuple):
                anchor_hidden = anchor_out[0]
            else:
                anchor_hidden = anchor_out

            anchor_embeds = pool(anchor_hidden, batch["anchor_attention_mask"])

            # Encode positives
            pos_out = model(
                batch["positive_input_ids"],
                attention_mask=batch["positive_attention_mask"],
            )
            if hasattr(pos_out, "last_hidden_state"):
                pos_hidden = pos_out.last_hidden_state
            elif isinstance(pos_out, tuple):
                pos_hidden = pos_out[0]
            else:
                pos_hidden = pos_out

            pos_embeds = pool(pos_hidden, batch["positive_attention_mask"])

            # Normalize
            if normalize:
                anchor_embeds = anchor_embeds / mx.maximum(
                    mx.linalg.norm(anchor_embeds, axis=-1, keepdims=True), 1e-9
                )
                pos_embeds = pos_embeds / mx.maximum(
                    mx.linalg.norm(pos_embeds, axis=-1, keepdims=True), 1e-9
                )

            # Compute loss
            if loss_type == "cosine":
                neg_embeds = None
                if "negative_input_ids" in batch:
                    neg_out = model(
                        batch["negative_input_ids"],
                        attention_mask=batch["negative_attention_mask"],
                    )
                    neg_hidden = neg_out.last_hidden_state if hasattr(neg_out, "last_hidden_state") else neg_out
                    neg_embeds = pool(neg_hidden, batch["negative_attention_mask"])
                    if normalize:
                        neg_embeds = neg_embeds / mx.maximum(
                            mx.linalg.norm(neg_embeds, axis=-1, keepdims=True), 1e-9
                        )
                return cosine_embedding_loss(anchor_embeds, pos_embeds, neg_embeds, margin)

            elif loss_type == "triplet":
                if "negative_input_ids" not in batch:
                    raise ValueError("Triplet loss requires negative samples")
                neg_out = model(
                    batch["negative_input_ids"],
                    attention_mask=batch["negative_attention_mask"],
                )
                neg_hidden = neg_out.last_hidden_state if hasattr(neg_out, "last_hidden_state") else neg_out
                neg_embeds = pool(neg_hidden, batch["negative_attention_mask"])
                return triplet_loss(anchor_embeds, pos_embeds, neg_embeds, margin)

            else:
                # Default: InfoNCE
                return infonce_loss(anchor_embeds, pos_embeds, temperature)

        loss_and_grad_fn = nn.value_and_grad(train_model, loss_fn)

        # Optimizer
        optimizer = optim.Adam(learning_rate=self.learning_rate)

        # Compiled step (mlx-lm pattern). `shapeless=True` because anchor /
        # positive / negative texts have variable seq length within a batch.
        # Closures over Python ints (grad_accum) are fine — they're folded
        # into the trace at compile time.
        grad_accum_local = self.gradient_accumulation_steps
        state = make_step_state(train_model, optimizer)

        def step(batch, prev_grad, do_update):
            loss, grad = loss_and_grad_fn(train_model, batch)
            if prev_grad is not None:
                grad = tree_map(lambda a, g: a + g, prev_grad, grad)
            if do_update:
                if grad_accum_local > 1:
                    grad = tree_map(lambda g: g / grad_accum_local, grad)
                optimizer.update(train_model, grad)
                return loss, None
            return loss, grad

        step = compiled_step(step, state)

        # Determine total steps
        dataset_len = len(self.train_dataset) if hasattr(self.train_dataset, "__len__") else 0
        if self.max_steps:
            total_steps = self.max_steps
        elif dataset_len > 0:
            steps_per_epoch = max(dataset_len // self.batch_size, 1)
            total_steps = steps_per_epoch * self.num_train_epochs
        else:
            total_steps = 100

        print(f"  Model: {self.wrapper.model_name if self.wrapper else 'unknown'}")
        print(f"  Architecture: {self.wrapper.architecture if self.wrapper else 'unknown'}")
        print(f"  Dataset: {dataset_len} samples")
        print(f"  Total steps: {total_steps}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Loss type: {self.loss_type}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Learning rate: {self.learning_rate}")

        grad_accum = self.gradient_accumulation_steps
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

                do_update = (micro_step + 1) >= grad_accum
                loss, accumulated_grads = step(batch, accumulated_grads, do_update)

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

                    if self.save_steps and step_counter % self.save_steps == 0 and self.wrapper:
                        ckpt_dir = Path(self.output_dir) / f"checkpoint-{step_counter}"
                        self.wrapper.save_pretrained(str(ckpt_dir))

        progress.close()

        # Save final adapters
        if self.wrapper:
            adapter_dir = Path(self.output_dir) / "adapters"
            self.wrapper.save_pretrained(str(adapter_dir))
            self.wrapper._adapter_path = adapter_dir

        avg_loss = total_loss / max(step_counter, 1)
        print(f"\nTraining complete! Average loss: {avg_loss:.4f}")

        return _TrainerStats({"train_loss": avg_loss, "train_runtime": 0})
