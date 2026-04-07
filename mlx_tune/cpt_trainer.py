"""
CPTTrainer - Continual Pretraining Trainer for MLX-Tune

Provides Unsloth-compatible continual pretraining (CPT) support.
CPT trains on raw text with loss on ALL tokens (no response masking),
and optionally includes embed_tokens + lm_head for vocabulary adaptation.

Key differences from SFT:
- Uses base models (NOT instruction-tuned)
- Raw text datasets (no chat template needed)
- Loss computed on ALL tokens
- Optional embed_tokens + lm_head training with decoupled learning rate
"""

from typing import Optional, Dict, Any, Union, List, Callable
from pathlib import Path
import json
import types
import warnings

import mlx.core as mx

# Try to import native training components
try:
    from mlx_lm.tuner.trainer import train as mlx_train, TrainingArgs
    from mlx_lm.tuner.datasets import load_dataset as mlx_load_dataset, CacheDataset
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_NATIVE_TRAINING = True
except ImportError:
    HAS_NATIVE_TRAINING = False
    warnings.warn(
        "Native training not available. Install with: uv pip install 'mlx-lm[train]'",
        ImportWarning
    )


class CPTConfig:
    """
    Configuration for Continual Pretraining.

    Compatible with Unsloth's CPT workflow. Key differences from SFTConfig:
    - ``include_embeddings``: auto-adds embed_tokens + lm_head to LoRA targets
    - ``embedding_learning_rate``: separate (typically smaller) LR for embedding layers

    Example:
        >>> config = CPTConfig(
        ...     learning_rate=5e-5,
        ...     embedding_learning_rate=5e-6,
        ...     max_steps=1000,
        ...     output_dir="cpt_output",
        ... )
    """

    def __init__(
        self,
        # CPT-specific
        include_embeddings: bool = True,
        embedding_learning_rate: Optional[float] = None,
        # Training args
        output_dir: str = "./cpt_outputs",
        learning_rate: float = 5e-5,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        num_train_epochs: int = 1,
        max_steps: int = -1,
        warmup_steps: int = 10,
        warmup_ratio: float = 0.0,
        lr_scheduler_type: str = "cosine",
        weight_decay: float = 0.01,
        logging_steps: int = 10,
        save_steps: int = 100,
        max_seq_length: int = 2048,
        dataset_text_field: str = "text",
        grad_checkpoint: bool = False,
        num_layers: Optional[int] = None,
        **kwargs,
    ):
        self.include_embeddings = include_embeddings
        # Default embedding LR to 1/5 of main LR (Unsloth recommends 2-10x smaller)
        self.embedding_learning_rate = embedding_learning_rate if embedding_learning_rate is not None else learning_rate / 5.0
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.lr_scheduler_type = lr_scheduler_type
        self.weight_decay = weight_decay
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_seq_length = max_seq_length
        self.dataset_text_field = dataset_text_field
        self.grad_checkpoint = grad_checkpoint
        self.num_layers = num_layers

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class CPTTrainer:
    """
    Continual Pretraining Trainer for MLX-Tune.

    Trains on raw text with loss on ALL tokens. Supports decoupled
    learning rates for embedding layers (embed_tokens + lm_head).

    Example:
        >>> from mlx_tune import FastLanguageModel, CPTTrainer, CPTConfig
        >>>
        >>> model, tokenizer = FastLanguageModel.from_pretrained(
        ...     model_name="mlx-community/Llama-3.2-1B-4bit",
        ...     max_seq_length=2048,
        ... )
        >>> model = FastLanguageModel.get_peft_model(
        ...     model, r=16,
        ...     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
        ...                     "gate_proj", "up_proj", "down_proj"],
        ... )
        >>>
        >>> config = CPTConfig(
        ...     learning_rate=5e-5,
        ...     embedding_learning_rate=5e-6,
        ...     max_steps=100,
        ... )
        >>> trainer = CPTTrainer(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     train_dataset=dataset,
        ...     args=config,
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        tokenizer: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        args: Optional[CPTConfig] = None,
        formatting_func: Optional[Callable] = None,
        # Direct params (overridden by args if provided)
        learning_rate: float = 5e-5,
        embedding_learning_rate: Optional[float] = None,
        include_embeddings: bool = True,
        max_steps: int = -1,
        num_train_epochs: int = 1,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 10,
        logging_steps: int = 10,
        save_steps: int = 100,
        output_dir: str = "./cpt_outputs",
        max_seq_length: int = 2048,
        dataset_text_field: str = "text",
        **kwargs,
    ):
        self.args = args

        # Extract from args if provided
        if args is not None:
            args_dict = args.to_dict() if hasattr(args, 'to_dict') else vars(args)
            learning_rate = args_dict.get('learning_rate', learning_rate)
            embedding_learning_rate = args_dict.get('embedding_learning_rate', embedding_learning_rate)
            include_embeddings = args_dict.get('include_embeddings', include_embeddings)
            max_steps = args_dict.get('max_steps', max_steps)
            num_train_epochs = args_dict.get('num_train_epochs', num_train_epochs)
            per_device_train_batch_size = args_dict.get('per_device_train_batch_size', per_device_train_batch_size)
            gradient_accumulation_steps = args_dict.get('gradient_accumulation_steps', gradient_accumulation_steps)
            warmup_steps = args_dict.get('warmup_steps', warmup_steps)
            logging_steps = args_dict.get('logging_steps', logging_steps)
            save_steps = args_dict.get('save_steps', save_steps)
            output_dir = args_dict.get('output_dir', output_dir)
            max_seq_length = args_dict.get('max_seq_length', max_seq_length)
            dataset_text_field = args_dict.get('dataset_text_field', dataset_text_field)

        self.model = model
        if tokenizer is None and hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.formatting_func = formatting_func

        # Training config
        self.learning_rate = learning_rate
        self.embedding_learning_rate = embedding_learning_rate if embedding_learning_rate is not None else learning_rate / 5.0
        self.include_embeddings = include_embeddings
        self.max_steps = max_steps
        self.num_train_epochs = num_train_epochs
        self.batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.output_dir = Path(output_dir)
        self.adapter_path = self.output_dir / "adapters"
        self.max_seq_length = max_seq_length
        self.dataset_text_field = dataset_text_field
        self.weight_decay = getattr(args, 'weight_decay', 0.01) if args else 0.01
        self.lr_scheduler_type = getattr(args, 'lr_scheduler_type', 'cosine') if args else 'cosine'
        self.grad_checkpoint = getattr(args, 'grad_checkpoint', False) if args else False
        self.num_layers = getattr(args, 'num_layers', None) if args else None

        # Calculate iterations
        if self.max_steps > 0:
            self.iters = self.max_steps
        elif train_dataset is not None:
            dataset_size = len(train_dataset) if hasattr(train_dataset, '__len__') else 1000
            self.iters = max(1, (dataset_size // self.batch_size) * self.num_train_epochs)
        else:
            self.iters = 100

        # Auto-add embed_tokens + lm_head to target modules
        if self.include_embeddings and hasattr(model, 'lora_config') and model.lora_config:
            targets = list(model.lora_config.get('target_modules', []))
            added = []
            if 'embed_tokens' not in targets:
                targets.append('embed_tokens')
                added.append('embed_tokens')
            if 'lm_head' not in targets:
                targets.append('lm_head')
                added.append('lm_head')
            if added:
                model.lora_config['target_modules'] = targets
                print(f"CPT: Auto-added {added} to target modules for vocabulary adaptation")

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_path.mkdir(parents=True, exist_ok=True)

        # Compute gradient scale for decoupled learning rate
        self._embedding_grad_scale = self.embedding_learning_rate / self.learning_rate if self.learning_rate > 0 else 1.0
        self._use_decoupled_lr = abs(self._embedding_grad_scale - 1.0) > 1e-8

        print(f"CPT Trainer initialized:")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Embedding learning rate: {self.embedding_learning_rate}")
        print(f"  Decoupled LR: {self._use_decoupled_lr} (scale={self._embedding_grad_scale:.4f})")
        print(f"  Include embeddings: {self.include_embeddings}")
        print(f"  Iterations: {self.iters}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Dataset text field: {self.dataset_text_field}")

    def _prepare_training_data(self) -> str:
        """Prepare raw text training data in MLX-LM format (JSONL)."""
        train_file = self.output_dir / "train.jsonl"
        valid_file = self.output_dir / "valid.jsonl"

        print("Preparing CPT training data (raw text)...")

        def format_sample(sample) -> dict:
            if self.formatting_func:
                formatted = self.formatting_func(sample)
                if isinstance(formatted, str):
                    return {"text": formatted}
                return formatted
            if 'text' in sample:
                return {"text": sample['text']}
            if self.dataset_text_field and self.dataset_text_field in sample:
                return {"text": sample[self.dataset_text_field]}
            # Try common field names
            for field in ['content', 'document', 'passage', 'body', 'data']:
                if field in sample:
                    return {"text": sample[field]}
            # Last resort: concatenate all string values
            text_parts = [str(v) for v in sample.values() if isinstance(v, str)]
            if text_parts:
                return {"text": " ".join(text_parts)}
            print(f"  Warning: Could not extract text from sample with keys {list(sample.keys())}")
            return sample

        with open(train_file, 'w') as f:
            for idx, sample in enumerate(self.train_dataset):
                formatted = format_sample(sample)
                f.write(json.dumps(formatted) + '\n')

        num_samples = idx + 1
        print(f"  Prepared {num_samples} training samples -> {train_file}")

        # Validation set
        if self.eval_dataset:
            with open(valid_file, 'w') as f:
                for sample in self.eval_dataset:
                    f.write(json.dumps(format_sample(sample)) + '\n')
            print(f"  Prepared validation set")
        else:
            import shutil
            shutil.copy(train_file, valid_file)
            print(f"  Created validation set (copied from train)")

        return str(self.output_dir)

    def _apply_embedding_lora(self):
        """
        Make embed_tokens and lm_head trainable for CPT.

        After model.freeze() + linear_to_lora_layers(), embedding layers
        remain frozen. For CPT, we unfreeze them to allow vocabulary adaptation.

        Note: Quantized layers (4-bit/8-bit) do NOT support gradients.
        For full CPT with embeddings, use bf16/fp16 models.
        """
        if not self.include_embeddings:
            return

        actual_model = self.model.model if hasattr(self.model, 'model') else self.model

        def _is_quantized(module):
            """Check if a module uses quantized weights (no gradient support)."""
            type_name = type(module).__name__
            return 'Quantized' in type_name

        # Find and unfreeze embed_tokens
        embed = None
        if hasattr(actual_model, 'model') and hasattr(actual_model.model, 'embed_tokens'):
            embed = actual_model.model.embed_tokens
        elif hasattr(actual_model, 'embed_tokens'):
            embed = actual_model.embed_tokens

        if embed is not None:
            if _is_quantized(embed):
                print(f"  Skipping embed_tokens: quantized layers don't support gradients")
                print(f"  Tip: Use a bf16/fp16 model for full CPT with embeddings")
            else:
                embed.unfreeze()
                print(f"  Unfroze embed_tokens for CPT")
        else:
            print(f"  Warning: Could not find embed_tokens to unfreeze")

        # Find and unfreeze lm_head
        lm_head = None
        if hasattr(actual_model, 'lm_head'):
            lm_head = actual_model.lm_head
        elif hasattr(actual_model, 'model') and hasattr(actual_model.model, 'lm_head'):
            lm_head = actual_model.model.lm_head

        if lm_head is not None:
            if _is_quantized(lm_head):
                print(f"  Skipping lm_head: quantized layers don't support gradients")
            else:
                lm_head.unfreeze()
                print(f"  Unfroze lm_head for CPT")
        else:
            # Some models tie embeddings (no separate lm_head)
            print(f"  Note: No separate lm_head found (may be tied to embed_tokens)")

    def _get_lr_schedule(self):
        """Get learning rate schedule."""
        if not HAS_NATIVE_TRAINING:
            return self.learning_rate
        if self.lr_scheduler_type == "cosine":
            return optim.cosine_decay(init=self.learning_rate, decay_steps=self.iters)
        elif self.lr_scheduler_type == "linear":
            return optim.linear_schedule(init=self.learning_rate, end=0.0, steps=self.iters)
        return self.learning_rate

    def _scale_embedding_gradients(self, grads):
        """
        Scale gradients for embedding/lm_head parameters.

        This implements decoupled learning rates by scaling gradients:
        effective_lr = main_lr * scale = main_lr * (embedding_lr / main_lr) = embedding_lr
        """
        from mlx.utils import tree_flatten, tree_unflatten

        flat_grads = tree_flatten(grads)
        scaled = []
        for key, value in flat_grads:
            if 'embed_tokens' in key or 'lm_head' in key:
                scaled.append((key, value * self._embedding_grad_scale))
            else:
                scaled.append((key, value))
        return tree_unflatten(scaled)

    def train(self):
        """
        Train the model using continual pretraining.

        Supports two modes:
        1. **LoRA CPT** (default): Apply LoRA + optionally unfreeze embeddings.
           Use get_peft_model() before creating trainer.
        2. **Full-weight CPT**: Train all parameters (no LoRA). Skip get_peft_model()
           and the trainer will train the full model.
        """
        if not HAS_NATIVE_TRAINING:
            raise RuntimeError(
                "Native training required for CPT. Install with: uv pip install 'mlx-lm[train]'"
            )

        print("=" * 70)
        print("Starting Continual Pretraining (CPT)")
        print("=" * 70)

        # Determine training mode
        has_lora = hasattr(self.model, 'lora_enabled') and self.model.lora_enabled
        self._full_weight_mode = not has_lora

        if self._full_weight_mode:
            print("\n[Full-Weight CPT Mode] — training all parameters (no LoRA)")
        else:
            # Step 1: Apply LoRA if not already done
            if hasattr(self.model, '_apply_lora') and not self.model._lora_applied:
                print("\nApplying LoRA adapters...")
                self.model._apply_lora(num_layers=self.num_layers)

            # Step 2: Unfreeze embed_tokens + lm_head
            if self.include_embeddings:
                print("\nEnabling embedding layer training for CPT...")
                self._apply_embedding_lora()

        # Count trainable parameters
        actual_model = self.model.model if hasattr(self.model, 'model') else self.model
        from mlx.utils import tree_flatten
        trainable = tree_flatten(actual_model.trainable_parameters())
        total_params = sum(p.size for _, p in tree_flatten(actual_model.parameters()))
        trainable_count = sum(p.size for _, p in trainable)
        lora_params = [(k, p) for k, p in trainable if 'lora' in k]
        embed_params = [(k, p) for k, p in trainable if 'embed_tokens' in k or 'lm_head' in k]
        print(f"\nTrainable parameters: {trainable_count:,} / {total_params:,} "
              f"({100*trainable_count/total_params:.2f}%)")
        if lora_params:
            print(f"  LoRA parameters: {sum(p.size for _, p in lora_params):,}")
        if embed_params:
            print(f"  Embedding parameters: {sum(p.size for _, p in embed_params):,}")

        # Step 3: Set adapter path
        if hasattr(self.model, 'set_adapter_path'):
            self.model.set_adapter_path(str(self.adapter_path))

        # Step 4: Prepare data
        data_dir = self._prepare_training_data()

        # Choose training path
        if self._use_decoupled_lr:
            return self._train_with_decoupled_lr(data_dir)
        else:
            return self._train_standard(data_dir)

    def _train_standard(self, data_dir: str):
        """Train using standard mlx_lm training loop (same LR for all params)."""
        print("\n[Using Standard CPT Training (single learning rate)]")

        actual_model = self.model.model if hasattr(self.model, 'model') else self.model

        lr_schedule = self._get_lr_schedule()
        optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=self.weight_decay)

        adapter_file = str(self.adapter_path / "adapters.safetensors")
        training_args = TrainingArgs(
            batch_size=self.batch_size,
            iters=self.iters,
            val_batches=25,
            steps_per_report=self.logging_steps,
            steps_per_eval=max(self.save_steps, 100),
            steps_per_save=self.save_steps,
            max_seq_length=self.max_seq_length,
            adapter_file=adapter_file,
            grad_checkpoint=self.grad_checkpoint,
        )

        print(f"\nTraining configuration:")
        print(f"  Iterations: {self.iters}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Max seq length: {self.max_seq_length}")
        print()

        # Load datasets — no mask_prompt for CPT (train on all tokens)
        dataset_args = types.SimpleNamespace(
            data=data_dir, train=True, test=False,
            hf_dataset=None, mask_prompt=False,
        )

        train_set, valid_set, _ = mlx_load_dataset(
            args=dataset_args, tokenizer=self.tokenizer,
        )
        train_set = CacheDataset(train_set)
        valid_set = CacheDataset(valid_set)
        print(f"Loaded {len(train_set)} training samples, {len(valid_set)} validation samples")

        print("Starting training loop...")
        mlx_train(
            model=actual_model, optimizer=optimizer,
            train_dataset=train_set, val_dataset=valid_set,
            args=training_args,
        )

        self._save_adapters()

        print("\n" + "=" * 70)
        print("CPT Training Complete!")
        print("=" * 70)
        print(f"  Adapters saved to: {self.adapter_path}")
        return {"status": "success", "adapter_path": str(self.adapter_path)}

    def _train_with_decoupled_lr(self, data_dir: str):
        """
        Train with decoupled learning rates using custom loop.

        Embedding/lm_head gradients are scaled by (embedding_lr / main_lr)
        before the optimizer step, achieving separate effective learning rates.
        """
        print(f"\n[Using Decoupled LR Training]")
        print(f"  Main LR: {self.learning_rate}")
        print(f"  Embedding LR: {self.embedding_learning_rate}")
        print(f"  Gradient scale: {self._embedding_grad_scale:.4f}")

        actual_model = self.model.model if hasattr(self.model, 'model') else self.model

        # Switch to training mode
        if hasattr(self.model, 'train'):
            self.model.train()
        if hasattr(actual_model, 'train'):
            actual_model.train()

        # Load and tokenize dataset
        dataset_args = types.SimpleNamespace(
            data=data_dir, train=True, test=False,
            hf_dataset=None, mask_prompt=False,
        )
        train_set, valid_set, _ = mlx_load_dataset(
            args=dataset_args, tokenizer=self.tokenizer,
        )
        train_set = CacheDataset(train_set)
        print(f"Loaded {len(train_set)} training samples")

        # Create optimizer with main learning rate
        lr_schedule = self._get_lr_schedule()
        optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=self.weight_decay)

        # Define loss function (standard next-token prediction)
        def loss_fn(model, input_ids, lengths):
            logits = model(input_ids)
            logits = logits.astype(mx.float32)

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            # Cross-entropy loss
            loss = nn.losses.cross_entropy(shift_logits, shift_labels, reduction='none')

            # Create mask based on sequence lengths
            seq_len = shift_labels.shape[1]
            mask = mx.arange(seq_len)[None, :] < (lengths[:, None] - 1)
            loss = (loss * mask).sum() / mask.sum()
            return loss

        loss_and_grad = nn.value_and_grad(actual_model, loss_fn)

        print(f"\nStarting training for {self.iters} iterations...")
        total_loss = 0.0

        for step in range(self.iters):
            # Get batch
            batch_indices = [
                (step * self.batch_size + i) % len(train_set)
                for i in range(self.batch_size)
            ]
            batch_items = [train_set[idx] for idx in batch_indices]

            # Unpack — CacheDataset returns (token_ids, offset) tuples
            tokens_list = []
            lengths = []
            for item in batch_items:
                if isinstance(item, tuple):
                    toks = item[0]
                else:
                    toks = item
                tokens_list.append(toks)
                lengths.append(len(toks))

            # Pad to max length in batch
            max_len = min(max(lengths), self.max_seq_length)
            pad_id = self.tokenizer.pad_token_id or 0
            padded = []
            actual_lengths = []
            for toks, ln in zip(tokens_list, lengths):
                t = list(toks[:max_len])
                actual_len = min(ln, max_len)
                while len(t) < max_len:
                    t.append(pad_id)
                padded.append(t)
                actual_lengths.append(actual_len)

            input_ids = mx.array(padded)
            length_arr = mx.array(actual_lengths)

            # Forward + backward
            loss, grads = loss_and_grad(actual_model, input_ids, length_arr)

            # Scale embedding gradients for decoupled LR
            grads = self._scale_embedding_gradients(grads)

            # Optimizer step
            optimizer.update(actual_model, grads)
            mx.eval(actual_model.parameters(), optimizer.state)

            total_loss += loss.item()

            # Logging
            if (step + 1) % self.logging_steps == 0:
                avg_loss = total_loss / self.logging_steps
                print(f"  Step {step + 1}/{self.iters} | Loss: {avg_loss:.4f}")
                total_loss = 0.0

            # Save checkpoint
            if (step + 1) % self.save_steps == 0:
                self._save_adapters()
                print(f"  Saved checkpoint at step {step + 1}")

        # Final save
        self._save_adapters()

        print("\n" + "=" * 70)
        print("CPT Training Complete!")
        print("=" * 70)
        print(f"  Adapters saved to: {self.adapter_path}")
        return {"status": "success", "adapter_path": str(self.adapter_path)}

    def _save_adapters(self):
        """Save adapter weights (LoRA mode) or full model weights."""
        if getattr(self, '_full_weight_mode', False):
            self._save_full_weights()
        else:
            from mlx_tune.rl_trainers import _save_adapters_and_config
            _save_adapters_and_config(self.model, self.adapter_path)

    def _save_full_weights(self):
        """Save full model weights for full-weight CPT."""
        from mlx.utils import tree_flatten
        actual_model = self.model.model if hasattr(self.model, 'model') else self.model
        self.adapter_path.mkdir(parents=True, exist_ok=True)

        weights = dict(tree_flatten(actual_model.parameters()))
        save_path = self.adapter_path / "model.safetensors"
        mx.save_safetensors(str(save_path), weights)
        print(f"  Full model weights saved to: {save_path}")
