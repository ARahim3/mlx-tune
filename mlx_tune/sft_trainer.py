"""
SFTTrainer - Supervised Fine-Tuning Trainer for MLX-Tune

Provides Unsloth/TRL-compatible training interface using MLX under the hood.
Supports both native MLX training and subprocess fallback.
"""

from typing import Optional, Dict, Any, Union, List, Callable
from pathlib import Path
import json
import subprocess
import tempfile
import os
import types
import warnings
import yaml

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
    mlx_load_dataset = None
    CacheDataset = None
    warnings.warn(
        "Native training not available. Install with: pip install 'mlx-lm[train]'. "
        "Falling back to subprocess-based training.",
        ImportWarning
    )


class SFTConfig:
    """
    TRL-compatible SFTConfig for Supervised Fine-Tuning configuration.

    This class provides compatibility with TRL's SFTConfig, allowing users
    to use the same configuration pattern as the original Unsloth.

    Example:
        >>> from mlx_tune import SFTTrainer, SFTConfig
        >>>
        >>> config = SFTConfig(
        ...     per_device_train_batch_size=2,
        ...     gradient_accumulation_steps=4,
        ...     learning_rate=2e-4,
        ...     max_steps=100,
        ...     output_dir="outputs",
        ... )
        >>>
        >>> trainer = SFTTrainer(
        ...     model=model,
        ...     train_dataset=dataset,
        ...     args=config,  # Pass SFTConfig here!
        ... )
    """

    def __init__(
        self,
        output_dir: str = "./outputs",
        # Batch and accumulation
        per_device_train_batch_size: int = 2,
        per_device_eval_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        # Learning rate
        learning_rate: float = 2e-4,
        lr_scheduler_type: str = "cosine",  # cosine, linear, constant
        warmup_steps: int = 10,
        warmup_ratio: float = 0.0,
        # Training duration
        num_train_epochs: int = 3,
        max_steps: int = -1,  # -1 means use num_train_epochs
        # Logging and saving
        logging_steps: int = 10,
        save_steps: int = 100,
        save_total_limit: Optional[int] = None,
        # Precision
        fp16: bool = False,
        bf16: bool = False,
        # Optimizer
        optim: str = "adamw_8bit",
        weight_decay: float = 0.01,
        # Misc
        max_seq_length: int = 2048,
        dataset_text_field: Optional[str] = None,
        packing: bool = False,
        # MLX-specific options
        use_native_training: bool = True,  # Use native MLX training vs subprocess
        grad_checkpoint: bool = False,  # Enable gradient checkpointing
        num_layers: Optional[int] = None,  # Number of layers to apply LoRA to
        # Validation cost knobs (Session 38)
        # mlx-lm hardcoded val_batches=25 / steps_per_eval=max(save_steps, 100) — at
        # ctx=4096 a single eval costs ~26s, so the validation pass can be 5-10% of
        # short-run wall time. Defaults below preserve correctness but cut the cost:
        # 5 val batches is plenty for a moving-average signal, and steps_per_eval=200
        # halves how often eval runs vs the previous default. Set val_batches=0 to
        # skip validation entirely.
        val_batches: int = 5,
        steps_per_eval: Optional[int] = None,
        # HuggingFace dataset integration (Unsloth-compatible)
        hf_dataset: Optional[Union[Dict[str, Any], Any]] = None,  # HFDatasetConfig or dict
        **kwargs
    ):
        self.output_dir = output_dir
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.fp16 = fp16
        self.bf16 = bf16
        self.optim = optim
        self.weight_decay = weight_decay
        self.max_seq_length = max_seq_length
        self.dataset_text_field = dataset_text_field
        self.packing = packing
        self.use_native_training = use_native_training
        self.grad_checkpoint = grad_checkpoint
        self.num_layers = num_layers
        self.val_batches = val_batches
        self.steps_per_eval = steps_per_eval
        self.hf_dataset = hf_dataset

        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class SFTTrainer:
    """
    Supervised Fine-Tuning Trainer compatible with Unsloth's API.

    This class provides a simplified interface for fine-tuning models using
    MLX's training capabilities under the hood.

    Example:
        >>> from mlx_tune import FastLanguageModel, SFTTrainer
        >>>
        >>> model, tokenizer = FastLanguageModel.from_pretrained(...)
        >>> model = FastLanguageModel.get_peft_model(model, r=16)
        >>>
        >>> trainer = SFTTrainer(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     train_dataset=dataset,
        ...     max_seq_length=2048,
        ... )
        >>>
        >>> trainer.train()
    """

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        tokenizer: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        args: Optional[Union["SFTConfig", "TrainingArguments", Any]] = None,
        max_seq_length: int = 2048,
        dataset_text_field: Optional[str] = None,
        formatting_func: Optional[callable] = None,
        # Training arguments (can be overridden by args)
        learning_rate: float = 2e-4,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 10,
        save_steps: int = 100,
        logging_steps: int = 10,
        output_dir: str = "./lora_finetuned",
        max_steps: int = -1,
        # LoRA arguments (from model config)
        lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        lora_dropout: float = 0.05,
        # MLX-specific
        adapter_path: str = "./adapters",
        iters: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the SFT Trainer.

        Args:
            model: The model to train (should have LoRA configured)
            train_dataset: Training dataset (HF dataset or list of dicts)
            tokenizer: The tokenizer (optional if model has it)
            eval_dataset: Evaluation dataset (optional)
            args: SFTConfig or TrainingArguments object (TRL-compatible)
            max_seq_length: Maximum sequence length
            dataset_text_field: Field name containing text (for text datasets)
            formatting_func: Function to format dataset samples
            learning_rate: Learning rate
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Number of warmup steps
            save_steps: Save checkpoint every N steps
            logging_steps: Log every N steps
            output_dir: Directory to save model
            max_steps: Max training steps (-1 to use epochs)
            lora_r: LoRA rank (from model config if not specified)
            lora_alpha: LoRA alpha (from model config if not specified)
            lora_dropout: LoRA dropout
            adapter_path: Path to save LoRA adapters
            iters: Number of iterations (alternative to epochs)
            **kwargs: Additional training arguments
        """
        # Store the args object for later access
        self.args = args

        # If args is provided (SFTConfig), extract values from it
        if args is not None:
            if hasattr(args, 'to_dict'):
                args_dict = args.to_dict()
            else:
                args_dict = {k: getattr(args, k) for k in dir(args) if not k.startswith('_')}

            # Override defaults with args values
            learning_rate = args_dict.get('learning_rate', learning_rate)
            num_train_epochs = args_dict.get('num_train_epochs', num_train_epochs)
            per_device_train_batch_size = args_dict.get('per_device_train_batch_size', per_device_train_batch_size)
            gradient_accumulation_steps = args_dict.get('gradient_accumulation_steps', gradient_accumulation_steps)
            warmup_steps = args_dict.get('warmup_steps', warmup_steps)
            save_steps = args_dict.get('save_steps', save_steps)
            logging_steps = args_dict.get('logging_steps', logging_steps)
            output_dir = args_dict.get('output_dir', output_dir)
            max_steps = args_dict.get('max_steps', max_steps)
            max_seq_length = args_dict.get('max_seq_length', max_seq_length)
            dataset_text_field = args_dict.get('dataset_text_field', dataset_text_field)

        # MLX-specific options
        self.use_native_training = getattr(args, 'use_native_training', True) if args else True
        self.grad_checkpoint = getattr(args, 'grad_checkpoint', False) if args else False
        self.lr_scheduler_type = getattr(args, 'lr_scheduler_type', 'cosine') if args else 'cosine'
        self.num_layers = getattr(args, 'num_layers', None) if args else None
        self.weight_decay = getattr(args, 'weight_decay', 0.01) if args else 0.01
        # Validation cost knobs (Session 38). val_batches default lowered from 25 → 5;
        # steps_per_eval falls back to max(save_steps, 200) (was max(save_steps, 100))
        # so validation runs half as often on short runs.
        self.val_batches = getattr(args, 'val_batches', 5) if args else 5
        self.steps_per_eval = getattr(args, 'steps_per_eval', None) if args else None

        self.model = model
        # Get tokenizer from model if not provided
        if tokenizer is None and hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.max_seq_length = max_seq_length
        self.dataset_text_field = dataset_text_field
        self.formatting_func = formatting_func
        self.max_steps = max_steps

        # Training config
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.output_dir = Path(output_dir)
        # Make adapter_path relative to output_dir if it's a relative path
        adapter_path = Path(adapter_path)
        if not adapter_path.is_absolute():
            self.adapter_path = self.output_dir / adapter_path
        else:
            self.adapter_path = adapter_path

        # LoRA config
        if hasattr(model, 'lora_config') and model.lora_config:
            self.lora_r = lora_r or model.lora_config.get('r', 16)
            self.lora_alpha = lora_alpha or model.lora_config.get('lora_alpha', 32)
            self.lora_dropout = model.lora_config.get('lora_dropout', lora_dropout)
        else:
            self.lora_r = lora_r or 16
            self.lora_alpha = lora_alpha or 32
            self.lora_dropout = lora_dropout

        # Calculate iters: priority is max_steps > iters > calculated from epochs
        if self.max_steps > 0:
            self.iters = self.max_steps
        elif iters is not None:
            self.iters = iters
        elif train_dataset is not None:
            dataset_size = len(train_dataset) if hasattr(train_dataset, '__len__') else 1000
            self.iters = max(1, (dataset_size // self.batch_size) * self.num_train_epochs)
        else:
            self.iters = 100

        self.kwargs = kwargs

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_path.mkdir(parents=True, exist_ok=True)

        print(f"Trainer initialized:")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Adapter path: {self.adapter_path}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Iterations: {self.iters}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  LoRA r={self.lora_r}, alpha={self.lora_alpha}")
        print(f"  Native training: {self.use_native_training and HAS_NATIVE_TRAINING}")
        print(f"  LR scheduler: {self.lr_scheduler_type}")
        print(f"  Grad checkpoint: {self.grad_checkpoint}")

    def _get_lr_schedule(self):
        """
        Get learning rate schedule based on config.

        Returns:
            Learning rate schedule function or constant value.
        """
        if not HAS_NATIVE_TRAINING:
            return self.learning_rate

        if self.lr_scheduler_type == "cosine":
            return optim.cosine_decay(
                init=self.learning_rate,
                decay_steps=self.iters,
            )
        elif self.lr_scheduler_type == "linear":
            return optim.linear_schedule(
                init=self.learning_rate,
                end=0.0,
                steps=self.iters,
            )
        elif self.lr_scheduler_type == "constant":
            return self.learning_rate
        else:
            # Default to cosine
            return optim.cosine_decay(
                init=self.learning_rate,
                decay_steps=self.iters,
            )

    def _should_use_grad_checkpoint(self) -> bool:
        """
        Determine if gradient checkpointing should be enabled.

        Returns:
            True if gradient checkpointing should be used.
        """
        # Check explicit config
        if self.grad_checkpoint:
            return True

        # Check model's LoRA config
        if hasattr(self.model, 'lora_config') and self.model.lora_config:
            gc = self.model.lora_config.get('use_gradient_checkpointing', False)
            if gc == "unsloth" or gc is True:
                return True

        return False

    def _save_adapter_config(self) -> None:
        """
        Save adapter_config.json file required by mlx_lm for loading adapters.

        This file is needed when using mlx_lm.fuse to merge adapters or
        when exporting to GGUF format.
        """
        import json

        # Get LoRA config from model or use defaults from trainer
        lora_config = {}
        if hasattr(self.model, 'lora_config') and self.model.lora_config:
            lora_config = self.model.lora_config.copy()

        # Determine number of layers in the model
        num_layers = None
        actual_model = self.model.model if hasattr(self.model, 'model') else self.model
        if hasattr(actual_model, 'layers'):
            num_layers = len(actual_model.layers)
        elif hasattr(actual_model, 'model') and hasattr(actual_model.model, 'layers'):
            num_layers = len(actual_model.model.layers)

        r = lora_config.get('r', self.lora_r)
        alpha = lora_config.get('lora_alpha', self.lora_alpha)

        # Build adapter config in mlx_lm format
        adapter_config = {
            "fine_tune_type": "lora",
            "num_layers": num_layers,
            "lora_parameters": {
                "rank": r,
                "scale": alpha / r,
                "dropout": lora_config.get('lora_dropout', 0.0),
            },
        }

        # Add target modules (keys) if available
        # Prefer cached resolved keys (set during _apply_lora() before types change)
        if hasattr(self.model, '_resolved_lora_keys') and self.model._resolved_lora_keys:
            adapter_config["lora_parameters"]["keys"] = self.model._resolved_lora_keys
        else:
            target_modules = lora_config.get('target_modules', [])
            if target_modules:
                from mlx_tune.model import _resolve_target_modules
                model_for_resolve = actual_model.model if hasattr(actual_model, 'model') else actual_model
                keys = _resolve_target_modules(model_for_resolve, target_modules)
                adapter_config["lora_parameters"]["keys"] = keys

        # Save to adapter_config.json
        config_path = self.adapter_path / "adapter_config.json"
        with open(config_path, 'w') as f:
            json.dump(adapter_config, f, indent=2)

        print(f"  Adapter config saved to: {config_path}")

    def _prepare_training_data(self) -> str:
        """Prepare training data in MLX-LM compatible format.

        Supports automatic conversion of various dataset formats:
        - Alpaca: {"instruction": "...", "input": "...", "output": "..."}
        - ShareGPT: {"conversations": [{"from": "human", "value": "..."}]}
        - ChatML: {"messages": [{"role": "user", "content": "..."}]}
        - Text: {"text": "..."}
        - Completions: {"prompt": "...", "completion": "..."}
        """
        from mlx_tune.chat_templates import (
            detect_dataset_format,
            alpaca_to_text,
            apply_chat_template_to_sample,
        )

        # Create training and validation data files
        train_file = self.output_dir / "train.jsonl"
        valid_file = self.output_dir / "valid.jsonl"

        print(f"Preparing training data...")

        # Detect format from first sample
        if len(self.train_dataset) > 0:
            detected_format = detect_dataset_format(self.train_dataset[0])
            print(f"  Detected format: {detected_format}")

        def format_sample(sample) -> dict:
            """Convert a sample to mlx-lm compatible format."""
            # 1. User-provided formatting function takes priority
            if self.formatting_func:
                formatted = self.formatting_func(sample)
                if isinstance(formatted, str):
                    return {"text": formatted}
                return formatted

            # 2. Already in mlx-lm compatible format
            if 'text' in sample:
                return {"text": sample['text']}
            if 'messages' in sample:
                return {"messages": sample['messages']}
            if 'prompt' in sample and 'completion' in sample:
                return {"prompt": sample['prompt'], "completion": sample['completion']}

            # 3. Custom text field specified
            if self.dataset_text_field and self.dataset_text_field in sample:
                return {"text": sample[self.dataset_text_field]}

            # 4. Auto-convert known formats
            # Alpaca format: instruction/input/output -> text
            if 'instruction' in sample and 'output' in sample:
                text = alpaca_to_text(sample)
                return {"text": text}

            # ShareGPT format: conversations -> messages (ChatML)
            if 'conversations' in sample:
                role_mapping = {'human': 'user', 'gpt': 'assistant', 'system': 'system'}
                messages = []
                for turn in sample['conversations']:
                    role = role_mapping.get(turn.get('from', '').lower(), 'user')
                    messages.append({'role': role, 'content': turn.get('value', '')})
                return {"messages": messages}

            # 5. Fallback: try to apply chat template if messages-like structure
            if 'content' in sample or 'response' in sample:
                return {"text": sample.get('content') or sample.get('response', '')}

            # 6. Last resort: warn and use raw sample (will likely fail)
            print(f"  Warning: Unknown format for sample with keys {list(sample.keys())}")
            print(f"  Consider using formatting_func or dataset_text_field parameter")
            return sample

        with open(train_file, 'w') as f:
            for idx, sample in enumerate(self.train_dataset):
                formatted_sample = format_sample(sample)
                f.write(json.dumps(formatted_sample) + '\n')

        num_samples = idx + 1
        print(f"✓ Prepared {num_samples} training samples")
        print(f"  Saved to: {train_file}")

        # Create validation set (use eval_dataset if provided, otherwise reuse train)
        if self.eval_dataset:
            with open(valid_file, 'w') as f:
                for sample in self.eval_dataset:
                    formatted_sample = format_sample(sample)
                    f.write(json.dumps(formatted_sample) + '\n')
            print(f"✓ Prepared validation set")
        else:
            # Reuse training data for validation
            import shutil
            shutil.copy(train_file, valid_file)
            print(f"✓ Created validation set (copied from train)")

        # Return directory path (not file path)
        return str(self.output_dir)

    def train(self, use_native: Optional[bool] = None):
        """
        Train the model using MLX-LM.

        This method supports two training modes:
        1. Native MLX training (recommended) - Uses mlx_lm.tuner.train() directly
        2. Subprocess training (fallback) - Calls mlx_lm.lora CLI

        Args:
            use_native: Override for native training. If None, uses config value.

        Returns:
            Training result object.
        """
        # Determine training mode
        if use_native is None:
            use_native = self.use_native_training

        print("=" * 70)
        print("Starting Fine-Tuning")
        print("=" * 70)

        if use_native and HAS_NATIVE_TRAINING:
            return self._train_native()
        else:
            if use_native and not HAS_NATIVE_TRAINING:
                warnings.warn(
                    "Native training requested but mlx_lm.tuner not available. "
                    "Falling back to subprocess training. "
                    "Install with: pip install 'mlx-lm[train]'",
                    UserWarning
                )
            return self._train_subprocess()

    def _train_native(self):
        """
        Train using native MLX training loop.

        This is the recommended training method that provides:
        - Direct control over the training process
        - Custom loss functions (for DPO, GRPO, etc.)
        - Better error handling and debugging
        """
        print("\n[Using Native MLX Training]")

        # Step 1: Apply LoRA to model if not already done
        if hasattr(self.model, '_apply_lora') and not self.model._lora_applied:
            print("\nApplying LoRA adapters...")
            self.model._apply_lora(num_layers=self.num_layers)

        # Step 2: Set adapter path on model for later saving
        if hasattr(self.model, 'set_adapter_path'):
            self.model.set_adapter_path(str(self.adapter_path))

        # Step 3: Prepare training data
        data_dir = self._prepare_training_data()

        # Step 4: Create learning rate schedule
        lr_schedule = self._get_lr_schedule()

        # Step 5: Create optimizer
        optimizer = optim.AdamW(
            learning_rate=lr_schedule,
            weight_decay=self.weight_decay,
        )

        # Step 6: Create training args
        adapter_file = str(self.adapter_path / "adapters.safetensors")
        # Resolve validation knobs (Session 38).
        # val_batches: 0 means skip validation; ≥1 controls how many batches per eval.
        # steps_per_eval: None → default heuristic max(save_steps, 200). Previously this
        # was max(save_steps, 100); the bump halves eval frequency on short runs where
        # save_steps < 100, cutting wall-time without changing the train signal.
        val_batches = max(0, int(self.val_batches))
        steps_per_eval = (
            int(self.steps_per_eval)
            if self.steps_per_eval is not None
            else max(self.save_steps, 200)
        )
        training_args = TrainingArgs(
            batch_size=self.batch_size,
            iters=self.iters,
            val_batches=val_batches,
            steps_per_report=self.logging_steps,
            steps_per_eval=steps_per_eval,
            steps_per_save=self.save_steps,
            max_seq_length=self.max_seq_length,
            adapter_file=adapter_file,
            grad_checkpoint=self._should_use_grad_checkpoint(),
        )

        print(f"\nTraining configuration:")
        print(f"  Iterations: {self.iters}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  LR scheduler: {self.lr_scheduler_type}")
        print(f"  Grad checkpoint: {training_args.grad_checkpoint}")
        print(f"  Adapter file: {adapter_file}")
        print()

        # Step 7: Load datasets using mlx_lm dataset utilities
        # mlx_load_dataset expects an args object with specific attributes
        # Check if response-only training is enabled
        mask_prompt = getattr(self, '_train_on_responses_only', False)
        if mask_prompt:
            print("  Response-only training enabled (mask_prompt=True)")

        dataset_args = types.SimpleNamespace(
            data=data_dir,
            train=True,
            test=False,
            hf_dataset=None,
            mask_prompt=mask_prompt,
        )

        try:
            train_set, valid_set, _ = mlx_load_dataset(
                args=dataset_args,
                tokenizer=self.tokenizer,
            )
            # Wrap datasets in CacheDataset for proper iteration
            # CacheDataset processes items lazily and caches the results
            train_set = CacheDataset(train_set)
            valid_set = CacheDataset(valid_set)
            print(f"Loaded {len(train_set)} training samples, {len(valid_set)} validation samples")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to subprocess training...")
            return self._train_subprocess()

        # Step 8: Get the actual model to train
        # MLXModelWrapper wraps the actual model
        actual_model = self.model.model if hasattr(self.model, 'model') else self.model

        # Force a Python GC sweep before entering the training loop. Setup
        # creates transient Python objects (JSONL serialization buffers,
        # dataset_args namespace, the user's Python sample dicts kept alive
        # by `self.train_dataset`) that aren't needed during training. mlx-lm
        # CLI dodges this by living in its own short-lived process; when we
        # run inline, those refs survive into the loop and appear to fragment
        # the unified-memory allocator under long-ctx pressure (observed:
        # SFTTrainer iter-70 OOM at ctx=4096 no-GC, while direct mlx-lm at
        # the same config completes 100 iters cleanly). One sweep at this
        # boundary is harmless on short runs and load-bearing on long ones.
        import gc
        gc.collect()

        # Step 9: Run training
        try:
            print("Starting training loop...")
            # If user opted out of validation (val_batches=0), skip eval entirely
            # by passing val_dataset=None. mlx-lm's `evaluate()` would otherwise
            # iterate 0 batches and emit a NaN val loss — visible noise for no benefit.
            effective_val_set = valid_set if val_batches > 0 else None
            mlx_train(
                model=actual_model,
                optimizer=optimizer,
                train_dataset=train_set,
                val_dataset=effective_val_set,
                args=training_args,
            )

            # Save adapter_config.json (required by mlx_lm for loading adapters)
            self._save_adapter_config()

            print("\n" + "=" * 70)
            print("Training Complete!")
            print("=" * 70)
            print(f"  Adapters saved to: {self.adapter_path}")

            return {"status": "success", "adapter_path": str(self.adapter_path)}

        except Exception as e:
            print(f"\nNative training failed: {e}")
            print("Falling back to subprocess training...")
            return self._train_subprocess()

    def _train_subprocess(self):
        """
        Train using subprocess call to mlx_lm.lora CLI.

        This is the fallback training method for compatibility.
        """
        if self.use_native_training:
            warnings.warn(
                "Subprocess training is deprecated and will be removed in v0.4.0. "
                "Use native training (default) for better performance.",
                DeprecationWarning
            )

        print("\n[Using Subprocess Training (Legacy)]")

        # Prepare training data
        data_dir = self._prepare_training_data()

        # Get model name
        model_name = self.model.model_name if hasattr(self.model, 'model_name') else "model"

        # Create config file for LoRA settings (mlx_lm.lora uses config file for LoRA params)
        config_file = self.output_dir / "lora_config.yaml"
        lora_config = {
            "lora_parameters": {
                "rank": self.lora_r,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
                "scale": self.lora_alpha / self.lora_r,
            }
        }

        with open(config_file, 'w') as f:
            yaml.dump(lora_config, f)

        print(f"Created LoRA config: {config_file}")

        # Build MLX-LM training command
        # Note: LoRA rank/alpha are set via config file, not CLI args
        cmd = [
            "mlx_lm.lora",
            "--model", model_name,
            "--train",
            "--data", data_dir,
            "--iters", str(self.iters),
            "--learning-rate", str(self.learning_rate),
            "--batch-size", str(self.batch_size),
            "--adapter-path", str(self.adapter_path),
            "-c", str(config_file),  # Config file for LoRA settings
        ]

        # Add num-layers if specified (how many layers to apply LoRA to)
        if self.num_layers:
            cmd.extend(["--num-layers", str(self.num_layers)])

        # Note: mlx_lm.lora doesn't support --warmup, warmup is handled internally

        # Add optional arguments
        if self.save_steps:
            cmd.extend(["--save-every", str(self.save_steps)])

        # Add gradient checkpointing if enabled
        if self._should_use_grad_checkpoint():
            cmd.append("--grad-checkpoint")

        # Add prompt masking for response-only training
        if getattr(self, '_train_on_responses_only', False):
            cmd.append("--mask-prompt")
            print("  Response-only training enabled (--mask-prompt)")

        if self.eval_dataset:
            cmd.append("--test")

        print(f"\nRunning training command:")
        print(" ".join(cmd))
        print()

        # Run training
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True
            )

            # Set adapter path on model for later saving
            if hasattr(self.model, 'set_adapter_path'):
                self.model.set_adapter_path(str(self.adapter_path))

            print("\n" + "=" * 70)
            print("Training Complete!")
            print("=" * 70)
            print(f"  Adapters saved to: {self.adapter_path}")

            return result

        except subprocess.CalledProcessError as e:
            print(f"\nTraining failed with error code {e.returncode}")
            print("This might be because mlx_lm.lora needs to be run differently.")
            print("\nAlternative: Run training manually:")
            print(" ".join(cmd))
            raise

    def save_model(self, output_dir: Optional[str] = None):
        """
        Save the fine-tuned model.

        Args:
            output_dir: Directory to save model (defaults to self.output_dir)
        """

        if output_dir is None:
            output_dir = self.output_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving model to {output_dir}...")

        # For MLX, we typically fuse the adapters with the base model
        try:
            # Run mlx_lm.fuse to merge adapters
            cmd = [
                "mlx_lm.fuse",
                "--model", self.model.model_name,
                "--adapter-path", str(self.adapter_path),
                "--save-path", str(output_dir),
            ]

            subprocess.run(cmd, check=True)
            print(f"✓ Model saved to {output_dir}")

        except Exception as e:
            print(f"Error saving model: {e}")
            print("You can manually fuse adapters using:")
            print(f"  mlx_lm.fuse --model {self.model.model_name} --adapter-path {self.adapter_path}")


class TrainingArguments:
    """
    Training arguments compatible with Unsloth/TRL API.

    This is a simplified version for compatibility.
    """

    def __init__(
        self,
        output_dir: str = "./lora_finetuned",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 2e-4,
        warmup_steps: int = 10,
        save_steps: int = 100,
        logging_steps: int = 10,
        **kwargs
    ):
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps

        for key, value in kwargs.items():
            setattr(self, key, value)
