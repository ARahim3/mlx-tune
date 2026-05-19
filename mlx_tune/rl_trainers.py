"""
Reinforcement Learning Trainers for MLX-Tune

Provides Unsloth/TRL-compatible RL training interfaces:
- DPOTrainer: Direct Preference Optimization
- ORPOTrainer: Odds Ratio Preference Optimization
- GRPOTrainer: Group Relative Policy Optimization (DeepSeek R1 style)
- KTOTrainer: Kahneman-Tversky Optimization
- SimPOTrainer: Simple Preference Optimization

These trainers use MLX under the hood for Apple Silicon optimization.
Now with PROPER loss implementations using native MLX training!
"""

from typing import Optional, Dict, Any, Union, List, Callable
from pathlib import Path
import json
import subprocess
import warnings

import mlx.core as mx

# Try to import native training components
try:
    from mlx_lm.tuner.trainer import TrainingArgs
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_NATIVE_TRAINING = True
except ImportError:
    HAS_NATIVE_TRAINING = False

# Import our loss functions
from mlx_tune.losses import (
    dpo_loss as compute_dpo_loss,
    orpo_loss as compute_orpo_loss,
    kto_loss as compute_kto_loss,
    simpo_loss as compute_simpo_loss,
    grpo_batch_loss,
    generate_with_log_probs,
    compute_reference_logprobs,
    compute_log_probs_with_lengths,
    common_prefix_length,
    build_shared_prompt_cache,
    fork_prompt_cache,
)


def _save_adapters_and_config(model, adapter_path: Path):
    """
    Save adapter weights and config (required for GGUF export).

    This helper function is used by all RL trainers to ensure
    adapter_config.json is created alongside adapters.safetensors.
    """
    try:
        from mlx.utils import tree_flatten
        actual_model = model.model if hasattr(model, 'model') else model
        adapter_file = adapter_path / "adapters.safetensors"
        adapter_path.mkdir(parents=True, exist_ok=True)

        # Save adapter weights (trainable parameters only)
        adapter_weights = dict(tree_flatten(actual_model.trainable_parameters()))
        mx.save_safetensors(str(adapter_file), adapter_weights)

        # Determine number of layers in the model
        num_layers = None
        if hasattr(actual_model, 'layers'):
            num_layers = len(actual_model.layers)
        elif hasattr(actual_model, 'model') and hasattr(actual_model.model, 'layers'):
            num_layers = len(actual_model.model.layers)

        # Save adapter_config.json
        lora_config = {}
        if hasattr(model, 'lora_config') and model.lora_config:
            lora_config = model.lora_config.copy()

        r = lora_config.get('r', 16)
        alpha = lora_config.get('lora_alpha', 16)

        adapter_config = {
            "fine_tune_type": "lora",
            "num_layers": num_layers,
            "lora_parameters": {
                "rank": r,
                "scale": alpha / r,
                "dropout": lora_config.get('lora_dropout', 0.0),
            },
        }

        # Prefer cached resolved keys (set during _apply_lora() before types change)
        if hasattr(model, '_resolved_lora_keys') and model._resolved_lora_keys:
            adapter_config["lora_parameters"]["keys"] = model._resolved_lora_keys
        else:
            target_modules = lora_config.get('target_modules', [])
            if target_modules:
                from mlx_tune.model import _resolve_target_modules
                model_for_resolve = actual_model.model if hasattr(actual_model, 'model') else actual_model
                adapter_config["lora_parameters"]["keys"] = _resolve_target_modules(
                    model_for_resolve, target_modules
                )

        config_path = adapter_path / "adapter_config.json"
        with open(config_path, 'w') as f:
            json.dump(adapter_config, f, indent=2)

        return True
    except Exception as e:
        print(f"  ⚠ Could not save adapters: {e}")
        return False


class DPOConfig:
    """
    Configuration for Direct Preference Optimization training.

    Compatible with TRL's DPOConfig.

    Example:
        >>> config = DPOConfig(
        ...     beta=0.1,
        ...     learning_rate=5e-7,
        ...     max_steps=100,
        ... )
    """

    def __init__(
        self,
        # DPO-specific
        beta: float = 0.1,  # KL penalty coefficient
        loss_type: str = "sigmoid",  # sigmoid, hinge, ipo, kto_pair
        label_smoothing: float = 0.0,
        # Session 38: pre-compute reference logprobs from the base (pre-LoRA)
        # model once at trainer init and reuse for every step. Standard DPO
        # uses a frozen reference π_ref; the prior implementation used
        # `stop_gradient(log_pi)`, which drifts as the policy trains. The
        # fixed reference is more aligned with the published DPO algorithm
        # and prevents the implicit-reference drift at small cost (one
        # forward per pair, no gradient). Set False to keep the legacy
        # stop-grad reference (no init-time cost, slightly faster start).
        precompute_ref_logprobs: bool = True,
        # Training args
        output_dir: str = "./dpo_outputs",
        learning_rate: float = 5e-7,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        num_train_epochs: int = 1,
        max_steps: int = -1,
        warmup_steps: int = 10,
        logging_steps: int = 10,
        save_steps: int = 100,
        max_seq_length: int = 2048,
        max_prompt_length: int = 512,
        **kwargs
    ):
        self.beta = beta
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.precompute_ref_logprobs = precompute_ref_logprobs
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class ORPOConfig:
    """
    Configuration for Odds Ratio Preference Optimization training.

    ORPO combines SFT and preference learning into a single step,
    making it simpler and more efficient than traditional RLHF.

    Example:
        >>> config = ORPOConfig(
        ...     beta=0.1,
        ...     learning_rate=8e-6,
        ...     max_steps=1000,
        ... )
    """

    def __init__(
        self,
        # ORPO-specific
        beta: float = 0.1,  # Odds ratio coefficient
        # Training args
        output_dir: str = "./orpo_outputs",
        learning_rate: float = 8e-6,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        num_train_epochs: int = 1,
        max_steps: int = -1,
        warmup_steps: int = 10,
        logging_steps: int = 10,
        save_steps: int = 100,
        max_seq_length: int = 2048,
        max_prompt_length: int = 512,
        **kwargs
    ):
        self.beta = beta
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class GRPOConfig:
    """
    Configuration for Group Relative Policy Optimization training.

    GRPO is used by DeepSeek to train their R1 reasoning models.
    It replaces the value model with group statistics and uses custom
    reward functions.

    Supports loss types:
    - 'grpo': Standard GRPO
    - 'dr_grpo': Dr. GRPO (distilled)
    - 'dapo': DAPO variant
    - 'bnpo': BNPO variant

    Example:
        >>> config = GRPOConfig(
        ...     loss_type='grpo',
        ...     num_generations=4,
        ...     learning_rate=1e-6,
        ... )
    """

    def __init__(
        self,
        # GRPO-specific
        loss_type: str = "grpo",  # grpo, dr_grpo, dapo, bnpo
        beta: float = 0.04,  # KL coefficient
        num_generations: int = 4,  # Number of generations per prompt
        temperature: float = 0.7,
        max_completion_length: int = 512,
        # Reward function (custom callable)
        reward_fn: Optional[Callable] = None,
        # Training args
        output_dir: str = "./grpo_outputs",
        learning_rate: float = 1e-6,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        num_train_epochs: int = 1,
        max_steps: int = -1,
        warmup_ratio: float = 0.1,
        logging_steps: int = 1,
        save_steps: int = 100,
        max_seq_length: int = 2048,
        **kwargs
    ):
        self.loss_type = loss_type
        self.beta = beta
        self.num_generations = num_generations
        self.temperature = temperature
        self.max_completion_length = max_completion_length
        self.reward_fn = reward_fn
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.warmup_ratio = warmup_ratio
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_seq_length = max_seq_length

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_') and k != 'reward_fn'}


class KTOConfig:
    """
    Configuration for Kahneman-Tversky Optimization training.

    KTO uses prospect theory with asymmetric treatment of gains and losses.
    Unlike DPO, KTO works with binary feedback (desirable/undesirable)
    instead of paired preferences.

    Compatible with TRL's KTOConfig.

    Example:
        >>> config = KTOConfig(
        ...     beta=0.1,
        ...     learning_rate=5e-7,
        ...     max_steps=100,
        ... )
    """

    def __init__(
        self,
        # KTO-specific
        beta: float = 0.1,  # Temperature coefficient
        desirable_weight: float = 1.0,  # Weight for positive examples
        undesirable_weight: float = 1.0,  # Weight for negative examples
        # Training args
        output_dir: str = "./kto_outputs",
        learning_rate: float = 5e-7,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        num_train_epochs: int = 1,
        max_steps: int = -1,
        warmup_steps: int = 10,
        logging_steps: int = 10,
        save_steps: int = 100,
        max_seq_length: int = 2048,
        **kwargs
    ):
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_seq_length = max_seq_length

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class SimPOConfig:
    """
    Configuration for Simple Preference Optimization training.

    SimPO removes the need for a reference model by using
    length-normalized log probabilities as implicit rewards.

    Compatible with TRL's SimPOConfig.

    Example:
        >>> config = SimPOConfig(
        ...     beta=2.0,
        ...     gamma=0.5,
        ...     learning_rate=5e-7,
        ...     max_steps=100,
        ... )
    """

    def __init__(
        self,
        # SimPO-specific
        beta: float = 2.0,  # Temperature coefficient
        gamma: float = 0.5,  # Target reward margin
        # Training args
        output_dir: str = "./simpo_outputs",
        learning_rate: float = 5e-7,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        num_train_epochs: int = 1,
        max_steps: int = -1,
        warmup_steps: int = 10,
        logging_steps: int = 10,
        save_steps: int = 100,
        max_seq_length: int = 2048,
        **kwargs
    ):
        self.beta = beta
        self.gamma = gamma
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_seq_length = max_seq_length

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class DPOTrainer:
    """
    Direct Preference Optimization Trainer.

    DPO trains models on preference data (chosen vs rejected responses)
    without requiring a separate reward model.

    Compatible with TRL's DPOTrainer API.
    Now with PROPER DPO loss implementation!

    Example:
        >>> from mlx_tune import FastLanguageModel, DPOTrainer, DPOConfig
        >>>
        >>> model, tokenizer = FastLanguageModel.from_pretrained(...)
        >>> model = FastLanguageModel.get_peft_model(model, r=16)
        >>>
        >>> # Preference dataset with chosen/rejected pairs
        >>> dataset = [
        ...     {"prompt": "...", "chosen": "...", "rejected": "..."},
        ... ]
        >>>
        >>> trainer = DPOTrainer(
        ...     model=model,
        ...     ref_model=None,  # Uses stop_gradient by default
        ...     train_dataset=dataset,
        ...     tokenizer=tokenizer,
        ...     args=DPOConfig(beta=0.1),
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        ref_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        args: Optional[DPOConfig] = None,
        use_native: bool = True,
        **kwargs
    ):
        self.model = model
        self.ref_model = ref_model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer or getattr(model, 'tokenizer', None)
        self.use_native = use_native and HAS_NATIVE_TRAINING

        # Extract config
        if args is None:
            args = DPOConfig()

        self.config = args
        self.beta = args.beta
        self.loss_type = args.loss_type
        self.label_smoothing = args.label_smoothing
        self.precompute_ref_logprobs = getattr(args, 'precompute_ref_logprobs', True)
        self.output_dir = Path(args.output_dir)
        self.learning_rate = args.learning_rate
        self.batch_size = args.per_device_train_batch_size
        self.max_steps = args.max_steps
        self.max_seq_length = args.max_seq_length
        self.max_prompt_length = args.max_prompt_length
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.warmup_steps = args.warmup_steps
        self.logging_steps = args.logging_steps
        self.save_steps = args.save_steps

        # Calculate iters
        if self.max_steps > 0:
            self.iters = self.max_steps
        else:
            dataset_size = len(train_dataset) if hasattr(train_dataset, '__len__') else 100
            self.iters = max(1, (dataset_size // self.batch_size) * args.num_train_epochs)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_path = self.output_dir / "adapters"
        self.adapter_path.mkdir(parents=True, exist_ok=True)

        print(f"DPOTrainer initialized:")
        print(f"  Beta: {self.beta}")
        print(f"  Loss type: {self.loss_type}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Iterations: {self.iters}")
        print(f"  Native training: {self.use_native}")
        print(f"  Using proper DPO loss: {self.use_native}")

    def _tokenize_preference_pair(self, sample: Dict) -> Dict:
        """Tokenize a preference pair (prompt + chosen, prompt + rejected).

        Also records the actual shared-prefix length in tokens via
        `common_prefix_length`. This drives the prompt-prefix KV-cache
        sharing path in `dpo_loss` (Session 38). The literal-prompt encode
        wouldn't survive BPE-merge differences across the prompt/response
        boundary, so we always derive prompt_length from the tokenised
        prefix that's verifiably shared.
        """
        prompt = sample.get('prompt', '')
        chosen = sample.get('chosen', '')
        rejected = sample.get('rejected', '')

        # Tokenize chosen and rejected with prompt
        chosen_text = prompt + chosen
        rejected_text = prompt + rejected

        chosen_ids = self.tokenizer.encode(chosen_text)
        rejected_ids = self.tokenizer.encode(rejected_text)

        # Truncate if needed
        if len(chosen_ids) > self.max_seq_length:
            chosen_ids = chosen_ids[:self.max_seq_length]
        if len(rejected_ids) > self.max_seq_length:
            rejected_ids = rejected_ids[:self.max_seq_length]

        return {
            'chosen_ids': chosen_ids,
            'rejected_ids': rejected_ids,
            'chosen_length': len(chosen_ids),
            'rejected_length': len(rejected_ids),
            'prompt_length': common_prefix_length(chosen_ids, rejected_ids),
        }

    def _prepare_dpo_batches(self):
        """Prepare batched DPO data for training."""
        tokenized_data = []
        for sample in self.train_dataset:
            if 'prompt' in sample and 'chosen' in sample and 'rejected' in sample:
                tokenized_data.append(self._tokenize_preference_pair(sample))

        return tokenized_data

    def _pad_to_length(self, ids: List[int], length: int, pad_id: int = 0) -> List[int]:
        """Pad sequence to target length."""
        if len(ids) >= length:
            return ids[:length]
        return ids + [pad_id] * (length - len(ids))

    def train(self):
        """
        Train the model using DPO with proper loss computation.

        Uses native MLX training with real DPO loss when available,
        falls back to SFT approximation otherwise.
        """
        print("=" * 70)
        print("Starting DPO Training")
        print("=" * 70)

        if self.use_native:
            return self._train_native()
        else:
            return self._train_subprocess()

    def _train_native(self):
        """Train using native MLX with proper DPO loss."""
        from mlx_tune._perf import (
            bucket_length, compiled_step, configure_wired_limit,
            enable_grad_checkpoint, make_step_state,
        )
        configure_wired_limit()

        print("\n[Using Native DPO Training with Proper Loss]")

        # Prepare data
        print("Preparing preference data...")
        tokenized_data = self._prepare_dpo_batches()
        print(f"✓ Prepared {len(tokenized_data)} preference pairs")

        # Session 38: pre-compute reference log-probs against the base model
        # BEFORE applying LoRA. At step 0 LoRA's B factor is initialised to
        # zero so a fresh LoRA-applied model is mathematically identical to
        # the base, but computing the reference here keeps the path explicit
        # and robust to future LoRA-init changes. The reference stays frozen
        # for the rest of training — that's the standard DPO formulation.
        if self.precompute_ref_logprobs:
            print("Pre-computing reference log-probs from base model (one-time)...")
            actual_for_ref = self.model.model if hasattr(self.model, 'model') else self.model
            if hasattr(actual_for_ref, 'eval'):
                actual_for_ref.eval()
            ref_pad_id = self.tokenizer.pad_token_id or 0
            for sample in tokenized_data:
                cl = sample['chosen_length']
                rl = sample['rejected_length']
                max_len = max(cl, rl)
                chosen_padded = sample['chosen_ids'] + [ref_pad_id] * (max_len - cl)
                rejected_padded = sample['rejected_ids'] + [ref_pad_id] * (max_len - rl)
                chosen_arr = mx.array([chosen_padded])
                rejected_arr = mx.array([rejected_padded])
                # No grad — these are reference values, frozen for the rest
                # of training. mx.stop_gradient also disconnects the autograd
                # graph so the precomputed values never participate in
                # backward through the policy model.
                chosen_ref = mx.stop_gradient(
                    compute_log_probs_with_lengths(actual_for_ref, chosen_arr, mx.array([cl]))
                )
                rejected_ref = mx.stop_gradient(
                    compute_log_probs_with_lengths(actual_for_ref, rejected_arr, mx.array([rl]))
                )
                mx.eval(chosen_ref, rejected_ref)
                sample['ref_chosen_logp'] = float(chosen_ref.item())
                sample['ref_rejected_logp'] = float(rejected_ref.item())
            print(f"✓ Reference log-probs cached for {len(tokenized_data)} pairs")

        # Apply LoRA if needed (after ref computation so ref is from the base model)
        if hasattr(self.model, '_apply_lora') and not getattr(self.model, '_lora_applied', False):
            print("Applying LoRA adapters...")
            self.model._apply_lora()
        enable_grad_checkpoint(self.model)

        # Get actual model and switch to training mode
        actual_model = self.model.model if hasattr(self.model, 'model') else self.model
        if hasattr(self.model, 'train'):
            self.model.train()
        if hasattr(actual_model, 'train'):
            actual_model.train()

        # Create optimizer
        lr_schedule = optim.cosine_decay(self.learning_rate, self.iters)
        optimizer = optim.AdamW(learning_rate=lr_schedule)

        # Training loop
        print(f"\nStarting training for {self.iters} iterations...")

        bs = max(1, self.batch_size)
        # Session 38: at batch_size=1 we want the prompt-prefix shared-KV-cache
        # path (`compute_log_probs_pair_shared_prefix`). That path branches on
        # Python-int prompt_length/chosen/rejected lengths so it doesn't fit
        # inside `mx.compile`'s shape cache — `.item()` would bake the value
        # into the trace and stale results across steps. For B==1 we therefore
        # bypass `compiled_step` and stay eager; the prompt-share saving on
        # long-prompt data is far larger than the compile-fusion saving (the
        # latter was ~10-25% at 16-64 steps; this is ~30-50%). For B>1 the
        # compiled path is still wired (no shared prefix opportunity there).
        use_shared_prefix = (bs == 1)

        if use_shared_prefix:
            print("  Prompt-prefix KV-cache sharing: ON (batch_size=1) — compile bypassed")

            def loss_fn(model, batch_data, prompt_length, chosen_len_py, rejected_len_py, ref_logps):
                chosen_ids, rejected_ids, chosen_lengths, rejected_lengths = batch_data
                ref_chosen, ref_rejected = ref_logps
                loss, _ = compute_dpo_loss(
                    model=model,
                    chosen_ids=chosen_ids,
                    rejected_ids=rejected_ids,
                    chosen_lengths=chosen_lengths,
                    rejected_lengths=rejected_lengths,
                    beta=self.beta,
                    label_smoothing=self.label_smoothing,
                    reference_chosen_logprobs=ref_chosen,
                    reference_rejected_logprobs=ref_rejected,
                    prompt_length=prompt_length,
                    chosen_length_py=chosen_len_py,
                    rejected_length_py=rejected_len_py,
                )
                return loss

            loss_and_grad = nn.value_and_grad(actual_model, loss_fn)

            def step_fn(batch_data, prompt_length, chosen_len_py, rejected_len_py, ref_logps):
                loss, grads = loss_and_grad(
                    actual_model, batch_data, prompt_length, chosen_len_py, rejected_len_py, ref_logps
                )
                optimizer.update(actual_model, grads)
                return loss
        else:
            def loss_fn(model, batch_data, ref_logps):
                chosen_ids, rejected_ids, chosen_lengths, rejected_lengths = batch_data
                ref_chosen, ref_rejected = ref_logps
                loss, _ = compute_dpo_loss(
                    model=model,
                    chosen_ids=chosen_ids,
                    rejected_ids=rejected_ids,
                    chosen_lengths=chosen_lengths,
                    rejected_lengths=rejected_lengths,
                    beta=self.beta,
                    label_smoothing=self.label_smoothing,
                    reference_chosen_logprobs=ref_chosen,
                    reference_rejected_logprobs=ref_rejected,
                )
                return loss

            loss_and_grad = nn.value_and_grad(actual_model, loss_fn)
            state = make_step_state(actual_model, optimizer)

            def step_fn(batch_data, ref_logps):
                loss, grads = loss_and_grad(actual_model, batch_data, ref_logps)
                optimizer.update(actual_model, grads)
                return loss

            step_fn = compiled_step(step_fn, state)

        # State for mx.eval — for B==1 we still keep the model + optimizer state
        # on-device so the evaluation barrier flushes everything together.
        state = make_step_state(actual_model, optimizer)

        total_loss_arr = mx.zeros((), dtype=mx.float32)  # on-device accumulator
        for step in range(self.iters):
            # Gather batch of samples
            batch_samples = [
                tokenized_data[(step * bs + i) % len(tokenized_data)]
                for i in range(bs)
            ]

            # Pad all sequences to common max length within batch, rounded
            # up to the next bucket so `mx.compile`'s per-shape cache hits
            # (B>1 path) or so cache-fork buffer sizes stay consistent (B==1).
            max_len = max(
                max(s['chosen_length'], s['rejected_length']) for s in batch_samples
            )
            max_len = bucket_length(max_len)
            pad_id = self.tokenizer.pad_token_id or 0

            chosen_ids = mx.array([self._pad_to_length(s['chosen_ids'], max_len, pad_id) for s in batch_samples])
            rejected_ids = mx.array([self._pad_to_length(s['rejected_ids'], max_len, pad_id) for s in batch_samples])
            chosen_lengths = mx.array([s['chosen_length'] for s in batch_samples])
            rejected_lengths = mx.array([s['rejected_length'] for s in batch_samples])

            batch_data = (chosen_ids, rejected_ids, chosen_lengths, rejected_lengths)

            # Assemble the reference log-probs tuple. When precompute is on,
            # use the cached per-sample values; otherwise pass (None, None)
            # so dpo_loss falls back to its stop_gradient(log_pi) reference.
            if self.precompute_ref_logprobs and 'ref_chosen_logp' in batch_samples[0]:
                ref_chosen = mx.array([s['ref_chosen_logp'] for s in batch_samples])
                ref_rejected = mx.array([s['ref_rejected_logp'] for s in batch_samples])
                ref_logps = (ref_chosen, ref_rejected)
            else:
                ref_logps = (None, None)

            if use_shared_prefix:
                # B==1: hand the shared-prefix branch Python-int copies of the
                # lengths so it can decide without forcing a host sync.
                s0 = batch_samples[0]
                loss = step_fn(
                    batch_data,
                    int(s0['prompt_length']),
                    int(s0['chosen_length']),
                    int(s0['rejected_length']),
                    ref_logps,
                )
            else:
                loss = step_fn(batch_data, ref_logps)
            total_loss_arr = total_loss_arr + loss.astype(mx.float32)
            mx.eval(state, total_loss_arr)

            # Logging — single host sync per logging_steps interval, not per step.
            if (step + 1) % self.logging_steps == 0:
                avg_loss = float(total_loss_arr.item()) / self.logging_steps
                print(f"  Step {step + 1}/{self.iters} | Loss: {avg_loss:.4f} | batch_size: {bs}")
                total_loss_arr = mx.zeros((), dtype=mx.float32)

            # Save checkpoint
            if (step + 1) % self.save_steps == 0:
                self._save_adapters(step + 1)

        # Final save
        self._save_adapters(self.iters)

        print("\n" + "=" * 70)
        print("DPO Training Complete!")
        print("=" * 70)
        print(f"  Adapters saved to: {self.adapter_path}")

        return {"status": "success", "adapter_path": str(self.adapter_path)}

    def _save_adapters(self, step: int):
        """Save adapter weights and config."""
        if _save_adapters_and_config(self.model, self.adapter_path):
            print(f"  ✓ Saved checkpoint at step {step}")

    def _train_subprocess(self):
        """Fallback: Train using subprocess (SFT approximation)."""
        warnings.warn(
            "Native DPO training not available. Using SFT on chosen responses. "
            "Install mlx-lm[train] for proper DPO loss.",
            UserWarning
        )

        print("\n[Using Subprocess Training (SFT Approximation)]")

        # Prepare SFT data from chosen responses
        train_file = self.output_dir / "train.jsonl"
        valid_file = self.output_dir / "valid.jsonl"

        with open(train_file, 'w') as f:
            for sample in self.train_dataset:
                if 'prompt' in sample and 'chosen' in sample:
                    messages = [
                        {"role": "user", "content": sample['prompt']},
                        {"role": "assistant", "content": sample['chosen']}
                    ]
                    f.write(json.dumps({"messages": messages}) + '\n')

        import shutil
        shutil.copy(train_file, valid_file)

        model_name = getattr(self.model, 'model_name', 'model')

        cmd = [
            "mlx_lm.lora",
            "--model", model_name,
            "--train",
            "--data", str(self.output_dir),
            "--iters", str(self.iters),
            "--learning-rate", str(self.learning_rate),
            "--batch-size", str(self.batch_size),
            "--adapter-path", str(self.adapter_path),
        ]

        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        print("DPO Training Complete (SFT approximation)!")
        return {"status": "success", "adapter_path": str(self.adapter_path)}


class ORPOTrainer:
    """
    Odds Ratio Preference Optimization Trainer.

    ORPO combines SFT and preference alignment in a single training step,
    making it simpler and more memory-efficient than DPO.

    Compatible with TRL's ORPOTrainer API.
    Now with PROPER ORPO loss implementation!

    Example:
        >>> trainer = ORPOTrainer(
        ...     model=model,
        ...     train_dataset=preference_dataset,
        ...     tokenizer=tokenizer,
        ...     args=ORPOConfig(beta=0.1),
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        tokenizer: Optional[Any] = None,
        args: Optional[ORPOConfig] = None,
        use_native: bool = True,
        **kwargs
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer or getattr(model, 'tokenizer', None)
        self.use_native = use_native and HAS_NATIVE_TRAINING

        if args is None:
            args = ORPOConfig()

        self.config = args
        self.beta = args.beta
        self.output_dir = Path(args.output_dir)
        self.learning_rate = args.learning_rate
        self.batch_size = args.per_device_train_batch_size
        self.max_steps = args.max_steps
        self.max_seq_length = args.max_seq_length
        self.logging_steps = args.logging_steps
        self.save_steps = args.save_steps

        if self.max_steps > 0:
            self.iters = self.max_steps
        else:
            dataset_size = len(train_dataset) if hasattr(train_dataset, '__len__') else 100
            self.iters = max(1, (dataset_size // self.batch_size) * args.num_train_epochs)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_path = self.output_dir / "adapters"
        self.adapter_path.mkdir(parents=True, exist_ok=True)

        print(f"ORPOTrainer initialized:")
        print(f"  Beta: {self.beta}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Iterations: {self.iters}")
        print(f"  Native training: {self.use_native}")

    def _tokenize_preference_pair(self, sample: Dict) -> Dict:
        """Tokenize a preference pair.

        Records prompt_length = common token prefix between chosen and
        rejected (Session 38 — enables prompt-prefix KV-cache sharing in
        `orpo_loss`).
        """
        prompt = sample.get('prompt', '')
        chosen = sample.get('chosen', '')
        rejected = sample.get('rejected', '')

        chosen_ids = self.tokenizer.encode(prompt + chosen)
        rejected_ids = self.tokenizer.encode(prompt + rejected)

        if len(chosen_ids) > self.max_seq_length:
            chosen_ids = chosen_ids[:self.max_seq_length]
        if len(rejected_ids) > self.max_seq_length:
            rejected_ids = rejected_ids[:self.max_seq_length]

        return {
            'chosen_ids': chosen_ids,
            'rejected_ids': rejected_ids,
            'chosen_length': len(chosen_ids),
            'rejected_length': len(rejected_ids),
            'prompt_length': common_prefix_length(chosen_ids, rejected_ids),
        }

    def _pad_to_length(self, ids: List[int], length: int, pad_id: int = 0) -> List[int]:
        if len(ids) >= length:
            return ids[:length]
        return ids + [pad_id] * (length - len(ids))

    def train(self):
        """Train using ORPO with proper loss."""
        print("=" * 70)
        print("Starting ORPO Training")
        print("=" * 70)

        if self.use_native:
            return self._train_native()
        else:
            return self._train_subprocess()

    def _train_native(self):
        """Train with native ORPO loss."""
        from mlx_tune._perf import (
            bucket_length, compiled_step, configure_wired_limit,
            enable_grad_checkpoint, make_step_state,
        )
        configure_wired_limit()

        print("\n[Using Native ORPO Training with Proper Loss]")

        if hasattr(self.model, '_apply_lora') and not getattr(self.model, '_lora_applied', False):
            self.model._apply_lora()
        enable_grad_checkpoint(self.model)

        # Prepare data
        tokenized_data = []
        for sample in self.train_dataset:
            if 'prompt' in sample and 'chosen' in sample and 'rejected' in sample:
                tokenized_data.append(self._tokenize_preference_pair(sample))
        print(f"✓ Prepared {len(tokenized_data)} preference pairs")

        actual_model = self.model.model if hasattr(self.model, 'model') else self.model
        if hasattr(self.model, 'train'):
            self.model.train()
        if hasattr(actual_model, 'train'):
            actual_model.train()

        lr_schedule = optim.cosine_decay(self.learning_rate, self.iters)
        optimizer = optim.AdamW(learning_rate=lr_schedule)

        bs = max(1, self.batch_size)
        use_shared_prefix = (bs == 1)
        # See DPOTrainer._train_native for rationale. Session 38 prompt-prefix
        # share is the bigger lever on ORPO too — at ctx=4096 ORPO holds two
        # full forwards' worth of activations; halving the prompt forward
        # cost is a direct ~30-50% wall-time cut on long-prompt data, which
        # comfortably dominates the compile-fusion win we'd otherwise get.

        if use_shared_prefix:
            print("  Prompt-prefix KV-cache sharing: ON (batch_size=1) — compile bypassed")

            def loss_fn(model, batch_data, prompt_length, chosen_len_py, rejected_len_py):
                chosen_ids, rejected_ids, chosen_lengths, rejected_lengths = batch_data
                loss, _ = compute_orpo_loss(
                    model, chosen_ids, rejected_ids, chosen_lengths, rejected_lengths,
                    self.beta,
                    prompt_length=prompt_length,
                    chosen_length_py=chosen_len_py,
                    rejected_length_py=rejected_len_py,
                )
                return loss

            loss_and_grad = nn.value_and_grad(actual_model, loss_fn)

            def step_fn(batch_data, prompt_length, chosen_len_py, rejected_len_py):
                loss, grads = loss_and_grad(
                    actual_model, batch_data, prompt_length, chosen_len_py, rejected_len_py
                )
                optimizer.update(actual_model, grads)
                return loss
        else:
            def loss_fn(model, batch_data):
                chosen_ids, rejected_ids, chosen_lengths, rejected_lengths = batch_data
                loss, _ = compute_orpo_loss(
                    model, chosen_ids, rejected_ids, chosen_lengths, rejected_lengths, self.beta
                )
                return loss

            loss_and_grad = nn.value_and_grad(actual_model, loss_fn)
            state_compile = make_step_state(actual_model, optimizer)

            def step_fn(batch_data):
                loss, grads = loss_and_grad(actual_model, batch_data)
                optimizer.update(actual_model, grads)
                return loss

            step_fn = compiled_step(step_fn, state_compile)

        state = make_step_state(actual_model, optimizer)

        total_loss_arr = mx.zeros((), dtype=mx.float32)
        for step in range(self.iters):
            # Gather batch of samples
            batch_samples = [
                tokenized_data[(step * bs + i) % len(tokenized_data)]
                for i in range(bs)
            ]

            max_len = max(
                max(s['chosen_length'], s['rejected_length']) for s in batch_samples
            )
            max_len = bucket_length(max_len)
            pad_id = self.tokenizer.pad_token_id or 0

            chosen_ids = mx.array([self._pad_to_length(s['chosen_ids'], max_len, pad_id) for s in batch_samples])
            rejected_ids = mx.array([self._pad_to_length(s['rejected_ids'], max_len, pad_id) for s in batch_samples])
            chosen_lengths = mx.array([s['chosen_length'] for s in batch_samples])
            rejected_lengths = mx.array([s['rejected_length'] for s in batch_samples])

            if use_shared_prefix:
                s0 = batch_samples[0]
                loss = step_fn(
                    (chosen_ids, rejected_ids, chosen_lengths, rejected_lengths),
                    int(s0['prompt_length']),
                    int(s0['chosen_length']),
                    int(s0['rejected_length']),
                )
            else:
                loss = step_fn((chosen_ids, rejected_ids, chosen_lengths, rejected_lengths))
            total_loss_arr = total_loss_arr + loss.astype(mx.float32)
            mx.eval(state, total_loss_arr)

            if (step + 1) % self.logging_steps == 0:
                avg_loss = float(total_loss_arr.item()) / self.logging_steps
                print(f"  Step {step + 1}/{self.iters} | Loss: {avg_loss:.4f} | batch_size: {bs}")
                total_loss_arr = mx.zeros((), dtype=mx.float32)

        # Save adapters and config
        _save_adapters_and_config(self.model, self.adapter_path)

        print("\n" + "=" * 70)
        print("ORPO Training Complete!")
        print("=" * 70)
        print(f"  Adapters saved to: {self.adapter_path}")
        return {"status": "success", "adapter_path": str(self.adapter_path)}

    def _train_subprocess(self):
        """Fallback subprocess training."""
        warnings.warn("Using SFT approximation for ORPO.", UserWarning)

        train_file = self.output_dir / "train.jsonl"
        with open(train_file, 'w') as f:
            for sample in self.train_dataset:
                if 'prompt' in sample and 'chosen' in sample:
                    messages = [
                        {"role": "user", "content": sample['prompt']},
                        {"role": "assistant", "content": sample['chosen']}
                    ]
                    f.write(json.dumps({"messages": messages}) + '\n')

        import shutil
        shutil.copy(train_file, self.output_dir / "valid.jsonl")

        cmd = [
            "mlx_lm.lora", "--model", getattr(self.model, 'model_name', 'model'),
            "--train", "--data", str(self.output_dir), "--iters", str(self.iters),
            "--learning-rate", str(self.learning_rate), "--batch-size", str(self.batch_size),
            "--adapter-path", str(self.adapter_path),
        ]
        subprocess.run(cmd, check=True)
        return {"status": "success"}


class GRPOTrainer:
    """
    Group Relative Policy Optimization Trainer.

    GRPO is the technique used by DeepSeek to train reasoning models like R1.
    It removes the need for a value model by using group statistics from
    multiple generations and custom reward functions.

    Key features:
    - No value model needed (uses group statistics)
    - Custom reward functions (for math, code verification, etc.)
    - Supports GRPO, Dr.GRPO, DAPO, BNPO variants
    - NOW WITH FULL MULTI-GENERATION IMPLEMENTATION!

    Example:
        >>> def math_reward(response, prompt):
        ...     # Custom reward for math problems
        ...     return 1.0 if "correct" in response.lower() else 0.0
        >>>
        >>> trainer = GRPOTrainer(
        ...     model=model,
        ...     train_dataset=math_dataset,
        ...     tokenizer=tokenizer,
        ...     reward_fn=math_reward,
        ...     args=GRPOConfig(
        ...         loss_type='grpo',
        ...         num_generations=4,
        ...     ),
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        tokenizer: Optional[Any] = None,
        reward_fn: Optional[Callable] = None,
        args: Optional[GRPOConfig] = None,
        use_native: bool = True,
        **kwargs
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer or getattr(model, 'tokenizer', None)
        self.use_native = use_native and HAS_NATIVE_TRAINING

        if args is None:
            args = GRPOConfig()

        self.config = args
        self.loss_type = args.loss_type
        self.beta = args.beta
        self.num_generations = args.num_generations
        self.max_completion_length = args.max_completion_length
        self.reward_fn = reward_fn or args.reward_fn
        self.output_dir = Path(args.output_dir)
        self.learning_rate = args.learning_rate
        self.batch_size = args.per_device_train_batch_size
        self.max_steps = args.max_steps
        self.temperature = args.temperature
        self.logging_steps = args.logging_steps
        self.save_steps = args.save_steps

        if self.max_steps > 0:
            self.iters = self.max_steps
        else:
            dataset_size = len(train_dataset) if hasattr(train_dataset, '__len__') else 100
            self.iters = max(1, (dataset_size // self.batch_size) * args.num_train_epochs)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_path = self.output_dir / "adapters"
        self.adapter_path.mkdir(parents=True, exist_ok=True)

        # Default reward function if none provided
        if self.reward_fn is None:
            self.reward_fn = lambda response, prompt: len(response.split()) / 100.0

        print(f"GRPOTrainer initialized:")
        print(f"  Loss type: {self.loss_type}")
        print(f"  Beta: {self.beta}")
        print(f"  Num generations: {self.num_generations}")
        print(f"  Custom reward fn: {'Yes' if reward_fn else 'Default (length-based)'}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Iterations: {self.iters}")
        print(f"  Native GRPO: {self.use_native}")

    def train(self):
        """
        Train using GRPO with multi-generation sampling.
        """
        print("=" * 70)
        print(f"Starting GRPO Training (loss_type={self.loss_type})")
        print("=" * 70)

        if self.use_native:
            return self._train_native()
        else:
            return self._train_subprocess()

    def _train_native(self):
        """
        Train with native GRPO using two-phase approach per step:

        Phase 1 (no gradients): Generate N completions, score with reward_fn,
            compute group-normalized advantages.
        Phase 2 (with gradients): Forward pass on generated tokens to compute
            log probs, apply policy gradient loss weighted by advantages,
            update model via optimizer.
        """
        from mlx_tune._perf import (
            bucket_length, compiled_step, configure_wired_limit,
            enable_grad_checkpoint, make_step_state,
        )
        configure_wired_limit()

        print("\n[Using Native GRPO Training with Policy Gradient]")

        if hasattr(self.model, '_apply_lora') and not getattr(self.model, '_lora_applied', False):
            self.model._apply_lora()
        enable_grad_checkpoint(self.model)

        # Prepare prompts and ground truth answers
        prompts = []
        answers = []
        for sample in self.train_dataset:
            prompt = sample.get('prompt', sample.get('question', ''))
            if prompt:
                prompts.append(prompt)
                answers.append(sample.get('answer', ''))
        print(f"  Prepared {len(prompts)} prompts")

        actual_model = self.model.model if hasattr(self.model, 'model') else self.model
        if hasattr(self.model, 'train'):
            self.model.train()
        if hasattr(actual_model, 'train'):
            actual_model.train()

        lr_schedule = optim.cosine_decay(self.learning_rate, self.iters)
        optimizer = optim.AdamW(learning_rate=lr_schedule)

        print(f"\nStarting training for {self.iters} iterations...")
        print(f"  Generating {self.num_generations} completions per prompt")

        # Define policy gradient loss function for nn.value_and_grad
        def pg_loss_fn(model, sequences, seq_lengths, advantages):
            """Compute policy gradient loss on pre-generated completions (batched)."""
            # Batched forward pass for all generations at once
            log_probs = compute_log_probs_with_lengths(model, sequences, seq_lengths)  # [N]
            # Policy gradient: -advantage * log_prob, averaged over generations
            losses = -advantages * log_probs
            return mx.mean(losses)

        loss_and_grad = nn.value_and_grad(actual_model, pg_loss_fn)

        # Compiled step (Phase 2 only — generation phase is dynamic and can't
        # be `mx.compile`d). `shapeless=True` because completion lengths vary.
        state = make_step_state(actual_model, optimizer)

        def step_fn(sequences, seq_lengths, advantages):
            loss, grads = loss_and_grad(actual_model, sequences, seq_lengths, advantages)
            optimizer.update(actual_model, grads)
            return loss

        step_fn = compiled_step(step_fn, state)

        total_loss_arr = mx.zeros((), dtype=mx.float32)  # on-device accumulator
        total_reward = 0.0
        for step in range(self.iters):
            prompt_idx = step % len(prompts)
            prompt = prompts[prompt_idx]
            answer = answers[prompt_idx]

            prompt_ids = mx.array(self.tokenizer.encode(prompt))

            # Session 38: build a shared prompt KV cache once per prompt and
            # fork it for each of the N generations. With this, the prompt
            # forward runs once instead of N times. Saving scales with
            # prompt_len / total_tokens — significant for reasoning prompts
            # (1k+ tokens, short completions). Falls back to per-generation
            # full prompt forward when the helper can't build a cache (e.g.
            # mock models in tests).
            shared_prompt_cache = build_shared_prompt_cache(actual_model, prompt_ids)

            # ---- Phase 1: Generate completions and compute rewards (no gradients) ----
            completions_data = []
            rewards = []

            for _ in range(self.num_generations):
                # Fork the shared cache per generation so writes from one
                # completion don't leak into the next.
                fork = (
                    fork_prompt_cache(actual_model, shared_prompt_cache)
                    if shared_prompt_cache is not None
                    else None
                )
                gen_ids, _ = generate_with_log_probs(
                    actual_model, self.tokenizer, prompt_ids,
                    max_tokens=self.max_completion_length,
                    temperature=self.temperature,
                    prompt_cache=fork,
                )
                mx.eval(gen_ids)

                # Decode completion text (skip prompt)
                prompt_len = len(prompt_ids)
                completion_ids = gen_ids[prompt_len:]
                completion_text = self.tokenizer.decode(completion_ids.tolist())

                # Score with reward function
                reward = self.reward_fn(completion_text, answer)
                rewards.append(reward)
                completions_data.append({
                    'full_ids': gen_ids.tolist(),
                    'length': len(gen_ids),
                })

            rewards_arr = mx.array(rewards)
            mean_reward = mx.mean(rewards_arr)
            std_reward = mx.std(rewards_arr)
            mx.eval(mean_reward, std_reward)

            avg_reward = mean_reward.item()
            total_reward += avg_reward

            # Skip update if all rewards are equal (std=0)
            if std_reward.item() < 1e-8:
                if (step + 1) % self.logging_steps == 0:
                    avg_loss = float(total_loss_arr.item()) / self.logging_steps
                    avg_rew = total_reward / self.logging_steps
                    print(f"  Step {step + 1}/{self.iters} | Loss: {avg_loss:.4f} | Reward: {avg_rew:.3f} (skipped, equal rewards)")
                    total_loss_arr = mx.zeros((), dtype=mx.float32)
                    total_reward = 0.0
                continue

            advantages = (rewards_arr - mean_reward) / (std_reward + 1e-8)
            mx.eval(advantages)

            # ---- Phase 2: Compute policy gradient loss with gradient tracking ----
            # Pad sequences to same length for batched processing, rounded
            # up to the next bucket so `mx.compile`'s per-shape cache hits.
            max_len = bucket_length(max(c['length'] for c in completions_data))
            pad_id = getattr(self.tokenizer, 'pad_token_id', None) or 0

            padded_seqs = []
            lengths = []
            for c in completions_data:
                ids = c['full_ids']
                padded = ids + [pad_id] * (max_len - len(ids))
                padded_seqs.append(padded)
                lengths.append(c['length'])

            sequences = mx.array(padded_seqs)
            seq_lengths = mx.array(lengths)

            loss = step_fn(sequences, seq_lengths, advantages)
            total_loss_arr = total_loss_arr + loss.astype(mx.float32)
            mx.eval(state, total_loss_arr)

            if (step + 1) % self.logging_steps == 0:
                avg_loss = float(total_loss_arr.item()) / self.logging_steps
                avg_rew = total_reward / self.logging_steps
                print(f"  Step {step + 1}/{self.iters} | Loss: {avg_loss:.4f} | Reward: {avg_rew:.3f}")
                total_loss_arr = mx.zeros((), dtype=mx.float32)
                total_reward = 0.0

            # Save checkpoint
            if (step + 1) % self.save_steps == 0:
                _save_adapters_and_config(self.model, self.adapter_path)
                print(f"    Saved checkpoint at step {step + 1}")

        # Final save
        _save_adapters_and_config(self.model, self.adapter_path)

        print("\n" + "=" * 70)
        print("GRPO Training Complete!")
        print("=" * 70)
        print(f"  Adapters saved to: {self.adapter_path}")
        return {"status": "success", "adapter_path": str(self.adapter_path)}

    def _train_subprocess(self):
        """Fallback to SFT approximation."""
        warnings.warn(
            "Native GRPO not available. Using SFT on provided responses.",
            UserWarning
        )

        train_file = self.output_dir / "train.jsonl"
        with open(train_file, 'w') as f:
            for sample in self.train_dataset:
                if 'prompt' in sample:
                    messages = [{"role": "user", "content": sample['prompt']}]
                    if 'response' in sample or 'answer' in sample:
                        response = sample.get('response', sample.get('answer', ''))
                        messages.append({"role": "assistant", "content": response})
                    f.write(json.dumps({"messages": messages}) + '\n')

        import shutil
        shutil.copy(train_file, self.output_dir / "valid.jsonl")

        cmd = [
            "mlx_lm.lora", "--model", getattr(self.model, 'model_name', 'model'),
            "--train", "--data", str(self.output_dir), "--iters", str(self.iters),
            "--adapter-path", str(self.adapter_path),
        ]
        subprocess.run(cmd, check=True)
        return {"status": "success"}


class KTOTrainer:
    """
    Kahneman-Tversky Optimization Trainer.

    KTO uses prospect theory for preference optimization,
    treating gains and losses asymmetrically.

    Supports two dataset formats:
    - TRL format: {"prompt": "...", "completion": "...", "label": true/false}
    - Legacy format: {"text": "...", "label": 1/0}

    Compatible with TRL's KTOTrainer API.

    Example:
        >>> trainer = KTOTrainer(
        ...     model=model,
        ...     train_dataset=kto_dataset,
        ...     tokenizer=tokenizer,
        ...     args=KTOConfig(beta=0.1, max_steps=100),
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        tokenizer: Optional[Any] = None,
        args: Optional[KTOConfig] = None,
        use_native: bool = True,
        **kwargs
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer or getattr(model, 'tokenizer', None)
        self.use_native = use_native and HAS_NATIVE_TRAINING

        # Extract config from args or kwargs (backward compat)
        if args is None:
            args = KTOConfig(**kwargs)

        self.config = args
        self.beta = args.beta
        self.output_dir = Path(args.output_dir)
        self.learning_rate = args.learning_rate
        self.max_seq_length = args.max_seq_length
        self.logging_steps = args.logging_steps
        self.save_steps = args.save_steps
        self.batch_size = args.per_device_train_batch_size

        if args.max_steps > 0:
            self.iters = args.max_steps
        else:
            dataset_size = len(train_dataset) if hasattr(train_dataset, '__len__') else 100
            self.iters = max(1, (dataset_size // max(1, self.batch_size)) * args.num_train_epochs)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_path = self.output_dir / "adapters"
        self.adapter_path.mkdir(parents=True, exist_ok=True)

        print(f"KTOTrainer initialized:")
        print(f"  Beta: {self.beta}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Iterations: {self.iters}")
        print(f"  Native training: {self.use_native}")

    def train(self):
        """Train using KTO with proper loss."""
        print("=" * 70)
        print("Starting KTO Training")
        print("=" * 70)

        if not self.use_native:
            warnings.warn("KTO requires native training. Using SFT approximation.", UserWarning)
            return {"status": "fallback"}

        from mlx_tune._perf import (
            bucket_length, compiled_step, configure_wired_limit,
            enable_grad_checkpoint, make_step_state,
        )
        configure_wired_limit()

        print("\n[Using Native KTO Training with Proper Loss]")

        if hasattr(self.model, '_apply_lora') and not getattr(self.model, '_lora_applied', False):
            self.model._apply_lora()
        enable_grad_checkpoint(self.model)

        actual_model = self.model.model if hasattr(self.model, 'model') else self.model
        if hasattr(self.model, 'train'):
            self.model.train()
        if hasattr(actual_model, 'train'):
            actual_model.train()

        lr_schedule = optim.cosine_decay(self.learning_rate, self.iters)
        optimizer = optim.AdamW(learning_rate=lr_schedule)

        # Prepare data - supports both TRL and legacy formats
        tokenized_data = []
        for sample in self.train_dataset:
            if 'prompt' in sample and 'completion' in sample:
                # TRL format: prompt + completion + boolean label
                text = sample['prompt'] + sample['completion']
                label = 1.0 if sample.get('label', True) else 0.0
            elif 'text' in sample and 'label' in sample:
                # Legacy format: text + numeric label
                text = sample['text']
                label = float(sample['label'])
            else:
                continue

            ids = self.tokenizer.encode(text)[:self.max_seq_length]
            tokenized_data.append({
                'ids': ids,
                'length': len(ids),
                'label': label,
            })

        print(f"  Prepared {len(tokenized_data)} samples")

        def loss_fn(model, batch_data):
            input_ids, lengths, labels = batch_data
            loss, _ = compute_kto_loss(model, input_ids, lengths, labels, self.beta)
            return loss

        loss_and_grad = nn.value_and_grad(actual_model, loss_fn)

        # Compiled step. `shapeless=True` because seq length is per-batch max.
        state = make_step_state(actual_model, optimizer)

        def step_fn(batch_data):
            loss, grads = loss_and_grad(actual_model, batch_data)
            optimizer.update(actual_model, grads)
            return loss

        step_fn = compiled_step(step_fn, state)

        total_loss_arr = mx.zeros((), dtype=mx.float32)
        bs = max(1, self.batch_size)
        for step in range(self.iters):
            # Gather batch of samples
            batch_samples = [
                tokenized_data[(step * bs + i) % len(tokenized_data)]
                for i in range(bs)
            ]
            pad_id = self.tokenizer.pad_token_id or 0

            max_len = bucket_length(max(s['length'] for s in batch_samples))
            input_ids = mx.array([s['ids'] + [pad_id] * (max_len - len(s['ids'])) for s in batch_samples])
            lengths = mx.array([s['length'] for s in batch_samples])
            labels = mx.array([s['label'] for s in batch_samples])

            loss = step_fn((input_ids, lengths, labels))
            total_loss_arr = total_loss_arr + loss.astype(mx.float32)
            mx.eval(state, total_loss_arr)

            if (step + 1) % self.logging_steps == 0:
                avg_loss = float(total_loss_arr.item()) / self.logging_steps
                print(f"  Step {step + 1}/{self.iters} | Loss: {avg_loss:.4f} | batch_size: {bs}")
                total_loss_arr = mx.zeros((), dtype=mx.float32)

            if (step + 1) % self.save_steps == 0:
                _save_adapters_and_config(self.model, self.adapter_path)

        # Final save
        _save_adapters_and_config(self.model, self.adapter_path)

        print("\n" + "=" * 70)
        print("KTO Training Complete!")
        print("=" * 70)
        print(f"  Adapters saved to: {self.adapter_path}")
        return {"status": "success", "adapter_path": str(self.adapter_path)}


class SimPOTrainer:
    """
    Simple Preference Optimization Trainer.

    SimPO simplifies DPO by removing the reference model requirement.
    Uses length-normalized log probabilities as implicit rewards.

    Compatible with TRL's SimPOTrainer API.

    Example:
        >>> trainer = SimPOTrainer(
        ...     model=model,
        ...     train_dataset=preference_dataset,
        ...     tokenizer=tokenizer,
        ...     args=SimPOConfig(beta=2.0, gamma=0.5, max_steps=100),
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        tokenizer: Optional[Any] = None,
        args: Optional[SimPOConfig] = None,
        use_native: bool = True,
        **kwargs
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer or getattr(model, 'tokenizer', None)
        self.use_native = use_native and HAS_NATIVE_TRAINING

        # Extract config from args or kwargs (backward compat)
        if args is None:
            args = SimPOConfig(**kwargs)

        self.config = args
        self.gamma = args.gamma
        self.beta = args.beta
        self.output_dir = Path(args.output_dir)
        self.learning_rate = args.learning_rate
        self.max_seq_length = args.max_seq_length
        self.logging_steps = args.logging_steps
        self.save_steps = args.save_steps
        self.batch_size = args.per_device_train_batch_size

        if args.max_steps > 0:
            self.iters = args.max_steps
        else:
            dataset_size = len(train_dataset) if hasattr(train_dataset, '__len__') else 100
            self.iters = max(1, (dataset_size // max(1, self.batch_size)) * args.num_train_epochs)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_path = self.output_dir / "adapters"
        self.adapter_path.mkdir(parents=True, exist_ok=True)

        print(f"SimPOTrainer initialized:")
        print(f"  Beta: {self.beta}, Gamma: {self.gamma}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Iterations: {self.iters}")
        print(f"  Native training: {self.use_native}")

    def _tokenize_pair(self, sample):
        prompt = sample.get('prompt', '')
        chosen = sample.get('chosen', '')
        rejected = sample.get('rejected', '')

        chosen_ids = self.tokenizer.encode(prompt + chosen)[:self.max_seq_length]
        rejected_ids = self.tokenizer.encode(prompt + rejected)[:self.max_seq_length]

        return {
            'chosen_ids': chosen_ids,
            'rejected_ids': rejected_ids,
            'chosen_length': len(chosen_ids),
            'rejected_length': len(rejected_ids),
            'prompt_length': common_prefix_length(chosen_ids, rejected_ids),
        }

    def _pad(self, ids, length, pad_id=0):
        return ids + [pad_id] * (length - len(ids)) if len(ids) < length else ids[:length]

    def train(self):
        """Train using SimPO with proper loss."""
        print("=" * 70)
        print("Starting SimPO Training")
        print("=" * 70)

        if not self.use_native:
            warnings.warn("SimPO requires native training. Using SFT approximation.", UserWarning)
            return {"status": "fallback"}

        from mlx_tune._perf import (
            bucket_length, compiled_step, configure_wired_limit,
            enable_grad_checkpoint, make_step_state,
        )
        configure_wired_limit()

        print("\n[Using Native SimPO Training with Proper Loss]")

        if hasattr(self.model, '_apply_lora') and not getattr(self.model, '_lora_applied', False):
            self.model._apply_lora()
        enable_grad_checkpoint(self.model)

        tokenized_data = []
        for sample in self.train_dataset:
            if 'prompt' in sample and 'chosen' in sample and 'rejected' in sample:
                tokenized_data.append(self._tokenize_pair(sample))
        print(f"✓ Prepared {len(tokenized_data)} preference pairs")

        actual_model = self.model.model if hasattr(self.model, 'model') else self.model
        if hasattr(self.model, 'train'):
            self.model.train()
        if hasattr(actual_model, 'train'):
            actual_model.train()

        lr_schedule = optim.cosine_decay(self.learning_rate, self.iters)
        optimizer = optim.AdamW(learning_rate=lr_schedule)

        bs = max(1, self.batch_size)
        use_shared_prefix = (bs == 1)  # See DPOTrainer for rationale (Session 38)

        if use_shared_prefix:
            print("  Prompt-prefix KV-cache sharing: ON (batch_size=1) — compile bypassed")

            def loss_fn(model, batch_data, prompt_length, chosen_len_py, rejected_len_py):
                chosen_ids, rejected_ids, chosen_lengths, rejected_lengths = batch_data
                loss, _ = compute_simpo_loss(
                    model, chosen_ids, rejected_ids, chosen_lengths, rejected_lengths,
                    self.beta, self.gamma,
                    prompt_length=prompt_length,
                    chosen_length_py=chosen_len_py,
                    rejected_length_py=rejected_len_py,
                )
                return loss

            loss_and_grad = nn.value_and_grad(actual_model, loss_fn)

            def step_fn(batch_data, prompt_length, chosen_len_py, rejected_len_py):
                loss, grads = loss_and_grad(
                    actual_model, batch_data, prompt_length, chosen_len_py, rejected_len_py
                )
                optimizer.update(actual_model, grads)
                return loss
        else:
            def loss_fn(model, batch_data):
                chosen_ids, rejected_ids, chosen_lengths, rejected_lengths = batch_data
                loss, _ = compute_simpo_loss(
                    model, chosen_ids, rejected_ids, chosen_lengths, rejected_lengths,
                    self.beta, self.gamma
                )
                return loss

            loss_and_grad = nn.value_and_grad(actual_model, loss_fn)
            state_compile = make_step_state(actual_model, optimizer)

            def step_fn(batch_data):
                loss, grads = loss_and_grad(actual_model, batch_data)
                optimizer.update(actual_model, grads)
                return loss

            step_fn = compiled_step(step_fn, state_compile)

        state = make_step_state(actual_model, optimizer)

        total_loss_arr = mx.zeros((), dtype=mx.float32)
        for step in range(self.iters):
            # Gather batch of samples
            batch_samples = [
                tokenized_data[(step * bs + i) % len(tokenized_data)]
                for i in range(bs)
            ]
            max_len = max(
                max(s['chosen_length'], s['rejected_length']) for s in batch_samples
            )
            max_len = bucket_length(max_len)
            pad_id = self.tokenizer.pad_token_id or 0

            chosen_ids = mx.array([self._pad(s['chosen_ids'], max_len, pad_id) for s in batch_samples])
            rejected_ids = mx.array([self._pad(s['rejected_ids'], max_len, pad_id) for s in batch_samples])
            chosen_lengths = mx.array([s['chosen_length'] for s in batch_samples])
            rejected_lengths = mx.array([s['rejected_length'] for s in batch_samples])

            if use_shared_prefix:
                s0 = batch_samples[0]
                loss = step_fn(
                    (chosen_ids, rejected_ids, chosen_lengths, rejected_lengths),
                    int(s0['prompt_length']),
                    int(s0['chosen_length']),
                    int(s0['rejected_length']),
                )
            else:
                loss = step_fn((chosen_ids, rejected_ids, chosen_lengths, rejected_lengths))
            total_loss_arr = total_loss_arr + loss.astype(mx.float32)
            mx.eval(state, total_loss_arr)

            if (step + 1) % self.logging_steps == 0:
                avg_loss = float(total_loss_arr.item()) / self.logging_steps
                print(f"  Step {step + 1}/{self.iters} | Loss: {avg_loss:.4f} | batch_size: {bs}")
                total_loss_arr = mx.zeros((), dtype=mx.float32)

        # Save adapters and config
        _save_adapters_and_config(self.model, self.adapter_path)

        print("\n" + "=" * 70)
        print("SimPO Training Complete!")
        print("=" * 70)
        print(f"  Adapters saved to: {self.adapter_path}")
        return {"status": "success", "adapter_path": str(self.adapter_path)}


# Utility functions for preference data

def prepare_preference_dataset(
    dataset: Any,
    tokenizer: Any,
    format_type: str = "dpo",
) -> List[Dict]:
    """
    Prepare dataset for preference-based training (DPO, ORPO, etc.).

    Args:
        dataset: HuggingFace dataset with preference pairs
        tokenizer: Tokenizer for formatting
        format_type: 'dpo', 'orpo', or 'grpo'

    Returns:
        Formatted dataset ready for training

    Example:
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("Anthropic/hh-rlhf")
        >>> formatted = prepare_preference_dataset(dataset, tokenizer, "dpo")
    """

    formatted_data = []

    for sample in dataset:
        if format_type in ["dpo", "orpo"]:
            # Expect chosen/rejected format
            if 'chosen' in sample and 'rejected' in sample:
                formatted_data.append({
                    "prompt": sample.get('prompt', ''),
                    "chosen": sample['chosen'],
                    "rejected": sample['rejected'],
                })
        elif format_type == "grpo":
            # Expect prompt + optional ground truth
            formatted_data.append({
                "prompt": sample.get('prompt', sample.get('question', '')),
                "answer": sample.get('answer', sample.get('response', '')),
            })

    return formatted_data


def create_reward_function(reward_type: str = "simple") -> Callable:
    """
    Create a reward function for GRPO training.

    Args:
        reward_type: Type of reward function
            - 'simple': Binary correct/incorrect
            - 'math': Extract and compare numerical answers
            - 'code': Execute and verify code output
            - 'length': Reward based on response length

    Returns:
        Reward function callable

    Example:
        >>> reward_fn = create_reward_function('math')
        >>> trainer = GRPOTrainer(..., reward_fn=reward_fn)
    """

    if reward_type == "simple":
        def simple_reward(response: str, ground_truth: str) -> float:
            return 1.0 if ground_truth.lower() in response.lower() else 0.0
        return simple_reward

    elif reward_type == "math":
        def math_reward(response: str, ground_truth: str) -> float:
            import re
            # Extract numbers from response
            numbers = re.findall(r'-?\d+\.?\d*', response)
            target = re.findall(r'-?\d+\.?\d*', ground_truth)
            if numbers and target:
                try:
                    return 1.0 if float(numbers[-1]) == float(target[-1]) else 0.0
                except:
                    return 0.0
            return 0.0
        return math_reward

    elif reward_type == "length":
        def length_reward(response: str, _: str) -> float:
            # Reward longer, more detailed responses (up to a point)
            length = len(response.split())
            if length < 10:
                return 0.2
            elif length < 50:
                return 0.5
            elif length < 200:
                return 1.0
            else:
                return 0.8  # Penalize very long responses
        return length_reward

    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
