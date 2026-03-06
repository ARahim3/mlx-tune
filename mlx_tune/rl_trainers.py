"""
Reinforcement learning trainers for MLX-Tune.

Provides TRL-style trainer interfaces for:
- DPO
- ORPO
- GRPO
- KTO
- SimPO
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple
import json
import subprocess
import warnings

import mlx.core as mx

try:
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
    HAS_NATIVE_TRAINING = True
except ImportError:
    HAS_NATIVE_TRAINING = False

from mlx_tune.losses import (
    dpo_loss as compute_dpo_loss,
    grpo_recompute_loss,
    kto_loss as compute_kto_loss,
    orpo_loss as compute_orpo_loss,
    simpo_loss as compute_simpo_loss,
)
from mlx_tune._rl_runtime import (
    PolicyEvalBatch,
    PreferenceBatch,
    RolloutBatch,
    assemble_minibatches,
    collect_rollouts,
    compute_advantages,
    evaluate_rewards,
    make_policy_eval_batch,
    make_preference_batch,
    pad_sequences,
    score_policy_in_chunks,
)
from mlx_tune.model import ReferencePolicy
from mlx_tune.model import (
    RewardModel,
    ValueModel,
    build_reference_policy,
    build_reward_model,
    build_value_model,
)


STATE_FILE = "trainer_state.safetensors"
METADATA_FILE = "trainer_state.json"
REFERENCE_FILE = "reference_model.safetensors"
REFERENCE_METADATA_FILE = "reference_metadata.json"
MANIFEST_FILE = "manifest.json"
CHECKPOINT_FORMAT_NAME = "mlx_tune_rl_checkpoint"
CHECKPOINT_FORMAT_VERSION = 3
MLX_TUNE_VERSION = "0.4.0"
GRPO_PHASE1_LOSS_TYPES = {"grpo", "dr_grpo", "dapo", "bnpo"}


def _actual_model(model: Any) -> Any:
    return model.model if hasattr(model, "model") else model


def _pad_token_id(tokenizer: Any) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    return 0 if pad_id is None else pad_id


def _save_adapters_and_config(model: Any, adapter_path: Path) -> bool:
    """
    Save trainable parameters and adapter config in mlx_lm-compatible layout.
    """
    try:
        if hasattr(model, "save_adapter_snapshot"):
            return bool(model.save_adapter_snapshot(str(adapter_path)))

        actual_model = _actual_model(model)
        adapter_path.mkdir(parents=True, exist_ok=True)
        adapter_file = adapter_path / "adapters.safetensors"
        adapter_weights = dict(tree_flatten(actual_model.trainable_parameters()))
        mx.save_safetensors(str(adapter_file), adapter_weights)
        return True
    except Exception as exc:
        print(f"  Warning: could not save adapters: {exc}")
        return False


def _save_full_model_state(model: Any, path: Path) -> None:
    actual_model = _actual_model(model)
    mx.save_safetensors(str(path), dict(tree_flatten(actual_model.parameters())))


def _load_parameter_tree(model: Any, path: Path, strict: bool = False) -> None:
    if not path.exists():
        return
    actual_model = _actual_model(model)
    weights = mx.load(str(path))
    actual_model.update(tree_unflatten(list(weights.items())), strict=strict)
    mx.eval(actual_model.parameters())


def _flatten_prefixed_tree(prefix: str, tree: Dict[str, Any]) -> Dict[str, mx.array]:
    return {f"{prefix}.{key}": value for key, value in tree_flatten(tree)}


def _extract_prefixed_tree(prefix: str, flat_state: Dict[str, mx.array]) -> Dict[str, Any]:
    items = []
    prefix_with_dot = f"{prefix}."
    for key, value in flat_state.items():
        if key.startswith(prefix_with_dot):
            items.append((key[len(prefix_with_dot):], value))
    return tree_unflatten(items) if items else {}


def _rng_state_to_dict() -> Dict[str, mx.array]:
    return {f"rng.{idx}": state for idx, state in enumerate(mx.random.state)}


def _restore_rng_state(flat_state: Dict[str, mx.array]) -> None:
    rng_items = []
    idx = 0
    while f"rng.{idx}" in flat_state:
        rng_items.append(mx.array(flat_state[f"rng.{idx}"]))
        idx += 1
    if rng_items:
        mx.random.state = rng_items


def _pad_sequences(sequences: List[List[int]], pad_id: int) -> Tuple[mx.array, mx.array]:
    return pad_sequences(sequences, pad_id)


def _read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    with open(path) as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    history: List[Dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                history.append(json.loads(line))
    return history


def _save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class _RLTrainerBase:
    algorithm = "rl"

    def _init_native_state(self) -> None:
        self.global_step = 0
        self.dataset_cursor = 0
        self.reference_policy: Optional[ReferencePolicy] = None
        self.cache_metadata: Dict[str, Any] = {}
        self.optimizer = None
        self.reward_model: Optional[RewardModel] = getattr(self, "reward_model", None)
        self.value_model: Optional[ValueModel] = getattr(self, "value_model", None)
        self.metrics_history: List[Dict[str, Any]] = []
        self.loaded_checkpoint_manifest: Optional[Dict[str, Any]] = None

    def _apply_lora_if_needed(self) -> None:
        if hasattr(self.model, "_apply_lora") and not getattr(self.model, "_lora_applied", False):
            self.model._apply_lora()

    def _optimizer_for_training(self):
        lr_schedule = optim.cosine_decay(self.learning_rate, self.iters)
        return optim.AdamW(learning_rate=lr_schedule)

    def _next_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not samples:
            raise ValueError(f"{self.algorithm} training dataset is empty.")

        batch = []
        for _ in range(max(1, self.batch_size)):
            batch.append(samples[self.dataset_cursor])
            self.dataset_cursor = (self.dataset_cursor + 1) % len(samples)
        return batch

    def _checkpoint_dir(self, checkpoint_dir: Optional[Path] = None) -> Path:
        return checkpoint_dir or self.output_dir

    def _manifest_path(self, checkpoint_dir: Optional[Path] = None) -> Path:
        return self._checkpoint_dir(checkpoint_dir) / MANIFEST_FILE

    def _role_dir(self, role_name: str, checkpoint_dir: Optional[Path] = None) -> Path:
        return self._checkpoint_dir(checkpoint_dir) / role_name

    def _optimizer_state_path(self, checkpoint_dir: Optional[Path] = None) -> Path:
        return self._checkpoint_dir(checkpoint_dir) / "optimizer" / "state.safetensors"

    def _scheduler_state_path(self, checkpoint_dir: Optional[Path] = None) -> Path:
        return self._checkpoint_dir(checkpoint_dir) / "scheduler" / "state.json"

    def _trainer_state_path(self, checkpoint_dir: Optional[Path] = None) -> Path:
        return self._checkpoint_dir(checkpoint_dir) / "trainer" / "state.json"

    def _trainer_rng_path(self, checkpoint_dir: Optional[Path] = None) -> Path:
        return self._checkpoint_dir(checkpoint_dir) / "trainer" / "rng.safetensors"

    def _runtime_cache_path(self, checkpoint_dir: Optional[Path] = None) -> Path:
        return self._checkpoint_dir(checkpoint_dir) / "runtime" / "cache.safetensors"

    def _metrics_path(self, checkpoint_dir: Optional[Path] = None) -> Path:
        return self._checkpoint_dir(checkpoint_dir) / "metrics" / "history.jsonl"

    def _legacy_state_path(self, checkpoint_dir: Optional[Path] = None) -> Path:
        return self._checkpoint_dir(checkpoint_dir) / STATE_FILE

    def _legacy_metadata_path(self, checkpoint_dir: Optional[Path] = None) -> Path:
        return self._checkpoint_dir(checkpoint_dir) / METADATA_FILE

    def _legacy_reference_path(self, checkpoint_dir: Optional[Path] = None) -> Path:
        return self._checkpoint_dir(checkpoint_dir) / REFERENCE_FILE

    def _legacy_reference_metadata_path(self, checkpoint_dir: Optional[Path] = None) -> Path:
        return self._checkpoint_dir(checkpoint_dir) / REFERENCE_METADATA_FILE

    def _has_manifest_checkpoint(self, checkpoint_dir: Path) -> bool:
        return self._manifest_path(checkpoint_dir).exists()

    def _has_legacy_checkpoint(self, checkpoint_dir: Path) -> bool:
        return (
            (checkpoint_dir / "adapters" / "adapters.safetensors").exists()
            or self._legacy_state_path(checkpoint_dir).exists()
            or self._legacy_reference_path(checkpoint_dir).exists()
        )

    def _save_current_policy(self, checkpoint_dir: Optional[Path] = None) -> None:
        policy_dir = self._role_dir("policy", checkpoint_dir)
        _save_adapters_and_config(self.model, policy_dir)
        if hasattr(self.model, "set_adapter_path"):
            self.model.set_adapter_path(str(policy_dir))
        _write_json(
            policy_dir / "role.json",
            {
                "role": "policy",
                "weight_format": "adapters.safetensors",
            },
        )

    def _save_reference_policy(self, checkpoint_dir: Optional[Path] = None) -> None:
        if self.reference_policy is None:
            return
        reference_dir = self._role_dir("reference", checkpoint_dir)
        reference_dir.mkdir(parents=True, exist_ok=True)
        _save_full_model_state(self.reference_policy.model, reference_dir / "weights.safetensors")
        metadata = {
            "source": self.reference_policy.source,
            "metadata": self.reference_policy.metadata,
        }
        _write_json(reference_dir / "metadata.json", metadata)
        _write_json(
            reference_dir / "role.json",
            {
                "role": "reference",
                "weight_format": "weights.safetensors",
            },
        )

    def _save_optional_scalar_role(self, role_name: str, role_model: Optional[Any], checkpoint_dir: Optional[Path] = None) -> None:
        if role_model is None:
            return
        role_dir = self._role_dir(role_name, checkpoint_dir)
        role_model.save_pretrained(str(role_dir))

    def _build_training_metadata(self) -> Dict[str, Any]:
        config = self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config)
        return {
            "algorithm": self.algorithm,
            "config": config,
            "global_step": self.global_step,
            "dataset_cursor": self.dataset_cursor,
            "cache_metadata": self.cache_metadata,
        }

    def _build_scheduler_state(self, optimizer: Any) -> Dict[str, Any]:
        step_value = 0
        if optimizer is not None and getattr(optimizer, "state", None):
            step = optimizer.state.get("step", 0)
            step_value = int(step.item()) if hasattr(step, "item") else int(step)
        return {
            "name": "cosine_decay",
            "initial_learning_rate": self.learning_rate,
            "total_steps": self.iters,
            "step": step_value,
        }

    def _build_manifest(self) -> Dict[str, Any]:
        reward_base = getattr(self.reward_model, "base_model", None)
        value_base = getattr(self.value_model, "base_model", None)
        roles_present = ["policy"]
        role_weight_formats = {
            "policy": "adapters.safetensors",
        }
        reference_provenance = None
        if self.reference_policy is not None:
            roles_present.append("reference")
            role_weight_formats["reference"] = "weights.safetensors"
            reference_provenance = {
                "source": self.reference_policy.source,
                "metadata": self.reference_policy.metadata,
            }
        if self.reward_model is not None:
            roles_present.append("reward_model")
            role_weight_formats["reward_model"] = {
                "backbone": "weights.safetensors",
                "head": "head.safetensors",
                "adapters": "adapters.safetensors"
                if hasattr(reward_base, "has_adapters") and reward_base.has_adapters()
                else None,
            }
        if self.value_model is not None:
            roles_present.append("value_model")
            role_weight_formats["value_model"] = {
                "backbone": "weights.safetensors",
                "head": "head.safetensors",
                "adapters": "adapters.safetensors"
                if hasattr(value_base, "has_adapters") and value_base.has_adapters()
                else None,
            }
        return {
            "format_name": CHECKPOINT_FORMAT_NAME,
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "algorithm": self.algorithm,
            "roles_present": roles_present,
            "mlx_tune_version": MLX_TUNE_VERSION,
            "role_weight_formats": role_weight_formats,
            "trainer_state_locations": {
                "optimizer": "optimizer/state.safetensors",
                "scheduler": "scheduler/state.json",
                "trainer": "trainer/state.json",
                "rng": "trainer/rng.safetensors",
                "runtime_cache": "runtime/cache.safetensors",
            },
            "metrics_path": "metrics/history.jsonl",
            "reference_provenance": reference_provenance,
        }

    def _record_metric(self, **metrics: Any) -> None:
        if not metrics:
            return
        row = {"step": self.global_step}
        row.update(metrics)
        self.metrics_history.append(row)

    def save_state(
        self,
        optimizer: Any,
        extra_arrays: Optional[Dict[str, mx.array]] = None,
    ) -> None:
        checkpoint_dir = self._checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._save_current_policy(checkpoint_dir)
        self._save_reference_policy(checkpoint_dir)
        self._save_optional_scalar_role("reward_model", self.reward_model, checkpoint_dir)
        self._save_optional_scalar_role("value_model", self.value_model, checkpoint_dir)

        optimizer_path = self._optimizer_state_path(checkpoint_dir)
        optimizer_path.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(optimizer_path), _flatten_prefixed_tree("optimizer", optimizer.state))

        rng_path = self._trainer_rng_path(checkpoint_dir)
        rng_path.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(rng_path), _rng_state_to_dict())

        runtime_path = self._runtime_cache_path(checkpoint_dir)
        if extra_arrays:
            runtime_path.parent.mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(str(runtime_path), extra_arrays)

        _write_json(self._trainer_state_path(checkpoint_dir), self._build_training_metadata())
        _write_json(self._scheduler_state_path(checkpoint_dir), self._build_scheduler_state(optimizer))
        _save_jsonl(self._metrics_path(checkpoint_dir), self.metrics_history)
        _write_json(self._manifest_path(checkpoint_dir), self._build_manifest())

    def _ensure_reference_policy(self) -> None:
        if self.reference_policy is None:
            ref_model = getattr(self, "ref_model", None)
            self.reference_policy = build_reference_policy(self.model, ref_model=ref_model, snapshot=True)

    def _load_reference_policy(self, checkpoint_dir: Path) -> None:
        reference_path = self._role_dir("reference", checkpoint_dir) / "weights.safetensors"
        if reference_path.exists():
            self.reference_policy = build_reference_policy(self.model, snapshot=True)
            _load_parameter_tree(self.reference_policy.model, reference_path, strict=False)
            _actual_model(self.reference_policy.model).freeze()
            mx.eval(_actual_model(self.reference_policy.model).parameters())
            metadata = _read_json(self._role_dir("reference", checkpoint_dir) / "metadata.json")
            if metadata:
                self.reference_policy.source = metadata.get("source", self.reference_policy.source)
                self.reference_policy.metadata = metadata.get("metadata", self.reference_policy.metadata)
        else:
            self._ensure_reference_policy()

    def _load_optional_scalar_role(self, checkpoint_dir: Path, role_name: str) -> Optional[Any]:
        role_dir = self._role_dir(role_name, checkpoint_dir)
        if not role_dir.exists():
            return None
        with open(role_dir / "head_config.json") as handle:
            config = json.load(handle)
        if role_name == "reward_model":
            if self.reward_model is None:
                self.reward_model = build_reward_model(
                    self.model,
                    pooling=config.get("pooling", "last_token"),
                    target=config.get("target", "completion"),
                )
            self.reward_model.load_pretrained(str(role_dir))
            return self.reward_model
        if role_name == "value_model":
            if self.value_model is None:
                self.value_model = build_value_model(
                    self.model,
                    pooling=config.get("pooling", "last_token"),
                    target=config.get("target", "completion"),
                )
            self.value_model.load_pretrained(str(role_dir))
            return self.value_model
        return None

    def _load_manifest_state(self, optimizer: Any, checkpoint_dir: Path) -> Dict[str, mx.array]:
        manifest = _read_json(self._manifest_path(checkpoint_dir))
        if not manifest:
            raise FileNotFoundError(f"Checkpoint manifest not found under {checkpoint_dir}")
        self.loaded_checkpoint_manifest = manifest

        policy_adapter_file = self._role_dir("policy", checkpoint_dir) / "adapters.safetensors"
        if policy_adapter_file.exists():
            _load_parameter_tree(self.model, policy_adapter_file, strict=False)
            if hasattr(self.model, "set_adapter_path"):
                self.model.set_adapter_path(str(self._role_dir("policy", checkpoint_dir)))

        optimizer_path = self._optimizer_state_path(checkpoint_dir)
        if optimizer_path.exists():
            optimizer_state = _extract_prefixed_tree("optimizer", mx.load(str(optimizer_path)))
            if optimizer_state:
                optimizer.state = optimizer_state

        rng_path = self._trainer_rng_path(checkpoint_dir)
        if rng_path.exists():
            _restore_rng_state(mx.load(str(rng_path)))

        metadata = _read_json(self._trainer_state_path(checkpoint_dir))
        self.global_step = metadata.get("global_step", 0)
        self.dataset_cursor = metadata.get("dataset_cursor", 0)
        self.cache_metadata = metadata.get("cache_metadata", {})
        self.metrics_history = _load_jsonl(self._metrics_path(checkpoint_dir))

        self._load_reference_policy(checkpoint_dir)
        self._load_optional_scalar_role(checkpoint_dir, "reward_model")
        self._load_optional_scalar_role(checkpoint_dir, "value_model")

        runtime_path = self._runtime_cache_path(checkpoint_dir)
        return mx.load(str(runtime_path)) if runtime_path.exists() else {}

    def _load_legacy_state(self, optimizer: Any, checkpoint_dir: Path) -> Dict[str, mx.array]:
        adapter_file = checkpoint_dir / "adapters" / "adapters.safetensors"
        if adapter_file.exists():
            _load_parameter_tree(self.model, adapter_file, strict=False)
            if hasattr(self.model, "set_adapter_path"):
                self.model.set_adapter_path(str(checkpoint_dir / "adapters"))

        state_path = self._legacy_state_path(checkpoint_dir)
        metadata_path = self._legacy_metadata_path(checkpoint_dir)
        if not state_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Checkpoint state not found under {checkpoint_dir}")

        metadata = _read_json(metadata_path)
        flat_state = mx.load(str(state_path))
        optimizer_state = _extract_prefixed_tree("optimizer", flat_state)
        if optimizer_state:
            optimizer.state = optimizer_state
        _restore_rng_state(flat_state)

        self.global_step = metadata.get("global_step", 0)
        self.dataset_cursor = metadata.get("dataset_cursor", 0)
        self.cache_metadata = metadata.get("cache_metadata", {})
        self.metrics_history = []

        reference_path = self._legacy_reference_path(checkpoint_dir)
        if reference_path.exists():
            self.reference_policy = build_reference_policy(self.model, snapshot=True)
            _load_parameter_tree(self.reference_policy.model, reference_path, strict=False)
            reference_metadata = _read_json(self._legacy_reference_metadata_path(checkpoint_dir))
            if reference_metadata:
                self.reference_policy.source = reference_metadata.get("source", self.reference_policy.source)
                self.reference_policy.metadata = reference_metadata.get("metadata", self.reference_policy.metadata)
        else:
            self._ensure_reference_policy()
        return flat_state

    def load_state(self, optimizer: Any, checkpoint_dir: Path) -> Dict[str, mx.array]:
        checkpoint_dir = Path(checkpoint_dir)
        if self._has_manifest_checkpoint(checkpoint_dir):
            return self._load_manifest_state(optimizer, checkpoint_dir)
        if self._has_legacy_checkpoint(checkpoint_dir):
            return self._load_legacy_state(optimizer, checkpoint_dir)
        raise FileNotFoundError(f"Checkpoint state not found under {checkpoint_dir}")


class DPOConfig:
    def __init__(
        self,
        beta: float = 0.1,
        loss_type: str = "sigmoid",
        label_smoothing: float = 0.0,
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
        **kwargs,
    ):
        self.beta = beta
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
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
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}


class ORPOConfig:
    def __init__(
        self,
        beta: float = 0.1,
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
        **kwargs,
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
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}


class GRPOConfig:
    def __init__(
        self,
        loss_type: str = "grpo",
        beta: float = 0.04,
        num_generations: int = 4,
        temperature: float = 0.7,
        max_completion_length: int = 512,
        reward_fn: Optional[Callable] = None,
        reward_model: Optional[Any] = None,
        value_model: Optional[Any] = None,
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
        **kwargs,
    ):
        self.loss_type = loss_type
        self.beta = beta
        self.num_generations = num_generations
        self.temperature = temperature
        self.max_completion_length = max_completion_length
        self.reward_fn = reward_fn
        self.reward_model = reward_model
        self.value_model = value_model
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
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_") and key not in {"reward_fn", "reward_model", "value_model"}
        }


class DPOTrainer(_RLTrainerBase):
    algorithm = "dpo"

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        ref_model: Optional[Any] = None,
        reward_model: Optional[Any] = None,
        value_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        args: Optional[DPOConfig] = None,
        use_native: bool = True,
        **kwargs,
    ):
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.value_model = value_model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.use_native = use_native and HAS_NATIVE_TRAINING
        self.config = args or DPOConfig()
        self.beta = self.config.beta
        self.loss_type = self.config.loss_type
        self.label_smoothing = self.config.label_smoothing
        self.output_dir = Path(self.config.output_dir)
        self.learning_rate = self.config.learning_rate
        self.batch_size = self.config.per_device_train_batch_size
        self.max_steps = self.config.max_steps
        self.max_seq_length = self.config.max_seq_length
        self.max_prompt_length = self.config.max_prompt_length
        self.gradient_accumulation_steps = self.config.gradient_accumulation_steps
        self.warmup_steps = self.config.warmup_steps
        self.logging_steps = self.config.logging_steps
        self.save_steps = self.config.save_steps
        dataset_size = len(train_dataset) if hasattr(train_dataset, "__len__") else 100
        self.iters = self.max_steps if self.max_steps > 0 else max(
            1, (dataset_size // max(1, self.batch_size)) * self.config.num_train_epochs
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_path = self.output_dir / "policy"
        self.adapter_path.mkdir(parents=True, exist_ok=True)
        self.train_samples: List[Dict[str, Any]] = []
        self._init_native_state()

    def _tokenize_preference_pair(self, sample: Dict[str, Any], sample_index: int) -> Dict[str, Any]:
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")

        chosen_ids = self.tokenizer.encode(prompt + chosen)[: self.max_seq_length]
        rejected_ids = self.tokenizer.encode(prompt + rejected)[: self.max_seq_length]
        return {
            "sample_index": sample_index,
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,
            "chosen_length": len(chosen_ids),
            "rejected_length": len(rejected_ids),
        }

    def _prepare_training_samples(self) -> None:
        self.train_samples = []
        for sample_index, sample in enumerate(self.train_dataset):
            if {"prompt", "chosen", "rejected"} <= set(sample.keys()):
                self.train_samples.append(self._tokenize_preference_pair(sample, sample_index))
        if not self.train_samples:
            raise ValueError("DPOTrainer requires prompt/chosen/rejected samples.")

    def _precompute_reference_cache(self) -> None:
        self._ensure_reference_policy()
        preference_batch = make_preference_batch(
            chosen_sequences=[sample["chosen_ids"] for sample in self.train_samples],
            rejected_sequences=[sample["rejected_ids"] for sample in self.train_samples],
            pad_id=_pad_token_id(self.tokenizer),
            sample_indices=[sample["sample_index"] for sample in self.train_samples],
        )
        ref_chosen = score_policy_in_chunks(
            _actual_model(self.reference_policy.model),
            preference_batch.chosen,
            batch_size=max(1, self.batch_size),
            mode="sequence",
        ).summed_logprobs
        ref_rejected = score_policy_in_chunks(
            _actual_model(self.reference_policy.model),
            preference_batch.rejected,
            batch_size=max(1, self.batch_size),
            mode="sequence",
        ).summed_logprobs
        ref_chosen = mx.stop_gradient(ref_chosen)
        ref_rejected = mx.stop_gradient(ref_rejected)
        for idx, sample in enumerate(self.train_samples):
            sample["reference_chosen_logprobs"] = ref_chosen[idx]
            sample["reference_rejected_logprobs"] = ref_rejected[idx]
        self.cache_metadata = {
            "type": "inline_preference_reference_logprobs",
            "num_samples": len(self.train_samples),
        }

    def _restore_reference_cache(self, flat_state: Dict[str, mx.array]) -> None:
        if "dpo.reference_chosen_logprobs" not in flat_state:
            self._precompute_reference_cache()
            return
        ref_chosen = flat_state["dpo.reference_chosen_logprobs"]
        ref_rejected = flat_state["dpo.reference_rejected_logprobs"]
        if ref_chosen.shape[0] != len(self.train_samples):
            raise ValueError("Saved DPO cache does not match current dataset ordering.")
        for idx, sample in enumerate(self.train_samples):
            sample["reference_chosen_logprobs"] = ref_chosen[idx]
            sample["reference_rejected_logprobs"] = ref_rejected[idx]

    def _build_batch(self, samples: List[Dict[str, Any]]) -> PreferenceBatch:
        return make_preference_batch(
            chosen_sequences=[sample["chosen_ids"] for sample in samples],
            rejected_sequences=[sample["rejected_ids"] for sample in samples],
            pad_id=_pad_token_id(self.tokenizer),
            sample_indices=[sample["sample_index"] for sample in samples],
            chosen_reference_logprobs=mx.array(
                [sample["reference_chosen_logprobs"] for sample in samples]
            ),
            rejected_reference_logprobs=mx.array(
                [sample["reference_rejected_logprobs"] for sample in samples]
            ),
        )

    def _extra_state_arrays(self) -> Dict[str, mx.array]:
        return {
            "dpo.reference_chosen_logprobs": mx.array(
                [sample["reference_chosen_logprobs"] for sample in self.train_samples]
            ),
            "dpo.reference_rejected_logprobs": mx.array(
                [sample["reference_rejected_logprobs"] for sample in self.train_samples]
            ),
        }

    def train(self, resume_from_checkpoint: Optional[str] = None):
        if self.use_native:
            return self._train_native(resume_from_checkpoint=resume_from_checkpoint)
        return self._train_subprocess()

    def _train_native(self, resume_from_checkpoint: Optional[str] = None):
        self._apply_lora_if_needed()
        self._prepare_training_samples()

        actual_model = _actual_model(self.model)
        optimizer = self._optimizer_for_training()
        self.optimizer = optimizer

        if resume_from_checkpoint is not None:
            flat_state = self.load_state(optimizer, Path(resume_from_checkpoint))
            self._restore_reference_cache(flat_state)
        else:
            self._precompute_reference_cache()

        def loss_fn(model, batch):
            loss, _ = compute_dpo_loss(
                model=model,
                chosen_ids=batch.chosen.input_ids,
                rejected_ids=batch.rejected.input_ids,
                chosen_lengths=batch.chosen.sequence_lengths,
                rejected_lengths=batch.rejected.sequence_lengths,
                beta=self.beta,
                reference_chosen_logprobs=batch.chosen.reference_logprobs,
                reference_rejected_logprobs=batch.rejected.reference_logprobs,
                label_smoothing=self.label_smoothing,
            )
            return loss

        value_and_grad = nn.value_and_grad(actual_model, loss_fn)
        running_loss = 0.0
        last_loss = None

        while self.global_step < self.iters:
            batch_samples = self._next_samples(self.train_samples)
            batch = self._build_batch(batch_samples)
            loss, grads = value_and_grad(actual_model, batch)
            optimizer.update(actual_model, grads)
            mx.eval(actual_model.parameters(), optimizer.state)

            last_loss = loss.item()
            running_loss += last_loss
            self.global_step += 1
            self._record_metric(loss=last_loss)

            if self.global_step % self.logging_steps == 0:
                print(
                    f"DPO step {self.global_step}/{self.iters} | "
                    f"loss={running_loss / self.logging_steps:.4f}"
                )
                running_loss = 0.0

            if self.global_step % self.save_steps == 0:
                self.save_state(optimizer, self._extra_state_arrays())

        self.save_state(optimizer, self._extra_state_arrays())
        return {
            "status": "success",
            "adapter_path": str(self.adapter_path),
            "global_step": self.global_step,
            "final_loss": last_loss,
        }

    def _train_subprocess(self):
        warnings.warn(
            "Native DPO training not available. Using SFT-on-chosen approximation.",
            UserWarning,
        )
        train_file = self.output_dir / "train.jsonl"
        valid_file = self.output_dir / "valid.jsonl"
        with open(train_file, "w") as handle:
            for sample in self.train_dataset:
                if "prompt" in sample and "chosen" in sample:
                    messages = [
                        {"role": "user", "content": sample["prompt"]},
                        {"role": "assistant", "content": sample["chosen"]},
                    ]
                    handle.write(json.dumps({"messages": messages}) + "\n")
        valid_file.write_text(train_file.read_text())
        cmd = [
            "mlx_lm.lora",
            "--model",
            getattr(self.model, "model_name", "model"),
            "--train",
            "--data",
            str(self.output_dir),
            "--iters",
            str(self.iters),
            "--learning-rate",
            str(self.learning_rate),
            "--batch-size",
            str(self.batch_size),
            "--adapter-path",
            str(self.adapter_path),
        ]
        subprocess.run(cmd, check=True)
        return {"status": "success", "adapter_path": str(self.adapter_path)}


class ORPOTrainer:
    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        tokenizer: Optional[Any] = None,
        args: Optional[ORPOConfig] = None,
        use_native: bool = True,
        **kwargs,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.use_native = use_native and HAS_NATIVE_TRAINING
        self.config = args or ORPOConfig()
        self.beta = self.config.beta
        self.output_dir = Path(self.config.output_dir)
        self.learning_rate = self.config.learning_rate
        self.batch_size = self.config.per_device_train_batch_size
        self.max_steps = self.config.max_steps
        self.max_seq_length = self.config.max_seq_length
        self.logging_steps = self.config.logging_steps
        self.save_steps = self.config.save_steps
        dataset_size = len(train_dataset) if hasattr(train_dataset, "__len__") else 100
        self.iters = self.max_steps if self.max_steps > 0 else max(
            1, (dataset_size // max(1, self.batch_size)) * self.config.num_train_epochs
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_path = self.output_dir / "policy"
        self.adapter_path.mkdir(parents=True, exist_ok=True)

    def _tokenize_preference_pair(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")
        chosen_ids = self.tokenizer.encode(prompt + chosen)[: self.max_seq_length]
        rejected_ids = self.tokenizer.encode(prompt + rejected)[: self.max_seq_length]
        return {
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,
            "chosen_length": len(chosen_ids),
            "rejected_length": len(rejected_ids),
        }

    def train(self):
        if self.use_native:
            return self._train_native()
        return self._train_subprocess()

    def _train_native(self):
        if hasattr(self.model, "_apply_lora") and not getattr(self.model, "_lora_applied", False):
            self.model._apply_lora()

        tokenized_data = [
            self._tokenize_preference_pair(sample)
            for sample in self.train_dataset
            if {"prompt", "chosen", "rejected"} <= set(sample.keys())
        ]
        actual_model = _actual_model(self.model)
        optimizer = optim.AdamW(learning_rate=optim.cosine_decay(self.learning_rate, self.iters))

        def loss_fn(model, batch):
            chosen_ids, rejected_ids, chosen_lengths, rejected_lengths = batch
            loss, _ = compute_orpo_loss(
                model,
                chosen_ids,
                rejected_ids,
                chosen_lengths,
                rejected_lengths,
                self.beta,
            )
            return loss

        value_and_grad = nn.value_and_grad(actual_model, loss_fn)
        last_loss = None
        pad_id = _pad_token_id(self.tokenizer)

        for step in range(self.iters):
            samples = tokenized_data[step % len(tokenized_data): step % len(tokenized_data) + self.batch_size]
            if len(samples) < self.batch_size:
                samples += tokenized_data[: self.batch_size - len(samples)]
            chosen_ids, chosen_lengths = _pad_sequences([sample["chosen_ids"] for sample in samples], pad_id)
            rejected_ids, rejected_lengths = _pad_sequences([sample["rejected_ids"] for sample in samples], pad_id)
            loss, grads = value_and_grad(actual_model, (chosen_ids, rejected_ids, chosen_lengths, rejected_lengths))
            optimizer.update(actual_model, grads)
            mx.eval(actual_model.parameters(), optimizer.state)
            last_loss = loss.item()

        _save_adapters_and_config(self.model, self.adapter_path)
        return {"status": "success", "adapter_path": str(self.adapter_path), "final_loss": last_loss}

    def _train_subprocess(self):
        warnings.warn("Using SFT approximation for ORPO.", UserWarning)
        train_file = self.output_dir / "train.jsonl"
        valid_file = self.output_dir / "valid.jsonl"
        with open(train_file, "w") as handle:
            for sample in self.train_dataset:
                if "prompt" in sample and "chosen" in sample:
                    messages = [
                        {"role": "user", "content": sample["prompt"]},
                        {"role": "assistant", "content": sample["chosen"]},
                    ]
                    handle.write(json.dumps({"messages": messages}) + "\n")
        valid_file.write_text(train_file.read_text())
        cmd = [
            "mlx_lm.lora",
            "--model",
            getattr(self.model, "model_name", "model"),
            "--train",
            "--data",
            str(self.output_dir),
            "--iters",
            str(self.iters),
            "--learning-rate",
            str(self.learning_rate),
            "--batch-size",
            str(self.batch_size),
            "--adapter-path",
            str(self.adapter_path),
        ]
        subprocess.run(cmd, check=True)
        return {"status": "success", "adapter_path": str(self.adapter_path)}


class GRPOTrainer(_RLTrainerBase):
    algorithm = "grpo"

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        tokenizer: Optional[Any] = None,
        reward_fn: Optional[Callable] = None,
        reward_model: Optional[Any] = None,
        ref_model: Optional[Any] = None,
        value_model: Optional[Any] = None,
        args: Optional[GRPOConfig] = None,
        use_native: bool = True,
        **kwargs,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.ref_model = ref_model
        self.reward_model = reward_model or getattr(args, "reward_model", None)
        self.value_model = value_model or getattr(args, "value_model", None)
        self.use_native = use_native and HAS_NATIVE_TRAINING
        self.config = args or GRPOConfig()
        self.loss_type = self.config.loss_type
        self.phase1_loss_type = self._resolve_loss_type(self.loss_type)
        self.beta = self.config.beta
        self.num_generations = self.config.num_generations
        self.max_completion_length = self.config.max_completion_length
        self.reward_fn = reward_fn if reward_fn is not None else self.config.reward_fn
        self.output_dir = Path(self.config.output_dir)
        self.learning_rate = self.config.learning_rate
        self.batch_size = self.config.per_device_train_batch_size
        self.max_steps = self.config.max_steps
        self.temperature = self.config.temperature
        self.logging_steps = self.config.logging_steps
        self.save_steps = self.config.save_steps
        self.max_seq_length = self.config.max_seq_length
        dataset_size = len(train_dataset) if hasattr(train_dataset, "__len__") else 100
        self.iters = self.max_steps if self.max_steps > 0 else max(
            1, (dataset_size // max(1, self.batch_size)) * self.config.num_train_epochs
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_path = self.output_dir / "policy"
        self.adapter_path.mkdir(parents=True, exist_ok=True)
        self.prompt_samples: List[Dict[str, Any]] = []
        self._last_rollout_batch: Optional[RolloutBatch] = None
        self._init_native_state()

        if self.reward_model is None and self.reward_fn is None:
            self.reward_fn = lambda response, context: len(response.split()) / 100.0

    def _resolve_loss_type(self, loss_type: str) -> str:
        if loss_type not in GRPO_PHASE1_LOSS_TYPES:
            raise ValueError(
                f"Unsupported GRPO loss_type '{loss_type}'. "
                f"Supported values: {sorted(GRPO_PHASE1_LOSS_TYPES)}"
            )
        if loss_type != "grpo":
            warnings.warn(
                f"GRPO loss_type='{loss_type}' is accepted in Phase 1 but currently "
                "routes through the shared rollout/recompute objective.",
                UserWarning,
            )
        return "phase1_shared_rollout_recompute"

    def _prepare_prompt_samples(self) -> None:
        self.prompt_samples = []
        for sample_index, sample in enumerate(self.train_dataset):
            prompt = sample.get("prompt", sample.get("question", ""))
            if not prompt:
                continue
            reward_context = sample.get("answer", sample.get("response", prompt))
            prompt_ids = self.tokenizer.encode(prompt)
            self.prompt_samples.append(
                {
                    "sample_index": sample_index,
                    "prompt": prompt,
                    "prompt_ids": prompt_ids,
                    "reward_context": reward_context,
                }
            )
        if not self.prompt_samples:
            raise ValueError("GRPOTrainer requires prompt or question fields.")

    def _resolve_reward_evaluator(self) -> Any:
        if self.reward_model is not None:
            return self.reward_model
        if self.reward_fn is not None:
            return self.reward_fn
        config_reward_fn = getattr(self.config, "reward_fn", None)
        if config_reward_fn is not None:
            return config_reward_fn
        return lambda response, context: len(response.split()) / 100.0

    def _collect_rollout_batch(self, prompt_samples: List[Dict[str, Any]]) -> RolloutBatch:
        reward_evaluator = self._resolve_reward_evaluator()
        rollout_batch = collect_rollouts(
            _actual_model(self.model),
            self.tokenizer,
            prompt_samples=prompt_samples,
            sampling_config={
                "num_generations": self.num_generations,
                "temperature": self.temperature,
                "max_completion_length": self.max_completion_length,
                "max_seq_length": self.max_seq_length,
            },
            reward_evaluator=reward_evaluator,
            collect_sample_stats=False,
        )
        reward_batch = evaluate_rewards(rollout_batch, reward_evaluator)
        advantages = compute_advantages(reward_batch)
        rollout_batch.rewards = reward_batch.scalar_rewards
        rollout_batch.advantages = advantages
        rollout_batch.policy_eval.advantages = advantages
        return rollout_batch

    def train(self, resume_from_checkpoint: Optional[str] = None):
        if self.use_native:
            return self._train_native(resume_from_checkpoint=resume_from_checkpoint)
        return self._train_subprocess()

    def _train_native(self, resume_from_checkpoint: Optional[str] = None):
        self._apply_lora_if_needed()
        self._prepare_prompt_samples()

        actual_model = _actual_model(self.model)
        optimizer = self._optimizer_for_training()
        self.optimizer = optimizer

        if resume_from_checkpoint is not None:
            self.load_state(optimizer, Path(resume_from_checkpoint))
        else:
            self._ensure_reference_policy()

        reference_model = _actual_model(self.reference_policy.model)

        def loss_fn(model, batch):
            if self.phase1_loss_type != "phase1_shared_rollout_recompute":
                raise ValueError(f"Unhandled GRPO Phase 1 loss routing: {self.phase1_loss_type}")
            loss, _ = grpo_recompute_loss(
                model=model,
                reference_model=reference_model,
                input_ids=batch.input_ids,
                prompt_lengths=batch.prompt_lengths,
                completion_lengths=batch.completion_lengths,
                rollout_logprobs=batch.rollout_logprobs,
                advantages=batch.advantages,
                beta=self.beta,
                temperature=self.temperature,
            )
            return loss

        value_and_grad = nn.value_and_grad(actual_model, loss_fn)
        running_loss = 0.0
        last_loss = None

        while self.global_step < self.iters:
            prompt_samples = self._next_samples(self.prompt_samples)
            rollout_batch = self._collect_rollout_batch(prompt_samples)
            self._last_rollout_batch = rollout_batch
            minibatches = assemble_minibatches(
                rollout_batch.policy_eval,
                minibatch_size=max(1, self.batch_size * self.num_generations),
                shuffle=False,
            )

            step_loss = 0.0
            for minibatch in minibatches:
                loss, grads = value_and_grad(actual_model, minibatch)
                optimizer.update(actual_model, grads)
                mx.eval(actual_model.parameters(), optimizer.state)
                step_loss += loss.item()

            last_loss = step_loss / max(1, len(minibatches))
            running_loss += last_loss
            self.global_step += 1
            mean_reward = float(mx.mean(rollout_batch.rewards).item()) if rollout_batch.rewards is not None else None
            self._record_metric(loss=last_loss, reward_mean=mean_reward)

            if self.global_step % self.logging_steps == 0:
                print(
                    f"GRPO step {self.global_step}/{self.iters} | "
                    f"loss={running_loss / self.logging_steps:.4f}"
                )
                running_loss = 0.0

            if self.global_step % self.save_steps == 0:
                self.save_state(optimizer)

        self.save_state(optimizer)
        return {
            "status": "success",
            "adapter_path": str(self.adapter_path),
            "global_step": self.global_step,
            "final_loss": last_loss,
        }

    def _train_subprocess(self):
        warnings.warn(
            "Native GRPO training not available. Using SFT approximation.",
            UserWarning,
        )
        train_file = self.output_dir / "train.jsonl"
        valid_file = self.output_dir / "valid.jsonl"
        with open(train_file, "w") as handle:
            for sample in self.train_dataset:
                prompt = sample.get("prompt", sample.get("question", ""))
                if not prompt:
                    continue
                messages = [{"role": "user", "content": prompt}]
                if "response" in sample or "answer" in sample:
                    response = sample.get("response", sample.get("answer", ""))
                    messages.append({"role": "assistant", "content": response})
                handle.write(json.dumps({"messages": messages}) + "\n")
        valid_file.write_text(train_file.read_text())
        cmd = [
            "mlx_lm.lora",
            "--model",
            getattr(self.model, "model_name", "model"),
            "--train",
            "--data",
            str(self.output_dir),
            "--iters",
            str(self.iters),
            "--adapter-path",
            str(self.adapter_path),
        ]
        subprocess.run(cmd, check=True)
        return {"status": "success", "adapter_path": str(self.adapter_path)}


class KTOTrainer(_RLTrainerBase):
    algorithm = "kto"

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        tokenizer: Optional[Any] = None,
        beta: float = 0.1,
        ref_model: Optional[Any] = None,
        reward_model: Optional[Any] = None,
        value_model: Optional[Any] = None,
        use_native: bool = True,
        **kwargs,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.beta = beta
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.value_model = value_model
        self.use_native = use_native and HAS_NATIVE_TRAINING
        self.output_dir = Path(kwargs.get("output_dir", "./kto_outputs"))
        self.learning_rate = kwargs.get("learning_rate", 5e-7)
        self.iters = kwargs.get("max_steps", 100)
        self.max_seq_length = kwargs.get("max_seq_length", 2048)
        self.batch_size = kwargs.get("per_device_train_batch_size", 1)
        self.logging_steps = kwargs.get("logging_steps", 10)
        self.save_steps = kwargs.get("save_steps", 100)
        self.config = {
            "beta": beta,
            "output_dir": str(self.output_dir),
            "learning_rate": self.learning_rate,
            "max_steps": self.iters,
            "max_seq_length": self.max_seq_length,
            "per_device_train_batch_size": self.batch_size,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_path = self.output_dir / "adapters"
        self.adapter_path.mkdir(parents=True, exist_ok=True)
        self.train_samples: List[Dict[str, Any]] = []
        self._init_native_state()

    def _prepare_training_samples(self) -> None:
        self.train_samples = []
        for sample_index, sample in enumerate(self.train_dataset):
            if "text" not in sample or "label" not in sample:
                continue
            ids = self.tokenizer.encode(sample["text"])[: self.max_seq_length]
            self.train_samples.append(
                {
                    "sample_index": sample_index,
                    "ids": ids,
                    "length": len(ids),
                    "label": float(sample["label"]),
                }
            )
        if not self.train_samples:
            raise ValueError("KTOTrainer requires text/label samples.")

    def _precompute_reference_cache(self) -> None:
        self._ensure_reference_policy()
        eval_batch = make_policy_eval_batch(
            [sample["ids"] for sample in self.train_samples],
            pad_id=_pad_token_id(self.tokenizer),
            mode="sequence",
            labels=mx.array([sample["label"] for sample in self.train_samples]),
            sample_indices=mx.array([sample["sample_index"] for sample in self.train_samples]),
        )
        reference_logprobs = score_policy_in_chunks(
            _actual_model(self.reference_policy.model),
            eval_batch,
            batch_size=max(1, self.batch_size),
            mode="sequence",
        ).summed_logprobs
        reference_logprobs = mx.stop_gradient(reference_logprobs)
        for idx, sample in enumerate(self.train_samples):
            sample["reference_logprobs"] = reference_logprobs[idx]
        self.cache_metadata = {
            "type": "inline_kto_reference_logprobs",
            "num_samples": len(self.train_samples),
        }

    def _restore_reference_cache(self, flat_state: Dict[str, mx.array]) -> None:
        if "kto.reference_logprobs" not in flat_state:
            self._precompute_reference_cache()
            return
        reference_logprobs = flat_state["kto.reference_logprobs"]
        if reference_logprobs.shape[0] != len(self.train_samples):
            raise ValueError("Saved KTO cache does not match current dataset ordering.")
        for idx, sample in enumerate(self.train_samples):
            sample["reference_logprobs"] = reference_logprobs[idx]

    def _build_batch(self, samples: List[Dict[str, Any]]) -> PolicyEvalBatch:
        return make_policy_eval_batch(
            [sample["ids"] for sample in samples],
            pad_id=_pad_token_id(self.tokenizer),
            mode="sequence",
            labels=mx.array([sample["label"] for sample in samples]),
            reference_logprobs=mx.array([sample["reference_logprobs"] for sample in samples]),
            sample_indices=mx.array([sample["sample_index"] for sample in samples]),
        )

    def _extra_state_arrays(self) -> Dict[str, mx.array]:
        return {
            "kto.reference_logprobs": mx.array(
                [sample["reference_logprobs"] for sample in self.train_samples]
            )
        }

    def train(self, resume_from_checkpoint: Optional[str] = None):
        if self.use_native:
            return self._train_native(resume_from_checkpoint=resume_from_checkpoint)
        warnings.warn("KTO requires native training. Using SFT approximation.", UserWarning)
        return {"status": "fallback"}

    def _train_native(self, resume_from_checkpoint: Optional[str] = None):
        self._apply_lora_if_needed()
        self._prepare_training_samples()

        actual_model = _actual_model(self.model)
        optimizer = self._optimizer_for_training()
        self.optimizer = optimizer

        if resume_from_checkpoint is not None:
            flat_state = self.load_state(optimizer, Path(resume_from_checkpoint))
            self._restore_reference_cache(flat_state)
        else:
            self._precompute_reference_cache()

        def loss_fn(model, batch):
            loss, _ = compute_kto_loss(
                model=model,
                input_ids=batch.input_ids,
                lengths=batch.sequence_lengths,
                labels=batch.labels,
                beta=self.beta,
                reference_logprobs=batch.reference_logprobs,
            )
            return loss

        value_and_grad = nn.value_and_grad(actual_model, loss_fn)
        running_loss = 0.0
        last_loss = None

        while self.global_step < self.iters:
            batch_samples = self._next_samples(self.train_samples)
            batch = self._build_batch(batch_samples)
            loss, grads = value_and_grad(actual_model, batch)
            optimizer.update(actual_model, grads)
            mx.eval(actual_model.parameters(), optimizer.state)

            last_loss = loss.item()
            running_loss += last_loss
            self.global_step += 1
            self._record_metric(loss=last_loss)

            if self.global_step % self.logging_steps == 0:
                print(
                    f"KTO step {self.global_step}/{self.iters} | "
                    f"loss={running_loss / self.logging_steps:.4f}"
                )
                running_loss = 0.0

            if self.global_step % self.save_steps == 0:
                self.save_state(optimizer, self._extra_state_arrays())

        self.save_state(optimizer, self._extra_state_arrays())
        return {
            "status": "success",
            "adapter_path": str(self.adapter_path),
            "global_step": self.global_step,
            "final_loss": last_loss,
        }


class SimPOTrainer:
    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        tokenizer: Optional[Any] = None,
        gamma: float = 0.5,
        beta: float = 2.0,
        use_native: bool = True,
        **kwargs,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.gamma = gamma
        self.beta = beta
        self.use_native = use_native and HAS_NATIVE_TRAINING
        self.output_dir = Path(kwargs.get("output_dir", "./simpo_outputs"))
        self.learning_rate = kwargs.get("learning_rate", 5e-7)
        self.iters = kwargs.get("max_steps", 100)
        self.max_seq_length = kwargs.get("max_seq_length", 2048)
        self.logging_steps = kwargs.get("logging_steps", 10)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_path = self.output_dir / "adapters"
        self.adapter_path.mkdir(parents=True, exist_ok=True)

    def _tokenize_pair(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")
        chosen_ids = self.tokenizer.encode(prompt + chosen)[: self.max_seq_length]
        rejected_ids = self.tokenizer.encode(prompt + rejected)[: self.max_seq_length]
        return {
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,
            "chosen_length": len(chosen_ids),
            "rejected_length": len(rejected_ids),
        }

    def train(self):
        if not self.use_native:
            warnings.warn("SimPO requires native training. Using SFT approximation.", UserWarning)
            return {"status": "fallback"}

        if hasattr(self.model, "_apply_lora") and not getattr(self.model, "_lora_applied", False):
            self.model._apply_lora()

        tokenized_data = [
            self._tokenize_pair(sample)
            for sample in self.train_dataset
            if {"prompt", "chosen", "rejected"} <= set(sample.keys())
        ]
        actual_model = _actual_model(self.model)
        optimizer = optim.AdamW(learning_rate=optim.cosine_decay(self.learning_rate, self.iters))
        pad_id = _pad_token_id(self.tokenizer)

        def loss_fn(model, batch):
            chosen_ids, rejected_ids, chosen_lengths, rejected_lengths = batch
            loss, _ = compute_simpo_loss(
                model,
                chosen_ids,
                rejected_ids,
                chosen_lengths,
                rejected_lengths,
                self.beta,
                self.gamma,
            )
            return loss

        value_and_grad = nn.value_and_grad(actual_model, loss_fn)
        last_loss = None

        for step in range(self.iters):
            samples = tokenized_data[step % len(tokenized_data): step % len(tokenized_data) + 1]
            chosen_ids, chosen_lengths = _pad_sequences([sample["chosen_ids"] for sample in samples], pad_id)
            rejected_ids, rejected_lengths = _pad_sequences([sample["rejected_ids"] for sample in samples], pad_id)
            loss, grads = value_and_grad(actual_model, (chosen_ids, rejected_ids, chosen_lengths, rejected_lengths))
            optimizer.update(actual_model, grads)
            mx.eval(actual_model.parameters(), optimizer.state)
            last_loss = loss.item()

        _save_adapters_and_config(self.model, self.adapter_path)
        return {"status": "success", "adapter_path": str(self.adapter_path), "final_loss": last_loss}


def prepare_preference_dataset(
    dataset: Any,
    tokenizer: Any,
    format_type: str = "dpo",
) -> List[Dict[str, Any]]:
    formatted_data = []
    for sample in dataset:
        if format_type in ["dpo", "orpo"]:
            if "chosen" in sample and "rejected" in sample:
                formatted_data.append(
                    {
                        "prompt": sample.get("prompt", ""),
                        "chosen": sample["chosen"],
                        "rejected": sample["rejected"],
                    }
                )
        elif format_type == "grpo":
            formatted_data.append(
                {
                    "prompt": sample.get("prompt", sample.get("question", "")),
                    "answer": sample.get("answer", sample.get("response", "")),
                }
            )
    return formatted_data


def create_reward_function(reward_type: str = "simple") -> Callable:
    if reward_type == "simple":
        def simple_reward(response: str, ground_truth: str) -> float:
            return 1.0 if ground_truth.lower() in response.lower() else 0.0

        return simple_reward

    if reward_type == "math":
        def math_reward(response: str, ground_truth: str) -> float:
            import re

            numbers = re.findall(r"-?\d+\.?\d*", response)
            target = re.findall(r"-?\d+\.?\d*", ground_truth)
            if numbers and target:
                try:
                    return 1.0 if float(numbers[-1]) == float(target[-1]) else 0.0
                except Exception:
                    return 0.0
            return 0.0

        return math_reward

    if reward_type == "length":
        def length_reward(response: str, _: str) -> float:
            length = len(response.split())
            if length < 10:
                return 0.2
            if length < 50:
                return 0.5
            if length < 200:
                return 1.0
            return 0.8

        return length_reward

    raise ValueError(f"Unknown reward type: {reward_type}")
