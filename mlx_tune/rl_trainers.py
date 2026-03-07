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
from contextlib import contextmanager
import hashlib
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
    ppo_sequence_loss,
    reward_model_pairwise_loss,
    reward_model_regression_loss,
    scalar_loss_metrics,
    pairwise_ranking_accuracy,
    simpo_loss as compute_simpo_loss,
    value_model_regression_loss,
)
from mlx_tune._rl_runtime import (
    PolicyEvalBatch,
    PreferenceBatch,
    RolloutBatch,
    assemble_minibatches,
    cap_prompt_and_completion_lengths,
    collect_rollouts,
    compute_advantages,
    compute_returns_and_advantages,
    evaluate_rewards,
    kl_against_reference,
    make_policy_eval_batch,
    make_preference_batch,
    pad_sequences,
    predict_rollout_values,
    rank_grouped_rollouts,
    score_rollout_references,
    score_policy_in_chunks,
    summarize_rollout_metrics,
)
from mlx_tune.model import ReferencePolicy
from mlx_tune.model import (
    RewardModel,
    ValueModel,
    build_reference_policy,
    build_reward_model,
    build_value_model,
)
from mlx_tune.rl_api import (
    RLCheckpointBundle,
    create_reward_function as public_create_reward_function,
    prepare_preference_dataset as public_prepare_preference_dataset,
    prepare_reward_dataset as public_prepare_reward_dataset,
    prepare_rl_dataset,
    resume_from_checkpoint,
)


STATE_FILE = "trainer_state.safetensors"
METADATA_FILE = "trainer_state.json"
REFERENCE_FILE = "reference_model.safetensors"
REFERENCE_METADATA_FILE = "reference_metadata.json"
MANIFEST_FILE = "manifest.json"
CHECKPOINT_FORMAT_NAME = "mlx_tune_rl_checkpoint"
CHECKPOINT_FORMAT_VERSION = 4
MLX_TUNE_VERSION = "0.4.0"
GRPO_LOSS_TYPES = {"grpo", "dr_grpo", "dapo", "bnpo", "gspo"}


def _actual_model(model: Any) -> Any:
    return model.model if hasattr(model, "model") else model


def _pad_token_id(tokenizer: Any) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    return 0 if pad_id is None else pad_id


def _encode_text(tokenizer: Any, text: str, add_special_tokens: bool = True) -> List[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))
    except TypeError:
        return list(tokenizer.encode(text))


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


def _load_flat_parameter_tree(model: Any, flat_state: Dict[str, Any], strict: bool = False) -> None:
    if not flat_state:
        return
    actual_model = _actual_model(model)
    actual_model.update(tree_unflatten(list(flat_state.items())), strict=strict)
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


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class _ScalarRoleTrainTarget(nn.Module):
    def __init__(self, backbone: Any, head: Any):
        super().__init__()
        self.backbone = backbone
        self.head = head


class _RLTrainerBase:
    algorithm = "rl"
    requires_reference_policy = False

    def _init_native_state(self) -> None:
        self.global_step = 0
        self.dataset_cursor = 0
        self.reference_policy: Optional[ReferencePolicy] = None
        self.cache_metadata: Dict[str, Any] = {}
        self.runtime_cache_arrays: Dict[str, mx.array] = {}
        self.optimizer = None
        self.optimizers: Dict[str, Any] = {}
        self.reward_model: Optional[RewardModel] = getattr(self, "reward_model", None)
        self.value_model: Optional[ValueModel] = getattr(self, "value_model", None)
        self.metrics_history: List[Dict[str, Any]] = []
        self.loaded_checkpoint_manifest: Optional[Dict[str, Any]] = None
        self.seed = getattr(self.config, "seed", getattr(self, "seed", 0))
        self._seed_initialized = False

    def _apply_lora_if_needed(self) -> None:
        if hasattr(self.model, "_apply_lora") and not getattr(self.model, "_lora_applied", False):
            self.model._apply_lora()

    def _optimizer_for_training(self, learning_rate: Optional[float] = None):
        lr_schedule = optim.cosine_decay(self.learning_rate if learning_rate is None else learning_rate, self.iters)
        return optim.AdamW(learning_rate=lr_schedule)

    def _primary_role_name(self) -> str:
        return "policy"

    def _primary_role_weight_format(self) -> Any:
        return "adapters.safetensors"

    def _primary_optimizer_name(self) -> str:
        return self._primary_role_name()

    def _trainer_cursor_state(self) -> Dict[str, int]:
        return {"dataset": int(self.dataset_cursor)}

    def _restore_trainer_cursors(self, cursors: Dict[str, Any]) -> None:
        self.dataset_cursor = int(cursors.get("dataset", cursors.get("dataset_cursor", self.dataset_cursor)))

    def _sampling_config_payload(self) -> Dict[str, Any]:
        return {}

    def _sampling_config_fingerprint(self) -> Optional[str]:
        payload = self._sampling_config_payload()
        if not payload:
            return None
        return _hash_payload(payload)

    def _validate_resume_sampling_fingerprint(self, trainer_state: Dict[str, Any]) -> None:
        current_fingerprint = self._sampling_config_fingerprint()
        saved_fingerprint = (
            trainer_state.get("trainer_state", {}).get("sampling_config_fingerprint")
            or trainer_state.get("sampling_config_fingerprint")
        )
        if current_fingerprint and saved_fingerprint and current_fingerprint != saved_fingerprint:
            raise ValueError(
                "Checkpoint sampling config fingerprint does not match the current trainer configuration."
            )

    def _seed_training_run(self) -> None:
        if self._seed_initialized:
            return
        mx.random.seed(int(self.seed))
        self._seed_initialized = True

    @contextmanager
    def _preserve_rng_state(self):
        saved_state = [mx.array(state) for state in mx.random.state]
        try:
            yield
        finally:
            mx.random.state = [mx.array(state) for state in saved_state]

    def _normalize_metric_value(self, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return float(value)
        return value

    def _record_metrics(
        self,
        namespace: str,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        normalized = {
            (key if "/" in key else f"{namespace}/{key}"): self._normalize_metric_value(value)
            for key, value in metrics.items()
            if value is not None
        }
        if not normalized:
            return {}
        row = {"step": self.global_step if step is None else int(step)}
        row.update(normalized)
        self.metrics_history.append(row)
        if hasattr(self, "output_dir"):
            _save_jsonl(self._metrics_path(), self.metrics_history)
        return row

    def _record_metric(self, **metrics: Any) -> None:
        self._record_metrics("train", metrics)

    def _format_metric_summary(
        self,
        row: Dict[str, Any],
        namespace: str = "train",
        keys: Optional[List[str]] = None,
    ) -> str:
        preferred = keys or [
            "policy_loss",
            "loss",
            "value_loss",
            "reward_loss",
            "reward_mean",
            "completion_length_mean",
            "completion_length_max",
            "eos_rate",
            "truncation_rate",
            "kl_to_reference_mean",
            "preference_win_rate",
        ]
        parts = [f"step={row['step']}"]
        for key in preferred:
            full_key = f"{namespace}/{key}"
            if full_key not in row:
                continue
            value = row[full_key]
            if isinstance(value, bool):
                parts.append(f"{key}={value}")
            elif isinstance(value, (int, float)):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")
        return " | ".join(parts)

    def _gather_runtime_cache_arrays(
        self,
        extra_arrays: Optional[Dict[str, mx.array]] = None,
    ) -> Dict[str, mx.array]:
        merged: Dict[str, mx.array] = {}
        merged.update(getattr(self, "runtime_cache_arrays", {}))
        if hasattr(self, "_extra_state_arrays"):
            merged.update(getattr(self, "_extra_state_arrays")())
        if extra_arrays:
            merged.update(extra_arrays)
        return merged

    def _save_primary_role(self, checkpoint_dir: Optional[Path] = None) -> None:
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

    def _load_primary_role(self, checkpoint_dir: Path) -> None:
        policy_adapter_file = self._role_dir("policy", checkpoint_dir) / "adapters.safetensors"
        if policy_adapter_file.exists():
            _load_parameter_tree(self.model, policy_adapter_file, strict=False)
            if hasattr(self.model, "set_adapter_path"):
                self.model.set_adapter_path(str(self._role_dir("policy", checkpoint_dir)))

    def _next_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not samples:
            raise ValueError(f"{self.algorithm} training dataset is empty.")

        batch = []
        for _ in range(max(1, self.batch_size)):
            batch.append(samples[self.dataset_cursor])
            self.dataset_cursor = (self.dataset_cursor + 1) % len(samples)
        return batch

    def _next_rollout_samples(self, samples: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        if not samples:
            raise ValueError(f"{self.algorithm} training dataset is empty.")

        batch = []
        for _ in range(max(1, count)):
            batch.append(samples[self.dataset_cursor])
            self.dataset_cursor = (self.dataset_cursor + 1) % len(samples)
        return batch

    def _observed_rollout_kl(self, rollout_batch: Optional[RolloutBatch]) -> Optional[float]:
        if rollout_batch is None:
            return None
        if rollout_batch.reference_logprobs is None or rollout_batch.rollout_logprobs is None:
            return None
        kl_values = kl_against_reference(
            rollout_batch.rollout_logprobs.astype(mx.float32),
            rollout_batch.reference_logprobs.astype(mx.float32),
        )
        return float(mx.mean(kl_values).item())

    def _effective_kl_beta(self, rollout_batch: Optional[RolloutBatch] = None) -> float:
        base_beta = float(getattr(self, "beta", 0.0))
        if getattr(self, "kl_penalty_mode", "kl") == "none":
            return 0.0
        kl_target = getattr(self, "kl_target", None)
        observed_kl = self._observed_rollout_kl(rollout_batch)
        if kl_target is None or observed_kl is None:
            return base_beta
        target = max(float(kl_target), 1e-8)
        scale = min(max(observed_kl / target, 0.0), 10.0)
        return base_beta * scale

    def _checkpoint_dir(self, checkpoint_dir: Optional[Path] = None) -> Path:
        return checkpoint_dir or self.output_dir

    def _manifest_path(self, checkpoint_dir: Optional[Path] = None) -> Path:
        return self._checkpoint_dir(checkpoint_dir) / MANIFEST_FILE

    def _role_dir(self, role_name: str, checkpoint_dir: Optional[Path] = None) -> Path:
        return self._checkpoint_dir(checkpoint_dir) / role_name

    def _optimizer_state_path(
        self,
        checkpoint_dir: Optional[Path] = None,
        optimizer_name: Optional[str] = None,
    ) -> Path:
        if optimizer_name is None:
            return self._checkpoint_dir(checkpoint_dir) / "optimizer" / "state.safetensors"
        return self._checkpoint_dir(checkpoint_dir) / "optimizers" / optimizer_name / "state.safetensors"

    def _scheduler_state_path(
        self,
        checkpoint_dir: Optional[Path] = None,
        optimizer_name: Optional[str] = None,
    ) -> Path:
        if optimizer_name is None:
            return self._checkpoint_dir(checkpoint_dir) / "scheduler" / "state.json"
        return self._checkpoint_dir(checkpoint_dir) / "schedulers" / optimizer_name / "state.json"

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

    def _save_reference_policy(self, checkpoint_dir: Optional[Path] = None) -> None:
        if self.reference_policy is None:
            return
        reference_dir = self._role_dir("reference", checkpoint_dir)
        reference_dir.mkdir(parents=True, exist_ok=True)
        _save_full_model_state(self.reference_policy.model, reference_dir / "weights.safetensors")
        _write_json(
            reference_dir / "metadata.json",
            {
                "source": self.reference_policy.source,
                "metadata": self.reference_policy.metadata,
            },
        )
        _write_json(
            reference_dir / "role.json",
            {
                "role": "reference",
                "weight_format": "weights.safetensors",
            },
        )

    def _save_optional_scalar_role(
        self,
        role_name: str,
        role_model: Optional[Any],
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        if role_model is None or role_name == self._primary_role_name():
            return
        role_model.save_pretrained(str(self._role_dir(role_name, checkpoint_dir)))

    def _build_training_metadata(self) -> Dict[str, Any]:
        config = self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config)
        trainer_state = {
            "cursors": self._trainer_cursor_state(),
            "sampling_config_fingerprint": self._sampling_config_fingerprint(),
            "step_boundary": {
                "completed_optimizer_step": int(self.global_step),
                "checkpoint_authoritative": True,
            },
        }
        runtime_state = {
            "cache_metadata": self.cache_metadata,
            "runtime_cache_keys": sorted(self.runtime_cache_arrays.keys()),
            "rng_state_path": "trainer/rng.safetensors",
        }
        return {
            "algorithm": self.algorithm,
            "config": config,
            "global_step": self.global_step,
            "dataset_cursor": self.dataset_cursor,
            "cache_metadata": self.cache_metadata,
            "seed": int(self.seed),
            "sampling_config_fingerprint": trainer_state["sampling_config_fingerprint"],
            "trainer_state": trainer_state,
            "runtime_state": runtime_state,
        }

    def _build_scheduler_state(self, optimizer: Any, learning_rate: Optional[float] = None) -> Dict[str, Any]:
        step_value = 0
        if optimizer is not None and getattr(optimizer, "state", None):
            step = optimizer.state.get("step", 0)
            step_value = int(step.item()) if hasattr(step, "item") else int(step)
        return {
            "name": "cosine_decay",
            "initial_learning_rate": self.learning_rate if learning_rate is None else learning_rate,
            "total_steps": self.iters,
            "step": step_value,
        }

    def _normalize_optimizers(
        self,
        optimizer: Any = None,
        optimizers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if optimizers:
            return optimizers
        if optimizer is not None:
            return {self._primary_optimizer_name(): optimizer}
        return getattr(self, "optimizers", {})

    def _optimizer_learning_rates(self) -> Dict[str, float]:
        return {self._primary_optimizer_name(): self.learning_rate}

    def _build_manifest(self, optimizers: Dict[str, Any]) -> Dict[str, Any]:
        reward_base = getattr(self.reward_model, "base_model", None)
        value_base = getattr(self.value_model, "base_model", None)
        primary_role = self._primary_role_name()
        roles_present = [primary_role]
        role_weight_formats = {
            primary_role: self._primary_role_weight_format(),
        }
        reference_provenance = None
        if self.reference_policy is not None:
            roles_present.append("reference")
            role_weight_formats["reference"] = "weights.safetensors"
            reference_provenance = {
                "source": self.reference_policy.source,
                "metadata": self.reference_policy.metadata,
            }
        if self.reward_model is not None and primary_role != "reward_model":
            roles_present.append("reward_model")
            role_weight_formats["reward_model"] = {
                "backbone": "weights.safetensors",
                "head": "head.safetensors",
                "adapters": "adapters.safetensors"
                if hasattr(reward_base, "has_adapters") and reward_base.has_adapters()
                else None,
            }
        if self.value_model is not None and primary_role != "value_model":
            roles_present.append("value_model")
            role_weight_formats["value_model"] = {
                "backbone": "weights.safetensors",
                "head": "head.safetensors",
                "adapters": "adapters.safetensors"
                if hasattr(value_base, "has_adapters") and value_base.has_adapters()
                else None,
            }

        trainer_state_locations = {
            "optimizers": {
                name: f"optimizers/{name}/state.safetensors"
                for name in optimizers
            },
            "schedulers": {
                name: f"schedulers/{name}/state.json"
                for name in optimizers
            },
            "trainer": "trainer/state.json",
            "rng": "trainer/rng.safetensors",
            "runtime_cache": "runtime/cache.safetensors",
        }
        if len(optimizers) == 1:
            trainer_state_locations["optimizer"] = "optimizer/state.safetensors"
            trainer_state_locations["scheduler"] = "scheduler/state.json"

        return {
            "format_name": CHECKPOINT_FORMAT_NAME,
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "algorithm": self.algorithm,
            "roles_present": roles_present,
            "mlx_tune_version": MLX_TUNE_VERSION,
            "role_weight_formats": role_weight_formats,
            "trainer_state_locations": trainer_state_locations,
            "metrics_path": "metrics/history.jsonl",
            "reference_provenance": reference_provenance,
        }

    def save_state(
        self,
        optimizer: Any = None,
        extra_arrays: Optional[Dict[str, mx.array]] = None,
        optimizers: Optional[Dict[str, Any]] = None,
    ) -> None:
        checkpoint_dir = self._checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        optimizer_map = self._normalize_optimizers(optimizer=optimizer, optimizers=optimizers)
        if not optimizer_map:
            raise ValueError("No optimizer state provided for checkpoint save.")
        self.optimizers = optimizer_map
        if len(optimizer_map) == 1:
            self.optimizer = next(iter(optimizer_map.values()))

        self._save_primary_role(checkpoint_dir)
        self._save_reference_policy(checkpoint_dir)
        self._save_optional_scalar_role("reward_model", self.reward_model, checkpoint_dir)
        self._save_optional_scalar_role("value_model", self.value_model, checkpoint_dir)

        learning_rates = self._optimizer_learning_rates()
        for name, current_optimizer in optimizer_map.items():
            optimizer_path = self._optimizer_state_path(checkpoint_dir, name)
            optimizer_path.parent.mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(str(optimizer_path), _flatten_prefixed_tree("optimizer", current_optimizer.state))
            if len(optimizer_map) == 1:
                legacy_path = self._optimizer_state_path(checkpoint_dir)
                legacy_path.parent.mkdir(parents=True, exist_ok=True)
                mx.save_safetensors(str(legacy_path), _flatten_prefixed_tree("optimizer", current_optimizer.state))
            scheduler_state = self._build_scheduler_state(
                current_optimizer,
                learning_rate=learning_rates.get(name, self.learning_rate),
            )
            _write_json(self._scheduler_state_path(checkpoint_dir, name), scheduler_state)
            if len(optimizer_map) == 1:
                _write_json(self._scheduler_state_path(checkpoint_dir), scheduler_state)

        rng_path = self._trainer_rng_path(checkpoint_dir)
        rng_path.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(rng_path), _rng_state_to_dict())

        runtime_arrays = self._gather_runtime_cache_arrays(extra_arrays)
        self.runtime_cache_arrays = dict(runtime_arrays)
        runtime_path = self._runtime_cache_path(checkpoint_dir)
        if runtime_arrays:
            runtime_path.parent.mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(str(runtime_path), runtime_arrays)

        _write_json(self._trainer_state_path(checkpoint_dir), self._build_training_metadata())
        _save_jsonl(self._metrics_path(checkpoint_dir), self.metrics_history)
        _write_json(self._manifest_path(checkpoint_dir), self._build_manifest(optimizer_map))

    def _ensure_reference_policy(self) -> None:
        if not self.requires_reference_policy:
            return
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
        if not role_dir.exists() or role_name == self._primary_role_name():
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

    def _apply_scalar_role_state(self, role_name: str, role_state: Any) -> Optional[Any]:
        if role_name == "reward_model":
            if self.reward_model is None:
                self.reward_model = build_reward_model(
                    self.model,
                    pooling=role_state.head_config.get("pooling", "last_token"),
                    target=role_state.head_config.get("target", "completion"),
                )
            role_model = self.reward_model
        elif role_name == "value_model":
            if self.value_model is None:
                self.value_model = build_value_model(
                    self.model,
                    pooling=role_state.head_config.get("pooling", "last_token"),
                    target=role_state.head_config.get("target", "completion"),
                )
            role_model = self.value_model
        else:
            return None

        _load_flat_parameter_tree(role_model.base_model, role_state.parameter_state, strict=False)
        if role_state.adapter_state:
            _load_flat_parameter_tree(role_model.base_model, role_state.adapter_state, strict=False)
        if role_state.head_state:
            role_model.head.update(tree_unflatten(list(role_state.head_state.items())), strict=False)
            mx.eval(role_model.head.parameters())
        if role_state.head_config:
            role_model.head_config.update(role_state.head_config)
            role_model.pooling = role_model.head_config.get("pooling", role_model.pooling)
            role_model.target = role_model.head_config.get("target", role_model.target)
        return role_model

    def _apply_checkpoint_bundle(
        self,
        bundle: RLCheckpointBundle,
        optimizer: Any = None,
        optimizers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, mx.array]:
        self.loaded_checkpoint_manifest = bundle.manifest
        optimizer_map = self._normalize_optimizers(optimizer=optimizer, optimizers=optimizers)
        if optimizer_map:
            for name, current_optimizer in optimizer_map.items():
                state_tree = bundle.optimizer_state_trees.get(name)
                if state_tree is None and len(bundle.optimizer_state_trees) == 1:
                    state_tree = next(iter(bundle.optimizer_state_trees.values()))
                if state_tree:
                    current_optimizer.state = state_tree
            self.optimizers = optimizer_map
            if len(optimizer_map) == 1:
                self.optimizer = next(iter(optimizer_map.values()))

        if bundle.rng_state:
            _restore_rng_state(bundle.rng_state)
            self._seed_initialized = True

        trainer_state = bundle.trainer_state
        self._validate_resume_sampling_fingerprint(trainer_state)
        self.global_step = trainer_state.get("global_step", 0)
        self.dataset_cursor = trainer_state.get("dataset_cursor", 0)
        self.cache_metadata = trainer_state.get("cache_metadata", {})
        self.seed = int(trainer_state.get("seed", self.seed))
        self._restore_trainer_cursors(
            trainer_state.get("trainer_state", {}).get("cursors", {})
        )
        self.metrics_history = list(bundle.metrics_history)
        self.runtime_cache_arrays = dict(bundle.runtime_cache)

        for role_name, role_state in bundle.restored_roles.items():
            if role_name == "policy":
                _load_flat_parameter_tree(self.model, role_state.parameter_state, strict=False)
            elif role_name == "reference":
                self.reference_policy = build_reference_policy(self.model, snapshot=True)
                _load_flat_parameter_tree(self.reference_policy.model, role_state.parameter_state, strict=False)
                reference_actual = _actual_model(self.reference_policy.model)
                if hasattr(reference_actual, "freeze"):
                    reference_actual.freeze()
                    mx.eval(reference_actual.parameters())
                if role_state.metadata:
                    self.reference_policy.source = role_state.metadata.get("source", self.reference_policy.source)
                    self.reference_policy.metadata = role_state.metadata.get("metadata", self.reference_policy.metadata)
            elif role_name in {"reward_model", "value_model"}:
                self._apply_scalar_role_state(role_name, role_state)

        if self.requires_reference_policy and self.reference_policy is None:
            self._ensure_reference_policy()
        return bundle.runtime_cache

    def _restore_optimizer_states(
        self,
        checkpoint_dir: Path,
        optimizer_map: Dict[str, Any],
        manifest: Optional[Dict[str, Any]] = None,
    ) -> None:
        trainer_locations = {} if manifest is None else manifest.get("trainer_state_locations", {})
        optimizer_locations = trainer_locations.get("optimizers") or {}
        for name, current_optimizer in optimizer_map.items():
            optimizer_path = self._optimizer_state_path(checkpoint_dir, name)
            if name in optimizer_locations:
                optimizer_path = checkpoint_dir / optimizer_locations[name]
            elif not optimizer_path.exists() and len(optimizer_map) == 1:
                optimizer_path = self._optimizer_state_path(checkpoint_dir)
            if not optimizer_path.exists():
                continue
            optimizer_state = _extract_prefixed_tree("optimizer", mx.load(str(optimizer_path)))
            if optimizer_state:
                current_optimizer.state = optimizer_state

    def _load_manifest_state(
        self,
        optimizer: Any = None,
        checkpoint_dir: Optional[Path] = None,
        optimizers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, mx.array]:
        checkpoint_dir = Path(checkpoint_dir)
        manifest = _read_json(self._manifest_path(checkpoint_dir))
        if not manifest:
            raise FileNotFoundError(f"Checkpoint manifest not found under {checkpoint_dir}")
        self.loaded_checkpoint_manifest = manifest

        self._load_primary_role(checkpoint_dir)

        optimizer_map = self._normalize_optimizers(optimizer=optimizer, optimizers=optimizers)
        self._restore_optimizer_states(checkpoint_dir, optimizer_map, manifest=manifest)
        self.optimizers = optimizer_map
        if len(optimizer_map) == 1:
            self.optimizer = next(iter(optimizer_map.values()))

        rng_path = self._trainer_rng_path(checkpoint_dir)
        if rng_path.exists():
            _restore_rng_state(mx.load(str(rng_path)))
            self._seed_initialized = True

        metadata = _read_json(self._trainer_state_path(checkpoint_dir))
        self._validate_resume_sampling_fingerprint(metadata)
        self.global_step = metadata.get("global_step", 0)
        self.dataset_cursor = metadata.get("dataset_cursor", 0)
        self.cache_metadata = metadata.get("cache_metadata", {})
        self.seed = int(metadata.get("seed", self.seed))
        self._restore_trainer_cursors(metadata.get("trainer_state", {}).get("cursors", {}))
        self.metrics_history = _load_jsonl(self._metrics_path(checkpoint_dir))

        self._load_reference_policy(checkpoint_dir)
        self._load_optional_scalar_role(checkpoint_dir, "reward_model")
        self._load_optional_scalar_role(checkpoint_dir, "value_model")

        runtime_path = self._runtime_cache_path(checkpoint_dir)
        self.runtime_cache_arrays = dict(mx.load(str(runtime_path))) if runtime_path.exists() else {}
        return self.runtime_cache_arrays

    def _load_legacy_state(
        self,
        optimizer: Any = None,
        checkpoint_dir: Optional[Path] = None,
        optimizers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, mx.array]:
        checkpoint_dir = Path(checkpoint_dir)
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
        optimizer_map = self._normalize_optimizers(optimizer=optimizer, optimizers=optimizers)
        if optimizer_map:
            first_optimizer = next(iter(optimizer_map.values()))
            optimizer_state = _extract_prefixed_tree("optimizer", flat_state)
            if optimizer_state:
                first_optimizer.state = optimizer_state
            self.optimizers = optimizer_map
            if len(optimizer_map) == 1:
                self.optimizer = first_optimizer
        _restore_rng_state(flat_state)
        self._seed_initialized = True

        self.global_step = metadata.get("global_step", 0)
        self.dataset_cursor = metadata.get("dataset_cursor", 0)
        self.cache_metadata = metadata.get("cache_metadata", {})
        self.seed = int(metadata.get("seed", self.seed))
        self.metrics_history = []
        self.runtime_cache_arrays = {
            key: value
            for key, value in flat_state.items()
            if not key.startswith("optimizer.") and not key.startswith("rng.")
        }

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

    def load_state(
        self,
        optimizer: Any = None,
        checkpoint_dir: Optional[Path] = None,
        optimizers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, mx.array]:
        bundle = resume_from_checkpoint(Path(checkpoint_dir))
        return self._apply_checkpoint_bundle(bundle, optimizer=optimizer, optimizers=optimizers)


def prepare_reward_dataset(dataset: Any) -> List[Dict[str, Any]]:
    return public_prepare_reward_dataset(dataset)


def _tokenize_reward_scalar_sample(
    tokenizer: Any,
    sample: Dict[str, Any],
    max_seq_length: int,
) -> Dict[str, Any]:
    prompt = sample.get("prompt", "")
    response = sample.get("response", "")
    sequence_ids = tokenizer.encode(prompt + response)[:max_seq_length]
    prompt_ids = tokenizer.encode(prompt) if prompt else []
    prompt_length = min(len(prompt_ids), len(sequence_ids))
    completion_length = max(len(sequence_ids) - prompt_length, 0)
    return {
        "ids": sequence_ids,
        "length": len(sequence_ids),
        "prompt_length": prompt_length,
        "completion_length": completion_length,
        "score": float(sample.get("score", 0.0)),
    }


def _tokenize_reward_pairwise_sample(
    tokenizer: Any,
    sample: Dict[str, Any],
    max_seq_length: int,
) -> Dict[str, Any]:
    prompt = sample.get("prompt", "")
    prompt_ids = tokenizer.encode(prompt) if prompt else []
    chosen_ids = tokenizer.encode(prompt + sample["chosen"])[:max_seq_length]
    rejected_ids = tokenizer.encode(prompt + sample["rejected"])[:max_seq_length]
    chosen_prompt_length = min(len(prompt_ids), len(chosen_ids))
    rejected_prompt_length = min(len(prompt_ids), len(rejected_ids))
    return {
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
        "chosen_length": len(chosen_ids),
        "rejected_length": len(rejected_ids),
        "chosen_prompt_length": chosen_prompt_length,
        "rejected_prompt_length": rejected_prompt_length,
        "chosen_completion_length": max(len(chosen_ids) - chosen_prompt_length, 0),
        "rejected_completion_length": max(len(rejected_ids) - rejected_prompt_length, 0),
    }


def score_reward_model(
    reward_model: RewardModel,
    samples: Any,
    batch_size: int = 8,
    tokenizer: Optional[Any] = None,
    max_seq_length: int = 2048,
) -> List[float]:
    scoring_tokenizer = tokenizer or getattr(reward_model, "tokenizer", None)
    if scoring_tokenizer is None:
        raise ValueError("score_reward_model requires a tokenizer on the reward model or as an argument.")

    normalized = list(prepare_rl_dataset(samples, mode="reward_scalar", tokenizer=scoring_tokenizer))
    if any(sample.get("type") != "scalar" for sample in normalized):
        raise ValueError("score_reward_model only supports scalar reward samples.")

    tokenized = [
        _tokenize_reward_scalar_sample(scoring_tokenizer, sample, max_seq_length)
        for sample in normalized
    ]
    scores: List[float] = []
    pad_id = _pad_token_id(scoring_tokenizer)
    for start in range(0, len(tokenized), max(1, batch_size)):
        chunk = tokenized[start:start + max(1, batch_size)]
        input_ids, lengths = _pad_sequences([sample["ids"] for sample in chunk], pad_id)
        chunk_scores = reward_model.score(
            input_ids,
            sequence_lengths=lengths,
            prompt_lengths=mx.array([sample["prompt_length"] for sample in chunk]),
            completion_lengths=mx.array([sample["completion_length"] for sample in chunk]),
        )
        scores.extend(float(value.item()) for value in chunk_scores)
    return scores


class RLConfigBase:
    _NON_SERIALIZED_FIELDS: set[str] = set()

    @staticmethod
    def _pop_alias(kwargs: Dict[str, Any], *names: str, default: Any = None) -> Any:
        found = [name for name in names if name in kwargs]
        if not found:
            return default
        value = kwargs.pop(found[0])
        for name in found[1:]:
            kwargs.pop(name)
        return value

    @staticmethod
    def _normalize_reward_sources(
        reward_sources: Optional[Any],
        reward_fn: Optional[Any],
        reward_model: Optional[Any],
    ) -> List[Any]:
        resolved: List[Any] = []
        if reward_sources is not None:
            if isinstance(reward_sources, (list, tuple)):
                resolved.extend(list(reward_sources))
            else:
                resolved.append(reward_sources)
        if reward_model is not None:
            resolved.append({"name": "reward_model", "source": reward_model})
        if reward_fn is not None:
            resolved.append({"name": "reward_fn", "source": reward_fn})
        return resolved

    def _set_remaining(self, kwargs: Dict[str, Any]) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._validate()

    def _validate_choice(self, field: str, value: Any, choices: set[str]) -> None:
        if value not in choices:
            raise ValueError(f"Unsupported {field} '{value}'. Supported values: {sorted(choices)}")

    def _validate(self) -> None:
        reward_source = getattr(self, "reward_source", None)
        if reward_source is not None:
            self._validate_choice("reward_source", reward_source, {"auto", "online", "offline", "hybrid"})
        advantage_estimator = getattr(self, "advantage_estimator", None)
        if advantage_estimator is not None:
            self._validate_choice("advantage_estimator", advantage_estimator, {"group_zscore", "rloo", "gae"})
        kl_penalty_mode = getattr(self, "kl_penalty_mode", None)
        if kl_penalty_mode is not None:
            self._validate_choice("kl_penalty_mode", kl_penalty_mode, {"kl", "none"})

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if key.startswith("_") or key in self._NON_SERIALIZED_FIELDS:
                continue
            if callable(value):
                continue
            if hasattr(value, "parameters") or hasattr(value, "score") or hasattr(value, "predict"):
                continue
            payload[key] = value
        return payload


class RewardConfig(RLConfigBase):
    def __init__(
        self,
        output_dir: str = "./reward_outputs",
        learning_rate: float = 5e-6,
        per_device_train_batch_size: int = 2,
        num_train_epochs: int = 1,
        max_steps: int = -1,
        logging_steps: int = 10,
        save_steps: int = 100,
        max_seq_length: int = 2048,
        pairwise_margin: float = 0.0,
        regression_loss_type: str = "mse",
        dataset_mode: Optional[str] = None,
        chat_template: Optional[Any] = None,
        auto_detect_dataset: bool = True,
        **kwargs,
    ):
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_seq_length = max_seq_length
        self.pairwise_margin = pairwise_margin
        self.regression_loss_type = regression_loss_type
        self.dataset_mode = dataset_mode
        self.chat_template = chat_template
        self.auto_detect_dataset = auto_detect_dataset
        self._set_remaining(kwargs)


class DPOConfig(RLConfigBase):
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
        dataset_mode: Optional[str] = None,
        chat_template: Optional[Any] = None,
        auto_detect_dataset: bool = True,
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
        self.dataset_mode = dataset_mode
        self.chat_template = chat_template
        self.auto_detect_dataset = auto_detect_dataset
        self._set_remaining(kwargs)


class ORPOConfig(RLConfigBase):
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
        dataset_mode: Optional[str] = None,
        chat_template: Optional[Any] = None,
        auto_detect_dataset: bool = True,
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
        self.dataset_mode = dataset_mode
        self.chat_template = chat_template
        self.auto_detect_dataset = auto_detect_dataset
        self._set_remaining(kwargs)


class GRPOConfig(RLConfigBase):
    _NON_SERIALIZED_FIELDS = {"reward_fn", "reward_model", "value_model"}

    def __init__(
        self,
        loss_type: str = "grpo",
        advantage_mode: str = "group_zscore",
        beta: float = 0.04,
        num_generations: int = 4,
        temperature: float = 0.7,
        max_completion_length: int = 512,
        reward_fn: Optional[Callable] = None,
        reward_model: Optional[Any] = None,
        value_model: Optional[Any] = None,
        output_dir: str = "./grpo_outputs",
        learning_rate: float = 1e-6,
        seed: int = 0,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        num_train_epochs: int = 1,
        max_steps: int = -1,
        warmup_ratio: float = 0.1,
        logging_steps: int = 1,
        save_steps: int = 100,
        max_seq_length: int = 2048,
        clip_epsilon: float = 0.2,
        epsilon_low: Optional[float] = None,
        epsilon_high: Optional[float] = None,
        rollout_batch_size: Optional[int] = None,
        scale_rewards: Optional[bool] = None,
        reward_normalization: str = "none",
        mask_truncated_completions: bool = False,
        minibatch_reuse_steps: int = 1,
        entropy_bonus: float = 0.0,
        advantage_estimator: Optional[str] = None,
        reward_source: str = "auto",
        reward_sources: Optional[Any] = None,
        kl_target: Optional[float] = None,
        kl_penalty_mode: str = "kl",
        eval_steps: Optional[int] = None,
        eval_num_batches: Optional[int] = None,
        eval_num_generations: Optional[int] = None,
        generation_batch_size: Optional[int] = None,
        score_chunk_size: Optional[int] = None,
        precompute_reference_scores: bool = False,
        dataset_mode: Optional[str] = None,
        chat_template: Optional[Any] = None,
        auto_detect_dataset: bool = True,
        **kwargs,
    ):
        self.loss_type = loss_type
        resolved_advantage = self._pop_alias(
            kwargs,
            "baseline_mode",
            "advantage_estimator",
            default=advantage_estimator or advantage_mode,
        )
        self.advantage_estimator = resolved_advantage
        self.advantage_mode = resolved_advantage
        self.kl_beta = self._pop_alias(kwargs, "kl_beta", default=beta)
        self.beta = self.kl_beta
        self.kl_target = kl_target
        self.kl_penalty_mode = kl_penalty_mode
        self.num_generations = self._pop_alias(kwargs, "generations_per_prompt", default=num_generations)
        self.temperature = temperature
        self.max_completion_length = max_completion_length
        self.reward_fn = reward_fn
        self.reward_model = reward_model
        self.value_model = value_model
        self.reward_sources = self._normalize_reward_sources(
            self._pop_alias(kwargs, "reward_sources", default=reward_sources),
            reward_fn,
            reward_model,
        )
        self.reward_source = reward_source
        self.reward_normalization = reward_normalization
        self.mask_truncated_completions = mask_truncated_completions
        self.minibatch_reuse_steps = minibatch_reuse_steps
        self.entropy_bonus = entropy_bonus
        self.rollout_batch_size = rollout_batch_size
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.seed = seed
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.warmup_ratio = warmup_ratio
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_seq_length = max_seq_length
        self.clip_epsilon = clip_epsilon
        self.epsilon_low = clip_epsilon if epsilon_low is None else epsilon_low
        self.epsilon_high = clip_epsilon if epsilon_high is None else epsilon_high
        self.scale_rewards = scale_rewards
        self.eval_steps = eval_steps
        self.eval_num_batches = eval_num_batches
        self.eval_num_generations = eval_num_generations
        self.generation_batch_size = generation_batch_size
        self.score_chunk_size = score_chunk_size
        self.precompute_reference_scores = precompute_reference_scores
        self.dataset_mode = dataset_mode
        self.chat_template = chat_template
        self.auto_detect_dataset = auto_detect_dataset
        if self.loss_type == "dapo":
            self.mask_truncated_completions = True
            self.epsilon_high = 0.28 if epsilon_high is None else epsilon_high
        if self.scale_rewards is None:
            self.scale_rewards = self.loss_type != "dr_grpo"
        self._set_remaining(kwargs)


class PPOConfig(RLConfigBase):
    _NON_SERIALIZED_FIELDS = {"reward_fn", "reward_model", "value_model"}

    def __init__(
        self,
        output_dir: str = "./ppo_outputs",
        learning_rate: float = 1e-6,
        seed: int = 0,
        value_learning_rate: Optional[float] = None,
        per_device_train_batch_size: int = 1,
        num_train_epochs: int = 1,
        max_steps: int = -1,
        logging_steps: int = 1,
        save_steps: int = 100,
        max_seq_length: int = 2048,
        max_completion_length: int = 256,
        num_generations: int = 4,
        ppo_epochs: int = 2,
        temperature: float = 0.7,
        clip_epsilon: float = 0.2,
        beta: float = 0.0,
        gamma: float = 1.0,
        gae_lambda: float = 1.0,
        reward_fn: Optional[Callable] = None,
        reward_model: Optional[Any] = None,
        value_model: Optional[Any] = None,
        normalize_advantages: bool = True,
        rollout_batch_size: Optional[int] = None,
        reward_normalization: str = "none",
        mask_truncated_completions: bool = False,
        minibatch_reuse_steps: Optional[int] = None,
        entropy_bonus: float = 0.0,
        advantage_estimator: str = "gae",
        reward_source: str = "auto",
        reward_sources: Optional[Any] = None,
        kl_target: Optional[float] = None,
        kl_penalty_mode: str = "kl",
        eval_steps: Optional[int] = None,
        eval_num_batches: Optional[int] = None,
        eval_num_generations: Optional[int] = None,
        generation_batch_size: Optional[int] = None,
        score_chunk_size: Optional[int] = None,
        precompute_reference_scores: bool = False,
        dataset_mode: Optional[str] = None,
        chat_template: Optional[Any] = None,
        auto_detect_dataset: bool = True,
        **kwargs,
    ):
        reuse_steps = ppo_epochs if minibatch_reuse_steps is None else minibatch_reuse_steps
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.seed = seed
        self.value_learning_rate = learning_rate if value_learning_rate is None else value_learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_seq_length = max_seq_length
        self.max_completion_length = max_completion_length
        self.num_generations = self._pop_alias(kwargs, "generations_per_prompt", default=num_generations)
        self.minibatch_reuse_steps = reuse_steps
        self.ppo_epochs = reuse_steps
        self.temperature = temperature
        self.clip_epsilon = clip_epsilon
        self.kl_beta = self._pop_alias(kwargs, "kl_beta", default=beta)
        self.beta = self.kl_beta
        self.kl_target = kl_target
        self.kl_penalty_mode = kl_penalty_mode
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reward_fn = reward_fn
        self.reward_model = reward_model
        self.value_model = value_model
        self.reward_sources = self._normalize_reward_sources(
            self._pop_alias(kwargs, "reward_sources", default=reward_sources),
            reward_fn,
            reward_model,
        )
        self.reward_source = reward_source
        self.reward_normalization = reward_normalization
        self.mask_truncated_completions = mask_truncated_completions
        self.entropy_bonus = entropy_bonus
        self.advantage_estimator = advantage_estimator
        self.rollout_batch_size = rollout_batch_size
        self.normalize_advantages = normalize_advantages
        self.eval_steps = eval_steps
        self.eval_num_batches = eval_num_batches
        self.eval_num_generations = eval_num_generations
        self.generation_batch_size = generation_batch_size
        self.score_chunk_size = score_chunk_size
        self.precompute_reference_scores = precompute_reference_scores
        self.dataset_mode = dataset_mode
        self.chat_template = chat_template
        self.auto_detect_dataset = auto_detect_dataset
        self._set_remaining(kwargs)


class OnlineDPOConfig(RLConfigBase):
    _NON_SERIALIZED_FIELDS = {"reward_fn", "reward_model"}

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        output_dir: str = "./online_dpo_outputs",
        learning_rate: float = 5e-7,
        seed: int = 0,
        per_device_train_batch_size: int = 1,
        num_train_epochs: int = 1,
        max_steps: int = -1,
        logging_steps: int = 1,
        save_steps: int = 100,
        max_seq_length: int = 2048,
        max_completion_length: int = 256,
        num_generations: int = 4,
        temperature: float = 0.7,
        reward_fn: Optional[Callable] = None,
        reward_model: Optional[Any] = None,
        rollout_batch_size: Optional[int] = None,
        reward_normalization: str = "none",
        mask_truncated_completions: bool = False,
        minibatch_reuse_steps: int = 1,
        entropy_bonus: float = 0.0,
        reward_source: str = "auto",
        reward_sources: Optional[Any] = None,
        kl_target: Optional[float] = None,
        kl_penalty_mode: str = "kl",
        eval_steps: Optional[int] = None,
        eval_num_batches: Optional[int] = None,
        eval_num_generations: Optional[int] = None,
        generation_batch_size: Optional[int] = None,
        score_chunk_size: Optional[int] = None,
        precompute_reference_scores: bool = False,
        dataset_mode: Optional[str] = None,
        chat_template: Optional[Any] = None,
        auto_detect_dataset: bool = True,
        **kwargs,
    ):
        self.kl_beta = self._pop_alias(kwargs, "kl_beta", default=beta)
        self.beta = self.kl_beta
        self.kl_target = kl_target
        self.kl_penalty_mode = kl_penalty_mode
        self.label_smoothing = label_smoothing
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.seed = seed
        self.per_device_train_batch_size = per_device_train_batch_size
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_seq_length = max_seq_length
        self.max_completion_length = max_completion_length
        self.num_generations = self._pop_alias(kwargs, "generations_per_prompt", default=num_generations)
        self.temperature = temperature
        self.reward_fn = reward_fn
        self.reward_model = reward_model
        self.reward_sources = self._normalize_reward_sources(
            self._pop_alias(kwargs, "reward_sources", default=reward_sources),
            reward_fn,
            reward_model,
        )
        self.reward_source = reward_source
        self.reward_normalization = reward_normalization
        self.mask_truncated_completions = mask_truncated_completions
        self.minibatch_reuse_steps = minibatch_reuse_steps
        self.entropy_bonus = entropy_bonus
        self.rollout_batch_size = rollout_batch_size
        self.eval_steps = eval_steps
        self.eval_num_batches = eval_num_batches
        self.eval_num_generations = eval_num_generations
        self.generation_batch_size = generation_batch_size
        self.score_chunk_size = score_chunk_size
        self.precompute_reference_scores = precompute_reference_scores
        self.dataset_mode = dataset_mode
        self.chat_template = chat_template
        self.auto_detect_dataset = auto_detect_dataset
        self._set_remaining(kwargs)


class KTOConfig(RLConfigBase):
    def __init__(
        self,
        beta: float = 0.1,
        output_dir: str = "./kto_outputs",
        learning_rate: float = 5e-7,
        per_device_train_batch_size: int = 1,
        num_train_epochs: int = 1,
        max_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 100,
        max_seq_length: int = 2048,
        dataset_mode: Optional[str] = None,
        chat_template: Optional[Any] = None,
        auto_detect_dataset: bool = True,
        **kwargs,
    ):
        self.beta = beta
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_seq_length = max_seq_length
        self.dataset_mode = dataset_mode
        self.chat_template = chat_template
        self.auto_detect_dataset = auto_detect_dataset
        self._set_remaining(kwargs)


class SimPOConfig(RLConfigBase):
    def __init__(
        self,
        gamma: float = 0.5,
        beta: float = 2.0,
        output_dir: str = "./simpo_outputs",
        learning_rate: float = 5e-7,
        per_device_train_batch_size: int = 1,
        num_train_epochs: int = 1,
        max_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 100,
        max_seq_length: int = 2048,
        dataset_mode: Optional[str] = None,
        chat_template: Optional[Any] = None,
        auto_detect_dataset: bool = True,
        **kwargs,
    ):
        self.gamma = gamma
        self.beta = beta
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_seq_length = max_seq_length
        self.dataset_mode = dataset_mode
        self.chat_template = chat_template
        self.auto_detect_dataset = auto_detect_dataset
        self._set_remaining(kwargs)


def _resolve_reward_evaluator(
    reward_model: Optional[Any],
    reward_fn: Optional[Any],
    reward_sources: Optional[List[Any]] = None,
) -> Any:
    if reward_sources:
        return public_create_reward_function(rewards=reward_sources)
    if reward_model is not None:
        return reward_model
    if reward_fn is not None:
        return reward_fn
    return None


def _normalize_reward_values(
    rewards: mx.array,
    prompt_group_indices: mx.array,
    mode: str,
) -> mx.array:
    if mode in {"none", "off", ""}:
        return rewards
    if mode not in {"center", "zscore"}:
        raise ValueError(f"Unsupported reward normalization mode: {mode}")
    reward_values = rewards.tolist()
    groups = prompt_group_indices.tolist()
    adjusted = [0.0] * len(reward_values)
    grouped: Dict[int, List[int]] = {}
    for index, group in enumerate(groups):
        grouped.setdefault(int(group), []).append(index)
    for positions in grouped.values():
        group_rewards = mx.array([reward_values[position] for position in positions], dtype=mx.float32)
        mean_value = mx.mean(group_rewards)
        centered = group_rewards - mean_value
        if mode == "zscore":
            std_value = mx.std(group_rewards)
            centered = centered if float(std_value.item()) < 1e-6 else centered / std_value
        for offset, position in enumerate(positions):
            adjusted[position] = float(centered[offset].item())
    return mx.array(adjusted, dtype=mx.float32)


def _apply_truncation_mask_to_rollout(rollout_batch: RolloutBatch) -> RolloutBatch:
    if rollout_batch.truncation_flags is None or not bool(mx.any(rollout_batch.truncation_flags).item()):
        return rollout_batch

    keep_mask = (~rollout_batch.truncation_flags).astype(mx.float32)
    zero_lengths = mx.where(rollout_batch.truncation_flags, mx.zeros_like(rollout_batch.completion_lengths), rollout_batch.completion_lengths)
    rollout_batch.completion_lengths = zero_lengths
    rollout_batch.policy_eval.completion_lengths = zero_lengths
    rollout_batch.rollout_logprobs = rollout_batch.rollout_logprobs * keep_mask
    rollout_batch.policy_eval.rollout_logprobs = rollout_batch.rollout_logprobs
    rollout_batch.policy_eval.old_logprobs = rollout_batch.rollout_logprobs
    if rollout_batch.old_logprobs is not None:
        rollout_batch.old_logprobs = rollout_batch.old_logprobs * keep_mask
    if rollout_batch.policy_eval.old_token_logprobs is not None:
        rollout_batch.policy_eval.old_token_logprobs = rollout_batch.policy_eval.old_token_logprobs * keep_mask[:, None]
    if rollout_batch.reference_logprobs is not None:
        rollout_batch.reference_logprobs = rollout_batch.reference_logprobs * keep_mask
        rollout_batch.policy_eval.reference_logprobs = rollout_batch.reference_logprobs
    if rollout_batch.value_predictions is not None:
        rollout_batch.value_predictions = rollout_batch.value_predictions * keep_mask
        rollout_batch.policy_eval.value_predictions = rollout_batch.value_predictions
    if rollout_batch.rewards is not None:
        rollout_batch.rewards = rollout_batch.rewards * keep_mask
    if rollout_batch.returns is not None:
        rollout_batch.returns = rollout_batch.returns * keep_mask
        rollout_batch.policy_eval.returns = rollout_batch.returns
    if rollout_batch.advantages is not None:
        rollout_batch.advantages = rollout_batch.advantages * keep_mask
        rollout_batch.policy_eval.advantages = rollout_batch.advantages
    return rollout_batch


def _prepare_on_policy_samples(
    dataset: Any,
    tokenizer: Any,
    config: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    prompt_samples: List[Dict[str, Any]] = []
    rollout_samples: List[Dict[str, Any]] = []
    prepared = prepare_rl_dataset(
        dataset,
        mode=config.dataset_mode,
        tokenizer=tokenizer,
        chat_template=getattr(config, "chat_template", None),
    )
    for sample_index, sample in enumerate(prepared):
        if prepared.mode == "prompt":
            prompt = sample.get("prompt", "")
            if not prompt:
                continue
            prompt_samples.append(
                {
                    "sample_index": sample_index,
                    "prompt": prompt,
                    "prompt_ids": _encode_text(tokenizer, prompt),
                    "reward_context": sample.get("reward_context", prompt),
                }
            )
        elif prepared.mode == "rollout":
            rollout_samples.append(
                {
                    "sample_index": sample_index,
                    "prompt": sample["prompt"],
                    "completion": sample["completion"],
                    "reward": sample.get("reward"),
                    "reward_context": sample.get("reward_context", sample["completion"]),
                }
            )
    return prompt_samples, rollout_samples, prepared.mode


def _next_cursor_batch(
    samples: List[Dict[str, Any]],
    count: int,
    cursor: int,
    algorithm: str,
) -> Tuple[List[Dict[str, Any]], int]:
    if not samples:
        raise ValueError(f"{algorithm} training dataset is empty.")

    batch: List[Dict[str, Any]] = []
    next_cursor = cursor
    for _ in range(max(1, count)):
        batch.append(samples[next_cursor])
        next_cursor = (next_cursor + 1) % len(samples)
    return batch, next_cursor


def _rollout_score_batch_size(trainer: Any, num_generations: Optional[int] = None) -> int:
    generations = trainer.num_generations if num_generations is None else num_generations
    return max(1, trainer.rollout_batch_size * max(1, generations))


def _fixed_rollout_cache_dataset_fingerprint(samples: List[Dict[str, Any]]) -> str:
    return _hash_payload(
        [
            {
                "sample_index": sample.get("sample_index"),
                "prompt": sample.get("prompt"),
                "completion": sample.get("completion"),
                "reward_context": sample.get("reward_context"),
            }
            for sample in samples
        ]
    )


def _preference_cache_dataset_fingerprint(samples: List[Dict[str, Any]]) -> str:
    return _hash_payload(
        [
            {
                "sample_index": sample.get("sample_index"),
                "chosen_ids": sample.get("chosen_ids"),
                "rejected_ids": sample.get("rejected_ids"),
            }
            for sample in samples
        ]
    )


def _fixed_rollout_reference_cache_valid(
    trainer: Any,
    cache_key: str,
    samples: List[Dict[str, Any]],
) -> bool:
    cached_scores = trainer.runtime_cache_arrays.get(cache_key)
    cached_indices = trainer.runtime_cache_arrays.get(f"{cache_key}.sample_indices")
    if cached_scores is None or cached_indices is None or cached_scores.shape[0] != len(samples):
        return False
    cache_info = trainer.cache_metadata.get("reference_score_caches", {}).get(cache_key, {})
    return (
        cached_indices.tolist() == [sample["sample_index"] for sample in samples]
        and cache_info.get("dataset_fingerprint") == _fixed_rollout_cache_dataset_fingerprint(samples)
    )


def _ensure_fixed_rollout_reference_cache(
    trainer: Any,
    cache_key: str,
    samples: List[Dict[str, Any]],
) -> Optional[mx.array]:
    if not getattr(trainer.config, "precompute_reference_scores", False):
        return None
    if trainer.reference_policy is None or not samples:
        return None
    if _fixed_rollout_reference_cache_valid(trainer, cache_key, samples):
        return trainer.runtime_cache_arrays[cache_key]

    prompt_ids = [_encode_text(trainer.tokenizer, sample["prompt"]) for sample in samples]
    completion_ids = [
        _encode_text(trainer.tokenizer, sample["completion"], add_special_tokens=False)
        for sample in samples
    ]
    full_sequences = [prompt + completion for prompt, completion in zip(prompt_ids, completion_ids)]
    prompt_lengths = [len(prompt) for prompt in prompt_ids]
    completion_lengths = [len(completion) for completion in completion_ids]
    eval_batch = make_policy_eval_batch(
        full_sequences,
        pad_id=_pad_token_id(trainer.tokenizer),
        mode="completion",
        prompt_lengths=prompt_lengths,
        completion_lengths=completion_lengths,
        sample_indices=mx.array([sample["sample_index"] for sample in samples]),
    )
    reference_logprobs = score_policy_in_chunks(
        _actual_model(trainer.reference_policy.model),
        eval_batch,
        batch_size=_rollout_score_batch_size(trainer),
        token_budget=getattr(trainer.config, "score_chunk_size", None),
        mode="completion",
    ).summed_logprobs
    reference_logprobs = mx.stop_gradient(reference_logprobs.astype(mx.float32))
    trainer.runtime_cache_arrays[cache_key] = reference_logprobs
    trainer.runtime_cache_arrays[f"{cache_key}.sample_indices"] = mx.array(
        [sample["sample_index"] for sample in samples],
        dtype=mx.int32,
    )
    trainer.cache_metadata.setdefault("reference_score_caches", {})[cache_key] = {
        "num_samples": len(samples),
        "sampling_config_fingerprint": trainer._sampling_config_fingerprint(),
        "dataset_fingerprint": _fixed_rollout_cache_dataset_fingerprint(samples),
    }
    return reference_logprobs


def _preference_reference_cache_valid(
    trainer: Any,
    cache_key: str,
    samples: List[Dict[str, Any]],
) -> bool:
    chosen = trainer.runtime_cache_arrays.get(f"{cache_key}.chosen")
    rejected = trainer.runtime_cache_arrays.get(f"{cache_key}.rejected")
    sample_indices = trainer.runtime_cache_arrays.get(f"{cache_key}.sample_indices")
    if chosen is None or rejected is None or sample_indices is None:
        return False
    if chosen.shape[0] != len(samples) or rejected.shape[0] != len(samples):
        return False
    cache_info = trainer.cache_metadata.get("reference_score_caches", {}).get(cache_key, {})
    return (
        sample_indices.tolist() == [sample["sample_index"] for sample in samples]
        and cache_info.get("dataset_fingerprint") == _preference_cache_dataset_fingerprint(samples)
    )


def _ensure_preference_reference_cache(
    trainer: Any,
    cache_key: str,
    samples: List[Dict[str, Any]],
) -> Tuple[Optional[mx.array], Optional[mx.array]]:
    if not getattr(trainer.config, "precompute_reference_scores", False):
        return None, None
    if trainer.reference_policy is None or not samples:
        return None, None
    if _preference_reference_cache_valid(trainer, cache_key, samples):
        return (
            trainer.runtime_cache_arrays[f"{cache_key}.chosen"],
            trainer.runtime_cache_arrays[f"{cache_key}.rejected"],
        )

    preference_batch = make_preference_batch(
        chosen_sequences=[sample["chosen_ids"] for sample in samples],
        rejected_sequences=[sample["rejected_ids"] for sample in samples],
        pad_id=_pad_token_id(trainer.tokenizer),
        sample_indices=[sample["sample_index"] for sample in samples],
    )
    reference_model = _actual_model(trainer.reference_policy.model)
    chosen_reference = score_policy_in_chunks(
        reference_model,
        preference_batch.chosen,
        batch_size=max(1, trainer.batch_size),
        token_budget=getattr(trainer.config, "score_chunk_size", None),
        mode="sequence",
    ).summed_logprobs
    rejected_reference = score_policy_in_chunks(
        reference_model,
        preference_batch.rejected,
        batch_size=max(1, trainer.batch_size),
        token_budget=getattr(trainer.config, "score_chunk_size", None),
        mode="sequence",
    ).summed_logprobs
    trainer.runtime_cache_arrays[f"{cache_key}.chosen"] = mx.stop_gradient(chosen_reference.astype(mx.float32))
    trainer.runtime_cache_arrays[f"{cache_key}.rejected"] = mx.stop_gradient(rejected_reference.astype(mx.float32))
    trainer.runtime_cache_arrays[f"{cache_key}.sample_indices"] = mx.array(
        [sample["sample_index"] for sample in samples],
        dtype=mx.int32,
    )
    trainer.cache_metadata.setdefault("reference_score_caches", {})[cache_key] = {
        "num_samples": len(samples),
        "sampling_config_fingerprint": trainer._sampling_config_fingerprint(),
        "dataset_fingerprint": _preference_cache_dataset_fingerprint(samples),
    }
    return (
        trainer.runtime_cache_arrays[f"{cache_key}.chosen"],
        trainer.runtime_cache_arrays[f"{cache_key}.rejected"],
    )


def _collect_fixed_rollout_batch(
    trainer: Any,
    samples: List[Dict[str, Any]],
    cached_reference_logprobs: Optional[mx.array] = None,
) -> RolloutBatch:
    prompt_ids = []
    completion_ids = []
    truncation_flags = []
    max_seq_length = getattr(trainer, "max_seq_length", getattr(trainer.config, "max_seq_length", None))
    max_completion_length = getattr(
        trainer,
        "max_completion_length",
        getattr(trainer.config, "max_completion_length", None),
    )
    for sample in samples:
        raw_prompt_ids = _encode_text(trainer.tokenizer, sample["prompt"])
        raw_completion_ids = _encode_text(trainer.tokenizer, sample["completion"], add_special_tokens=False)
        capped_prompt_ids, capped_completion_ids, truncated = cap_prompt_and_completion_lengths(
            raw_prompt_ids,
            raw_completion_ids,
            max_seq_length=max_seq_length,
            max_completion_length=max_completion_length,
        )
        prompt_ids.append(capped_prompt_ids)
        completion_ids.append(capped_completion_ids)
        truncation_flags.append(bool(truncated))
    full_sequences = [prompt + completion for prompt, completion in zip(prompt_ids, completion_ids)]
    prompt_lengths = [len(prompt) for prompt in prompt_ids]
    completion_lengths = [len(completion) for completion in completion_ids]
    prompt_group_map: Dict[str, int] = {}
    grouped_indices = []
    for sample in samples:
        grouped_indices.append(prompt_group_map.setdefault(sample["prompt"], len(prompt_group_map)))
    eval_batch = make_policy_eval_batch(
        full_sequences,
        pad_id=_pad_token_id(trainer.tokenizer),
        mode="completion",
        prompt_lengths=prompt_lengths,
        completion_lengths=completion_lengths,
        sample_indices=mx.array([sample["sample_index"] for sample in samples]),
        prompt_group_indices=mx.array(grouped_indices),
        reference_logprobs=cached_reference_logprobs,
    )
    scored = score_policy_in_chunks(
        _actual_model(trainer.model),
        eval_batch,
        batch_size=_rollout_score_batch_size(trainer),
        token_budget=getattr(trainer.config, "score_chunk_size", None),
        mode="completion",
    )
    reward_values = [float(sample.get("reward", 0.0) or 0.0) for sample in samples]
    rollout_batch = RolloutBatch(
        prompt_ids=prompt_ids,
        prompt_lengths=mx.array(prompt_lengths),
        completion_ids=completion_ids,
        completion_lengths=mx.array(completion_lengths),
        prompt_texts=[sample["prompt"] for sample in samples],
        original_prompt_texts=[sample["prompt"] for sample in samples],
        completion_texts=[sample["completion"] for sample in samples],
        reward_contexts=[sample.get("reward_context") for sample in samples],
        sampled_token_logprobs=mx.zeros(
            (len(samples), max(completion_lengths) if completion_lengths else 0),
            dtype=mx.float32,
        ),
        rollout_logprobs=scored.summed_logprobs,
        eos_flags=mx.array([True] * len(samples)),
        truncation_flags=mx.array(truncation_flags),
        prompt_group_indices=mx.array(grouped_indices),
        policy_eval=scored,
        sample_indices=mx.array([sample["sample_index"] for sample in samples]),
        old_logprobs=scored.summed_logprobs,
        rewards=mx.array(reward_values, dtype=mx.float32),
        reference_logprobs=cached_reference_logprobs,
    )
    rollout_batch.policy_eval.old_logprobs = scored.summed_logprobs
    rollout_batch.policy_eval.old_token_logprobs = scored.token_logprobs
    if cached_reference_logprobs is not None:
        rollout_batch.policy_eval.reference_logprobs = cached_reference_logprobs
    return rollout_batch


class RewardTrainer(_RLTrainerBase):
    algorithm = "reward"

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        tokenizer: Optional[Any] = None,
        args: Optional[RewardConfig] = None,
        reward_model: Optional[RewardModel] = None,
        use_native: bool = True,
        **kwargs,
    ):
        self.reward_model = model if isinstance(model, RewardModel) else reward_model or build_reward_model(model)
        self.model = self.reward_model.base_model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer or getattr(self.reward_model, "tokenizer", None) or getattr(self.model, "tokenizer", None)
        self.use_native = use_native and HAS_NATIVE_TRAINING
        self.config = args or RewardConfig()
        self.output_dir = Path(self.config.output_dir)
        self.learning_rate = self.config.learning_rate
        self.batch_size = self.config.per_device_train_batch_size
        self.max_steps = self.config.max_steps
        self.max_seq_length = self.config.max_seq_length
        self.pairwise_margin = self.config.pairwise_margin
        self.regression_loss_type = self.config.regression_loss_type
        self.logging_steps = self.config.logging_steps
        self.save_steps = self.config.save_steps
        dataset_size = len(train_dataset) if hasattr(train_dataset, "__len__") else 100
        self.iters = self.max_steps if self.max_steps > 0 else max(
            1, (dataset_size // max(1, self.batch_size)) * self.config.num_train_epochs
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_samples: List[Dict[str, Any]] = []
        self.dataset_type: Optional[str] = None
        self._train_target = _ScalarRoleTrainTarget(
            _actual_model(self.reward_model.base_model),
            self.reward_model.head,
        )
        self._init_native_state()

    def _primary_role_name(self) -> str:
        return "reward_model"

    def _primary_role_weight_format(self) -> Any:
        reward_base = getattr(self.reward_model, "base_model", None)
        return {
            "backbone": "weights.safetensors",
            "head": "head.safetensors",
            "adapters": "adapters.safetensors"
            if hasattr(reward_base, "has_adapters") and reward_base.has_adapters()
            else None,
        }

    def _save_primary_role(self, checkpoint_dir: Optional[Path] = None) -> None:
        self.reward_model.save_pretrained(str(self._role_dir("reward_model", checkpoint_dir)))

    def _load_primary_role(self, checkpoint_dir: Path) -> None:
        role_dir = self._role_dir("reward_model", checkpoint_dir)
        if role_dir.exists():
            self.reward_model.load_pretrained(str(role_dir))

    def _prepare_training_samples(self) -> None:
        records = list(self.train_dataset)
        dataset_mode = self.config.dataset_mode
        if dataset_mode is None and records:
            first_sample = records[0]
            dataset_mode = "reward_pairwise" if {"chosen", "rejected"} <= set(first_sample.keys()) else "reward_scalar"
        prepared = prepare_rl_dataset(
            records,
            mode=dataset_mode,
            tokenizer=self.tokenizer,
            chat_template=getattr(self.config, "chat_template", None),
        )
        if not prepared:
            raise ValueError("RewardTrainer requires pairwise or scalar reward samples.")
        if prepared.mode not in {"reward_scalar", "reward_pairwise"}:
            raise ValueError("RewardTrainer requires reward_scalar or reward_pairwise datasets.")
        normalized = list(prepared)
        sample_types = {sample["type"] for sample in normalized}
        if len(sample_types) != 1:
            raise ValueError("RewardTrainer currently requires all reward samples to use the same supervision type.")
        self.dataset_type = next(iter(sample_types))
        self.train_samples = []
        for sample in normalized:
            if self.dataset_type == "pairwise":
                self.train_samples.append(
                    _tokenize_reward_pairwise_sample(self.tokenizer, sample, self.max_seq_length)
                )
            else:
                if "score" not in sample:
                    raise ValueError("RewardTrainer scalar samples require a score field.")
                self.train_samples.append(
                    _tokenize_reward_scalar_sample(self.tokenizer, sample, self.max_seq_length)
                )

    def _build_pairwise_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, mx.array]:
        pad_id = _pad_token_id(self.tokenizer)
        chosen_ids, chosen_lengths = _pad_sequences([sample["chosen_ids"] for sample in samples], pad_id)
        rejected_ids, rejected_lengths = _pad_sequences([sample["rejected_ids"] for sample in samples], pad_id)
        return {
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,
            "chosen_lengths": chosen_lengths,
            "rejected_lengths": rejected_lengths,
            "chosen_prompt_lengths": mx.array([sample["chosen_prompt_length"] for sample in samples]),
            "rejected_prompt_lengths": mx.array([sample["rejected_prompt_length"] for sample in samples]),
            "chosen_completion_lengths": mx.array([sample["chosen_completion_length"] for sample in samples]),
            "rejected_completion_lengths": mx.array([sample["rejected_completion_length"] for sample in samples]),
        }

    def _build_scalar_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, mx.array]:
        pad_id = _pad_token_id(self.tokenizer)
        input_ids, lengths = _pad_sequences([sample["ids"] for sample in samples], pad_id)
        return {
            "input_ids": input_ids,
            "lengths": lengths,
            "prompt_lengths": mx.array([sample["prompt_length"] for sample in samples]),
            "completion_lengths": mx.array([sample["completion_length"] for sample in samples]),
            "targets": mx.array([sample["score"] for sample in samples], dtype=mx.float32),
        }

    def train(self, resume_from_checkpoint: Optional[str] = None):
        if not self.use_native:
            raise ValueError("RewardTrainer requires native MLX training support.")
        return self._train_native(resume_from_checkpoint=resume_from_checkpoint)

    def _train_native(self, resume_from_checkpoint: Optional[str] = None):
        self._apply_lora_if_needed()
        self._prepare_training_samples()

        optimizer = self._optimizer_for_training()
        self.optimizer = optimizer
        self.optimizers = {self._primary_optimizer_name(): optimizer}

        if resume_from_checkpoint is not None:
            self.load_state(optimizer=optimizer, checkpoint_dir=Path(resume_from_checkpoint))

        if self.dataset_type == "pairwise":
            def loss_fn(_, batch):
                loss, outputs = reward_model_pairwise_loss(
                    self.reward_model,
                    chosen_input_ids=batch["chosen_ids"],
                    rejected_input_ids=batch["rejected_ids"],
                    chosen_sequence_lengths=batch["chosen_lengths"],
                    rejected_sequence_lengths=batch["rejected_lengths"],
                    chosen_prompt_lengths=batch["chosen_prompt_lengths"],
                    rejected_prompt_lengths=batch["rejected_prompt_lengths"],
                    chosen_completion_lengths=batch["chosen_completion_lengths"],
                    rejected_completion_lengths=batch["rejected_completion_lengths"],
                    margin=self.pairwise_margin,
                )
                return loss, outputs
        else:
            def loss_fn(_, batch):
                loss, predictions = reward_model_regression_loss(
                    self.reward_model,
                    input_ids=batch["input_ids"],
                    sequence_lengths=batch["lengths"],
                    targets=batch["targets"],
                    prompt_lengths=batch["prompt_lengths"],
                    completion_lengths=batch["completion_lengths"],
                    loss_type=self.regression_loss_type,
                )
                return loss, predictions

        value_and_grad = nn.value_and_grad(self._train_target, lambda modules, batch: loss_fn(modules, batch)[0])
        running_loss = 0.0
        last_loss = None

        while self.global_step < self.iters:
            batch_samples = self._next_samples(self.train_samples)
            batch = (
                self._build_pairwise_batch(batch_samples)
                if self.dataset_type == "pairwise"
                else self._build_scalar_batch(batch_samples)
            )
            loss, grads = value_and_grad(self._train_target, batch)
            optimizer.update(self._train_target, grads)
            mx.eval(self._train_target.parameters(), optimizer.state)

            last_loss = float(loss.item())
            metric_payload: Dict[str, Any] = {"loss": last_loss}
            if self.dataset_type == "pairwise":
                _, outputs = loss_fn(self._train_target, batch)
                metric_payload["ranking_accuracy"] = pairwise_ranking_accuracy(
                    outputs["chosen_scores"],
                    outputs["rejected_scores"],
                )
            else:
                _, predictions = loss_fn(self._train_target, batch)
                metric_payload.update(
                    scalar_loss_metrics(loss, predictions, batch["targets"])
                )

            running_loss += last_loss
            self.global_step += 1
            self._record_metric(**metric_payload)

            if self.global_step % self.logging_steps == 0:
                print(
                    f"Reward step {self.global_step}/{self.iters} | "
                    f"loss={running_loss / self.logging_steps:.4f}"
                )
                running_loss = 0.0

            if self.global_step % self.save_steps == 0:
                self.save_state(optimizer=optimizer)

        self.save_state(optimizer=optimizer)
        return {
            "status": "success",
            "global_step": self.global_step,
            "final_loss": last_loss,
            "reward_model_path": str(self._role_dir("reward_model")),
        }


class DPOTrainer(_RLTrainerBase):
    algorithm = "dpo"
    requires_reference_policy = True

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
        prepared = prepare_rl_dataset(
            self.train_dataset,
            mode=self.config.dataset_mode or "preference",
            tokenizer=self.tokenizer,
            chat_template=getattr(self.config, "chat_template", None),
        )
        for sample_index, sample in enumerate(prepared):
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

        prepared = prepare_rl_dataset(
            self.train_dataset,
            mode=self.config.dataset_mode or "preference",
            tokenizer=self.tokenizer,
            chat_template=getattr(self.config, "chat_template", None),
        )
        tokenized_data = [self._tokenize_preference_pair(sample) for sample in prepared]
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
    requires_reference_policy = True

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        eval_dataset: Any = None,
        eval_preference_dataset: Any = None,
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
        self.eval_dataset = eval_dataset
        self.eval_preference_dataset = eval_preference_dataset
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.ref_model = ref_model
        self.reward_model = reward_model or getattr(args, "reward_model", None)
        self.value_model = value_model or getattr(args, "value_model", None)
        self.use_native = use_native and HAS_NATIVE_TRAINING
        self.config = args or GRPOConfig()
        self.loss_type = self.config.loss_type
        self.phase1_loss_type = "phase1_shared_rollout_recompute"
        self.resolved_loss_type = self._resolve_loss_type(self.loss_type)
        self.advantage_mode = self.config.advantage_estimator
        self.beta = self.config.kl_beta
        self.num_generations = self.config.num_generations
        self.max_completion_length = self.config.max_completion_length
        self.reward_fn = reward_fn if reward_fn is not None else self.config.reward_fn
        self.reward_sources = self.config.reward_sources
        self.reward_source = self.config.reward_source
        self.kl_target = self.config.kl_target
        self.kl_penalty_mode = self.config.kl_penalty_mode
        self.reward_normalization = self.config.reward_normalization
        self.mask_truncated_completions = self.config.mask_truncated_completions
        self.minibatch_reuse_steps = self.config.minibatch_reuse_steps
        self.entropy_bonus = self.config.entropy_bonus
        self.output_dir = Path(self.config.output_dir)
        self.learning_rate = self.config.learning_rate
        self.batch_size = self.config.per_device_train_batch_size
        self.rollout_batch_size = self.config.rollout_batch_size or self.batch_size
        self.max_steps = self.config.max_steps
        self.temperature = self.config.temperature
        self.clip_epsilon = self.config.clip_epsilon
        self.epsilon_low = self.config.epsilon_low
        self.epsilon_high = self.config.epsilon_high
        self.scale_rewards = self.config.scale_rewards
        self.eval_steps = self.config.eval_steps
        self.eval_num_batches = self.config.eval_num_batches
        self.eval_num_generations = self.config.eval_num_generations or self.num_generations
        self.generation_batch_size = self.config.generation_batch_size
        self.score_chunk_size = self.config.score_chunk_size
        self.precompute_reference_scores = self.config.precompute_reference_scores
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
        self.rollout_samples: List[Dict[str, Any]] = []
        self.eval_prompt_samples: List[Dict[str, Any]] = []
        self.eval_rollout_samples: List[Dict[str, Any]] = []
        self.eval_preference_samples: List[Dict[str, Any]] = []
        self.prepared_dataset_mode: Optional[str] = None
        self.prompt_dataset_cursor = 0
        self.rollout_dataset_cursor = 0
        self._last_rollout_batch: Optional[RolloutBatch] = None
        self._init_native_state()

        if self.reward_model is None and self.reward_fn is None:
            self.reward_fn = lambda response, context: len(response.split()) / 100.0

    def _resolve_loss_type(self, loss_type: str) -> str:
        if loss_type not in GRPO_LOSS_TYPES:
            raise ValueError(
                f"Unsupported GRPO loss_type '{loss_type}'. "
                f"Supported values: {sorted(GRPO_LOSS_TYPES)}"
            )
        return loss_type

    def _trainer_cursor_state(self) -> Dict[str, int]:
        return {
            "dataset": int(self.dataset_cursor),
            "prompt_dataset": int(self.prompt_dataset_cursor),
            "offline_rollout_dataset": int(self.rollout_dataset_cursor),
        }

    def _restore_trainer_cursors(self, cursors: Dict[str, Any]) -> None:
        super()._restore_trainer_cursors(cursors)
        self.prompt_dataset_cursor = int(cursors.get("prompt_dataset", self.prompt_dataset_cursor))
        self.rollout_dataset_cursor = int(cursors.get("offline_rollout_dataset", self.rollout_dataset_cursor))

    def _sampling_config_payload(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "loss_type": self.resolved_loss_type,
            "beta": self.beta,
            "clip_epsilon": self.clip_epsilon,
            "epsilon_low": self.epsilon_low,
            "epsilon_high": self.epsilon_high,
            "rollout_batch_size": self.rollout_batch_size,
            "minibatch_reuse_steps": self.minibatch_reuse_steps,
            "advantage_mode": self.advantage_mode,
            "scale_rewards": self.scale_rewards,
            "kl_target": self.kl_target,
            "kl_penalty_mode": self.kl_penalty_mode,
            "reward_source": self.reward_source,
            "reward_normalization": self.reward_normalization,
            "mask_truncated_completions": self.mask_truncated_completions,
            "entropy_bonus": self.entropy_bonus,
            "temperature": self.temperature,
            "num_generations": self.num_generations,
            "max_completion_length": self.max_completion_length,
            "max_seq_length": self.max_seq_length,
            "generation_batch_size": self.generation_batch_size,
            "score_chunk_size": self.score_chunk_size,
        }

    def _prepare_prompt_samples(self) -> None:
        self.prompt_samples, self.rollout_samples, self.prepared_dataset_mode = _prepare_on_policy_samples(
            self.train_dataset,
            self.tokenizer,
            self.config,
        )
        if not self.prompt_samples and not self.rollout_samples:
            raise ValueError("GRPOTrainer requires prompt or rollout samples.")

    def _prepare_eval_datasets(self) -> None:
        self.eval_prompt_samples = []
        self.eval_rollout_samples = []
        self.eval_preference_samples = []
        if self.eval_dataset is not None:
            self.eval_prompt_samples, self.eval_rollout_samples, _ = _prepare_on_policy_samples(
                self.eval_dataset,
                self.tokenizer,
                self.config,
            )
        if self.eval_preference_dataset is not None:
            prepared = prepare_rl_dataset(
                self.eval_preference_dataset,
                mode="preference",
                tokenizer=self.tokenizer,
                chat_template=getattr(self.config, "chat_template", None),
            )
            for sample_index, sample in enumerate(prepared):
                prompt = sample.get("prompt", "")
                prompt_ids = _encode_text(self.tokenizer, prompt)
                chosen_ids = _encode_text(self.tokenizer, sample["chosen"], add_special_tokens=False)
                rejected_ids = _encode_text(self.tokenizer, sample["rejected"], add_special_tokens=False)
                self.eval_preference_samples.append(
                    {
                        "sample_index": sample_index,
                        "prompt_ids": prompt_ids,
                        "chosen_ids": prompt_ids + chosen_ids,
                        "rejected_ids": prompt_ids + rejected_ids,
                        "prompt_length": len(prompt_ids),
                        "chosen_completion_length": len(chosen_ids),
                        "rejected_completion_length": len(rejected_ids),
                    }
                )

    def _resolve_reward_evaluator(self) -> Any:
        evaluator = _resolve_reward_evaluator(
            self.reward_model,
            self.reward_fn,
            self.reward_sources,
        )
        if evaluator is not None:
            return evaluator
        return lambda response, context: len(response.split()) / 100.0

    def _next_prompt_batch(self) -> List[Dict[str, Any]]:
        batch, self.prompt_dataset_cursor = _next_cursor_batch(
            self.prompt_samples,
            self.rollout_batch_size,
            self.prompt_dataset_cursor,
            self.algorithm,
        )
        self.dataset_cursor = self.prompt_dataset_cursor
        return batch

    def _next_offline_rollout_batch(self) -> List[Dict[str, Any]]:
        batch, self.rollout_dataset_cursor = _next_cursor_batch(
            self.rollout_samples,
            self.rollout_batch_size * self.num_generations,
            self.rollout_dataset_cursor,
            self.algorithm,
        )
        self.dataset_cursor = self.rollout_dataset_cursor
        return batch

    def _collect_fixed_rollout_batch(
        self,
        samples: List[Dict[str, Any]],
        cache_key: Optional[str] = None,
    ) -> RolloutBatch:
        cached_reference_logprobs = None
        if cache_key is not None:
            cached_reference_logprobs = _ensure_fixed_rollout_reference_cache(self, cache_key, samples)
        return _collect_fixed_rollout_batch(
            self,
            samples,
            cached_reference_logprobs=cached_reference_logprobs,
        )

    def _collect_rollout_batch(
        self,
        prompt_samples: List[Dict[str, Any]],
        num_generations: Optional[int] = None,
        cache_key: Optional[str] = None,
    ) -> RolloutBatch:
        generations = self.num_generations if num_generations is None else num_generations
        reward_evaluator = self._resolve_reward_evaluator()
        if prompt_samples and "completion" in prompt_samples[0]:
            rollout_batch = self._collect_fixed_rollout_batch(prompt_samples, cache_key=cache_key)
        else:
            rollout_batch = collect_rollouts(
                _actual_model(self.model),
                self.tokenizer,
                prompt_samples=prompt_samples,
                sampling_config={
                    "num_generations": generations,
                    "temperature": self.temperature,
                    "max_completion_length": self.max_completion_length,
                    "max_seq_length": self.max_seq_length,
                    "generation_batch_size": self.generation_batch_size,
                },
                reward_evaluator=None,
                collect_sample_stats=self.entropy_bonus != 0.0,
            )
            if self.reward_source == "offline":
                raise ValueError("reward_source='offline' requires rollout samples with completion/reward fields.")
            reward_batch = evaluate_rewards(rollout_batch, reward_evaluator)
            rollout_batch.rewards = reward_batch.scalar_rewards
        if prompt_samples and "completion" in prompt_samples[0]:
            if self.reward_source == "online":
                reward_batch = evaluate_rewards(rollout_batch, reward_evaluator)
                rollout_batch.rewards = reward_batch.scalar_rewards
            elif self.reward_source == "hybrid":
                reward_batch = evaluate_rewards(rollout_batch, reward_evaluator)
                rollout_batch.rewards = rollout_batch.rewards + reward_batch.scalar_rewards
        if rollout_batch.rewards is None:
            rollout_batch.rewards = mx.zeros((len(rollout_batch.prompt_ids),), dtype=mx.float32)
        if self.entropy_bonus and rollout_batch.token_entropies is not None:
            entropy_bonus = mx.mean(rollout_batch.token_entropies, axis=-1) * self.entropy_bonus
            rollout_batch.rewards = rollout_batch.rewards + entropy_bonus.astype(mx.float32)
        if self.reward_normalization != "none":
            rollout_batch.rewards = _normalize_reward_values(
                rollout_batch.rewards,
                rollout_batch.prompt_group_indices,
                self.reward_normalization,
            )
        rollout_batch = score_rollout_references(
            _actual_model(self.reference_policy.model) if self.reference_policy is not None else None,
            rollout_batch,
            batch_size=_rollout_score_batch_size(self, num_generations=generations),
            token_budget=self.score_chunk_size,
        )
        if self.mask_truncated_completions:
            rollout_batch = _apply_truncation_mask_to_rollout(rollout_batch)
        returns_mode = "rloo" if self.advantage_mode == "rloo" else "group_zscore"
        if returns_mode == "group_zscore" and self.scale_rewards is False:
            returns_mode = "group_center"
        returns, advantages = compute_returns_and_advantages(
            rewards=rollout_batch.rewards,
            prompt_group_indices=rollout_batch.prompt_group_indices,
            mode=returns_mode,
        )
        rollout_batch.returns = returns
        rollout_batch.advantages = advantages
        rollout_batch.policy_eval.returns = returns
        rollout_batch.policy_eval.advantages = advantages
        return rollout_batch

    def _evaluate_rollout_metrics(self) -> Dict[str, Any]:
        prompt_limit = max(1, self.eval_num_batches or 1) * self.rollout_batch_size
        rollout_limit = max(1, self.eval_num_batches or 1) * self.rollout_batch_size * self.num_generations
        if self.eval_rollout_samples:
            rollout_batch = self._collect_rollout_batch(
                self.eval_rollout_samples[:rollout_limit],
                cache_key=f"{self.algorithm}.eval_rollout_reference_logprobs",
            )
        elif self.eval_prompt_samples:
            rollout_batch = self._collect_rollout_batch(
                self.eval_prompt_samples[:prompt_limit],
                num_generations=self.eval_num_generations,
            )
        else:
            return {}
        reference_model = _actual_model(self.reference_policy.model)
        effective_beta = self._effective_kl_beta(rollout_batch)
        loss, _ = grpo_recompute_loss(
            model=_actual_model(self.model),
            reference_model=reference_model,
            input_ids=rollout_batch.policy_eval.input_ids,
            prompt_lengths=rollout_batch.policy_eval.prompt_lengths,
            completion_lengths=rollout_batch.policy_eval.completion_lengths,
            rollout_logprobs=rollout_batch.policy_eval.rollout_logprobs,
            old_token_logprobs=rollout_batch.policy_eval.old_token_logprobs,
            reference_logprobs=rollout_batch.policy_eval.reference_logprobs,
            advantages=rollout_batch.policy_eval.advantages,
            beta=effective_beta,
            clip_epsilon=self.clip_epsilon,
            epsilon_low=self.epsilon_low,
            epsilon_high=self.epsilon_high,
            temperature=self.temperature,
            loss_type=self.resolved_loss_type,
            max_completion_length=self.max_completion_length,
        )
        return summarize_rollout_metrics(rollout_batch, policy_loss=float(loss.item()))

    def _evaluate_preference_metrics(self) -> Dict[str, Any]:
        if not self.eval_preference_samples:
            return {}
        limit = max(1, self.eval_num_batches or 1) * self.batch_size
        samples = self.eval_preference_samples[:limit]
        chosen_batch = make_policy_eval_batch(
            [sample["chosen_ids"] for sample in samples],
            pad_id=_pad_token_id(self.tokenizer),
            mode="completion",
            prompt_lengths=[sample["prompt_length"] for sample in samples],
            completion_lengths=[sample["chosen_completion_length"] for sample in samples],
            sample_indices=mx.array([sample["sample_index"] for sample in samples]),
        )
        rejected_batch = make_policy_eval_batch(
            [sample["rejected_ids"] for sample in samples],
            pad_id=_pad_token_id(self.tokenizer),
            mode="completion",
            prompt_lengths=[sample["prompt_length"] for sample in samples],
            completion_lengths=[sample["rejected_completion_length"] for sample in samples],
            sample_indices=mx.array([sample["sample_index"] for sample in samples]),
        )
        chosen_scores = score_policy_in_chunks(
            _actual_model(self.model),
            chosen_batch,
            batch_size=max(1, self.batch_size),
            token_budget=self.score_chunk_size,
            mode="completion",
        ).summed_logprobs
        rejected_scores = score_policy_in_chunks(
            _actual_model(self.model),
            rejected_batch,
            batch_size=max(1, self.batch_size),
            token_budget=self.score_chunk_size,
            mode="completion",
        ).summed_logprobs
        return {
            "preference_win_rate": float(mx.mean((chosen_scores > rejected_scores).astype(mx.float32)).item())
        }

    def evaluate(self) -> Dict[str, Any]:
        self._prepare_eval_datasets()
        if self.reference_policy is None:
            self._ensure_reference_policy()
        with self._preserve_rng_state():
            mx.random.seed(int(self.seed) + 100000 + int(self.global_step))
            metrics: Dict[str, Any] = {}
            metrics.update(self._evaluate_rollout_metrics())
            metrics.update(self._evaluate_preference_metrics())
        if not metrics:
            return {}
        return self._record_metrics("eval", metrics)

    def train(self, resume_from_checkpoint: Optional[str] = None):
        if self.use_native:
            return self._train_native(resume_from_checkpoint=resume_from_checkpoint)
        return self._train_subprocess()

    def _train_native(self, resume_from_checkpoint: Optional[str] = None):
        self._apply_lora_if_needed()
        self._prepare_prompt_samples()
        self._prepare_eval_datasets()
        if resume_from_checkpoint is None:
            self._seed_training_run()

        actual_model = _actual_model(self.model)
        optimizer = self._optimizer_for_training()
        self.optimizer = optimizer

        if resume_from_checkpoint is not None:
            self.load_state(optimizer=optimizer, checkpoint_dir=Path(resume_from_checkpoint))
        else:
            self._ensure_reference_policy()

        reference_model = _actual_model(self.reference_policy.model)
        if self.rollout_samples:
            _ensure_fixed_rollout_reference_cache(
                self,
                f"{self.algorithm}.train_rollout_reference_logprobs",
                self.rollout_samples,
            )

        effective_beta = self.beta

        def loss_fn(model, batch):
            loss, _ = grpo_recompute_loss(
                model=model,
                reference_model=reference_model,
                input_ids=batch.input_ids,
                prompt_lengths=batch.prompt_lengths,
                completion_lengths=batch.completion_lengths,
                rollout_logprobs=batch.rollout_logprobs,
                old_token_logprobs=batch.old_token_logprobs,
                reference_logprobs=batch.reference_logprobs,
                advantages=batch.advantages,
                beta=effective_beta,
                clip_epsilon=self.clip_epsilon,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                loss_type=self.resolved_loss_type,
                max_completion_length=self.max_completion_length,
            )
            return loss

        value_and_grad = nn.value_and_grad(actual_model, loss_fn)
        running_loss = 0.0
        last_loss = None

        while self.global_step < self.iters:
            if self.reward_source == "offline" and self.rollout_samples:
                prompt_samples = self._next_offline_rollout_batch()
                rollout_batch = self._collect_rollout_batch(
                    prompt_samples,
                    cache_key=f"{self.algorithm}.train_rollout_reference_logprobs",
                )
            else:
                prompt_samples = self._next_prompt_batch()
                rollout_batch = self._collect_rollout_batch(prompt_samples)
            self._last_rollout_batch = rollout_batch
            effective_beta = self._effective_kl_beta(rollout_batch)
            step_loss = 0.0
            step_updates = 0
            for _ in range(max(1, self.minibatch_reuse_steps)):
                minibatches = assemble_minibatches(
                    rollout_batch.policy_eval,
                    minibatch_size=_rollout_score_batch_size(self),
                    shuffle=False,
                    mode="completion",
                    token_budget=self.score_chunk_size,
                )
                for minibatch in minibatches:
                    loss, grads = value_and_grad(actual_model, minibatch)
                    optimizer.update(actual_model, grads)
                    mx.eval(actual_model.parameters(), optimizer.state)
                    step_loss += loss.item()
                    step_updates += 1

            last_loss = step_loss / max(1, step_updates)
            running_loss += last_loss
            self.global_step += 1
            train_row = self._record_metrics(
                "train",
                summarize_rollout_metrics(rollout_batch, policy_loss=last_loss),
            )

            if self.global_step % self.logging_steps == 0:
                print(f"GRPO step {self.global_step}/{self.iters} | {self._format_metric_summary(train_row)}")
                running_loss = 0.0

            if self.eval_steps and self.global_step % self.eval_steps == 0:
                eval_row = self.evaluate()
                if eval_row:
                    print(f"GRPO eval | {self._format_metric_summary(eval_row, namespace='eval')}")

            if self.global_step % self.save_steps == 0:
                self.save_state(optimizer=optimizer)

        self.save_state(optimizer=optimizer)
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


class PPOTrainer(_RLTrainerBase):
    algorithm = "ppo"
    requires_reference_policy = True

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        eval_dataset: Any = None,
        eval_preference_dataset: Any = None,
        tokenizer: Optional[Any] = None,
        reward_fn: Optional[Callable] = None,
        reward_model: Optional[Any] = None,
        ref_model: Optional[Any] = None,
        value_model: Optional[Any] = None,
        args: Optional[PPOConfig] = None,
        use_native: bool = True,
        **kwargs,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.eval_preference_dataset = eval_preference_dataset
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.ref_model = ref_model
        self.config = args or PPOConfig()
        self.reward_model = reward_model or self.config.reward_model
        self.reward_fn = reward_fn if reward_fn is not None else self.config.reward_fn
        self.reward_sources = self.config.reward_sources
        self.reward_source = self.config.reward_source
        self.value_model = value_model or self.config.value_model or build_value_model(model)
        self.use_native = use_native and HAS_NATIVE_TRAINING
        self.output_dir = Path(self.config.output_dir)
        self.learning_rate = self.config.learning_rate
        self.value_learning_rate = self.config.value_learning_rate
        self.batch_size = self.config.per_device_train_batch_size
        self.rollout_batch_size = self.config.rollout_batch_size or self.batch_size
        self.max_steps = self.config.max_steps
        self.max_seq_length = self.config.max_seq_length
        self.max_completion_length = self.config.max_completion_length
        self.num_generations = self.config.num_generations
        self.ppo_epochs = self.config.ppo_epochs
        self.minibatch_reuse_steps = self.config.minibatch_reuse_steps
        self.temperature = self.config.temperature
        self.clip_epsilon = self.config.clip_epsilon
        self.beta = self.config.kl_beta
        self.reward_normalization = self.config.reward_normalization
        self.mask_truncated_completions = self.config.mask_truncated_completions
        self.entropy_bonus = self.config.entropy_bonus
        self.gamma = self.config.gamma
        self.gae_lambda = self.config.gae_lambda
        self.normalize_advantages = self.config.normalize_advantages
        self.advantage_estimator = self.config.advantage_estimator
        self.kl_target = self.config.kl_target
        self.kl_penalty_mode = self.config.kl_penalty_mode
        self.eval_steps = self.config.eval_steps
        self.eval_num_batches = self.config.eval_num_batches
        self.eval_num_generations = self.config.eval_num_generations or self.num_generations
        self.generation_batch_size = self.config.generation_batch_size
        self.score_chunk_size = self.config.score_chunk_size
        self.precompute_reference_scores = self.config.precompute_reference_scores
        self.logging_steps = self.config.logging_steps
        self.save_steps = self.config.save_steps
        dataset_size = len(train_dataset) if hasattr(train_dataset, "__len__") else 100
        self.iters = self.max_steps if self.max_steps > 0 else max(
            1, (dataset_size // max(1, self.batch_size)) * self.config.num_train_epochs
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_path = self.output_dir / "policy"
        self.adapter_path.mkdir(parents=True, exist_ok=True)
        self.prompt_samples: List[Dict[str, Any]] = []
        self.rollout_samples: List[Dict[str, Any]] = []
        self.eval_prompt_samples: List[Dict[str, Any]] = []
        self.eval_rollout_samples: List[Dict[str, Any]] = []
        self.eval_preference_samples: List[Dict[str, Any]] = []
        self.prepared_dataset_mode: Optional[str] = None
        self.prompt_dataset_cursor = 0
        self.rollout_dataset_cursor = 0
        self._last_rollout_batch: Optional[RolloutBatch] = None
        self._value_train_target = _ScalarRoleTrainTarget(
            _actual_model(self.value_model.base_model),
            self.value_model.head,
        )
        self._init_native_state()

    def _optimizer_learning_rates(self) -> Dict[str, float]:
        return {"policy": self.learning_rate, "value": self.value_learning_rate}

    def _trainer_cursor_state(self) -> Dict[str, int]:
        return {
            "dataset": int(self.dataset_cursor),
            "prompt_dataset": int(self.prompt_dataset_cursor),
            "offline_rollout_dataset": int(self.rollout_dataset_cursor),
        }

    def _restore_trainer_cursors(self, cursors: Dict[str, Any]) -> None:
        super()._restore_trainer_cursors(cursors)
        self.prompt_dataset_cursor = int(cursors.get("prompt_dataset", self.prompt_dataset_cursor))
        self.rollout_dataset_cursor = int(cursors.get("offline_rollout_dataset", self.rollout_dataset_cursor))

    def _sampling_config_payload(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "beta": self.beta,
            "clip_epsilon": self.clip_epsilon,
            "rollout_batch_size": self.rollout_batch_size,
            "minibatch_reuse_steps": self.minibatch_reuse_steps,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "normalize_advantages": self.normalize_advantages,
            "value_learning_rate": self.value_learning_rate,
            "kl_target": self.kl_target,
            "kl_penalty_mode": self.kl_penalty_mode,
            "reward_source": self.reward_source,
            "reward_normalization": self.reward_normalization,
            "mask_truncated_completions": self.mask_truncated_completions,
            "entropy_bonus": self.entropy_bonus,
            "advantage_estimator": self.advantage_estimator,
            "temperature": self.temperature,
            "num_generations": self.num_generations,
            "max_completion_length": self.max_completion_length,
            "max_seq_length": self.max_seq_length,
            "generation_batch_size": self.generation_batch_size,
            "score_chunk_size": self.score_chunk_size,
        }

    def _prepare_prompt_samples(self) -> None:
        self.prompt_samples, self.rollout_samples, self.prepared_dataset_mode = _prepare_on_policy_samples(
            self.train_dataset,
            self.tokenizer,
            self.config,
        )
        if not self.prompt_samples and not self.rollout_samples:
            raise ValueError("PPOTrainer requires prompt or rollout samples.")

    def _prepare_eval_datasets(self) -> None:
        self.eval_prompt_samples = []
        self.eval_rollout_samples = []
        self.eval_preference_samples = []
        if self.eval_dataset is not None:
            self.eval_prompt_samples, self.eval_rollout_samples, _ = _prepare_on_policy_samples(
                self.eval_dataset,
                self.tokenizer,
                self.config,
            )
        if self.eval_preference_dataset is not None:
            prepared = prepare_rl_dataset(
                self.eval_preference_dataset,
                mode="preference",
                tokenizer=self.tokenizer,
                chat_template=getattr(self.config, "chat_template", None),
            )
            for sample_index, sample in enumerate(prepared):
                prompt = sample.get("prompt", "")
                prompt_ids = _encode_text(self.tokenizer, prompt)
                chosen_ids = _encode_text(self.tokenizer, sample["chosen"], add_special_tokens=False)
                rejected_ids = _encode_text(self.tokenizer, sample["rejected"], add_special_tokens=False)
                self.eval_preference_samples.append(
                    {
                        "sample_index": sample_index,
                        "prompt_ids": prompt_ids,
                        "chosen_ids": prompt_ids + chosen_ids,
                        "rejected_ids": prompt_ids + rejected_ids,
                        "prompt_length": len(prompt_ids),
                        "chosen_completion_length": len(chosen_ids),
                        "rejected_completion_length": len(rejected_ids),
                    }
                )

    def _resolve_reward_evaluator(self) -> Any:
        evaluator = _resolve_reward_evaluator(
            self.reward_model,
            self.reward_fn,
            self.reward_sources,
        )
        if evaluator is not None:
            return evaluator
        return lambda response, context: len(response.split()) / 100.0

    def _next_prompt_batch(self) -> List[Dict[str, Any]]:
        batch, self.prompt_dataset_cursor = _next_cursor_batch(
            self.prompt_samples,
            self.rollout_batch_size,
            self.prompt_dataset_cursor,
            self.algorithm,
        )
        self.dataset_cursor = self.prompt_dataset_cursor
        return batch

    def _next_offline_rollout_batch(self) -> List[Dict[str, Any]]:
        batch, self.rollout_dataset_cursor = _next_cursor_batch(
            self.rollout_samples,
            self.rollout_batch_size * self.num_generations,
            self.rollout_dataset_cursor,
            self.algorithm,
        )
        self.dataset_cursor = self.rollout_dataset_cursor
        return batch

    def _collect_fixed_rollout_batch(
        self,
        samples: List[Dict[str, Any]],
        cache_key: Optional[str] = None,
    ) -> RolloutBatch:
        cached_reference_logprobs = None
        if cache_key is not None:
            cached_reference_logprobs = _ensure_fixed_rollout_reference_cache(self, cache_key, samples)
        return _collect_fixed_rollout_batch(
            self,
            samples,
            cached_reference_logprobs=cached_reference_logprobs,
        )

    def _collect_rollout_batch(
        self,
        prompt_samples: List[Dict[str, Any]],
        num_generations: Optional[int] = None,
        cache_key: Optional[str] = None,
    ) -> RolloutBatch:
        generations = self.num_generations if num_generations is None else num_generations
        reward_evaluator = self._resolve_reward_evaluator()
        if prompt_samples and "completion" in prompt_samples[0]:
            rollout_batch = self._collect_fixed_rollout_batch(prompt_samples, cache_key=cache_key)
        else:
            rollout_batch = collect_rollouts(
                _actual_model(self.model),
                self.tokenizer,
                prompt_samples=prompt_samples,
                sampling_config={
                    "num_generations": generations,
                    "temperature": self.temperature,
                    "max_completion_length": self.max_completion_length,
                    "max_seq_length": self.max_seq_length,
                    "generation_batch_size": self.generation_batch_size,
                },
                reward_evaluator=None,
                collect_sample_stats=self.entropy_bonus != 0.0,
            )
            if self.reward_source == "offline":
                raise ValueError("reward_source='offline' requires rollout samples with completion/reward fields.")
            reward_batch = evaluate_rewards(rollout_batch, reward_evaluator)
            rollout_batch.rewards = reward_batch.scalar_rewards
        if prompt_samples and "completion" in prompt_samples[0]:
            if self.reward_source == "online":
                reward_batch = evaluate_rewards(rollout_batch, reward_evaluator)
                rollout_batch.rewards = reward_batch.scalar_rewards
            elif self.reward_source == "hybrid":
                reward_batch = evaluate_rewards(rollout_batch, reward_evaluator)
                rollout_batch.rewards = rollout_batch.rewards + reward_batch.scalar_rewards
        if rollout_batch.rewards is None:
            rollout_batch.rewards = mx.zeros((len(rollout_batch.prompt_ids),), dtype=mx.float32)
        if self.entropy_bonus and rollout_batch.token_entropies is not None:
            entropy_bonus = mx.mean(rollout_batch.token_entropies, axis=-1) * self.entropy_bonus
            rollout_batch.rewards = rollout_batch.rewards + entropy_bonus.astype(mx.float32)
        if self.reward_normalization != "none":
            rollout_batch.rewards = _normalize_reward_values(
                rollout_batch.rewards,
                rollout_batch.prompt_group_indices,
                self.reward_normalization,
            )
        rollout_batch = score_rollout_references(
            _actual_model(self.reference_policy.model),
            rollout_batch,
            batch_size=_rollout_score_batch_size(self, num_generations=generations),
            token_budget=self.score_chunk_size,
        )
        if self.mask_truncated_completions:
            rollout_batch = _apply_truncation_mask_to_rollout(rollout_batch)
        rollout_batch = predict_rollout_values(
            self.value_model,
            rollout_batch,
            batch_size=_rollout_score_batch_size(self, num_generations=generations),
            token_budget=self.score_chunk_size,
        )
        advantage_mode = self.advantage_estimator
        rollout_values = rollout_batch.value_predictions if advantage_mode == "gae" else None
        returns, advantages = compute_returns_and_advantages(
            rewards=rollout_batch.rewards,
            values=rollout_values,
            prompt_group_indices=rollout_batch.prompt_group_indices,
            mode=advantage_mode,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            normalize=self.normalize_advantages,
        )
        rollout_batch.returns = returns
        rollout_batch.advantages = advantages
        rollout_batch.policy_eval.returns = returns
        rollout_batch.policy_eval.advantages = advantages
        rollout_batch.policy_eval.reference_logprobs = rollout_batch.reference_logprobs
        rollout_batch.policy_eval.value_predictions = rollout_batch.value_predictions
        return rollout_batch

    def _evaluate_rollout_metrics(self) -> Dict[str, Any]:
        prompt_limit = max(1, self.eval_num_batches or 1) * self.rollout_batch_size
        rollout_limit = max(1, self.eval_num_batches or 1) * self.rollout_batch_size * self.num_generations
        if self.eval_rollout_samples:
            rollout_batch = self._collect_rollout_batch(
                self.eval_rollout_samples[:rollout_limit],
                cache_key=f"{self.algorithm}.eval_rollout_reference_logprobs",
            )
        elif self.eval_prompt_samples:
            rollout_batch = self._collect_rollout_batch(
                self.eval_prompt_samples[:prompt_limit],
                num_generations=self.eval_num_generations,
            )
        else:
            return {}
        effective_beta = self._effective_kl_beta(rollout_batch)
        policy_loss, _ = ppo_sequence_loss(
            model=_actual_model(self.model),
            batch=rollout_batch.policy_eval,
            beta=effective_beta,
            clip_epsilon=self.clip_epsilon,
            temperature=self.temperature,
        )
        value_loss, _ = value_model_regression_loss(
            self.value_model,
            input_ids=rollout_batch.policy_eval.input_ids,
            sequence_lengths=rollout_batch.policy_eval.sequence_lengths,
            targets=rollout_batch.policy_eval.returns,
            prompt_lengths=rollout_batch.policy_eval.prompt_lengths,
            completion_lengths=rollout_batch.policy_eval.completion_lengths,
        )
        return summarize_rollout_metrics(
            rollout_batch,
            policy_loss=float(policy_loss.item()),
            value_loss=float(value_loss.item()),
        )

    def _evaluate_preference_metrics(self) -> Dict[str, Any]:
        if not self.eval_preference_samples:
            return {}
        limit = max(1, self.eval_num_batches or 1) * self.batch_size
        samples = self.eval_preference_samples[:limit]
        chosen_batch = make_policy_eval_batch(
            [sample["chosen_ids"] for sample in samples],
            pad_id=_pad_token_id(self.tokenizer),
            mode="completion",
            prompt_lengths=[sample["prompt_length"] for sample in samples],
            completion_lengths=[sample["chosen_completion_length"] for sample in samples],
            sample_indices=mx.array([sample["sample_index"] for sample in samples]),
        )
        rejected_batch = make_policy_eval_batch(
            [sample["rejected_ids"] for sample in samples],
            pad_id=_pad_token_id(self.tokenizer),
            mode="completion",
            prompt_lengths=[sample["prompt_length"] for sample in samples],
            completion_lengths=[sample["rejected_completion_length"] for sample in samples],
            sample_indices=mx.array([sample["sample_index"] for sample in samples]),
        )
        chosen_scores = score_policy_in_chunks(
            _actual_model(self.model),
            chosen_batch,
            batch_size=max(1, self.batch_size),
            token_budget=self.score_chunk_size,
            mode="completion",
        ).summed_logprobs
        rejected_scores = score_policy_in_chunks(
            _actual_model(self.model),
            rejected_batch,
            batch_size=max(1, self.batch_size),
            token_budget=self.score_chunk_size,
            mode="completion",
        ).summed_logprobs
        return {
            "preference_win_rate": float(mx.mean((chosen_scores > rejected_scores).astype(mx.float32)).item())
        }

    def evaluate(self) -> Dict[str, Any]:
        self._prepare_eval_datasets()
        if self.reference_policy is None:
            self._ensure_reference_policy()
        with self._preserve_rng_state():
            mx.random.seed(int(self.seed) + 100000 + int(self.global_step))
            metrics: Dict[str, Any] = {}
            metrics.update(self._evaluate_rollout_metrics())
            metrics.update(self._evaluate_preference_metrics())
        if not metrics:
            return {}
        return self._record_metrics("eval", metrics)

    def train(self, resume_from_checkpoint: Optional[str] = None):
        if not self.use_native:
            raise ValueError("PPOTrainer requires native MLX training support.")
        return self._train_native(resume_from_checkpoint=resume_from_checkpoint)

    def _train_native(self, resume_from_checkpoint: Optional[str] = None):
        self._apply_lora_if_needed()
        if hasattr(self.value_model.base_model, "_apply_lora") and not getattr(self.value_model.base_model, "_lora_applied", False):
            self.value_model.base_model._apply_lora()
        self._prepare_prompt_samples()
        self._prepare_eval_datasets()
        if resume_from_checkpoint is None:
            self._seed_training_run()

        policy_model = _actual_model(self.model)
        policy_optimizer = self._optimizer_for_training(self.learning_rate)
        value_optimizer = self._optimizer_for_training(self.value_learning_rate)
        self.optimizer = policy_optimizer
        self.optimizers = {"policy": policy_optimizer, "value": value_optimizer}

        if resume_from_checkpoint is not None:
            self.load_state(
                checkpoint_dir=Path(resume_from_checkpoint),
                optimizers=self.optimizers,
            )
        else:
            self._ensure_reference_policy()
        if self.rollout_samples:
            _ensure_fixed_rollout_reference_cache(
                self,
                f"{self.algorithm}.train_rollout_reference_logprobs",
                self.rollout_samples,
            )

        effective_beta = self.beta

        def policy_loss_fn(model, batch):
            loss, _ = ppo_sequence_loss(
                model=model,
                batch=batch,
                beta=effective_beta,
                clip_epsilon=self.clip_epsilon,
                temperature=self.temperature,
            )
            return loss

        def value_loss_fn(_, batch):
            loss, _ = value_model_regression_loss(
                self.value_model,
                input_ids=batch.input_ids,
                sequence_lengths=batch.sequence_lengths,
                targets=batch.returns,
                prompt_lengths=batch.prompt_lengths,
                completion_lengths=batch.completion_lengths,
            )
            return loss

        policy_value_and_grad = nn.value_and_grad(policy_model, policy_loss_fn)
        value_value_and_grad = nn.value_and_grad(self._value_train_target, value_loss_fn)
        running_loss = 0.0
        last_policy_loss = None
        last_value_loss = None

        while self.global_step < self.iters:
            if self.reward_source == "offline" and self.rollout_samples:
                prompt_samples = self._next_offline_rollout_batch()
                rollout_batch = self._collect_rollout_batch(
                    prompt_samples,
                    cache_key=f"{self.algorithm}.train_rollout_reference_logprobs",
                )
            else:
                prompt_samples = self._next_prompt_batch()
                rollout_batch = self._collect_rollout_batch(prompt_samples)
            self._last_rollout_batch = rollout_batch
            effective_beta = self._effective_kl_beta(rollout_batch)

            total_policy_loss = 0.0
            total_value_loss = 0.0
            update_count = 0
            for _ in range(max(1, self.minibatch_reuse_steps)):
                minibatches = assemble_minibatches(
                    rollout_batch.policy_eval,
                    minibatch_size=_rollout_score_batch_size(self),
                    shuffle=True,
                    mode="completion",
                    token_budget=self.score_chunk_size,
                )
                for minibatch in minibatches:
                    policy_loss, policy_grads = policy_value_and_grad(policy_model, minibatch)
                    policy_optimizer.update(policy_model, policy_grads)
                    mx.eval(policy_model.parameters(), policy_optimizer.state)

                    value_loss, value_grads = value_value_and_grad(self._value_train_target, minibatch)
                    value_optimizer.update(self._value_train_target, value_grads)
                    mx.eval(self._value_train_target.parameters(), value_optimizer.state)

                    total_policy_loss += float(policy_loss.item())
                    total_value_loss += float(value_loss.item())
                    update_count += 1

            last_policy_loss = total_policy_loss / max(1, update_count)
            last_value_loss = total_value_loss / max(1, update_count)
            running_loss += last_policy_loss
            self.global_step += 1
            train_row = self._record_metrics(
                "train",
                summarize_rollout_metrics(
                    rollout_batch,
                    policy_loss=last_policy_loss,
                    value_loss=last_value_loss,
                ),
            )

            if self.global_step % self.logging_steps == 0:
                print(f"PPO step {self.global_step}/{self.iters} | {self._format_metric_summary(train_row)}")
                running_loss = 0.0

            if self.eval_steps and self.global_step % self.eval_steps == 0:
                eval_row = self.evaluate()
                if eval_row:
                    print(f"PPO eval | {self._format_metric_summary(eval_row, namespace='eval')}")

            if self.global_step % self.save_steps == 0:
                self.save_state(optimizers=self.optimizers)

        self.save_state(optimizers=self.optimizers)
        return {
            "status": "success",
            "adapter_path": str(self.adapter_path),
            "global_step": self.global_step,
            "final_loss": last_policy_loss,
            "final_value_loss": last_value_loss,
        }


class OnlineDPOTrainer(_RLTrainerBase):
    algorithm = "online_dpo"
    requires_reference_policy = True

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        eval_dataset: Any = None,
        eval_preference_dataset: Any = None,
        tokenizer: Optional[Any] = None,
        reward_fn: Optional[Callable] = None,
        reward_model: Optional[Any] = None,
        ref_model: Optional[Any] = None,
        args: Optional[OnlineDPOConfig] = None,
        use_native: bool = True,
        **kwargs,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.eval_preference_dataset = eval_preference_dataset
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.ref_model = ref_model
        self.config = args or OnlineDPOConfig()
        self.reward_model = reward_model or self.config.reward_model
        self.reward_fn = reward_fn if reward_fn is not None else self.config.reward_fn
        self.reward_sources = self.config.reward_sources
        self.reward_source = self.config.reward_source
        self.use_native = use_native and HAS_NATIVE_TRAINING
        self.beta = self.config.beta
        self.kl_target = self.config.kl_target
        self.kl_penalty_mode = self.config.kl_penalty_mode
        self.label_smoothing = self.config.label_smoothing
        self.output_dir = Path(self.config.output_dir)
        self.learning_rate = self.config.learning_rate
        self.batch_size = self.config.per_device_train_batch_size
        self.rollout_batch_size = self.config.rollout_batch_size or self.batch_size
        self.max_steps = self.config.max_steps
        self.max_seq_length = self.config.max_seq_length
        self.max_completion_length = self.config.max_completion_length
        self.num_generations = self.config.num_generations
        self.temperature = self.config.temperature
        self.reward_normalization = self.config.reward_normalization
        self.mask_truncated_completions = self.config.mask_truncated_completions
        self.minibatch_reuse_steps = self.config.minibatch_reuse_steps
        self.entropy_bonus = self.config.entropy_bonus
        self.eval_steps = self.config.eval_steps
        self.eval_num_batches = self.config.eval_num_batches
        self.eval_num_generations = self.config.eval_num_generations or self.num_generations
        self.generation_batch_size = self.config.generation_batch_size
        self.score_chunk_size = self.config.score_chunk_size
        self.precompute_reference_scores = self.config.precompute_reference_scores
        self.logging_steps = self.config.logging_steps
        self.save_steps = self.config.save_steps
        dataset_size = len(train_dataset) if hasattr(train_dataset, "__len__") else 100
        self.iters = self.max_steps if self.max_steps > 0 else max(
            1, (dataset_size // max(1, self.batch_size)) * self.config.num_train_epochs
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_path = self.output_dir / "policy"
        self.adapter_path.mkdir(parents=True, exist_ok=True)
        self.prompt_samples: List[Dict[str, Any]] = []
        self.rollout_samples: List[Dict[str, Any]] = []
        self.eval_prompt_samples: List[Dict[str, Any]] = []
        self.eval_rollout_samples: List[Dict[str, Any]] = []
        self.eval_preference_samples: List[Dict[str, Any]] = []
        self.prepared_dataset_mode: Optional[str] = None
        self.prompt_dataset_cursor = 0
        self.rollout_dataset_cursor = 0
        self._last_rollout_batch: Optional[RolloutBatch] = None
        self._init_native_state()
        if self.num_generations < 2:
            raise ValueError("OnlineDPOTrainer requires num_generations >= 2.")

    def _prepare_prompt_samples(self) -> None:
        self.prompt_samples, self.rollout_samples, self.prepared_dataset_mode = _prepare_on_policy_samples(
            self.train_dataset,
            self.tokenizer,
            self.config,
        )
        if not self.prompt_samples and not self.rollout_samples:
            raise ValueError("OnlineDPOTrainer requires prompt or rollout samples.")

    def _trainer_cursor_state(self) -> Dict[str, int]:
        return {
            "dataset": int(self.dataset_cursor),
            "prompt_dataset": int(self.prompt_dataset_cursor),
            "offline_rollout_dataset": int(self.rollout_dataset_cursor),
        }

    def _restore_trainer_cursors(self, cursors: Dict[str, Any]) -> None:
        super()._restore_trainer_cursors(cursors)
        self.prompt_dataset_cursor = int(cursors.get("prompt_dataset", self.prompt_dataset_cursor))
        self.rollout_dataset_cursor = int(cursors.get("offline_rollout_dataset", self.rollout_dataset_cursor))

    def _sampling_config_payload(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "beta": self.beta,
            "label_smoothing": self.label_smoothing,
            "kl_target": self.kl_target,
            "kl_penalty_mode": self.kl_penalty_mode,
            "rollout_batch_size": self.rollout_batch_size,
            "minibatch_reuse_steps": self.minibatch_reuse_steps,
            "reward_source": self.reward_source,
            "reward_normalization": self.reward_normalization,
            "mask_truncated_completions": self.mask_truncated_completions,
            "entropy_bonus": self.entropy_bonus,
            "temperature": self.temperature,
            "num_generations": self.num_generations,
            "max_completion_length": self.max_completion_length,
            "max_seq_length": self.max_seq_length,
            "generation_batch_size": self.generation_batch_size,
            "score_chunk_size": self.score_chunk_size,
        }

    def _prepare_eval_datasets(self) -> None:
        self.eval_prompt_samples = []
        self.eval_rollout_samples = []
        self.eval_preference_samples = []
        if self.eval_dataset is not None:
            self.eval_prompt_samples, self.eval_rollout_samples, _ = _prepare_on_policy_samples(
                self.eval_dataset,
                self.tokenizer,
                self.config,
            )
        if self.eval_preference_dataset is not None:
            prepared = prepare_rl_dataset(
                self.eval_preference_dataset,
                mode="preference",
                tokenizer=self.tokenizer,
                chat_template=getattr(self.config, "chat_template", None),
            )
            for sample_index, sample in enumerate(prepared):
                prompt = sample.get("prompt", "")
                prompt_ids = _encode_text(self.tokenizer, prompt)
                chosen_ids = prompt_ids + _encode_text(self.tokenizer, sample["chosen"], add_special_tokens=False)
                rejected_ids = prompt_ids + _encode_text(self.tokenizer, sample["rejected"], add_special_tokens=False)
                self.eval_preference_samples.append(
                    {
                        "sample_index": sample_index,
                        "chosen_ids": chosen_ids,
                        "rejected_ids": rejected_ids,
                        "chosen_length": len(chosen_ids),
                        "rejected_length": len(rejected_ids),
                    }
                )

    def _resolve_reward_evaluator(self) -> Any:
        evaluator = _resolve_reward_evaluator(
            self.reward_model,
            self.reward_fn,
            self.reward_sources,
        )
        if evaluator is not None:
            return evaluator
        return lambda response, context: len(response.split()) / 100.0

    def _next_prompt_batch(self) -> List[Dict[str, Any]]:
        batch, self.prompt_dataset_cursor = _next_cursor_batch(
            self.prompt_samples,
            self.rollout_batch_size,
            self.prompt_dataset_cursor,
            self.algorithm,
        )
        self.dataset_cursor = self.prompt_dataset_cursor
        return batch

    def _next_offline_rollout_batch(self) -> List[Dict[str, Any]]:
        batch, self.rollout_dataset_cursor = _next_cursor_batch(
            self.rollout_samples,
            self.rollout_batch_size * self.num_generations,
            self.rollout_dataset_cursor,
            self.algorithm,
        )
        self.dataset_cursor = self.rollout_dataset_cursor
        return batch

    def _collect_fixed_rollout_batch(
        self,
        samples: List[Dict[str, Any]],
        cache_key: Optional[str] = None,
    ) -> RolloutBatch:
        cached_reference_logprobs = None
        if cache_key is not None:
            cached_reference_logprobs = _ensure_fixed_rollout_reference_cache(self, cache_key, samples)
        return _collect_fixed_rollout_batch(
            self,
            samples,
            cached_reference_logprobs=cached_reference_logprobs,
        )

    def _build_online_preference_batch(self, rollout_batch: RolloutBatch) -> Optional[PreferenceBatch]:
        rankings = rank_grouped_rollouts(rollout_batch)
        chosen_sequences: List[List[int]] = []
        rejected_sequences: List[List[int]] = []
        sample_indices: List[int] = []
        for ranking in rankings:
            if ranking["all_tied"]:
                continue
            best = ranking["best_position"]
            worst = ranking["worst_position"]
            chosen_sequences.append(rollout_batch.prompt_ids[best] + rollout_batch.completion_ids[best])
            rejected_sequences.append(rollout_batch.prompt_ids[worst] + rollout_batch.completion_ids[worst])
            sample_indices.append(int(rollout_batch.sample_indices[best].item()))
        if not chosen_sequences:
            return None

        preference_batch = make_preference_batch(
            chosen_sequences=chosen_sequences,
            rejected_sequences=rejected_sequences,
            pad_id=_pad_token_id(self.tokenizer),
            sample_indices=sample_indices,
        )
        reference_model = _actual_model(self.reference_policy.model)
        preference_batch.chosen_reference_logprobs = score_policy_in_chunks(
            reference_model,
            preference_batch.chosen,
            batch_size=max(1, self.batch_size),
            token_budget=self.score_chunk_size,
            mode="sequence",
        ).summed_logprobs
        preference_batch.rejected_reference_logprobs = score_policy_in_chunks(
            reference_model,
            preference_batch.rejected,
            batch_size=max(1, self.batch_size),
            token_budget=self.score_chunk_size,
            mode="sequence",
        ).summed_logprobs
        preference_batch.chosen.reference_logprobs = preference_batch.chosen_reference_logprobs
        preference_batch.rejected.reference_logprobs = preference_batch.rejected_reference_logprobs
        return preference_batch

    def _collect_rollout_batch(
        self,
        prompt_samples: List[Dict[str, Any]],
        num_generations: Optional[int] = None,
        cache_key: Optional[str] = None,
    ) -> RolloutBatch:
        generations = self.num_generations if num_generations is None else num_generations
        if prompt_samples and "completion" in prompt_samples[0]:
            rollout_batch = self._collect_fixed_rollout_batch(prompt_samples, cache_key=cache_key)
        else:
            rollout_batch = collect_rollouts(
                _actual_model(self.model),
                self.tokenizer,
                prompt_samples=prompt_samples,
                sampling_config={
                    "num_generations": generations,
                    "temperature": self.temperature,
                    "max_completion_length": self.max_completion_length,
                    "max_seq_length": self.max_seq_length,
                    "generation_batch_size": self.generation_batch_size,
                },
                reward_evaluator=None,
                collect_sample_stats=self.entropy_bonus != 0.0,
            )
            if self.reward_source == "offline":
                raise ValueError("reward_source='offline' requires rollout samples with completion/reward fields.")
            reward_batch = evaluate_rewards(rollout_batch, self._resolve_reward_evaluator())
            rollout_batch.rewards = reward_batch.scalar_rewards
        if prompt_samples and "completion" in prompt_samples[0]:
            if self.reward_source == "online":
                reward_batch = evaluate_rewards(rollout_batch, self._resolve_reward_evaluator())
                rollout_batch.rewards = reward_batch.scalar_rewards
            elif self.reward_source == "hybrid":
                reward_batch = evaluate_rewards(rollout_batch, self._resolve_reward_evaluator())
                rollout_batch.rewards = rollout_batch.rewards + reward_batch.scalar_rewards
        if self.entropy_bonus and rollout_batch.token_entropies is not None:
            entropy_bonus = mx.mean(rollout_batch.token_entropies, axis=-1) * self.entropy_bonus
            rollout_batch.rewards = rollout_batch.rewards + entropy_bonus.astype(mx.float32)
        if self.reward_normalization != "none":
            rollout_batch.rewards = _normalize_reward_values(
                rollout_batch.rewards,
                rollout_batch.prompt_group_indices,
                self.reward_normalization,
            )
        if self.kl_penalty_mode != "none" or self.kl_target is not None:
            rollout_batch = score_rollout_references(
                _actual_model(self.reference_policy.model),
                rollout_batch,
                batch_size=_rollout_score_batch_size(self, num_generations=generations),
                token_budget=self.score_chunk_size,
            )
        if self.mask_truncated_completions:
            rollout_batch = _apply_truncation_mask_to_rollout(rollout_batch)
        return rollout_batch

    def _evaluate_rollout_metrics(self) -> Dict[str, Any]:
        prompt_limit = max(1, self.eval_num_batches or 1) * self.rollout_batch_size
        rollout_limit = max(1, self.eval_num_batches or 1) * self.rollout_batch_size * self.num_generations
        if self.eval_rollout_samples:
            rollout_batch = self._collect_rollout_batch(self.eval_rollout_samples[:rollout_limit])
        elif self.eval_prompt_samples:
            rollout_batch = self._collect_rollout_batch(
                self.eval_prompt_samples[:prompt_limit],
                num_generations=self.eval_num_generations,
            )
        else:
            return {}
        preference_batch = self._build_online_preference_batch(rollout_batch)
        metrics = summarize_rollout_metrics(rollout_batch)
        if preference_batch is not None:
            effective_beta = self._effective_kl_beta(rollout_batch)
            loss, _ = compute_dpo_loss(
                model=_actual_model(self.model),
                chosen_ids=preference_batch.chosen.input_ids,
                rejected_ids=preference_batch.rejected.input_ids,
                chosen_lengths=preference_batch.chosen.sequence_lengths,
                rejected_lengths=preference_batch.rejected.sequence_lengths,
                beta=effective_beta,
                reference_chosen_logprobs=preference_batch.chosen.reference_logprobs,
                reference_rejected_logprobs=preference_batch.rejected.reference_logprobs,
                label_smoothing=self.label_smoothing,
            )
            metrics["policy_loss"] = float(loss.item())
        return metrics

    def _evaluate_preference_metrics(self) -> Dict[str, Any]:
        if not self.eval_preference_samples:
            return {}
        limit = max(1, self.eval_num_batches or 1) * self.batch_size
        samples = self.eval_preference_samples[:limit]
        preference_batch = make_preference_batch(
            chosen_sequences=[sample["chosen_ids"] for sample in samples],
            rejected_sequences=[sample["rejected_ids"] for sample in samples],
            pad_id=_pad_token_id(self.tokenizer),
            sample_indices=[sample["sample_index"] for sample in samples],
        )
        chosen_scores = score_policy_in_chunks(
            _actual_model(self.model),
            preference_batch.chosen,
            batch_size=max(1, self.batch_size),
            token_budget=self.score_chunk_size,
            mode="sequence",
        ).summed_logprobs
        rejected_scores = score_policy_in_chunks(
            _actual_model(self.model),
            preference_batch.rejected,
            batch_size=max(1, self.batch_size),
            token_budget=self.score_chunk_size,
            mode="sequence",
        ).summed_logprobs
        metrics = {
            "preference_win_rate": float(mx.mean((chosen_scores > rejected_scores).astype(mx.float32)).item())
        }
        chosen_reference, rejected_reference = _ensure_preference_reference_cache(
            self,
            f"{self.algorithm}.eval_preference_reference_logprobs",
            samples,
        )
        if chosen_reference is None or rejected_reference is None:
            reference_model = _actual_model(self.reference_policy.model)
            chosen_reference = score_policy_in_chunks(
                reference_model,
                preference_batch.chosen,
                batch_size=max(1, self.batch_size),
                token_budget=self.score_chunk_size,
                mode="sequence",
            ).summed_logprobs
            rejected_reference = score_policy_in_chunks(
                reference_model,
                preference_batch.rejected,
                batch_size=max(1, self.batch_size),
                token_budget=self.score_chunk_size,
                mode="sequence",
            ).summed_logprobs
        loss, _ = compute_dpo_loss(
            model=_actual_model(self.model),
            chosen_ids=preference_batch.chosen.input_ids,
            rejected_ids=preference_batch.rejected.input_ids,
            chosen_lengths=preference_batch.chosen.sequence_lengths,
            rejected_lengths=preference_batch.rejected.sequence_lengths,
            beta=self.beta,
            reference_chosen_logprobs=chosen_reference,
            reference_rejected_logprobs=rejected_reference,
            label_smoothing=self.label_smoothing,
        )
        metrics["policy_loss"] = float(loss.item())
        return metrics

    def evaluate(self) -> Dict[str, Any]:
        self._prepare_eval_datasets()
        if self.reference_policy is None:
            self._ensure_reference_policy()
        with self._preserve_rng_state():
            mx.random.seed(int(self.seed) + 100000 + int(self.global_step))
            metrics: Dict[str, Any] = {}
            metrics.update(self._evaluate_rollout_metrics())
            metrics.update(self._evaluate_preference_metrics())
        if not metrics:
            return {}
        return self._record_metrics("eval", metrics)

    def train(self, resume_from_checkpoint: Optional[str] = None):
        if not self.use_native:
            raise ValueError("OnlineDPOTrainer requires native MLX training support.")
        return self._train_native(resume_from_checkpoint=resume_from_checkpoint)

    def _train_native(self, resume_from_checkpoint: Optional[str] = None):
        self._apply_lora_if_needed()
        self._prepare_prompt_samples()
        self._prepare_eval_datasets()
        if resume_from_checkpoint is None:
            self._seed_training_run()

        actual_model = _actual_model(self.model)
        optimizer = self._optimizer_for_training()
        self.optimizer = optimizer
        self.optimizers = {"policy": optimizer}

        if resume_from_checkpoint is not None:
            self.load_state(optimizer=optimizer, checkpoint_dir=Path(resume_from_checkpoint))
        else:
            self._ensure_reference_policy()
        if self.rollout_samples:
            _ensure_fixed_rollout_reference_cache(
                self,
                f"{self.algorithm}.train_rollout_reference_logprobs",
                self.rollout_samples,
            )

        effective_beta = self.beta

        def loss_fn(model, batch):
            loss, _ = compute_dpo_loss(
                model=model,
                chosen_ids=batch.chosen.input_ids,
                rejected_ids=batch.rejected.input_ids,
                chosen_lengths=batch.chosen.sequence_lengths,
                rejected_lengths=batch.rejected.sequence_lengths,
                beta=effective_beta,
                reference_chosen_logprobs=batch.chosen.reference_logprobs,
                reference_rejected_logprobs=batch.rejected.reference_logprobs,
                label_smoothing=self.label_smoothing,
            )
            return loss

        value_and_grad = nn.value_and_grad(actual_model, loss_fn)
        running_loss = 0.0
        last_loss = None

        while self.global_step < self.iters:
            if self.reward_source == "offline" and self.rollout_samples:
                prompt_samples = self._next_offline_rollout_batch()
                rollout_batch = self._collect_rollout_batch(
                    prompt_samples,
                    cache_key=f"{self.algorithm}.train_rollout_reference_logprobs",
                )
            else:
                prompt_samples = self._next_prompt_batch()
                rollout_batch = self._collect_rollout_batch(prompt_samples)
            self._last_rollout_batch = rollout_batch
            effective_beta = self._effective_kl_beta(rollout_batch)
            preference_batch = self._build_online_preference_batch(rollout_batch)
            if preference_batch is None:
                self.global_step += 1
                train_row = self._record_metrics(
                    "train",
                    {**summarize_rollout_metrics(rollout_batch, policy_loss=0.0), "skipped_pairs": True},
                )
                if self.global_step % self.logging_steps == 0:
                    print(f"Online DPO step {self.global_step}/{self.iters} | {self._format_metric_summary(train_row)}")
                continue

            loss, grads = value_and_grad(actual_model, preference_batch)
            optimizer.update(actual_model, grads)
            mx.eval(actual_model.parameters(), optimizer.state)

            last_loss = float(loss.item())
            running_loss += last_loss
            self.global_step += 1
            train_row = self._record_metrics(
                "train",
                summarize_rollout_metrics(rollout_batch, policy_loss=last_loss),
            )

            if self.global_step % self.logging_steps == 0:
                print(f"Online DPO step {self.global_step}/{self.iters} | {self._format_metric_summary(train_row)}")
                running_loss = 0.0

            if self.eval_steps and self.global_step % self.eval_steps == 0:
                eval_row = self.evaluate()
                if eval_row:
                    print(f"Online DPO eval | {self._format_metric_summary(eval_row, namespace='eval')}")

            if self.global_step % self.save_steps == 0:
                self.save_state(optimizer=optimizer)

        self.save_state(optimizer=optimizer)
        return {
            "status": "success",
            "adapter_path": str(self.adapter_path),
            "global_step": self.global_step,
            "final_loss": last_loss,
        }


class KTOTrainer(_RLTrainerBase):
    algorithm = "kto"
    requires_reference_policy = True

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        tokenizer: Optional[Any] = None,
        beta: float = 0.1,
        args: Optional[KTOConfig] = None,
        ref_model: Optional[Any] = None,
        reward_model: Optional[Any] = None,
        value_model: Optional[Any] = None,
        use_native: bool = True,
        **kwargs,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.config = args or KTOConfig(beta=beta, **kwargs)
        self.beta = self.config.beta
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.value_model = value_model
        self.use_native = use_native and HAS_NATIVE_TRAINING
        self.output_dir = Path(self.config.output_dir)
        self.learning_rate = self.config.learning_rate
        self.iters = self.config.max_steps
        self.max_seq_length = self.config.max_seq_length
        self.batch_size = self.config.per_device_train_batch_size
        self.logging_steps = self.config.logging_steps
        self.save_steps = self.config.save_steps
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
        args: Optional[SimPOConfig] = None,
        use_native: bool = True,
        **kwargs,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.config = args or SimPOConfig(gamma=gamma, beta=beta, **kwargs)
        self.gamma = self.config.gamma
        self.beta = self.config.beta
        self.use_native = use_native and HAS_NATIVE_TRAINING
        self.output_dir = Path(self.config.output_dir)
        self.learning_rate = self.config.learning_rate
        self.batch_size = self.config.per_device_train_batch_size
        self.iters = self.config.max_steps
        self.max_seq_length = self.config.max_seq_length
        self.logging_steps = self.config.logging_steps
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

        prepared = prepare_rl_dataset(
            self.train_dataset,
            mode=self.config.dataset_mode or "preference",
            tokenizer=self.tokenizer,
            chat_template=getattr(self.config, "chat_template", None),
        )
        tokenized_data = [self._tokenize_pair(sample) for sample in prepared]
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
            start = (step * max(1, self.batch_size)) % len(tokenized_data)
            samples = tokenized_data[start:start + max(1, self.batch_size)]
            if len(samples) < max(1, self.batch_size):
                samples += tokenized_data[: max(1, self.batch_size) - len(samples)]
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
    return public_prepare_preference_dataset(dataset, tokenizer=tokenizer, format_type=format_type)


def create_reward_function(reward_type: Any = "simple", *, rewards: Optional[List[Any]] = None) -> Callable:
    return public_create_reward_function(reward_type, rewards=rewards)
