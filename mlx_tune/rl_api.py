from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence
import warnings

import mlx.core as mx
from mlx.utils import tree_unflatten

from mlx_tune.chat_templates import apply_chat_template_to_sample
from mlx_tune.model import build_reference_policy, build_reward_model


MANIFEST_FILE = "manifest.json"
STATE_FILE = "trainer_state.safetensors"
METADATA_FILE = "trainer_state.json"
REFERENCE_FILE = "reference_model.safetensors"
REFERENCE_METADATA_FILE = "reference_metadata.json"
CHECKPOINT_FORMAT_NAME = "mlx_tune_rl_checkpoint"
CHECKPOINT_FORMAT_VERSION = 4
SUPPORTED_RL_DATASET_MODES = (
    "prompt",
    "preference",
    "reward_scalar",
    "reward_pairwise",
    "rollout",
    "chat",
)


@dataclass
class PreparedRLDataset:
    samples: List[Dict[str, Any]]
    mode: str
    adapter_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]


@dataclass
class RLRoleState:
    role: str
    weight_format: Any
    parameter_state: Dict[str, mx.array] = field(default_factory=dict)
    head_state: Dict[str, mx.array] = field(default_factory=dict)
    adapter_state: Dict[str, mx.array] = field(default_factory=dict)
    head_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RLCheckpointBundle:
    manifest: Dict[str, Any]
    algorithm: str
    restored_roles: Dict[str, RLRoleState]
    optimizer_state_trees: Dict[str, Dict[str, Any]]
    scheduler_metadata: Dict[str, Dict[str, Any]]
    trainer_state: Dict[str, Any]
    rng_state: Dict[str, mx.array]
    runtime_cache: Dict[str, mx.array]
    metrics_history: List[Dict[str, Any]]
    source_format: str


def _read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    with open(path) as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_prefixed_tree(prefix: str, flat_state: Dict[str, mx.array]) -> Dict[str, Any]:
    prefix_with_dot = f"{prefix}."
    items = [(key[len(prefix_with_dot):], value) for key, value in flat_state.items() if key.startswith(prefix_with_dot)]
    return tree_unflatten(items) if items else {}


def _role_dir(checkpoint_dir: Path, role_name: str) -> Path:
    return checkpoint_dir / role_name


def _legacy_checkpoint_exists(checkpoint_dir: Path) -> bool:
    return (
        (checkpoint_dir / "adapters" / "adapters.safetensors").exists()
        or (checkpoint_dir / STATE_FILE).exists()
        or (checkpoint_dir / REFERENCE_FILE).exists()
    )


def _metrics_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "metrics" / "history.jsonl"


def _runtime_cache_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "runtime" / "cache.safetensors"


def _trainer_state_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "trainer" / "state.json"


def _rng_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "trainer" / "rng.safetensors"


def _scheduler_path(checkpoint_dir: Path, name: Optional[str] = None) -> Path:
    if name is None:
        return checkpoint_dir / "scheduler" / "state.json"
    return checkpoint_dir / "schedulers" / name / "state.json"


def _optimizer_path(checkpoint_dir: Path, name: Optional[str] = None) -> Path:
    if name is None:
        return checkpoint_dir / "optimizer" / "state.safetensors"
    return checkpoint_dir / "optimizers" / name / "state.safetensors"


def _load_role_state(checkpoint_dir: Path, role_name: str, weight_format: Any) -> RLRoleState:
    role_dir = _role_dir(checkpoint_dir, role_name)
    metadata = _read_json(role_dir / "metadata.json")
    head_config = _read_json(role_dir / "head_config.json")
    parameter_state: Dict[str, mx.array] = {}
    head_state: Dict[str, mx.array] = {}
    adapter_state: Dict[str, mx.array] = {}

    if isinstance(weight_format, str):
        weight_path = role_dir / weight_format
        if weight_path.exists():
            parameter_state = dict(mx.load(str(weight_path)))
    elif isinstance(weight_format, Mapping):
        backbone_path = role_dir / str(weight_format.get("backbone", "weights.safetensors"))
        head_path = role_dir / str(weight_format.get("head", "head.safetensors"))
        adapters_name = weight_format.get("adapters")
        if backbone_path.exists():
            parameter_state = dict(mx.load(str(backbone_path)))
        if head_path.exists():
            head_state = dict(mx.load(str(head_path)))
        if adapters_name:
            adapter_path = role_dir / str(adapters_name)
            if adapter_path.exists():
                adapter_state = dict(mx.load(str(adapter_path)))

    return RLRoleState(
        role=role_name,
        weight_format=weight_format,
        parameter_state=parameter_state,
        head_state=head_state,
        adapter_state=adapter_state,
        head_config=head_config,
        metadata=metadata,
    )


def _build_manifest_bundle(checkpoint_dir: Path) -> RLCheckpointBundle:
    manifest = _read_json(checkpoint_dir / MANIFEST_FILE)
    if not manifest:
        raise FileNotFoundError(f"Checkpoint manifest not found under {checkpoint_dir}")

    roles = {
        role_name: _load_role_state(
            checkpoint_dir,
            role_name,
            manifest.get("role_weight_formats", {}).get(role_name, "weights.safetensors"),
        )
        for role_name in manifest.get("roles_present", [])
    }

    trainer_locations = manifest.get("trainer_state_locations", {})
    optimizer_state_trees: Dict[str, Dict[str, Any]] = {}
    optimizer_locations = trainer_locations.get("optimizers") or {}
    if optimizer_locations:
        for name, relative_path in optimizer_locations.items():
            path = checkpoint_dir / relative_path
            if path.exists():
                optimizer_state_trees[name] = _extract_prefixed_tree("optimizer", dict(mx.load(str(path))))
    else:
        legacy_optimizer_path = _optimizer_path(checkpoint_dir)
        if legacy_optimizer_path.exists():
            optimizer_state_trees["default"] = _extract_prefixed_tree(
                "optimizer",
                dict(mx.load(str(legacy_optimizer_path))),
            )

    scheduler_metadata: Dict[str, Dict[str, Any]] = {}
    scheduler_locations = trainer_locations.get("schedulers") or {}
    if scheduler_locations:
        for name, relative_path in scheduler_locations.items():
            scheduler_metadata[name] = _read_json(checkpoint_dir / relative_path)
    else:
        default_scheduler = _read_json(_scheduler_path(checkpoint_dir))
        if default_scheduler:
            scheduler_metadata["default"] = default_scheduler

    rng_state = dict(mx.load(str(_rng_path(checkpoint_dir)))) if _rng_path(checkpoint_dir).exists() else {}
    runtime_cache = (
        dict(mx.load(str(_runtime_cache_path(checkpoint_dir))))
        if _runtime_cache_path(checkpoint_dir).exists()
        else {}
    )
    trainer_state = _read_json(_trainer_state_path(checkpoint_dir))
    metrics_history = _load_jsonl(_metrics_path(checkpoint_dir))

    return RLCheckpointBundle(
        manifest=manifest,
        algorithm=manifest.get("algorithm", trainer_state.get("algorithm", "rl")),
        restored_roles=roles,
        optimizer_state_trees=optimizer_state_trees,
        scheduler_metadata=scheduler_metadata,
        trainer_state=trainer_state,
        rng_state=rng_state,
        runtime_cache=runtime_cache,
        metrics_history=metrics_history,
        source_format="manifest",
    )


def _build_legacy_manifest(metadata: Dict[str, Any], has_reference: bool) -> Dict[str, Any]:
    roles_present = ["policy"]
    role_weight_formats: Dict[str, Any] = {"policy": "adapters.safetensors"}
    if has_reference:
        roles_present.append("reference")
        role_weight_formats["reference"] = "weights.safetensors"
    return {
        "format_name": CHECKPOINT_FORMAT_NAME,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "algorithm": metadata.get("algorithm", "rl"),
        "roles_present": roles_present,
        "role_weight_formats": role_weight_formats,
        "trainer_state_locations": {
            "optimizer": "trainer_state.safetensors",
            "rng": "trainer_state.safetensors",
            "runtime_cache": "trainer_state.safetensors",
            "trainer": "trainer_state.json",
        },
        "metrics_path": None,
    }


def _build_legacy_bundle(checkpoint_dir: Path) -> RLCheckpointBundle:
    state_path = checkpoint_dir / STATE_FILE
    metadata_path = checkpoint_dir / METADATA_FILE
    if not state_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Checkpoint state not found under {checkpoint_dir}")

    metadata = _read_json(metadata_path)
    flat_state = dict(mx.load(str(state_path)))
    policy_state: Dict[str, mx.array] = {}
    adapter_path = checkpoint_dir / "adapters" / "adapters.safetensors"
    if adapter_path.exists():
        policy_state = dict(mx.load(str(adapter_path)))

    roles = {
        "policy": RLRoleState(
            role="policy",
            weight_format="adapters.safetensors",
            parameter_state=policy_state,
        ),
    }
    reference_path = checkpoint_dir / REFERENCE_FILE
    if reference_path.exists():
        roles["reference"] = RLRoleState(
            role="reference",
            weight_format="weights.safetensors",
            parameter_state=dict(mx.load(str(reference_path))),
            metadata=_read_json(checkpoint_dir / REFERENCE_METADATA_FILE),
        )

    runtime_cache = {
        key: value
        for key, value in flat_state.items()
        if not key.startswith("optimizer.") and not key.startswith("rng.")
    }
    optimizer_state = _extract_prefixed_tree("optimizer", flat_state)
    rng_state = {key: value for key, value in flat_state.items() if key.startswith("rng.")}

    return RLCheckpointBundle(
        manifest=_build_legacy_manifest(metadata, has_reference="reference" in roles),
        algorithm=metadata.get("algorithm", "rl"),
        restored_roles=roles,
        optimizer_state_trees={"default": optimizer_state} if optimizer_state else {},
        scheduler_metadata={},
        trainer_state=metadata,
        rng_state=rng_state,
        runtime_cache=runtime_cache,
        metrics_history=[],
        source_format="legacy",
    )


def resume_from_checkpoint(checkpoint_dir: str | Path) -> RLCheckpointBundle:
    checkpoint_path = Path(checkpoint_dir)
    if (checkpoint_path / MANIFEST_FILE).exists():
        return _build_manifest_bundle(checkpoint_path)
    if _legacy_checkpoint_exists(checkpoint_path):
        return _build_legacy_bundle(checkpoint_path)
    raise FileNotFoundError(f"Checkpoint state not found under {checkpoint_path}")


def _tokenizer_encode(tokenizer: Any, text: str, add_special_tokens: bool) -> List[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))
    except TypeError:
        return list(tokenizer.encode(text))


def _is_message_sequence(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, Mapping) and "role" in item for item in value)


def _render_messages(
    messages: Sequence[Mapping[str, Any]],
    tokenizer: Any = None,
    chat_template: Optional[Any] = None,
    add_generation_prompt: bool = False,
) -> str:
    payload = {"messages": list(messages)}
    if callable(chat_template):
        return chat_template(payload, add_generation_prompt=add_generation_prompt)
    if isinstance(chat_template, str):
        if add_generation_prompt:
            return chat_template.format(messages=list(messages), add_generation_prompt=True)
        return chat_template.format(messages=list(messages))
    if tokenizer is not None:
        return apply_chat_template_to_sample(
            payload,
            tokenizer,
            add_generation_prompt=add_generation_prompt,
        )
    return "\n".join(f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages)


def _last_assistant_index(messages: Sequence[Mapping[str, Any]]) -> Optional[int]:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    return None


def _extract_chat_prompt_response(
    sample: Mapping[str, Any],
    tokenizer: Any = None,
    chat_template: Optional[Any] = None,
) -> tuple[str, str]:
    messages = list(sample.get("messages") or [])
    if not messages:
        raise ValueError("Chat adaptation requires a messages field.")
    assistant_index = _last_assistant_index(messages)
    if assistant_index is None:
        prompt_messages = messages
        response = ""
    else:
        prompt_messages = messages[:assistant_index]
        response = str(messages[assistant_index].get("content", ""))
    prompt = _render_messages(
        prompt_messages,
        tokenizer=tokenizer,
        chat_template=chat_template,
        add_generation_prompt=True,
    )
    return prompt, response


def _extract_preference_value(
    value: Any,
    prompt: str,
    tokenizer: Any = None,
    chat_template: Optional[Any] = None,
) -> tuple[str, str]:
    if isinstance(value, str):
        return prompt, value
    if isinstance(value, Mapping) and _is_message_sequence(value.get("messages")):
        prompt_text, response = _extract_chat_prompt_response(
            value,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
        return prompt_text or prompt, response
    if _is_message_sequence(value):
        prompt_text, response = _extract_chat_prompt_response(
            {"messages": value},
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
        return prompt_text or prompt, response
    raise ValueError("Unsupported preference value; expected string or chat message sequence.")


def _normalize_prompt_sample(
    sample: Mapping[str, Any],
    tokenizer: Any = None,
    chat_template: Optional[Any] = None,
) -> Dict[str, Any]:
    if _is_message_sequence(sample.get("messages")):
        prompt, response = _extract_chat_prompt_response(
            sample,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
        reward_context = sample.get("reward_context", sample.get("answer", sample.get("response", response or prompt)))
        return {
            "prompt": prompt,
            "reward_context": reward_context,
            "source_messages": list(sample.get("messages") or []),
        }

    prompt = str(sample.get("prompt", sample.get("question", "")))
    if not prompt:
        raise ValueError("Prompt samples require a prompt or question field.")
    return {
        "prompt": prompt,
        "reward_context": sample.get(
            "reward_context",
            sample.get("answer", sample.get("response", prompt)),
        ),
    }


def _normalize_preference_sample(
    sample: Mapping[str, Any],
    tokenizer: Any = None,
    chat_template: Optional[Any] = None,
    target_mode: str = "preference",
) -> Dict[str, Any]:
    prompt = str(sample.get("prompt", sample.get("question", "")))
    if _is_message_sequence(sample.get("messages")):
        prompt = _render_messages(
            sample.get("messages") or [],
            tokenizer=tokenizer,
            chat_template=chat_template,
            add_generation_prompt=True,
        )
    chosen_prompt, chosen = _extract_preference_value(
        sample.get("chosen"),
        prompt,
        tokenizer=tokenizer,
        chat_template=chat_template,
    )
    rejected_prompt, rejected = _extract_preference_value(
        sample.get("rejected"),
        chosen_prompt,
        tokenizer=tokenizer,
        chat_template=chat_template,
    )
    return {
        "type": "pairwise" if target_mode == "reward_pairwise" else "preference",
        "prompt": chosen_prompt or rejected_prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def _normalize_reward_scalar_sample(
    sample: Mapping[str, Any],
    tokenizer: Any = None,
    chat_template: Optional[Any] = None,
) -> Dict[str, Any]:
    if _is_message_sequence(sample.get("messages")):
        prompt, response = _extract_chat_prompt_response(
            sample,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
        return {
            "type": "scalar",
            "prompt": prompt,
            "response": response,
            "score": float(sample.get("score", sample.get("reward", 0.0))),
        }

    if "text" in sample:
        return {
            "type": "scalar",
            "prompt": "",
            "response": str(sample.get("text", "")),
            "score": float(sample.get("score", sample.get("reward", 0.0))),
        }

    prompt = str(sample.get("prompt", sample.get("question", "")))
    response = str(sample.get("response", sample.get("completion", sample.get("assistant", ""))))
    if not response:
        raise ValueError("Scalar reward samples require a response, completion, assistant, or text field.")
    return {
        "type": "scalar",
        "prompt": prompt,
        "response": response,
        "score": float(sample.get("score", sample.get("reward", 0.0))),
    }


def _normalize_rollout_sample(
    sample: Mapping[str, Any],
    tokenizer: Any = None,
    chat_template: Optional[Any] = None,
) -> Dict[str, Any]:
    if _is_message_sequence(sample.get("messages")):
        prompt, response = _extract_chat_prompt_response(
            sample,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
        completion = response or str(sample.get("completion", sample.get("response", "")))
    else:
        prompt = str(sample.get("prompt", sample.get("question", "")))
        completion = str(sample.get("completion", sample.get("response", "")))
    if not prompt or not completion:
        raise ValueError("Rollout samples require prompt/question and completion/response data.")
    reward = sample.get("reward", sample.get("score"))
    return {
        "prompt": prompt,
        "completion": completion,
        "reward": None if reward is None else float(reward),
        "reward_context": sample.get(
            "reward_context",
            sample.get("answer", sample.get("response", completion)),
        ),
    }


def _explicit_adapter_mode(mode: str) -> str:
    if mode not in SUPPORTED_RL_DATASET_MODES:
        raise ValueError(
            f"Unsupported RL dataset mode '{mode}'. Supported modes: {SUPPORTED_RL_DATASET_MODES}"
        )
    return mode


def _candidate_modes(sample: Mapping[str, Any]) -> List[str]:
    if _is_message_sequence(sample.get("messages")):
        messages = list(sample.get("messages") or [])
        assistant_index = _last_assistant_index(messages)
        if "chosen" in sample and "rejected" in sample:
            if "chosen_score" in sample or "rejected_score" in sample:
                return ["reward_pairwise"]
            return ["preference", "reward_pairwise"]
        if "score" in sample or "reward" in sample:
            if assistant_index is not None:
                return ["reward_scalar"]
        if "completion" in sample or "response" in sample or "rewards" in sample:
            return ["rollout"]
        if assistant_index is None:
            return ["prompt"]
        return []

    keys = set(sample.keys())
    if {"chosen", "rejected"} <= keys:
        if "chosen_score" in keys or "rejected_score" in keys:
            return ["reward_pairwise"]
        return ["preference", "reward_pairwise"]
    if "text" in keys and "score" in keys:
        return ["reward_scalar"]
    if ("prompt" in keys or "question" in keys) and ("completion" in keys) and ("reward" in keys or "score" in keys):
        return ["rollout"]
    if ("prompt" in keys or "question" in keys) and ("response" in keys) and ("score" in keys):
        return ["reward_scalar"]
    if ("prompt" in keys or "question" in keys) and not (
        {"chosen", "rejected", "response", "completion", "score", "reward"} & keys
    ):
        return ["prompt"]
    return []


def prepare_rl_dataset(
    data: Iterable[Mapping[str, Any]] | PreparedRLDataset,
    mode: Optional[str] = None,
    tokenizer: Any = None,
    chat_template: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PreparedRLDataset:
    if isinstance(data, PreparedRLDataset):
        if mode is None or mode == data.mode:
            return data
        data = data.samples

    records = list(data)
    if not records:
        resolved_mode = _explicit_adapter_mode(mode) if mode is not None else "prompt"
        return PreparedRLDataset(
            samples=[],
            mode=resolved_mode if resolved_mode != "chat" else "prompt",
            adapter_name=resolved_mode,
            metadata=dict(metadata or {}),
        )

    requested_mode = _explicit_adapter_mode(mode) if mode is not None else None
    if requested_mode == "chat":
        candidate_sets = [_candidate_modes(record) for record in records]
        unique_modes = sorted({candidate for candidates in candidate_sets for candidate in candidates})
        if len(unique_modes) != 1:
            raise ValueError(
                "Chat auto-adaptation is ambiguous. Choose one explicit RL mode from "
                f"{SUPPORTED_RL_DATASET_MODES[:-1]}."
            )
        requested_mode = unique_modes[0]

    if requested_mode is None:
        candidate_sets = [_candidate_modes(record) for record in records]
        unique_modes = {candidate for candidates in candidate_sets for candidate in candidates}
        if not unique_modes:
            raise ValueError(
                "Could not auto-detect RL dataset mode. Supported modes: "
                f"{SUPPORTED_RL_DATASET_MODES[:-1]}."
            )
        if len(unique_modes) != 1:
            raise ValueError(
                "Ambiguous RL dataset schema. Choose an explicit mode from "
                f"{sorted(unique_modes)}."
            )
        requested_mode = next(iter(unique_modes))

    normalized: List[Dict[str, Any]] = []
    adapter_name = requested_mode
    for record in records:
        if requested_mode == "prompt":
            sample = _normalize_prompt_sample(record, tokenizer=tokenizer, chat_template=chat_template)
        elif requested_mode == "preference":
            sample = _normalize_preference_sample(
                record,
                tokenizer=tokenizer,
                chat_template=chat_template,
                target_mode="preference",
            )
        elif requested_mode == "reward_scalar":
            sample = _normalize_reward_scalar_sample(
                record,
                tokenizer=tokenizer,
                chat_template=chat_template,
            )
        elif requested_mode == "reward_pairwise":
            sample = _normalize_preference_sample(
                record,
                tokenizer=tokenizer,
                chat_template=chat_template,
                target_mode="reward_pairwise",
            )
        elif requested_mode == "rollout":
            sample = _normalize_rollout_sample(
                record,
                tokenizer=tokenizer,
                chat_template=chat_template,
            )
        else:
            raise ValueError(f"Unsupported RL dataset mode: {requested_mode}")

        if _is_message_sequence(record.get("messages")) or _is_message_sequence(record.get("chosen")):
            adapter_name = f"chat_{requested_mode}"
        normalized.append(sample)

    dataset_metadata = dict(metadata or {})
    dataset_metadata.update(
        {
            "requested_mode": mode,
            "num_samples": len(normalized),
        }
    )
    return PreparedRLDataset(
        samples=normalized,
        mode=requested_mode,
        adapter_name=adapter_name,
        metadata=dataset_metadata,
    )


def prepare_reward_dataset(dataset: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    warnings.warn(
        "prepare_reward_dataset() is deprecated; use prepare_rl_dataset(..., mode='reward_scalar' or 'reward_pairwise') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    records = list(dataset)
    if not records:
        return []
    first = records[0]
    mode = "reward_pairwise" if {"chosen", "rejected"} <= set(first.keys()) else "reward_scalar"
    return list(prepare_rl_dataset(records, mode=mode))


def prepare_preference_dataset(
    dataset: Iterable[Mapping[str, Any]],
    tokenizer: Any = None,
    format_type: str = "dpo",
) -> List[Dict[str, Any]]:
    warnings.warn(
        "prepare_preference_dataset() is deprecated; use prepare_rl_dataset(..., mode='preference' or 'prompt') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    mode = "prompt" if format_type == "grpo" else "preference"
    return list(prepare_rl_dataset(dataset, mode=mode, tokenizer=tokenizer))


def _simple_reward_builder(reward_type: str) -> Any:
    if reward_type == "simple":
        return lambda response, ground_truth: 1.0 if str(ground_truth).lower() in str(response).lower() else 0.0
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
            length = len(str(response).split())
            if length < 10:
                return 0.2
            if length < 50:
                return 0.5
            if length < 200:
                return 1.0
            return 0.8

        return length_reward
    raise ValueError(f"Unknown reward type: {reward_type}")


class _WeightedRewardComposer:
    def __init__(self, components: Sequence[Dict[str, Any]]):
        self.components = [dict(component) for component in components]
        self._running_stats: Dict[str, Dict[str, float]] = {}

    def _resolve_source(self, source: Any) -> Any:
        if isinstance(source, str):
            return _simple_reward_builder(source)
        return source

    def _normalize(self, name: str, value: float, mode: Any) -> float:
        if mode in (None, False, "none"):
            return value
        if callable(mode):
            return float(mode(value))
        if mode not in (True, "zscore"):
            raise ValueError(f"Unsupported reward normalization mode: {mode}")
        stats = self._running_stats.setdefault(name, {"count": 0.0, "mean": 0.0, "m2": 0.0})
        stats["count"] += 1.0
        delta = value - stats["mean"]
        stats["mean"] += delta / stats["count"]
        delta2 = value - stats["mean"]
        stats["m2"] += delta * delta2
        variance = stats["m2"] / max(stats["count"] - 1.0, 1.0)
        std = variance ** 0.5
        if std < 1e-6:
            return value - stats["mean"]
        return (value - stats["mean"]) / std

    def _evaluate_source(self, source: Any, payload: Dict[str, Any]) -> tuple[float, Dict[str, float], Optional[Dict[str, Any]]]:
        evaluator = self._resolve_source(source)
        if hasattr(evaluator, "evaluate"):
            result = evaluator.evaluate(payload)
        elif callable(evaluator):
            try:
                signature = inspect.signature(evaluator)
            except (TypeError, ValueError):
                signature = None

            if signature is None:
                result = evaluator(payload["completion_text"], payload["reward_context"])
            else:
                required = [
                    parameter
                    for parameter in signature.parameters.values()
                    if parameter.kind in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                    and parameter.default is inspect._empty
                ]
                if len(required) >= 2:
                    result = evaluator(payload["completion_text"], payload["reward_context"])
                else:
                    result = evaluator(payload)
        else:
            raise TypeError("Reward component source must be a string, callable, or evaluator object.")

        if isinstance(result, Mapping):
            reward = float(result.get("reward", result.get("score", 0.0)))
            return reward, dict(result.get("components") or {}), result.get("diagnostics")
        return float(result), {}, None

    def evaluate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        total = 0.0
        named_components: Dict[str, float] = {}
        diagnostics: Dict[str, Any] = {}
        for index, component in enumerate(self.components):
            name = str(component.get("name") or f"reward_{index}")
            weight = float(component.get("weight", 1.0))
            raw_reward, nested_components, component_diagnostics = self._evaluate_source(component.get("source"), payload)
            reward = self._normalize(name, raw_reward, component.get("normalize"))
            total += weight * reward
            named_components[name] = reward
            for nested_name, nested_value in nested_components.items():
                named_components[f"{name}.{nested_name}"] = float(nested_value)
            if component_diagnostics is not None:
                diagnostics[name] = component_diagnostics
        result: Dict[str, Any] = {
            "reward": total,
            "components": named_components,
        }
        if diagnostics:
            result["diagnostics"] = diagnostics
        return result


def create_reward_function(
    reward_type: Any = "simple",
    *,
    rewards: Optional[Sequence[Any]] = None,
) -> Any:
    if rewards is not None:
        components: List[Dict[str, Any]] = []
        for index, component in enumerate(rewards):
            if isinstance(component, Mapping):
                item = dict(component)
                if "source" not in item:
                    raise ValueError("Reward components must include a source field.")
            else:
                item = {"source": component}
            item.setdefault("name", f"reward_{index}")
            item.setdefault("weight", 1.0)
            components.append(item)
        return _WeightedRewardComposer(components)

    if isinstance(reward_type, Mapping):
        if "rewards" in reward_type:
            return create_reward_function(rewards=reward_type["rewards"])
        return create_reward_function(rewards=[reward_type])

    if isinstance(reward_type, (list, tuple)):
        return create_reward_function(rewards=list(reward_type))

    if callable(reward_type) or hasattr(reward_type, "evaluate"):
        return reward_type

    return _simple_reward_builder(str(reward_type))


__all__ = [
    "RLCheckpointBundle",
    "RLRoleState",
    "PreparedRLDataset",
    "SUPPORTED_RL_DATASET_MODES",
    "build_reference_policy",
    "build_reward_model",
    "create_reward_function",
    "prepare_preference_dataset",
    "prepare_reward_dataset",
    "prepare_rl_dataset",
    "resume_from_checkpoint",
]
