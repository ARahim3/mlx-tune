"""
Compatibility patching for TRL/Unsloth-style imports.

This module exposes ``PatchFastRL`` so MLX-Tune can populate or mutate a
``trl`` module in-place with trainer/config shims backed by MLX-Tune's native
implementations.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import inspect
import sys
from types import ModuleType
from typing import Any, Dict, Mapping, Tuple, Type

from mlx_tune.rl_trainers import (
    DPOConfig as _DPOConfig,
    DPOTrainer as _DPOTrainer,
    GRPOConfig as _GRPOConfig,
    GRPOTrainer as _GRPOTrainer,
    KTOConfig as _KTOConfig,
    KTOTrainer as _KTOTrainer,
    OnlineDPOConfig as _OnlineDPOConfig,
    OnlineDPOTrainer as _OnlineDPOTrainer,
    ORPOConfig as _ORPOConfig,
    ORPOTrainer as _ORPOTrainer,
    PPOConfig as _PPOConfig,
    PPOTrainer as _PPOTrainer,
    RewardConfig as _RewardConfig,
    RewardTrainer as _RewardTrainer,
    SimPOConfig as _SimPOConfig,
    SimPOTrainer as _SimPOTrainer,
)
from mlx_tune.sft_trainer import SFTConfig as _SFTConfig
from mlx_tune.sft_trainer import SFTTrainer as _SFTTrainer


def _normalize_alias_kwargs(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(kwargs)

    if "processing_class" in normalized:
        normalized.setdefault("tokenizer", normalized["processing_class"])
        normalized.pop("processing_class", None)

    if "reward_funcs" in normalized:
        normalized.setdefault("reward_sources", normalized["reward_funcs"])
        normalized.pop("reward_funcs", None)

    if "reward_func" in normalized:
        normalized.setdefault("reward_fn", normalized["reward_func"])
        normalized.pop("reward_func", None)

    return normalized


def _extract_public_attrs(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return dict(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return dict(asdict(value))
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_") and not callable(item)
        }

    public: Dict[str, Any] = {}
    for key in dir(value):
        if key.startswith("_"):
            continue
        try:
            item = getattr(value, key)
        except Exception:
            continue
        if callable(item):
            continue
        public[key] = item
    return public


def _config_field_names(config_class: Type[Any]) -> set[str]:
    parameters = inspect.signature(config_class.__init__).parameters
    return {
        name
        for name, parameter in parameters.items()
        if name != "self" and parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }


def _trainer_param_names(trainer_class: Type[Any]) -> set[str]:
    parameters = inspect.signature(trainer_class.__init__).parameters
    return {
        name
        for name, parameter in parameters.items()
        if name != "self" and parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }


def _coerce_config(
    config_value: Any,
    config_class: Type[Any],
    config_overrides: Mapping[str, Any] | None = None,
) -> Any:
    if isinstance(config_value, config_class):
        if not config_overrides:
            return config_value
        payload = _extract_public_attrs(config_value)
    else:
        payload = _extract_public_attrs(config_value)

    if config_overrides:
        payload.update(config_overrides)
    return config_class(**_normalize_alias_kwargs(payload))


def _prepare_trainer_kwargs(
    kwargs: Mapping[str, Any],
    trainer_class: Type[Any],
    config_class: Type[Any],
) -> Dict[str, Any]:
    normalized = _normalize_alias_kwargs(kwargs)
    trainer_params = _trainer_param_names(trainer_class)
    config_fields = _config_field_names(config_class)

    config_overrides: Dict[str, Any] = {}
    for key in list(normalized):
        if key in {"args"} or key in trainer_params:
            continue
        if key in config_fields:
            config_overrides[key] = normalized.pop(key)

    if "args" not in normalized:
        if config_overrides:
            normalized["args"] = config_class(**config_overrides)
        return normalized

    args_value = normalized.get("args")
    if args_value is None:
        if config_overrides:
            normalized["args"] = config_class(**config_overrides)
        return normalized

    normalized["args"] = _coerce_config(args_value, config_class, config_overrides=config_overrides)
    return normalized


def _build_compat_config_class(export_name: str, config_class: Type[Any]) -> Type[Any]:
    class CompatConfig(config_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **_normalize_alias_kwargs(kwargs))

    CompatConfig.__name__ = export_name
    CompatConfig.__qualname__ = export_name
    CompatConfig.__module__ = "trl"
    CompatConfig.__doc__ = config_class.__doc__
    return CompatConfig


def _build_compat_trainer_class(
    export_name: str,
    trainer_class: Type[Any],
    config_class: Type[Any],
) -> Type[Any]:
    class CompatTrainer(trainer_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            prepared_kwargs = _prepare_trainer_kwargs(kwargs, trainer_class, config_class)
            super().__init__(*args, **prepared_kwargs)

    CompatTrainer.__name__ = export_name
    CompatTrainer.__qualname__ = export_name
    CompatTrainer.__module__ = "trl"
    CompatTrainer.__doc__ = trainer_class.__doc__
    return CompatTrainer


_COMPAT_PAIRS: Tuple[Tuple[str, Type[Any], Type[Any]], ...] = (
    ("SFT", _SFTTrainer, _SFTConfig),
    ("Reward", _RewardTrainer, _RewardConfig),
    ("DPO", _DPOTrainer, _DPOConfig),
    ("ORPO", _ORPOTrainer, _ORPOConfig),
    ("GRPO", _GRPOTrainer, _GRPOConfig),
    ("PPO", _PPOTrainer, _PPOConfig),
    ("OnlineDPO", _OnlineDPOTrainer, _OnlineDPOConfig),
    ("KTO", _KTOTrainer, _KTOConfig),
    ("SimPO", _SimPOTrainer, _SimPOConfig),
)

_COMPAT_CLASSES: Dict[str, Type[Any]] = {}
for stem, trainer_class, config_class in _COMPAT_PAIRS:
    compat_config = _build_compat_config_class(f"{stem}Config", config_class)
    compat_trainer = _build_compat_trainer_class(f"{stem}Trainer", trainer_class, config_class)
    _COMPAT_CLASSES[compat_config.__name__] = compat_config
    _COMPAT_CLASSES[compat_trainer.__name__] = compat_trainer

_COMPAT_EXPORT_NAMES = tuple(_COMPAT_CLASSES)


def _ensure_trl_modules() -> tuple[ModuleType, ModuleType]:
    trl_module = sys.modules.get("trl")
    if trl_module is None:
        trl_module = ModuleType("trl")
        trl_module.__package__ = "trl"
        trl_module.__path__ = []
        trl_module.__version__ = "0.0.0-mlx-tune"
        sys.modules["trl"] = trl_module
    else:
        if not hasattr(trl_module, "__package__"):
            trl_module.__package__ = "trl"
        if not hasattr(trl_module, "__path__"):
            trl_module.__path__ = []
        if not hasattr(trl_module, "__version__"):
            trl_module.__version__ = "0.0.0-mlx-tune"

    trainer_module = sys.modules.get("trl.trainer")
    if trainer_module is None:
        trainer_module = ModuleType("trl.trainer")
        trainer_module.__package__ = "trl"
        sys.modules["trl.trainer"] = trainer_module
    else:
        if not hasattr(trainer_module, "__package__"):
            trainer_module.__package__ = "trl"

    trl_module.trainer = trainer_module
    return trl_module, trainer_module


def PatchFastRL(algorithm: Any = None, FastLanguageModel: Any = None) -> None:
    """
    Patch ``trl`` so top-level trainer/config imports resolve to MLX-Tune.

    ``algorithm`` and ``FastLanguageModel`` are accepted for source compatibility
    with Unsloth's ``PatchFastRL`` signature.
    """
    del algorithm, FastLanguageModel

    trl_module, trainer_module = _ensure_trl_modules()
    for export_name, export_value in _COMPAT_CLASSES.items():
        setattr(trl_module, export_name, export_value)
        setattr(trainer_module, export_name, export_value)

    trl_module.__all__ = list(_COMPAT_EXPORT_NAMES)
    trainer_module.__all__ = list(_COMPAT_EXPORT_NAMES)
    trl_module.__MLX_TUNE_PATCHED__ = True
    trainer_module.__MLX_TUNE_PATCHED__ = True

