"""Internal performance helpers shared across custom trainers.

Keep this module dependency-light — it's imported by every trainer.
"""

import os
from functools import partial
from typing import Any, Callable, List, Optional

import mlx.core as mx


# Tracks layer classes already wrapped with `mx.checkpoint` via
# `enable_grad_checkpoint`. mlx-lm's `grad_checkpoint(layer)` rewrites
# `type(layer).__call__`, so calling it twice on the same class would nest
# checkpointing (recompute twice in backward — wrong). This set makes the
# wiring idempotent within a single Python process.
_GC_APPLIED_CLASSES: set = set()


def _compile_globally_disabled() -> bool:
    """Honour `MLX_TUNE_DISABLE_COMPILE=1` as a kill switch.

    Useful for A/B benchmarking, debugging recompile thrash, or working around
    a compile issue without code changes.
    """
    val = os.environ.get("MLX_TUNE_DISABLE_COMPILE", "")
    return val.strip().lower() in ("1", "true", "yes", "on")


def configure_wired_limit() -> None:
    """Mirror mlx-lm's training-time wired-memory configuration.

    `mx.set_wired_limit(max_recommended_working_set_size)` lets MLX hold more
    of the model in non-pageable memory, which is meaningful when the model is
    near the working-set boundary on Apple Silicon. The CUDA backend ignores
    this — guard with `mx.metal.is_available()`.

    Idempotent and safe to call repeatedly. Failures (e.g. running on a
    non-Metal backend, or older MLX without the API) are silenced.

    Source: `.venv/.../mlx_lm/tuner/trainer.py:214-215`.
    """
    try:
        if not mx.metal.is_available():
            return
        wired = mx.device_info().get("max_recommended_working_set_size")
        if wired is None:
            return
        if hasattr(mx, "set_wired_limit"):
            mx.set_wired_limit(wired)
        elif hasattr(mx.metal, "set_wired_limit"):
            mx.metal.set_wired_limit(wired)
    except Exception:
        pass


def bucket_length(max_len: int, bucket_size: int = 64) -> int:
    """Round `max_len` up to the next multiple of `bucket_size`.

    Used by trainers that pad batches to a per-batch max length. Without
    bucketing, consecutive batches almost always have different lengths, so
    `mx.compile`'s per-(shape, dtype) cache thrashes — each batch pays a
    ~0.4 s recompile. With `bucket_size=64`, a typical training run only sees
    a handful of unique shapes (max_seq_length / 64 of them), so the cache
    fills within a few steps and steady-state speedup kicks in quickly.

    Mirrors mlx-lm's `iterate_batches` step-padding (default 64). Honour
    `MLX_TUNE_BUCKET_SIZE` to override at runtime.
    """
    env = os.environ.get("MLX_TUNE_BUCKET_SIZE")
    if env:
        try:
            bucket_size = max(1, int(env))
        except ValueError:
            pass
    if bucket_size <= 1:
        return max_len
    return ((int(max_len) + bucket_size - 1) // bucket_size) * bucket_size


def make_step_state(model, optimizer) -> List:
    """Build the state list for `@mx.compile(inputs=state, outputs=state)`.

    Captures model parameters, optimizer state, and the global random state so
    a compiled step picks up parameter updates and is reproducible. This is the
    same triple mlx-lm uses (`mlx_lm/tuner/trainer.py:232`).
    """
    return [model.state, optimizer.state, mx.random.state]


def enable_grad_checkpoint(model_wrapper: Any, *, verbose: bool = True) -> bool:
    """Wire gradient checkpointing into the model's transformer block class.

    Reads `use_gradient_checkpointing` from `model_wrapper.lora_config`.
    When set to `True` or `"unsloth"`, wraps `type(block).__call__` with
    `mx.checkpoint` via mlx-lm's helper, which trades a 2× recompute cost in
    the backward pass for a roughly 2× cut in activation memory.

    This propagation is the only way `use_gradient_checkpointing` reaches
    non-SFT trainers. SFT inherits GC through mlx-lm's `TrainingArgs`, but
    ORPO/DPO/KTO/SimPO/GRPO/CPT/Embedding/VLM/TTS/STT/OCR call `model(...)`
    directly inside their loss functions and never see that flag — so they
    needed an explicit wiring step. Without it, ORPO at ctx=4096 OOMs on
    M4 Pro 48GB even though chosen+rejected double-forward only needs
    ~9 GB peak with GC enabled.

    Idempotent: tracks which layer classes have been wrapped this process
    so repeated calls (e.g. across multiple trainer instantiations) don't
    nest checkpointing.

    Args:
        model_wrapper: The user-facing model wrapper (e.g. `MLXModelWrapper`)
            with a `.lora_config` dict and a `.model` attribute pointing to
            the underlying mlx-lm `Model`.
        verbose: Print a one-line confirmation on first wire.

    Returns:
        True if GC was enabled (either now or already), False if the config
        didn't request it or no `.layers` attribute was found.
    """
    lora_config = getattr(model_wrapper, "lora_config", None)
    if not lora_config:
        return False
    flag = lora_config.get("use_gradient_checkpointing", False)
    if flag is not True and flag != "unsloth":
        return False

    try:
        from mlx_lm.tuner.trainer import grad_checkpoint as _grad_checkpoint
    except ImportError:
        return False

    target = getattr(model_wrapper, "model", model_wrapper)

    # Collect candidate `.layers` locations. Two structural patterns covered:
    #   - LLM/Embedding/TTS/STT: `target.layers` or `target.model.layers`
    #     (walk up to 3 levels of `.model`).
    #   - VLM/OCR: `target.language_model.layers` (mlx-vlm models nest the
    #     LM under `.language_model`; the outer model has no `.layers`).
    # We GC the LM blocks because they dominate activation memory; vision
    # encoder layers are much smaller and not worth the recompute cost.
    candidates: list = []
    current = target
    for _ in range(3):
        if hasattr(current, "layers"):
            candidates.append(("base", current.layers))
            break
        current = getattr(current, "model", None)
        if current is None:
            break
    lm = getattr(target, "language_model", None)
    if lm is not None and hasattr(lm, "layers"):
        candidates.append(("language_model", lm.layers))

    if not candidates:
        return False

    wrapped: list = []
    for name, layers in candidates:
        try:
            first = layers[0]
        except (TypeError, IndexError, KeyError):
            continue
        cls = type(first)
        if cls in _GC_APPLIED_CLASSES:
            continue
        _grad_checkpoint(first)
        _GC_APPLIED_CLASSES.add(cls)
        n = len(layers) if hasattr(layers, "__len__") else "?"
        wrapped.append(f"{name}:{cls.__name__}×{n}")

    if verbose and wrapped:
        print(f"[grad_checkpoint] enabled for {', '.join(wrapped)}", flush=True)
    return True


def compiled_step(
    step_fn: Callable,
    state: List,
    *,
    enabled: bool = True,
    shapeless: bool = False,
) -> Callable:
    """Wrap a training step function with `@mx.compile(inputs=state, outputs=state)`.

    Mirrors mlx-lm's pattern from `mlx_lm/tuner/trainer.py:234`. Returns a
    compiled callable when `enabled=True` and `mx.compile` accepts the
    function; falls back to the eager `step_fn` otherwise so a compile
    failure never breaks training.

    Args:
        step_fn: Pure function performing one training step
            (loss + grads + `optimizer.update`). Must use only mlx ops on
            its arguments and the captured `model`/`optimizer` to be safely
            traceable.
        state: List returned by `make_step_state(model, optimizer)`.
        enabled: Set False to bypass compilation (useful for debugging or for
            architectures that thrash). When False, returns `step_fn` as-is.
        shapeless: Pass through to `mx.compile`. Most of our losses use
            dynamic slicing (`arr[:, 1:]`) which `shapeless=True` rejects
            with `Primitive::output_shapes` errors; for those, leave at the
            default and accept per-shape retracing. Only set True if the
            step function operates on already-bucketed fixed-shape inputs.

    Returns:
        A callable with the same signature as `step_fn`.
    """
    if not enabled or _compile_globally_disabled():
        return step_fn
    try:
        if shapeless:
            return mx.compile(step_fn, inputs=state, outputs=state, shapeless=True)
        return mx.compile(step_fn, inputs=state, outputs=state)
    except Exception:
        # Older MLX without `shapeless=` kwarg, or compile rejects this fn
        # upfront — fall back to eager rather than crashing the trainer.
        try:
            return mx.compile(step_fn, inputs=state, outputs=state)
        except Exception:
            return step_fn
