from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_tune import (
    build_reference_policy,
    build_reward_model,
    build_value_model,
    create_rl_model_roles,
    pairwise_ranking_accuracy,
    reward_model_pairwise_loss,
    scalar_loss_metrics,
    value_model_regression_loss,
)
from mlx_tune.model import MLXModelWrapper


class MockTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True):
        ids = [((ord(char) % 20) + 3) for char in text[:16]]
        if add_special_tokens:
            return [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids, skip_special_tokens: bool = True):
        if skip_special_tokens:
            ids = [token for token in ids if token not in (0, 1, 2)]
        return "".join(chr(65 + (token % 26)) for token in ids)


class DeterministicBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 2

    def __call__(self, x):
        values = x.astype(mx.float32)
        return mx.stack([values, values * 10.0], axis=-1)


class DeterministicCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 64):
        super().__init__()
        self.model = DeterministicBackbone()
        self.output = nn.Linear(2, vocab_size)

    def __call__(self, x):
        return self.output(self.model(x))


def make_wrapper() -> MLXModelWrapper:
    model = DeterministicCausalLM()
    mx.eval(model.parameters())
    wrapper = MLXModelWrapper(
        model=model,
        tokenizer=MockTokenizer(),
        max_seq_length=32,
        model_name="deterministic-model",
    )
    return wrapper


def _set_scalar_head_to_first_feature(role_model) -> None:
    role_model.head.update(
        {
            "weight": mx.array([[1.0, 0.0]], dtype=mx.float32),
            "bias": mx.array([0.0], dtype=mx.float32),
        },
        strict=False,
    )
    mx.eval(role_model.head.parameters())


def _parameter_snapshot(model) -> dict[str, mx.array]:
    actual_model = model.model if hasattr(model, "model") else model
    return {name: mx.array(value) for name, value in tree_flatten(actual_model.parameters())}


def _parameters_match(before: dict[str, mx.array], after_model) -> bool:
    actual_model = after_model.model if hasattr(after_model, "model") else after_model
    after = {name: value for name, value in tree_flatten(actual_model.parameters())}
    return all(mx.allclose(before[name], after[name]) for name in before)


def test_reference_policy_clone_is_frozen_and_isolated():
    policy = make_wrapper()
    policy.lora_enabled = True
    policy._lora_applied = True
    policy.set_adapter_path("/tmp/live-policy")

    roles = create_rl_model_roles(policy)
    reference = roles.reference_policy

    assert reference.model is not policy
    assert reference.model.get_adapter_path() is None
    assert not tree_flatten(reference.model.model.trainable_parameters())

    reference_before = _parameter_snapshot(reference.model)
    policy.model.output.update(
        {
            "weight": mx.ones_like(policy.model.output.weight),
            "bias": mx.zeros_like(policy.model.output.bias),
        },
        strict=False,
    )
    mx.eval(policy.model.parameters())

    assert _parameters_match(reference_before, reference.model)


def test_scalar_roles_default_to_last_completion_token_and_support_mean_pooling():
    base_model = make_wrapper()

    reward_model = build_reward_model(base_model)
    _set_scalar_head_to_first_feature(reward_model)

    input_ids = mx.array([[2, 5, 7, 9], [2, 4, 6, 8]], dtype=mx.int32)
    sequence_lengths = mx.array([4, 4], dtype=mx.int32)
    prompt_lengths = mx.array([2, 3], dtype=mx.int32)
    completion_lengths = mx.array([2, 1], dtype=mx.int32)

    reward_scores = reward_model.score(
        input_ids,
        sequence_lengths=sequence_lengths,
        prompt_lengths=prompt_lengths,
        completion_lengths=completion_lengths,
    )
    assert mx.allclose(reward_scores, mx.array([9.0, 8.0], dtype=mx.float32))

    mean_completion_model = build_value_model(base_model, pooling="mean_completion")
    _set_scalar_head_to_first_feature(mean_completion_model)
    mean_completion = mean_completion_model.predict(
        input_ids,
        sequence_lengths=sequence_lengths,
        prompt_lengths=prompt_lengths,
        completion_lengths=completion_lengths,
    )
    assert mx.allclose(mean_completion, mx.array([8.0, 8.0], dtype=mx.float32))

    mean_sequence_model = build_value_model(base_model, pooling="mean_sequence", target="sequence")
    _set_scalar_head_to_first_feature(mean_sequence_model)
    mean_sequence = mean_sequence_model.predict(
        input_ids,
        sequence_lengths=sequence_lengths,
        prompt_lengths=prompt_lengths,
        completion_lengths=completion_lengths,
    )
    assert mx.allclose(mean_sequence, mx.array([5.75, 5.0], dtype=mx.float32))


def test_scalar_role_save_load_round_trip_preserves_head_and_adapter_state(tmp_path):
    base_model = make_wrapper()
    base_model.lora_enabled = True
    base_model._lora_applied = True

    reward_model = build_reward_model(base_model)
    _set_scalar_head_to_first_feature(reward_model)
    reward_model.base_model.model.output.update(
        {
            "weight": mx.ones_like(reward_model.base_model.model.output.weight) * 0.5,
            "bias": mx.ones_like(reward_model.base_model.model.output.bias) * 0.25,
        },
        strict=False,
    )
    mx.eval(reward_model.base_model.model.parameters())

    output_dir = Path(tmp_path) / "reward_role"
    reward_model.save_pretrained(str(output_dir))

    restored = build_reward_model(make_wrapper())
    restored.load_pretrained(str(output_dir))

    score_inputs = mx.array([[2, 4, 6]], dtype=mx.int32)
    sequence_lengths = mx.array([3], dtype=mx.int32)
    prompt_lengths = mx.array([1], dtype=mx.int32)
    completion_lengths = mx.array([2], dtype=mx.int32)

    assert (output_dir / "head.safetensors").exists()
    assert (output_dir / "head_config.json").exists()
    assert (output_dir / "adapters.safetensors").exists()
    assert (output_dir / "adapter_config.json").exists()
    assert mx.allclose(
        reward_model.score(
            score_inputs,
            sequence_lengths=sequence_lengths,
            prompt_lengths=prompt_lengths,
            completion_lengths=completion_lengths,
        ),
        restored.score(
            score_inputs,
            sequence_lengths=sequence_lengths,
            prompt_lengths=prompt_lengths,
            completion_lengths=completion_lengths,
        ),
    )


def test_scalar_objective_helpers_return_stable_losses_and_metrics():
    reward_model = build_reward_model(make_wrapper())
    _set_scalar_head_to_first_feature(reward_model)

    chosen_input_ids = mx.array([[2, 4, 9], [2, 3, 8]], dtype=mx.int32)
    rejected_input_ids = mx.array([[2, 4, 5], [2, 3, 6]], dtype=mx.int32)
    chosen_lengths = mx.array([3, 3], dtype=mx.int32)
    rejected_lengths = mx.array([3, 3], dtype=mx.int32)
    prompt_lengths = mx.array([1, 1], dtype=mx.int32)
    completion_lengths = mx.array([2, 2], dtype=mx.int32)

    reward_loss, reward_outputs = reward_model_pairwise_loss(
        reward_model,
        chosen_input_ids=chosen_input_ids,
        rejected_input_ids=rejected_input_ids,
        chosen_sequence_lengths=chosen_lengths,
        rejected_sequence_lengths=rejected_lengths,
        chosen_prompt_lengths=prompt_lengths,
        rejected_prompt_lengths=prompt_lengths,
        chosen_completion_lengths=completion_lengths,
        rejected_completion_lengths=completion_lengths,
    )
    assert float(reward_loss.item()) > 0.0
    assert pairwise_ranking_accuracy(
        reward_outputs["chosen_scores"],
        reward_outputs["rejected_scores"],
    ) == 1.0

    value_model = build_value_model(make_wrapper())
    _set_scalar_head_to_first_feature(value_model)
    value_targets = mx.array([9.0, 8.0], dtype=mx.float32)
    value_loss, predictions = value_model_regression_loss(
        value_model,
        input_ids=chosen_input_ids,
        sequence_lengths=chosen_lengths,
        targets=value_targets,
        prompt_lengths=prompt_lengths,
        completion_lengths=completion_lengths,
    )
    metrics = scalar_loss_metrics(value_loss, predictions, value_targets)
    assert metrics["loss"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["mse"] == 0.0


def test_build_reference_policy_public_builder_returns_compat_wrapper():
    policy = make_wrapper()
    reference = build_reference_policy(policy)

    assert reference.source == "policy_snapshot"
    assert reference.metadata["snapshot_strategy"] == "clone_and_freeze"
