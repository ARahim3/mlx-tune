"""
Unit tests for mlx_tune.losses.
"""

import mlx.core as mx
import mlx.nn as nn


class TinyModel(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def __call__(self, x):
        return self.output(self.embedding(x))


class TinyTokenizer:
    eos_token_id = 1


class ConstantRewardModel:
    def score(self, input_ids, **kwargs):
        del input_ids, kwargs
        return mx.array([1.0, 2.0], dtype=mx.float32)


def test_compute_log_probs_with_lengths_shape():
    from mlx_tune.losses import compute_log_probs_with_lengths

    model = TinyModel()
    mx.eval(model.parameters())

    input_ids = mx.array([[1, 2, 3, 4], [1, 5, 6, 0]])
    lengths = mx.array([3, 2])

    log_probs = compute_log_probs_with_lengths(model, input_ids, lengths)
    assert log_probs.shape == (2,)


def test_precompute_preference_reference_logprobs_matches_direct():
    from mlx_tune.losses import (
        compute_reference_logprobs,
        precompute_preference_reference_logprobs,
    )

    model = TinyModel()
    mx.eval(model.parameters())

    chosen_ids = mx.array([[1, 2, 3, 4], [1, 4, 5, 6]])
    rejected_ids = mx.array([[1, 3, 2, 4], [1, 6, 5, 4]])
    chosen_lengths = mx.array([3, 3])
    rejected_lengths = mx.array([3, 3])

    direct = compute_reference_logprobs(
        model,
        chosen_ids,
        rejected_ids,
        chosen_lengths,
        rejected_lengths,
    )
    batched = precompute_preference_reference_logprobs(
        model,
        chosen_ids,
        rejected_ids,
        chosen_lengths,
        rejected_lengths,
        batch_size=1,
    )

    assert mx.allclose(direct[0], batched[0])
    assert mx.allclose(direct[1], batched[1])


def test_precompute_kto_reference_logprobs_matches_direct():
    from mlx_tune.losses import compute_log_probs_with_lengths, precompute_kto_reference_logprobs

    model = TinyModel()
    mx.eval(model.parameters())

    input_ids = mx.array([[1, 2, 3, 4], [1, 6, 7, 8]])
    lengths = mx.array([3, 3])

    direct = compute_log_probs_with_lengths(model, input_ids, lengths)
    cached = precompute_kto_reference_logprobs(model, input_ids, lengths, batch_size=1)

    assert mx.allclose(direct, cached)


def test_compute_completion_log_probs_masks_prompt_tokens():
    from mlx_tune.losses import compute_completion_log_probs

    model = TinyModel()
    mx.eval(model.parameters())

    input_ids = mx.array([[1, 2, 3, 4, 5]])
    prompt_lengths = mx.array([3])
    completion_lengths = mx.array([2])

    completion_only = compute_completion_log_probs(
        model,
        input_ids,
        prompt_lengths,
        completion_lengths,
    )

    first_completion = compute_completion_log_probs(
        model,
        input_ids,
        mx.array([4]),
        mx.array([1]),
    )

    assert completion_only.shape == (1,)
    assert not mx.allclose(completion_only, first_completion)


def test_grpo_recompute_loss_is_finite_and_trainable():
    from mlx_tune.losses import (
        compute_completion_log_probs,
        grpo_recompute_loss,
    )

    policy = TinyModel()
    reference = TinyModel()
    mx.eval(policy.parameters(), reference.parameters())

    input_ids = mx.array([[1, 2, 3, 4, 5], [1, 4, 3, 2, 1]])
    prompt_lengths = mx.array([3, 2])
    completion_lengths = mx.array([2, 3])
    rollout_logprobs = compute_completion_log_probs(
        policy,
        input_ids,
        prompt_lengths,
        completion_lengths,
    )
    advantages = mx.array([1.0, -0.5])

    loss, ntoks = grpo_recompute_loss(
        model=policy,
        reference_model=reference,
        input_ids=input_ids,
        prompt_lengths=prompt_lengths,
        completion_lengths=completion_lengths,
        rollout_logprobs=rollout_logprobs,
        advantages=advantages,
        beta=0.04,
    )

    mx.eval(loss, ntoks)
    assert loss.shape == ()
    assert ntoks.item() == 5


def test_grpo_recompute_kl_penalty_is_temperature_invariant_when_advantages_are_zero():
    from mlx_tune.losses import grpo_recompute_loss

    policy = TinyModel()
    reference = TinyModel()
    mx.eval(policy.parameters(), reference.parameters())

    input_ids = mx.array([[1, 2, 3, 4, 5], [1, 4, 3, 2, 1]])
    prompt_lengths = mx.array([3, 2])
    completion_lengths = mx.array([2, 3])
    rollout_logprobs = mx.array([0.0, 0.0], dtype=mx.float32)
    advantages = mx.array([0.0, 0.0], dtype=mx.float32)

    loss_at_one, _ = grpo_recompute_loss(
        model=policy,
        reference_model=reference,
        input_ids=input_ids,
        prompt_lengths=prompt_lengths,
        completion_lengths=completion_lengths,
        rollout_logprobs=rollout_logprobs,
        advantages=advantages,
        beta=0.04,
        temperature=1.0,
    )
    loss_at_point_seven, _ = grpo_recompute_loss(
        model=policy,
        reference_model=reference,
        input_ids=input_ids,
        prompt_lengths=prompt_lengths,
        completion_lengths=completion_lengths,
        rollout_logprobs=rollout_logprobs,
        advantages=advantages,
        beta=0.04,
        temperature=0.7,
    )

    assert mx.allclose(loss_at_one, loss_at_point_seven)


def test_ppo_kl_penalty_is_temperature_invariant_when_advantages_are_zero():
    from mlx_tune._rl_runtime import make_policy_eval_batch, score_policy
    from mlx_tune.losses import ppo_sequence_loss

    policy = TinyModel()
    mx.eval(policy.parameters())

    input_ids = [[1, 2, 3, 4, 5], [1, 4, 3, 2, 1]]
    prompt_lengths = [3, 2]
    completion_lengths = [2, 3]
    batch = make_policy_eval_batch(
        input_ids,
        pad_id=0,
        mode="completion",
        prompt_lengths=prompt_lengths,
        completion_lengths=completion_lengths,
        old_logprobs=mx.array([0.0, 0.0], dtype=mx.float32),
        advantages=mx.array([0.0, 0.0], dtype=mx.float32),
    )
    raw_scores = score_policy(policy, batch, mode="completion", temperature=1.0)
    batch.reference_logprobs = raw_scores.summed_logprobs

    loss_at_one, metrics_at_one = ppo_sequence_loss(
        model=policy,
        batch=batch,
        beta=0.04,
        temperature=1.0,
    )
    loss_at_point_seven, metrics_at_point_seven = ppo_sequence_loss(
        model=policy,
        batch=batch,
        beta=0.04,
        temperature=0.7,
    )

    assert mx.allclose(loss_at_one, loss_at_point_seven)
    assert mx.allclose(metrics_at_one["kl_penalty"], mx.zeros_like(metrics_at_one["kl_penalty"]))
    assert mx.allclose(
        metrics_at_point_seven["kl_penalty"],
        mx.zeros_like(metrics_at_point_seven["kl_penalty"]),
    )


def test_grpo_rollout_and_recompute_logprobs_match_with_temperature():
    from mlx_tune.losses import compute_completion_log_probs, generate_with_log_probs

    model = TinyModel()
    tokenizer = TinyTokenizer()
    mx.eval(model.parameters())
    mx.random.seed(21)

    prompt_ids = mx.array([2, 7, 8])
    generated_ids, rollout_token_logprobs = generate_with_log_probs(
        model,
        tokenizer,
        prompt_ids,
        max_tokens=3,
        temperature=0.7,
    )
    completion_ids = generated_ids[len(prompt_ids):].tolist()
    input_ids = mx.array([prompt_ids.tolist() + completion_ids])

    recomputed = compute_completion_log_probs(
        model,
        input_ids,
        mx.array([len(prompt_ids)]),
        mx.array([len(completion_ids)]),
        temperature=0.7,
    )

    assert mx.allclose(recomputed, mx.array([rollout_token_logprobs.sum()]))


def test_reward_model_regression_loss_returns_expected_predictions():
    from mlx_tune.losses import reward_model_regression_loss

    loss, predictions = reward_model_regression_loss(
        ConstantRewardModel(),
        input_ids=mx.array([[1, 2], [3, 4]], dtype=mx.int32),
        sequence_lengths=mx.array([2, 2], dtype=mx.int32),
        targets=mx.array([1.0, 2.0], dtype=mx.float32),
    )

    assert float(loss.item()) == 0.0
    assert predictions.tolist() == [1.0, 2.0]


def test_grpo_loss_routes_produce_distinct_losses_from_same_rollout():
    from mlx_tune._rl_runtime import make_policy_eval_batch, score_policy
    from mlx_tune.losses import grpo_recompute_loss

    policy = TinyModel()
    reference = TinyModel()
    mx.eval(policy.parameters(), reference.parameters())

    input_ids = mx.array([[1, 2, 3, 4, 5], [1, 4, 3, 2, 1]], dtype=mx.int32)
    prompt_lengths = mx.array([3, 2], dtype=mx.int32)
    completion_lengths = mx.array([2, 3], dtype=mx.int32)
    batch = make_policy_eval_batch(
        input_ids.tolist(),
        pad_id=0,
        mode="completion",
        prompt_lengths=prompt_lengths.tolist(),
        completion_lengths=completion_lengths.tolist(),
    )
    scored = score_policy(policy, batch, mode="completion")
    advantages = mx.array([1.0, -0.5], dtype=mx.float32)

    losses = {
        name: grpo_recompute_loss(
            model=policy,
            reference_model=reference,
            input_ids=input_ids,
            prompt_lengths=prompt_lengths,
            completion_lengths=completion_lengths,
            rollout_logprobs=scored.summed_logprobs,
            old_token_logprobs=scored.token_logprobs * batch.token_mask.astype(mx.float32),
            advantages=advantages,
            loss_type=name,
            max_completion_length=4,
        )[0]
        for name in ["grpo", "dapo", "dr_grpo", "gspo"]
    }

    unique_losses = {round(float(loss.item()), 6) for loss in losses.values()}
    assert len(unique_losses) >= 3
