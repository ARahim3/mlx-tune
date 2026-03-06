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
