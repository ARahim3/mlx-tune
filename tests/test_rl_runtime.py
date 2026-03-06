import mlx.core as mx
import mlx.nn as nn


class TinyModel(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def __call__(self, x):
        return self.output(self.embedding(x))


class ScriptedModel(nn.Module):
    def __init__(self, next_tokens, vocab_size: int = 16):
        super().__init__()
        self.next_tokens = dict(next_tokens)
        self.vocab_size = vocab_size

    def __call__(self, x):
        batch, seq_len = x.shape
        logits = mx.full((batch, seq_len, self.vocab_size), -100.0)
        token_id = self.next_tokens.get(seq_len, self.next_tokens.get("default", 0))
        logits[:, -1, token_id] = 100.0
        return logits


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True):
        ids = [((ord(char) % 10) + 3) for char in text]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids, skip_special_tokens: bool = True):
        if skip_special_tokens:
            ids = [token for token in ids if token not in (self.pad_token_id, self.eos_token_id, self.bos_token_id)]
        return "".join(chr(65 + (token % 26)) for token in ids)


def test_collect_rollouts_returns_metadata_and_truncates_prompt_left():
    from mlx_tune._rl_runtime import collect_rollouts

    tokenizer = TinyTokenizer()
    model = ScriptedModel({2: 5, 3: 1, "default": 1})

    rollout = collect_rollouts(
        policy=model,
        tokenizer=tokenizer,
        prompt_samples=[
            {
                "sample_index": 7,
                "prompt": "abcdef",
                "prompt_ids": [3, 4, 5, 6, 7, 8],
                "reward_context": "ctx",
            }
        ],
        sampling_config={
            "num_generations": 2,
            "temperature": 0.0,
            "max_completion_length": 4,
            "max_seq_length": 6,
        },
    )

    assert rollout.prompt_ids == [[7, 8], [7, 8]]
    assert rollout.prompt_lengths.tolist() == [2, 2]
    assert rollout.prompt_texts == [tokenizer.decode([7, 8]), tokenizer.decode([7, 8])]
    assert rollout.original_prompt_texts == ["abcdef", "abcdef"]
    assert rollout.completion_ids == [[5, 1], [5, 1]]
    assert rollout.completion_lengths.tolist() == [2, 2]
    assert rollout.sampled_token_logprobs.shape == (2, 2)
    assert rollout.eos_flags.tolist() == [True, True]
    assert rollout.truncation_flags.tolist() == [False, False]
    assert rollout.prompt_group_indices.tolist() == [0, 0]
    assert rollout.sample_indices.tolist() == [7, 7]
    assert rollout.policy_eval.input_ids.shape == (2, 4)
    assert mx.allclose(
        rollout.rollout_logprobs,
        rollout.sampled_token_logprobs.sum(axis=-1),
    )


def test_collect_rollouts_can_capture_sample_stats_and_truncation():
    from mlx_tune._rl_runtime import collect_rollouts

    tokenizer = TinyTokenizer()
    model = ScriptedModel({2: 5, 3: 6, 4: 7, "default": 7})

    rollout = collect_rollouts(
        policy=model,
        tokenizer=tokenizer,
        prompt_samples=[
            {
                "sample_index": 1,
                "prompt": "ab",
                "prompt_ids": [3, 4],
                "reward_context": "ctx",
            }
        ],
        sampling_config={
            "num_generations": 1,
            "temperature": 0.0,
            "max_completion_length": 2,
            "max_seq_length": 8,
        },
        collect_sample_stats=True,
    )

    assert rollout.completion_ids == [[5, 6]]
    assert rollout.eos_flags.tolist() == [False]
    assert rollout.truncation_flags.tolist() == [True]
    assert rollout.sampled_token_logits.shape == (1, 2)
    assert rollout.token_entropies.shape == (1, 2)


def test_score_policy_matches_public_logprob_helpers():
    from mlx_tune._rl_runtime import make_policy_eval_batch, score_policy
    from mlx_tune.losses import compute_completion_log_probs, compute_log_probs_with_lengths

    model = TinyModel()
    mx.eval(model.parameters())

    sequence_batch = make_policy_eval_batch(
        [[1, 2, 3, 4], [1, 5, 6]],
        pad_id=0,
        mode="sequence",
    )
    sequence_scores = score_policy(model, sequence_batch, mode="sequence")
    direct_sequence = compute_log_probs_with_lengths(
        model,
        sequence_batch.input_ids,
        sequence_batch.sequence_lengths,
    )

    completion_batch = make_policy_eval_batch(
        [[1, 2, 3, 4, 5], [1, 4, 3, 2]],
        pad_id=0,
        mode="completion",
        prompt_lengths=[3, 2],
        completion_lengths=[2, 2],
    )
    completion_scores = score_policy(model, completion_batch, mode="completion")
    direct_completion = compute_completion_log_probs(
        model,
        completion_batch.input_ids,
        completion_batch.prompt_lengths,
        completion_batch.completion_lengths,
    )

    assert mx.allclose(sequence_scores.summed_logprobs, direct_sequence)
    assert mx.allclose(completion_scores.summed_logprobs, direct_completion)


def test_reference_precompute_helpers_match_runtime_scorer():
    from mlx_tune._rl_runtime import make_policy_eval_batch, score_policy_in_chunks
    from mlx_tune.losses import (
        precompute_kto_reference_logprobs,
        precompute_preference_reference_logprobs,
    )

    model = TinyModel()
    mx.eval(model.parameters())

    chosen = make_policy_eval_batch([[1, 2, 3, 4], [1, 4, 5]], pad_id=0, mode="sequence")
    rejected = make_policy_eval_batch([[1, 3, 2, 4], [1, 6, 5]], pad_id=0, mode="sequence")
    direct_chosen = score_policy_in_chunks(model, chosen, batch_size=1, mode="sequence").summed_logprobs
    direct_rejected = score_policy_in_chunks(model, rejected, batch_size=1, mode="sequence").summed_logprobs
    cached_chosen, cached_rejected = precompute_preference_reference_logprobs(
        model,
        chosen.input_ids,
        rejected.input_ids,
        chosen.sequence_lengths,
        rejected.sequence_lengths,
        batch_size=1,
    )

    kto_batch = make_policy_eval_batch([[1, 2, 3], [1, 4, 5, 6]], pad_id=0, mode="sequence")
    direct_kto = score_policy_in_chunks(model, kto_batch, batch_size=1, mode="sequence").summed_logprobs
    cached_kto = precompute_kto_reference_logprobs(
        model,
        kto_batch.input_ids,
        kto_batch.sequence_lengths,
        batch_size=1,
    )

    assert mx.allclose(direct_chosen, cached_chosen)
    assert mx.allclose(direct_rejected, cached_rejected)
    assert mx.allclose(direct_kto, cached_kto)


def test_reward_adapter_supports_legacy_and_structured_evaluators():
    from mlx_tune._rl_runtime import collect_rollouts, evaluate_rewards

    tokenizer = TinyTokenizer()
    model = ScriptedModel({2: 5, 3: 1, "default": 1})
    rollout = collect_rollouts(
        policy=model,
        tokenizer=tokenizer,
        prompt_samples=[
            {
                "sample_index": 0,
                "prompt": "ab",
                "prompt_ids": [3, 4],
                "reward_context": "ctx",
            }
        ],
        sampling_config={"num_generations": 1, "temperature": 0.0, "max_completion_length": 2},
    )

    legacy = evaluate_rewards(
        rollout,
        lambda response, context: float(len(response) + len(context)),
    )

    seen_payloads = []

    class StructuredEvaluator:
        def evaluate(self, payload):
            seen_payloads.append(payload)
            return {
                "reward": float(payload["completion_length"]),
                "components": {"length": float(payload["completion_length"])},
                "diagnostics": {"used_context": payload["reward_context"]},
            }

    structured = evaluate_rewards(rollout, StructuredEvaluator())

    assert legacy.scalar_rewards.tolist() == [float(len(rollout.completion_texts[0]) + 3)]
    assert structured.scalar_rewards.tolist() == [2.0]
    assert structured.named_reward_components == [{"length": 2.0}]
    assert structured.diagnostics == [{"used_context": "ctx"}]
    assert seen_payloads[0]["prompt_text"] == tokenizer.decode([3, 4])
    assert seen_payloads[0]["original_prompt_text"] == "ab"


def test_reward_payload_exposes_effective_and_original_prompt_when_truncated():
    from mlx_tune._rl_runtime import collect_rollouts, evaluate_rewards

    tokenizer = TinyTokenizer()
    model = ScriptedModel({2: 5, 3: 1, "default": 1})
    captured = []

    rollout = collect_rollouts(
        policy=model,
        tokenizer=tokenizer,
        prompt_samples=[
            {
                "sample_index": 0,
                "prompt": "abcdef",
                "prompt_ids": [3, 4, 5, 6, 7, 8],
                "reward_context": "ctx",
            }
        ],
        sampling_config={
            "num_generations": 1,
            "temperature": 0.0,
            "max_completion_length": 2,
            "max_seq_length": 4,
        },
    )

    evaluate_rewards(
        rollout,
        lambda payload: captured.append(payload) or float(payload["completion_length"]),
    )

    assert captured[0]["prompt_ids"] == [7, 8]
    assert captured[0]["prompt_text"] == tokenizer.decode([7, 8])
    assert captured[0]["original_prompt_text"] == "abcdef"


def test_compute_advantages_uses_zero_variance_fallback_per_prompt():
    from mlx_tune._rl_runtime import RewardBatch, compute_advantages

    reward_batch = RewardBatch(
        prompt_texts=["p0", "p0", "p1", "p1"],
        completion_texts=["a", "b", "c", "d"],
        reward_contexts=["c0", "c0", "c1", "c1"],
        scalar_rewards=mx.array([2.0, 2.0, 1.0, 3.0], dtype=mx.float32),
        prompt_group_indices=mx.array([0, 0, 1, 1]),
    )

    advantages = compute_advantages(reward_batch)

    assert mx.allclose(advantages, mx.array([0.0, 0.0, -1.0, 1.0], dtype=mx.float32))


def test_assemble_minibatches_preserves_order_and_prompt_groups():
    from mlx_tune._rl_runtime import assemble_minibatches, make_policy_eval_batch

    batch = make_policy_eval_batch(
        [[1, 2, 3], [1, 4, 5], [1, 6, 7]],
        pad_id=0,
        mode="sequence",
        rollout_logprobs=mx.array([0.1, 0.2, 0.3], dtype=mx.float32),
        advantages=mx.array([1.0, 2.0, 3.0], dtype=mx.float32),
        prompt_group_indices=mx.array([9, 9, 10]),
        sample_indices=mx.array([0, 1, 2]),
    )

    minibatches = list(assemble_minibatches(batch, minibatch_size=2, shuffle=False))

    assert len(minibatches) == 2
    assert minibatches[0].input_ids.tolist() == [[1, 2, 3], [1, 4, 5]]
    assert minibatches[0].prompt_group_indices.tolist() == [9, 9]
    assert minibatches[1].input_ids.tolist() == [[1, 6, 7]]
    assert minibatches[1].prompt_group_indices.tolist() == [10]


def test_length_normalization_and_kl_helpers_match_expected_math():
    from mlx_tune._rl_runtime import kl_against_reference, normalize_logprobs

    summed = mx.array([4.0, 6.0], dtype=mx.float32)
    lengths = mx.array([2, 3])
    normalized = normalize_logprobs(summed, lengths, mode="mean")
    kl = kl_against_reference(
        mx.array([0.0, 1.0], dtype=mx.float32),
        mx.array([0.0, 0.5], dtype=mx.float32),
    )

    assert mx.allclose(normalized, mx.array([2.0, 2.0], dtype=mx.float32))
    assert mx.allclose(
        kl,
        mx.array([0.0, float((mx.exp(mx.array(0.5)) - 0.5 - 1.0).item())], dtype=mx.float32),
    )
