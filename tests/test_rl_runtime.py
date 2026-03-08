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


class CacheOnlyScriptedModel(nn.Module):
    def __init__(self, next_tokens, vocab_size: int = 16):
        super().__init__()
        self.next_tokens = dict(next_tokens)
        self.vocab_size = vocab_size
        self.calls = []

    def make_cache(self):
        return [{"steps": 0}]

    def __call__(self, x, cache=None):
        if cache is None:
            raise ValueError("cache is required")
        batch, seq_len = x.shape
        cache[0]["steps"] += 1
        self.calls.append((seq_len, cache[0]["steps"]))
        logits = mx.full((batch, seq_len, self.vocab_size), -100.0)
        token_id = self.next_tokens.get(cache[0]["steps"], self.next_tokens.get("default", 0))
        logits[:, -1, token_id] = 100.0
        return logits


class BatchSizedCacheScriptedModel(nn.Module):
    def __init__(self, vocab_size: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.calls = []

    def make_cache(self):
        return [{"steps": 0, "batch_size": None}]

    def __call__(self, x, cache=None):
        if cache is None:
            raise ValueError("cache is required")
        batch, seq_len = x.shape
        if cache[0]["batch_size"] is None:
            cache[0]["batch_size"] = batch
        elif cache[0]["batch_size"] != batch:
            raise ValueError("cache batch size mismatch")
        cache[0]["steps"] += 1
        self.calls.append((batch, seq_len, cache[0]["steps"]))
        logits = mx.full((batch, seq_len, self.vocab_size), -100.0)
        for row_index in range(batch):
            if cache[0]["steps"] == 1:
                token_id = 1 if row_index == 0 else 5
            elif cache[0]["steps"] == 2:
                token_id = 1
            else:
                token_id = 1
            logits[row_index, -1, token_id] = 100.0
        return logits


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True):
        if text == "<|im_end|>":
            return [9]
        ids = [((ord(char) % 10) + 3) for char in text]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids, skip_special_tokens: bool = True):
        if skip_special_tokens:
            ids = [token for token in ids if token not in (self.pad_token_id, self.eos_token_id, self.bos_token_id)]
        return "".join(chr(65 + (token % 26)) for token in ids)

    def convert_tokens_to_ids(self, token: str):
        if token == "<|im_end|>":
            return 9
        return None

    def get_vocab(self):
        return {"<|im_end|>": 9}


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


def test_collect_rollouts_respects_max_seq_length_during_generation():
    from mlx_tune._rl_runtime import collect_rollouts

    tokenizer = TinyTokenizer()
    model = ScriptedModel({2: 5, 3: 6, 4: 7, 5: 8, 6: 9, "default": 9})

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
            "max_completion_length": 5,
            "max_seq_length": 3,
        },
    )

    assert rollout.prompt_lengths.tolist() == [1]
    assert rollout.completion_lengths.tolist() == [2]
    assert rollout.policy_eval.input_ids.shape == (1, 3)
    assert rollout.truncation_flags.tolist() == [True]


def test_collect_rollouts_treats_unsloth_stop_token_as_terminal():
    from mlx_tune._rl_runtime import collect_rollouts, sample_completion

    tokenizer = TinyTokenizer()
    tokenizer._unsloth_stop_token = "<|im_end|>"
    model = ScriptedModel({2: 9, 3: 7, 4: 7, "default": 7})

    sampled = sample_completion(
        policy=model,
        tokenizer=tokenizer,
        prompt_ids=[3, 4],
        max_tokens=4,
        temperature=0.0,
    )
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
        sampling_config={
            "num_generations": 1,
            "temperature": 0.0,
            "max_completion_length": 4,
        },
    )

    assert sampled["completion_ids"] == [9]
    assert sampled["eos_flag"] is True
    assert sampled["truncation_flag"] is False
    assert rollout.completion_ids == [[9]]
    assert rollout.eos_flags.tolist() == [True]
    assert rollout.truncation_flags.tolist() == [False]


def test_collect_rollouts_initializes_and_reuses_prompt_cache_for_logits_only_models():
    from mlx_tune._rl_runtime import collect_rollouts

    tokenizer = TinyTokenizer()
    model = CacheOnlyScriptedModel({1: 5, 2: 1, "default": 1})

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
        sampling_config={
            "num_generations": 2,
            "temperature": 0.0,
            "max_completion_length": 4,
            "generation_batch_size": 2,
        },
    )

    assert rollout.completion_ids == [[5, 1], [5, 1]]
    assert rollout.eos_flags.tolist() == [True, True]
    assert model.calls == [(2, 1), (1, 2)]


def test_collect_rollouts_rebuilds_prompt_cache_when_active_batch_shrinks():
    from mlx_tune._rl_runtime import collect_rollouts

    tokenizer = TinyTokenizer()
    model = BatchSizedCacheScriptedModel()

    rollout = collect_rollouts(
        policy=model,
        tokenizer=tokenizer,
        prompt_samples=[
            {
                "sample_index": 0,
                "prompt": "ab",
                "prompt_ids": [3, 4],
                "reward_context": "ctx-a",
            },
            {
                "sample_index": 1,
                "prompt": "ac",
                "prompt_ids": [3, 5],
                "reward_context": "ctx-b",
            },
        ],
        sampling_config={
            "num_generations": 1,
            "temperature": 0.0,
            "max_completion_length": 3,
            "generation_batch_size": 2,
        },
    )

    assert rollout.completion_ids == [[1], [5, 1]]
    assert rollout.eos_flags.tolist() == [True, True]
    assert rollout.truncation_flags.tolist() == [False, False]
    assert model.calls == [(2, 2, 1), (1, 3, 1)]


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


def test_post_rollout_helpers_attach_reference_values_and_rloo_advantages():
    from mlx_tune._rl_runtime import (
        collect_rollouts,
        compute_returns_and_advantages,
        predict_rollout_values,
        rank_grouped_rollouts,
        score_rollout_references,
    )

    class ConstantValueModel:
        def predict(self, input_ids, **kwargs):
            del kwargs
            return mx.array([0.5] * input_ids.shape[0], dtype=mx.float32)

    tokenizer = TinyTokenizer()
    policy = ScriptedModel({2: 5, 3: 1, "default": 1})
    reference = ScriptedModel({2: 5, 3: 1, "default": 1})

    rollout = collect_rollouts(
        policy=policy,
        tokenizer=tokenizer,
        prompt_samples=[
            {"sample_index": 0, "prompt": "ab", "prompt_ids": [3, 4], "reward_context": "x"},
            {"sample_index": 1, "prompt": "cd", "prompt_ids": [5, 6], "reward_context": "y"},
        ],
        sampling_config={"num_generations": 2, "temperature": 0.0, "max_completion_length": 2},
    )
    rollout.rewards = mx.array([3.0, 1.0, 4.0, 2.0], dtype=mx.float32)

    rollout = score_rollout_references(reference, rollout, batch_size=1)
    rollout = predict_rollout_values(ConstantValueModel(), rollout, batch_size=2)
    returns, advantages = compute_returns_and_advantages(
        rollout.rewards,
        prompt_group_indices=rollout.prompt_group_indices,
        mode="rloo",
    )
    rollout.returns = returns
    rollout.advantages = advantages
    rankings = rank_grouped_rollouts(rollout)

    assert rollout.reference_logprobs.shape == (4,)
    assert rollout.value_predictions.tolist() == [0.5, 0.5, 0.5, 0.5]
    assert returns.tolist() == [3.0, 1.0, 4.0, 2.0]
    assert advantages.tolist() == [2.0, -2.0, 2.0, -2.0]
    assert rankings[0]["best_position"] == 0
    assert rankings[0]["worst_position"] == 1


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


def test_collect_rollouts_batched_decode_matches_per_sequence_sampler():
    from mlx_tune._rl_runtime import collect_rollouts, sample_completion

    tokenizer = TinyTokenizer()
    model = ScriptedModel({2: 5, 3: 6, 4: 1, "default": 1})
    prompt_samples = [
        {"sample_index": 0, "prompt": "ab", "prompt_ids": [3, 4], "reward_context": "x"},
        {"sample_index": 1, "prompt": "cd", "prompt_ids": [5, 6], "reward_context": "y"},
    ]

    expected = []
    for sample in prompt_samples:
        for _ in range(2):
            expected.append(
                sample_completion(
                    policy=model,
                    tokenizer=tokenizer,
                    prompt_ids=sample["prompt_ids"],
                    max_tokens=3,
                    temperature=0.0,
                    collect_sample_stats=True,
                )
            )

    rollout = collect_rollouts(
        policy=model,
        tokenizer=tokenizer,
        prompt_samples=prompt_samples,
        sampling_config={
            "num_generations": 2,
            "temperature": 0.0,
            "max_completion_length": 3,
            "generation_batch_size": 2,
        },
        collect_sample_stats=True,
    )

    assert rollout.completion_ids == [item["completion_ids"] for item in expected]
    assert rollout.eos_flags.tolist() == [item["eos_flag"] for item in expected]
    assert rollout.truncation_flags.tolist() == [item["truncation_flag"] for item in expected]
    assert mx.allclose(
        rollout.sampled_token_logprobs,
        mx.array([item["sampled_logprobs"] for item in expected], dtype=mx.float32),
    )
    assert mx.allclose(
        rollout.sampled_token_logits,
        mx.array([item["sampled_logits"] for item in expected], dtype=mx.float32),
    )
    assert mx.allclose(
        rollout.token_entropies,
        mx.array([item["token_entropies"] for item in expected], dtype=mx.float32),
    )


def test_reward_adapter_prefers_evaluate_batch_when_available():
    from mlx_tune._rl_runtime import collect_rollouts, evaluate_rewards

    tokenizer = TinyTokenizer()
    model = ScriptedModel({2: 5, 3: 1, "default": 1})
    rollout = collect_rollouts(
        policy=model,
        tokenizer=tokenizer,
        prompt_samples=[
            {"sample_index": 0, "prompt": "ab", "prompt_ids": [3, 4], "reward_context": "ctx0"},
            {"sample_index": 1, "prompt": "cd", "prompt_ids": [5, 6], "reward_context": "ctx1"},
        ],
        sampling_config={"num_generations": 1, "temperature": 0.0, "max_completion_length": 2},
    )

    calls = []

    class BatchEvaluator:
        def evaluate_batch(self, payloads):
            calls.append([payload["reward_context"] for payload in payloads])
            return [
                {"reward": float(payload["completion_length"]), "components": {"len": float(payload["completion_length"])}}
                for payload in payloads
            ]

    reward_batch = evaluate_rewards(rollout, BatchEvaluator())

    assert calls == [["ctx0", "ctx1"]]
    assert reward_batch.scalar_rewards.tolist() == [2.0, 2.0]
    assert reward_batch.named_reward_components == [{"len": 2.0}, {"len": 2.0}]


def test_token_budget_chunking_matches_unchunked_policy_and_value_scoring():
    from mlx_tune._rl_runtime import (
        collect_rollouts,
        make_policy_eval_batch,
        predict_rollout_values,
        score_policy_in_chunks,
    )

    class LengthValueModel:
        def predict(self, input_ids, sequence_lengths, **kwargs):
            del input_ids, kwargs
            return sequence_lengths.astype(mx.float32)

    model = TinyModel()
    mx.eval(model.parameters())
    batch = make_policy_eval_batch(
        [[1, 2, 3, 4, 5], [1, 4, 5], [1, 6, 7, 8], [1, 9, 10, 11, 12, 13]],
        pad_id=0,
        mode="sequence",
    )

    full_scores = score_policy_in_chunks(model, batch, batch_size=8, mode="sequence")
    chunked_scores = score_policy_in_chunks(
        model,
        batch,
        batch_size=8,
        token_budget=4,
        mode="sequence",
    )

    tokenizer = TinyTokenizer()
    rollout = collect_rollouts(
        policy=ScriptedModel({2: 5, 3: 1, "default": 1}),
        tokenizer=tokenizer,
        prompt_samples=[
            {"sample_index": 0, "prompt": "ab", "prompt_ids": [3, 4], "reward_context": "x"},
            {"sample_index": 1, "prompt": "cde", "prompt_ids": [5, 6, 7], "reward_context": "y"},
        ],
        sampling_config={"num_generations": 2, "temperature": 0.0, "max_completion_length": 2},
    )
    full_values = predict_rollout_values(LengthValueModel(), rollout, batch_size=8).value_predictions
    chunked_values = predict_rollout_values(
        LengthValueModel(),
        rollout,
        batch_size=8,
        token_budget=4,
    ).value_predictions

    assert mx.allclose(full_scores.summed_logprobs, chunked_scores.summed_logprobs)
    assert mx.allclose(full_scores.token_logprobs, chunked_scores.token_logprobs)
    assert mx.allclose(full_values, chunked_values)


def test_summarize_rollout_metrics_includes_completion_and_stop_stats():
    from mlx_tune._rl_runtime import collect_rollouts, summarize_rollout_metrics

    tokenizer = TinyTokenizer()
    tokenizer._unsloth_stop_token = "<|im_end|>"
    rollout = collect_rollouts(
        policy=ScriptedModel({2: 9, 3: 7, 4: 7, "default": 7}),
        tokenizer=tokenizer,
        prompt_samples=[
            {"sample_index": 0, "prompt": "ab", "prompt_ids": [3, 4], "reward_context": "x"},
            {"sample_index": 1, "prompt": "cd", "prompt_ids": [5, 6], "reward_context": "y"},
        ],
        sampling_config={"num_generations": 1, "temperature": 0.0, "max_completion_length": 4},
    )
    rollout.rewards = mx.array([1.0, 0.0], dtype=mx.float32)

    metrics = summarize_rollout_metrics(rollout, policy_loss=0.5)

    assert metrics["completion_length_mean"] == 1.0
    assert metrics["completion_length_max"] == 1.0
    assert metrics["eos_rate"] == 1.0
    assert metrics["truncation_rate"] == 0.0
    assert metrics["reward_mean"] == 0.5
    assert metrics["policy_loss"] == 0.5
