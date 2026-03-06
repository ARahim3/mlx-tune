from dataclasses import dataclass, fields, replace
import inspect
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn


@dataclass
class PolicyEvalBatch:
    input_ids: mx.array
    sequence_lengths: mx.array
    token_mask: mx.array
    prompt_lengths: Optional[mx.array] = None
    completion_lengths: Optional[mx.array] = None
    rollout_logprobs: Optional[mx.array] = None
    reference_logprobs: Optional[mx.array] = None
    advantages: Optional[mx.array] = None
    labels: Optional[mx.array] = None
    prompt_group_indices: Optional[mx.array] = None
    sample_indices: Optional[mx.array] = None
    token_logprobs: Optional[mx.array] = None
    summed_logprobs: Optional[mx.array] = None


@dataclass
class RolloutBatch:
    prompt_ids: List[List[int]]
    prompt_lengths: mx.array
    completion_ids: List[List[int]]
    completion_lengths: mx.array
    prompt_texts: List[str]
    original_prompt_texts: Optional[List[str]]
    completion_texts: List[str]
    reward_contexts: List[Any]
    sampled_token_logprobs: mx.array
    rollout_logprobs: mx.array
    eos_flags: mx.array
    truncation_flags: mx.array
    prompt_group_indices: mx.array
    policy_eval: PolicyEvalBatch
    sample_indices: Optional[mx.array] = None
    sampled_token_logits: Optional[mx.array] = None
    token_entropies: Optional[mx.array] = None
    rewards: Optional[mx.array] = None
    advantages: Optional[mx.array] = None


@dataclass
class RewardBatch:
    prompt_texts: List[str]
    completion_texts: List[str]
    reward_contexts: List[Any]
    scalar_rewards: mx.array
    prompt_group_indices: mx.array
    original_prompt_texts: Optional[List[str]] = None
    named_reward_components: Optional[List[Dict[str, float]]] = None
    diagnostics: Optional[List[Dict[str, Any]]] = None


@dataclass
class PreferenceBatch:
    chosen: PolicyEvalBatch
    rejected: PolicyEvalBatch
    sample_indices: mx.array
    chosen_reference_logprobs: Optional[mx.array] = None
    rejected_reference_logprobs: Optional[mx.array] = None


def pad_sequences(sequences: Sequence[Sequence[int]], pad_id: int) -> tuple[mx.array, mx.array]:
    max_length = max(len(sequence) for sequence in sequences)
    padded = [list(sequence) + [pad_id] * (max_length - len(sequence)) for sequence in sequences]
    lengths = [len(sequence) for sequence in sequences]
    return mx.array(padded), mx.array(lengths)


def truncate_prompt_tokens(prompt_ids: Sequence[int], max_prompt_length: Optional[int]) -> List[int]:
    prompt_tokens = list(prompt_ids)
    if max_prompt_length is None or len(prompt_tokens) <= max_prompt_length:
        return prompt_tokens
    return prompt_tokens[-max_prompt_length:]


def truncate_completion_tokens(
    completion_ids: Sequence[int],
    max_completion_length: Optional[int],
) -> tuple[List[int], bool]:
    completion_tokens = list(completion_ids)
    if max_completion_length is None or len(completion_tokens) <= max_completion_length:
        return completion_tokens, False
    return completion_tokens[:max_completion_length], True


def length_mask(lengths: mx.array, width: int) -> mx.array:
    positions = mx.arange(width)[None, :]
    return positions < lengths[:, None]


def completion_token_mask(
    input_ids: mx.array,
    prompt_lengths: mx.array,
    completion_lengths: mx.array,
) -> mx.array:
    width = input_ids.shape[1] - 1
    positions = mx.arange(width)[None, :]
    start = mx.maximum(prompt_lengths - 1, 0)[:, None]
    end = mx.maximum(prompt_lengths + completion_lengths - 1, 0)[:, None]
    return (positions >= start) & (positions < end)


def build_token_mask(
    input_ids: mx.array,
    sequence_lengths: mx.array,
    mode: str = "sequence",
    prompt_lengths: Optional[mx.array] = None,
    completion_lengths: Optional[mx.array] = None,
) -> mx.array:
    if mode == "sequence":
        return length_mask(sequence_lengths, input_ids.shape[1] - 1)
    if mode == "completion":
        if prompt_lengths is None or completion_lengths is None:
            raise ValueError("Completion scoring requires prompt_lengths and completion_lengths.")
        return completion_token_mask(input_ids, prompt_lengths, completion_lengths)
    raise ValueError(f"Unsupported scoring mode: {mode}")


def make_policy_eval_batch(
    sequences: Sequence[Sequence[int]],
    pad_id: int,
    mode: str = "sequence",
    prompt_lengths: Optional[Sequence[int]] = None,
    completion_lengths: Optional[Sequence[int]] = None,
    rollout_logprobs: Optional[mx.array] = None,
    reference_logprobs: Optional[mx.array] = None,
    advantages: Optional[mx.array] = None,
    labels: Optional[mx.array] = None,
    prompt_group_indices: Optional[mx.array] = None,
    sample_indices: Optional[mx.array] = None,
) -> PolicyEvalBatch:
    input_ids, sequence_lengths = pad_sequences(sequences, pad_id)
    prompt_lengths_array = mx.array(prompt_lengths) if prompt_lengths is not None else None
    completion_lengths_array = mx.array(completion_lengths) if completion_lengths is not None else None
    token_mask = build_token_mask(
        input_ids=input_ids,
        sequence_lengths=sequence_lengths,
        mode=mode,
        prompt_lengths=prompt_lengths_array,
        completion_lengths=completion_lengths_array,
    )
    return PolicyEvalBatch(
        input_ids=input_ids,
        sequence_lengths=sequence_lengths,
        prompt_lengths=prompt_lengths_array,
        completion_lengths=completion_lengths_array,
        token_mask=token_mask,
        rollout_logprobs=rollout_logprobs,
        reference_logprobs=reference_logprobs,
        advantages=advantages,
        labels=labels,
        prompt_group_indices=prompt_group_indices,
        sample_indices=sample_indices,
    )


def make_preference_batch(
    chosen_sequences: Sequence[Sequence[int]],
    rejected_sequences: Sequence[Sequence[int]],
    pad_id: int,
    sample_indices: Sequence[int],
    chosen_reference_logprobs: Optional[mx.array] = None,
    rejected_reference_logprobs: Optional[mx.array] = None,
) -> PreferenceBatch:
    return PreferenceBatch(
        chosen=make_policy_eval_batch(
            chosen_sequences,
            pad_id=pad_id,
            mode="sequence",
            reference_logprobs=chosen_reference_logprobs,
            sample_indices=mx.array(sample_indices),
        ),
        rejected=make_policy_eval_batch(
            rejected_sequences,
            pad_id=pad_id,
            mode="sequence",
            reference_logprobs=rejected_reference_logprobs,
            sample_indices=mx.array(sample_indices),
        ),
        sample_indices=mx.array(sample_indices),
        chosen_reference_logprobs=chosen_reference_logprobs,
        rejected_reference_logprobs=rejected_reference_logprobs,
    )


def _token_log_probs(
    model: Any,
    input_ids: mx.array,
    temperature: float = 1.0,
) -> mx.array:
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]
    logits = model(inputs)
    if temperature != 1.0:
        logits = logits / temperature
    log_probs = nn.log_softmax(logits, axis=-1)
    return mx.take_along_axis(log_probs, targets[:, :, None], axis=-1).squeeze(-1)


def normalize_logprobs(
    summed_logprobs: mx.array,
    lengths: mx.array,
    mode: str = "sum",
) -> mx.array:
    if mode == "sum":
        return summed_logprobs
    if mode in {"mean", "token_mean"}:
        return summed_logprobs / mx.maximum(lengths.astype(summed_logprobs.dtype), 1.0)
    raise ValueError(f"Unsupported length normalization mode: {mode}")


def kl_against_reference(
    policy_logprobs: mx.array,
    reference_logprobs: mx.array,
) -> mx.array:
    log_ratio = policy_logprobs - reference_logprobs
    return mx.exp(log_ratio) - log_ratio - 1.0


def score_policy(
    model: Any,
    batch: PolicyEvalBatch,
    mode: str = "sequence",
    reference_model: Optional[Any] = None,
    temperature: float = 1.0,
) -> PolicyEvalBatch:
    token_mask = build_token_mask(
        input_ids=batch.input_ids,
        sequence_lengths=batch.sequence_lengths,
        mode=mode,
        prompt_lengths=batch.prompt_lengths,
        completion_lengths=batch.completion_lengths,
    )
    token_logprobs = _token_log_probs(model, batch.input_ids, temperature=temperature)
    summed_logprobs = (token_logprobs * token_mask.astype(token_logprobs.dtype)).sum(axis=-1)

    reference_logprobs = batch.reference_logprobs
    if reference_model is not None:
        reference_tokens = _token_log_probs(reference_model, batch.input_ids, temperature=temperature)
        reference_logprobs = mx.stop_gradient(
            (reference_tokens * token_mask.astype(reference_tokens.dtype)).sum(axis=-1)
        )

    return replace(
        batch,
        token_mask=token_mask,
        token_logprobs=token_logprobs,
        summed_logprobs=summed_logprobs,
        reference_logprobs=reference_logprobs,
    )


def score_policy_in_chunks(
    model: Any,
    batch: PolicyEvalBatch,
    batch_size: int,
    mode: str = "sequence",
    reference_model: Optional[Any] = None,
    temperature: float = 1.0,
) -> PolicyEvalBatch:
    if batch.input_ids.shape[0] <= batch_size:
        return score_policy(
            model,
            batch,
            mode=mode,
            reference_model=reference_model,
            temperature=temperature,
        )

    scored_chunks = []
    for minibatch in assemble_minibatches(batch, batch_size, shuffle=False):
        scored_chunks.append(
            score_policy(
                model,
                minibatch,
                mode=mode,
                reference_model=reference_model,
                temperature=temperature,
            )
        )
    return _concat_policy_eval_batches(scored_chunks)


def _concat_policy_eval_batches(chunks: Sequence[PolicyEvalBatch]) -> PolicyEvalBatch:
    def concat_attr(name: str):
        values = [getattr(chunk, name) for chunk in chunks]
        if values[0] is None:
            return None
        if hasattr(values[0], "shape"):
            return mx.concatenate(values, axis=0)
        if isinstance(values[0], list):
            merged = []
            for value in values:
                merged.extend(value)
            return merged
        raise TypeError(f"Unsupported PolicyEvalBatch field type for concat: {name}")

    return PolicyEvalBatch(
        input_ids=concat_attr("input_ids"),
        sequence_lengths=concat_attr("sequence_lengths"),
        prompt_lengths=concat_attr("prompt_lengths"),
        completion_lengths=concat_attr("completion_lengths"),
        token_mask=concat_attr("token_mask"),
        rollout_logprobs=concat_attr("rollout_logprobs"),
        reference_logprobs=concat_attr("reference_logprobs"),
        advantages=concat_attr("advantages"),
        labels=concat_attr("labels"),
        prompt_group_indices=concat_attr("prompt_group_indices"),
        sample_indices=concat_attr("sample_indices"),
        token_logprobs=concat_attr("token_logprobs"),
        summed_logprobs=concat_attr("summed_logprobs"),
    )


def sample_completion(
    policy: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    max_tokens: int,
    temperature: float,
    collect_sample_stats: bool = False,
) -> Dict[str, Any]:
    generated_ids = list(prompt_ids)
    sampled_logprobs: List[float] = []
    sampled_logits: List[float] = []
    token_entropies: List[float] = []
    saw_eos = False
    x = mx.array([generated_ids])

    for _ in range(max_tokens):
        logits = policy(x)[:, -1, :]
        if temperature > 0:
            scaled = logits / temperature
            probs = mx.softmax(scaled, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
            log_probs = nn.log_softmax(scaled, axis=-1)
            entropy = -mx.sum(probs * log_probs, axis=-1)[0]
        else:
            scaled = logits
            next_token = mx.argmax(logits, axis=-1)
            log_probs = nn.log_softmax(logits, axis=-1)
            probs = mx.softmax(logits, axis=-1)
            entropy = -mx.sum(probs * log_probs, axis=-1)[0]

        next_token_id = int(next_token.item())
        sampled_logprobs.append(float(log_probs[0, next_token_id].item()))
        if collect_sample_stats:
            sampled_logits.append(float(scaled[0, next_token_id].item()))
            token_entropies.append(float(entropy.item()))

        generated_ids.append(next_token_id)
        x = mx.array([generated_ids])
        if hasattr(tokenizer, "eos_token_id") and next_token_id == tokenizer.eos_token_id:
            saw_eos = True
            break

    completion_ids = generated_ids[len(prompt_ids):]
    return {
        "generated_ids": generated_ids,
        "completion_ids": completion_ids,
        "sampled_logprobs": sampled_logprobs,
        "sampled_logits": sampled_logits,
        "token_entropies": token_entropies,
        "eos_flag": saw_eos,
        "truncation_flag": (not saw_eos) and len(completion_ids) >= max_tokens,
    }


class _RewardEvaluatorAdapter:
    def __init__(self, evaluator: Any):
        self.evaluator = evaluator
        self.mode = self._resolve_mode(evaluator)

    def _resolve_mode(self, evaluator: Any) -> str:
        if evaluator is None:
            return "none"
        if hasattr(evaluator, "evaluate"):
            return "evaluate"
        if not callable(evaluator):
            raise TypeError("Reward evaluator must be callable or expose evaluate().")

        try:
            signature = inspect.signature(evaluator)
        except (TypeError, ValueError):
            return "legacy"

        positional = 0
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                return "legacy"
            if (
                parameter.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                and parameter.default is inspect._empty
            ):
                positional += 1
        return "legacy" if positional >= 2 else "structured"

    def evaluate(self, payload: Dict[str, Any]) -> tuple[float, Optional[Dict[str, float]], Optional[Dict[str, Any]]]:
        if self.mode == "none":
            result = 0.0
        elif self.mode == "evaluate":
            result = self.evaluator.evaluate(payload)
        elif self.mode == "structured":
            result = self.evaluator(payload)
        else:
            result = self.evaluator(payload["completion_text"], payload["reward_context"])
        return self._normalize_result(result)

    def _normalize_result(
        self,
        result: Any,
    ) -> tuple[float, Optional[Dict[str, float]], Optional[Dict[str, Any]]]:
        if isinstance(result, Mapping):
            reward = float(result.get("reward", result.get("score", 0.0)))
            components = result.get("components")
            diagnostics = result.get("diagnostics")
            return reward, dict(components) if components is not None else None, diagnostics
        return float(result), None, None


def collect_rollouts(
    policy: Any,
    tokenizer: Any,
    prompt_samples: Sequence[Dict[str, Any]],
    sampling_config: Dict[str, Any],
    reward_evaluator: Any = None,
    collect_sample_stats: bool = False,
) -> RolloutBatch:
    num_generations = int(sampling_config.get("num_generations", 1))
    max_completion_length = int(sampling_config.get("max_tokens", sampling_config.get("max_completion_length", 256)))
    max_seq_length = sampling_config.get("max_seq_length")
    temperature = float(sampling_config.get("temperature", 0.7))
    max_prompt_length = None
    if max_seq_length is not None:
        max_prompt_length = max(1, int(max_seq_length) - max_completion_length)

    prompt_texts: List[str] = []
    original_prompt_texts: List[str] = []
    prompt_ids: List[List[int]] = []
    prompt_lengths: List[int] = []
    completion_ids: List[List[int]] = []
    completion_lengths: List[int] = []
    completion_texts: List[str] = []
    reward_contexts: List[Any] = []
    rollout_logprobs: List[float] = []
    sampled_logprob_rows: List[List[float]] = []
    sampled_logit_rows: List[List[float]] = []
    entropy_rows: List[List[float]] = []
    eos_flags: List[bool] = []
    truncation_flags: List[bool] = []
    prompt_group_indices: List[int] = []
    sample_indices: List[int] = []

    for group_index, sample in enumerate(prompt_samples):
        sample_index = int(sample.get("sample_index", group_index))
        original_prompt_text = sample.get("prompt", sample.get("prompt_text", ""))
        reward_context = sample.get("reward_context")
        prepared_prompt_ids = truncate_prompt_tokens(sample.get("prompt_ids", []), max_prompt_length)
        effective_prompt_text = tokenizer.decode(prepared_prompt_ids)

        for _ in range(num_generations):
            sample_output = sample_completion(
                policy=policy,
                tokenizer=tokenizer,
                prompt_ids=prepared_prompt_ids,
                max_tokens=max_completion_length,
                temperature=temperature,
                collect_sample_stats=collect_sample_stats,
            )
            prepared_completion_ids, truncated = truncate_completion_tokens(
                sample_output["completion_ids"],
                max_completion_length,
            )
            sampled_logprobs = sample_output["sampled_logprobs"][: len(prepared_completion_ids)]
            sampled_logits = sample_output["sampled_logits"][: len(prepared_completion_ids)]
            entropies = sample_output["token_entropies"][: len(prepared_completion_ids)]

            prompt_texts.append(effective_prompt_text)
            original_prompt_texts.append(original_prompt_text)
            prompt_ids.append(prepared_prompt_ids)
            prompt_lengths.append(len(prepared_prompt_ids))
            reward_contexts.append(reward_context)
            completion_ids.append(prepared_completion_ids)
            completion_lengths.append(len(prepared_completion_ids))
            completion_texts.append(tokenizer.decode(prepared_completion_ids))
            sampled_logprob_rows.append(sampled_logprobs)
            rollout_logprobs.append(sum(sampled_logprobs))
            eos_flags.append(bool(sample_output["eos_flag"]))
            truncation_flags.append(bool(sample_output["truncation_flag"] or truncated))
            prompt_group_indices.append(group_index)
            sample_indices.append(sample_index)
            if collect_sample_stats:
                sampled_logit_rows.append(sampled_logits)
                entropy_rows.append(entropies)

    max_completion_width = max(completion_lengths) if completion_lengths else 0
    padded_token_logprobs, _ = pad_sequences(
        [
            [float(value) for value in row] + [0.0] * (max_completion_width - len(row))
            for row in sampled_logprob_rows
        ] if sampled_logprob_rows else [[0.0]],
        0,
    )
    if not sampled_logprob_rows:
        padded_token_logprobs = mx.zeros((0, 0), dtype=mx.float32)
    else:
        padded_token_logprobs = padded_token_logprobs.astype(mx.float32)

    sampled_token_logits = None
    token_entropies = None
    if collect_sample_stats:
        if sampled_logit_rows:
            sampled_token_logits, _ = pad_sequences(
                [
                    [float(value) for value in row] + [0.0] * (max_completion_width - len(row))
                    for row in sampled_logit_rows
                ],
                0,
            )
            token_entropies, _ = pad_sequences(
                [
                    [float(value) for value in row] + [0.0] * (max_completion_width - len(row))
                    for row in entropy_rows
                ],
                0,
            )
            sampled_token_logits = sampled_token_logits.astype(mx.float32)
            token_entropies = token_entropies.astype(mx.float32)
        else:
            sampled_token_logits = mx.zeros((0, 0), dtype=mx.float32)
            token_entropies = mx.zeros((0, 0), dtype=mx.float32)

    full_sequences = [
        prompt_sequence + completion_sequence
        for prompt_sequence, completion_sequence in zip(prompt_ids, completion_ids)
    ]
    policy_eval = make_policy_eval_batch(
        full_sequences,
        pad_id=int(getattr(tokenizer, "pad_token_id", 0) or 0),
        mode="completion",
        prompt_lengths=prompt_lengths,
        completion_lengths=completion_lengths,
        rollout_logprobs=mx.array(rollout_logprobs, dtype=mx.float32),
        prompt_group_indices=mx.array(prompt_group_indices),
        sample_indices=mx.array(sample_indices),
    )
    rollout_batch = RolloutBatch(
        prompt_ids=prompt_ids,
        prompt_lengths=mx.array(prompt_lengths),
        completion_ids=completion_ids,
        completion_lengths=mx.array(completion_lengths),
        prompt_texts=prompt_texts,
        original_prompt_texts=original_prompt_texts,
        completion_texts=completion_texts,
        reward_contexts=reward_contexts,
        sampled_token_logprobs=padded_token_logprobs,
        rollout_logprobs=mx.array(rollout_logprobs, dtype=mx.float32),
        eos_flags=mx.array(eos_flags),
        truncation_flags=mx.array(truncation_flags),
        prompt_group_indices=mx.array(prompt_group_indices),
        policy_eval=policy_eval,
        sample_indices=mx.array(sample_indices),
        sampled_token_logits=sampled_token_logits,
        token_entropies=token_entropies,
    )
    if reward_evaluator is not None:
        reward_batch = evaluate_rewards(rollout_batch, reward_evaluator)
        rollout_batch.rewards = reward_batch.scalar_rewards
    return rollout_batch


def evaluate_rewards(rollout_batch: RolloutBatch, evaluator: Any) -> RewardBatch:
    adapter = _RewardEvaluatorAdapter(evaluator)
    scalar_rewards: List[float] = []
    named_components: List[Dict[str, float]] = []
    diagnostics: List[Dict[str, Any]] = []

    for index in range(len(rollout_batch.prompt_texts)):
        payload = {
            "prompt_text": rollout_batch.prompt_texts[index],
            "original_prompt_text": (
                rollout_batch.original_prompt_texts[index]
                if rollout_batch.original_prompt_texts is not None
                else rollout_batch.prompt_texts[index]
            ),
            "completion_text": rollout_batch.completion_texts[index],
            "reward_context": rollout_batch.reward_contexts[index],
            "prompt_ids": list(rollout_batch.prompt_ids[index]),
            "completion_ids": list(rollout_batch.completion_ids[index]),
            "prompt_length": int(rollout_batch.prompt_lengths[index].item()),
            "completion_length": int(rollout_batch.completion_lengths[index].item()),
            "eos_flag": bool(rollout_batch.eos_flags[index].item()),
            "truncation_flag": bool(rollout_batch.truncation_flags[index].item()),
            "prompt_group_index": int(rollout_batch.prompt_group_indices[index].item()),
            "sample_index": (
                int(rollout_batch.sample_indices[index].item())
                if rollout_batch.sample_indices is not None
                else index
            ),
        }
        reward, components, sample_diagnostics = adapter.evaluate(payload)
        scalar_rewards.append(reward)
        named_components.append(components or {})
        diagnostics.append(sample_diagnostics or {})

    has_components = any(component for component in named_components)
    has_diagnostics = any(diagnostic for diagnostic in diagnostics)
    return RewardBatch(
        prompt_texts=list(rollout_batch.prompt_texts),
        completion_texts=list(rollout_batch.completion_texts),
        reward_contexts=list(rollout_batch.reward_contexts),
        scalar_rewards=mx.array(scalar_rewards, dtype=mx.float32),
        prompt_group_indices=mx.array(rollout_batch.prompt_group_indices),
        original_prompt_texts=list(rollout_batch.original_prompt_texts)
        if rollout_batch.original_prompt_texts is not None
        else None,
        named_reward_components=named_components if has_components else None,
        diagnostics=diagnostics if has_diagnostics else None,
    )


def compute_advantages(
    reward_batch: RewardBatch,
    grouping: str = "per_prompt",
    normalization: str = "zscore_if_nonzero_else_center",
) -> mx.array:
    rewards = reward_batch.scalar_rewards.astype(mx.float32)
    if grouping != "per_prompt":
        raise ValueError(f"Unsupported advantage grouping: {grouping}")
    if normalization != "zscore_if_nonzero_else_center":
        raise ValueError(f"Unsupported advantage normalization: {normalization}")

    reward_values = rewards.tolist()
    group_values = reward_batch.prompt_group_indices.tolist()
    advantages = [0.0] * len(reward_values)
    grouped_positions: Dict[int, List[int]] = {}
    for position, group_value in enumerate(group_values):
        grouped_positions.setdefault(int(group_value), []).append(position)

    for positions in grouped_positions.values():
        group_rewards = mx.array([reward_values[position] for position in positions], dtype=mx.float32)
        group_std = mx.std(group_rewards)
        if float(group_std.item()) < 1e-6:
            group_advantages = group_rewards - mx.mean(group_rewards)
        else:
            group_advantages = (group_rewards - mx.mean(group_rewards)) / (group_std + 1e-8)
        for offset, position in enumerate(positions):
            advantages[position] = float(group_advantages[offset].item())
    return mx.array(advantages, dtype=mx.float32)


def _slice_value(value: Any, indices: Sequence[int]) -> Any:
    if value is None:
        return None
    if hasattr(value, "shape"):
        return value[mx.array(indices)]
    if isinstance(value, list):
        return [value[index] for index in indices]
    if isinstance(value, tuple):
        return tuple(value[index] for index in indices)
    raise TypeError(f"Unsupported minibatch field type: {type(value)!r}")


def assemble_minibatches(
    batch: PolicyEvalBatch,
    minibatch_size: int,
    shuffle: bool = False,
) -> Iterable[PolicyEvalBatch]:
    batch_size = batch.input_ids.shape[0]
    if batch_size == 0:
        return []

    if shuffle:
        order = [int(value) for value in mx.random.permutation(batch_size).tolist()]
    else:
        order = list(range(batch_size))

    minibatches: List[PolicyEvalBatch] = []
    for start in range(0, batch_size, minibatch_size):
        indices = order[start:start + minibatch_size]
        values = {
            field.name: _slice_value(getattr(batch, field.name), indices)
            for field in fields(batch)
        }
        minibatches.append(PolicyEvalBatch(**values))
    return minibatches
