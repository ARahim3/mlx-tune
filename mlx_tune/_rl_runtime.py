from dataclasses import dataclass, fields, replace
import inspect
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    old_logprobs: Optional[mx.array] = None
    old_token_logprobs: Optional[mx.array] = None
    reference_logprobs: Optional[mx.array] = None
    value_predictions: Optional[mx.array] = None
    returns: Optional[mx.array] = None
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
    old_logprobs: Optional[mx.array] = None
    rewards: Optional[mx.array] = None
    reference_logprobs: Optional[mx.array] = None
    value_predictions: Optional[mx.array] = None
    returns: Optional[mx.array] = None
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
    old_logprobs: Optional[mx.array] = None,
    old_token_logprobs: Optional[mx.array] = None,
    reference_logprobs: Optional[mx.array] = None,
    value_predictions: Optional[mx.array] = None,
    returns: Optional[mx.array] = None,
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
        old_logprobs=old_logprobs if old_logprobs is not None else rollout_logprobs,
        old_token_logprobs=old_token_logprobs,
        reference_logprobs=reference_logprobs,
        value_predictions=value_predictions,
        returns=returns,
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
    batch_size: Optional[int],
    mode: str = "sequence",
    reference_model: Optional[Any] = None,
    temperature: float = 1.0,
    token_budget: Optional[int] = None,
) -> PolicyEvalBatch:
    if batch.input_ids.shape[0] == 0:
        return score_policy(
            model,
            batch,
            mode=mode,
            reference_model=reference_model,
            temperature=temperature,
        )

    if token_budget is None and batch_size is None:
        batch_size = batch.input_ids.shape[0]

    if token_budget is None and batch_size is not None and batch.input_ids.shape[0] <= batch_size:
        return score_policy(
            model,
            batch,
            mode=mode,
            reference_model=reference_model,
            temperature=temperature,
        )

    scored_chunks = []
    for minibatch in assemble_minibatches(
        batch,
        minibatch_size=batch_size,
        shuffle=False,
        mode=mode,
        token_budget=token_budget,
    ):
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


def _policy_eval_effective_lengths(batch: PolicyEvalBatch, mode: str) -> List[int]:
    if batch.input_ids.shape[0] == 0:
        return []
    if mode == "completion" and batch.completion_lengths is not None:
        return [max(1, int(value)) for value in batch.completion_lengths.tolist()]
    return [max(1, int(value) - 1) for value in batch.sequence_lengths.tolist()]


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
        old_logprobs=concat_attr("old_logprobs"),
        old_token_logprobs=concat_attr("old_token_logprobs"),
        reference_logprobs=concat_attr("reference_logprobs"),
        value_predictions=concat_attr("value_predictions"),
        returns=concat_attr("returns"),
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
        if hasattr(evaluator, "evaluate_batch"):
            return "evaluate_batch"
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
        elif self.mode == "evaluate_batch":
            result = self.evaluator.evaluate_batch([payload])[0]
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

    def evaluate_batch(
        self,
        payloads: Sequence[Dict[str, Any]],
    ) -> List[tuple[float, Optional[Dict[str, float]], Optional[Dict[str, Any]]]]:
        if self.mode == "none":
            return [(0.0, None, None) for _ in payloads]
        if hasattr(self.evaluator, "evaluate_batch"):
            results = self.evaluator.evaluate_batch(list(payloads))
            return [self._normalize_result(result) for result in results]
        return [self.evaluate(payload) for payload in payloads]


def _batched_last_token_logits(
    policy: Any,
    input_rows: Sequence[Sequence[int]],
    pad_id: int,
    cache_state: Any = None,
) -> Tuple[mx.array, Any]:
    if not input_rows:
        return mx.zeros((0, 0), dtype=mx.float32), cache_state

    lengths = [len(row) for row in input_rows]
    call_signature = None
    try:
        call_signature = inspect.signature(policy.__call__)
    except (TypeError, ValueError):
        call_signature = None
    use_cache = (
        cache_state is not False
        and len(set(lengths)) == 1
        and (
            hasattr(policy, "forward_with_cache")
            or (call_signature is not None and "cache" in call_signature.parameters)
        )
    )
    if use_cache:
        inputs = mx.array(input_rows if cache_state is None else [[row[-1]] for row in input_rows])
        try:
            if hasattr(policy, "forward_with_cache"):
                outputs = policy.forward_with_cache(inputs, cache=cache_state)
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    logits, next_cache = outputs
                    return logits[:, -1, :], next_cache
            outputs = policy(inputs, cache=cache_state)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                logits, next_cache = outputs
                return logits[:, -1, :], next_cache
        except Exception:
            cache_state = False

    padded, seq_lengths = pad_sequences(input_rows, pad_id)
    logits = policy(padded)
    row_positions = mx.arange(padded.shape[0])
    last_positions = seq_lengths - 1
    return logits[row_positions, last_positions, :], cache_state


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

    generation_batch_size = int(sampling_config.get("generation_batch_size") or 0)
    generation_batch_size = max(1, generation_batch_size or (len(prompt_samples) * max(1, num_generations)))
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)

    expanded_samples: List[Dict[str, Any]] = []
    for group_index, sample in enumerate(prompt_samples):
        sample_index = int(sample.get("sample_index", group_index))
        original_prompt_text = sample.get("prompt", sample.get("prompt_text", ""))
        reward_context = sample.get("reward_context")
        prepared_prompt_ids = truncate_prompt_tokens(sample.get("prompt_ids", []), max_prompt_length)
        effective_prompt_text = tokenizer.decode(prepared_prompt_ids)
        for _ in range(num_generations):
            expanded_samples.append(
                {
                    "prompt_ids": list(prepared_prompt_ids),
                    "prompt_text": effective_prompt_text,
                    "original_prompt_text": original_prompt_text,
                    "reward_context": reward_context,
                    "sample_index": sample_index,
                    "prompt_group_index": group_index,
                    "generated_ids": list(prepared_prompt_ids),
                    "sampled_logprobs": [],
                    "sampled_logits": [],
                    "token_entropies": [],
                    "eos_flag": False,
                    "done": False,
                }
            )

    for start in range(0, len(expanded_samples), generation_batch_size):
        chunk = expanded_samples[start:start + generation_batch_size]
        cache_state: Any = None
        for _ in range(max_completion_length):
            active_rows = [row for row in chunk if not row["done"]]
            if not active_rows:
                break

            logits, cache_state = _batched_last_token_logits(
                policy,
                [row["generated_ids"] for row in active_rows],
                pad_id=pad_id,
                cache_state=cache_state,
            )
            if temperature > 0:
                scaled = logits / temperature
                log_probs = nn.log_softmax(scaled, axis=-1)
                probs = mx.softmax(scaled, axis=-1)
                entropies = -mx.sum(probs * log_probs, axis=-1)
                next_tokens = []
                for row_index in range(logits.shape[0]):
                    sampled = mx.random.categorical(log_probs[row_index:row_index + 1])
                    next_tokens.append(int(sampled.item()))
            else:
                scaled = logits
                log_probs = nn.log_softmax(logits, axis=-1)
                probs = mx.softmax(logits, axis=-1)
                entropies = -mx.sum(probs * log_probs, axis=-1)
                next_tokens = [int(value) for value in mx.argmax(logits, axis=-1).tolist()]

            for row_index, row in enumerate(active_rows):
                token_id = next_tokens[row_index]
                row["generated_ids"].append(token_id)
                row["sampled_logprobs"].append(float(log_probs[row_index, token_id].item()))
                if collect_sample_stats:
                    row["sampled_logits"].append(float(scaled[row_index, token_id].item()))
                    row["token_entropies"].append(float(entropies[row_index].item()))
                if hasattr(tokenizer, "eos_token_id") and token_id == tokenizer.eos_token_id:
                    row["eos_flag"] = True
                    row["done"] = True

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

    for row in expanded_samples:
        raw_completion_ids = row["generated_ids"][len(row["prompt_ids"]):]
        prepared_completion_ids, truncated = truncate_completion_tokens(
            raw_completion_ids,
            max_completion_length,
        )
        sampled_logprobs = row["sampled_logprobs"][: len(prepared_completion_ids)]
        sampled_logits = row["sampled_logits"][: len(prepared_completion_ids)]
        entropies = row["token_entropies"][: len(prepared_completion_ids)]

        prompt_texts.append(row["prompt_text"])
        original_prompt_texts.append(row["original_prompt_text"])
        prompt_ids.append(list(row["prompt_ids"]))
        prompt_lengths.append(len(row["prompt_ids"]))
        reward_contexts.append(row["reward_context"])
        completion_ids.append(prepared_completion_ids)
        completion_lengths.append(len(prepared_completion_ids))
        completion_texts.append(tokenizer.decode(prepared_completion_ids))
        sampled_logprob_rows.append(sampled_logprobs)
        rollout_logprobs.append(sum(sampled_logprobs))
        eos_flags.append(bool(row["eos_flag"]))
        truncation_flags.append(bool((not row["eos_flag"]) and len(prepared_completion_ids) >= max_completion_length or truncated))
        prompt_group_indices.append(int(row["prompt_group_index"]))
        sample_indices.append(int(row["sample_index"]))
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
    aligned_old_token_rows = [
        [0.0] * max(prompt_length - 1, 0) + [float(value) for value in row]
        for prompt_length, row in zip(prompt_lengths, sampled_logprob_rows)
    ]
    aligned_old_token_logprobs, _ = pad_sequences(
        aligned_old_token_rows if aligned_old_token_rows else [[0.0]],
        0,
    )
    if not aligned_old_token_rows:
        aligned_old_token_logprobs = mx.zeros((0, 0), dtype=mx.float32)
    else:
        aligned_old_token_logprobs = aligned_old_token_logprobs.astype(mx.float32)
    policy_eval = make_policy_eval_batch(
        full_sequences,
        pad_id=pad_id,
        mode="completion",
        prompt_lengths=prompt_lengths,
        completion_lengths=completion_lengths,
        rollout_logprobs=mx.array(rollout_logprobs, dtype=mx.float32),
        old_logprobs=mx.array(rollout_logprobs, dtype=mx.float32),
        old_token_logprobs=aligned_old_token_logprobs,
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
        old_logprobs=mx.array(rollout_logprobs, dtype=mx.float32),
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
    payloads: List[Dict[str, Any]] = []
    for index in range(len(rollout_batch.prompt_texts)):
        payloads.append(
            {
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
        )

    scalar_rewards: List[float] = []
    named_components: List[Dict[str, float]] = []
    diagnostics: List[Dict[str, Any]] = []
    for reward, components, sample_diagnostics in adapter.evaluate_batch(payloads):
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


def score_rollout_references(
    reference_model: Any,
    rollout_batch: RolloutBatch,
    batch_size: Optional[int] = 8,
    temperature: float = 1.0,
    token_budget: Optional[int] = None,
) -> RolloutBatch:
    if reference_model is None:
        return rollout_batch

    scored = score_policy_in_chunks(
        reference_model,
        rollout_batch.policy_eval,
        batch_size=batch_size,
        token_budget=token_budget,
        mode="completion",
        temperature=temperature,
    )
    reference_logprobs = mx.stop_gradient(scored.summed_logprobs.astype(mx.float32))
    return replace(
        rollout_batch,
        reference_logprobs=reference_logprobs,
        policy_eval=replace(
            rollout_batch.policy_eval,
            reference_logprobs=reference_logprobs,
        ),
    )


def predict_rollout_values(
    value_model: Any,
    rollout_batch: RolloutBatch,
    batch_size: Optional[int] = 8,
    token_budget: Optional[int] = None,
) -> RolloutBatch:
    if value_model is None:
        return rollout_batch

    value_chunks = []
    for minibatch in assemble_minibatches(
        rollout_batch.policy_eval,
        minibatch_size=batch_size,
        shuffle=False,
        mode="completion",
        token_budget=token_budget,
    ):
        value_chunks.append(
            value_model.predict(
                minibatch.input_ids,
                sequence_lengths=minibatch.sequence_lengths,
                prompt_lengths=minibatch.prompt_lengths
                if minibatch.prompt_lengths is not None
                else None,
                completion_lengths=minibatch.completion_lengths
                if minibatch.completion_lengths is not None
                else None,
            )
        )
    value_predictions = (
        mx.concatenate(value_chunks, axis=0) if value_chunks else mx.zeros((0,), dtype=mx.float32)
    )
    value_predictions = mx.stop_gradient(value_predictions.astype(mx.float32))
    return replace(
        rollout_batch,
        value_predictions=value_predictions,
        policy_eval=replace(
            rollout_batch.policy_eval,
            value_predictions=value_predictions,
        ),
    )


def compute_returns_and_advantages(
    rewards: mx.array,
    values: Optional[mx.array] = None,
    prompt_group_indices: Optional[mx.array] = None,
    mode: str = "gae",
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    normalize: bool = False,
) -> tuple[mx.array, mx.array]:
    rewards = rewards.astype(mx.float32)
    values = mx.zeros_like(rewards) if values is None else values.astype(mx.float32)

    if mode == "gae":
        deltas = rewards - values
        advantages = deltas * gae_lambda + deltas * (1.0 - gae_lambda)
        returns = advantages + values
    elif mode == "group_zscore":
        if prompt_group_indices is None:
            raise ValueError("prompt_group_indices is required for grouped advantages.")
        reward_batch = RewardBatch(
            prompt_texts=[""] * rewards.shape[0],
            completion_texts=[""] * rewards.shape[0],
            reward_contexts=[None] * rewards.shape[0],
            scalar_rewards=rewards,
            prompt_group_indices=prompt_group_indices,
        )
        advantages = compute_advantages(reward_batch)
        returns = rewards
    elif mode == "group_center":
        if prompt_group_indices is None:
            raise ValueError("prompt_group_indices is required for grouped advantages.")
        reward_values = rewards.tolist()
        advantages_list = [0.0] * len(reward_values)
        grouped_positions: Dict[int, List[int]] = {}
        for position, group_value in enumerate(prompt_group_indices.tolist()):
            grouped_positions.setdefault(int(group_value), []).append(position)
        for positions in grouped_positions.values():
            group_rewards = [reward_values[position] for position in positions]
            baseline = sum(group_rewards) / float(len(group_rewards))
            for offset, position in enumerate(positions):
                advantages_list[position] = group_rewards[offset] - baseline
        advantages = mx.array(advantages_list, dtype=mx.float32)
        returns = rewards
    elif mode == "rloo":
        if prompt_group_indices is None:
            raise ValueError("prompt_group_indices is required for RLOO advantages.")
        reward_values = rewards.tolist()
        advantages_list = [0.0] * len(reward_values)
        grouped_positions: Dict[int, List[int]] = {}
        for position, group_value in enumerate(prompt_group_indices.tolist()):
            grouped_positions.setdefault(int(group_value), []).append(position)
        for positions in grouped_positions.values():
            if len(positions) <= 1:
                continue
            group_rewards = [reward_values[position] for position in positions]
            total = sum(group_rewards)
            denominator = float(len(positions) - 1)
            for offset, position in enumerate(positions):
                baseline = (total - group_rewards[offset]) / denominator
                advantages_list[position] = group_rewards[offset] - baseline
        advantages = mx.array(advantages_list, dtype=mx.float32)
        returns = rewards
    else:
        raise ValueError(f"Unsupported returns/advantages mode: {mode}")

    if normalize and advantages.shape[0] > 1:
        advantages = (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-8)

    if gamma != 1.0:
        returns = rewards + gamma * (returns - rewards)
    return returns.astype(mx.float32), advantages.astype(mx.float32)


def rank_grouped_rollouts(
    rollout_batch: RolloutBatch,
    score_tolerance: float = 1e-6,
) -> List[Dict[str, Any]]:
    if rollout_batch.rewards is None:
        raise ValueError("Rollout rewards are required for grouped ranking.")

    grouped_positions: Dict[int, List[int]] = {}
    for position, group_value in enumerate(rollout_batch.prompt_group_indices.tolist()):
        grouped_positions.setdefault(int(group_value), []).append(position)

    reward_values = rollout_batch.rewards.tolist()
    rankings: List[Dict[str, Any]] = []
    for group_index, positions in grouped_positions.items():
        ordered_positions = sorted(positions, key=lambda position: reward_values[position], reverse=True)
        ordered_scores = [reward_values[position] for position in ordered_positions]
        rankings.append(
            {
                "prompt_group_index": group_index,
                "positions": positions,
                "ordered_positions": ordered_positions,
                "scores": ordered_scores,
                "best_position": ordered_positions[0],
                "worst_position": ordered_positions[-1],
                "all_tied": (
                    len(ordered_scores) <= 1
                    or max(ordered_scores) - min(ordered_scores) <= score_tolerance
                ),
            }
        )
    return rankings


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
    minibatch_size: Optional[int],
    shuffle: bool = False,
    mode: str = "sequence",
    token_budget: Optional[int] = None,
) -> Iterable[PolicyEvalBatch]:
    batch_size = batch.input_ids.shape[0]
    if batch_size == 0:
        return []

    if shuffle:
        order = [int(value) for value in mx.random.permutation(batch_size).tolist()]
    else:
        order = list(range(batch_size))

    effective_lengths = _policy_eval_effective_lengths(batch, mode)
    row_budget = max(1, minibatch_size or batch_size)
    token_budget = max(1, token_budget) if token_budget is not None else None

    batches_of_indices: List[List[int]] = []
    current_indices: List[int] = []
    current_tokens = 0
    for index in order:
        row_tokens = effective_lengths[index]
        exceeds_token_budget = token_budget is not None and current_indices and current_tokens + row_tokens > token_budget
        exceeds_row_budget = current_indices and len(current_indices) >= row_budget
        if exceeds_token_budget or exceeds_row_budget:
            batches_of_indices.append(current_indices)
            current_indices = []
            current_tokens = 0
        current_indices.append(index)
        current_tokens += row_tokens
    if current_indices:
        batches_of_indices.append(current_indices)

    minibatches: List[PolicyEvalBatch] = []
    for indices in batches_of_indices:
        values = {
            field.name: _slice_value(getattr(batch, field.name), indices)
            for field in fields(batch)
        }
        minibatches.append(PolicyEvalBatch(**values))
    return minibatches


def summarize_rollout_metrics(
    rollout_batch: RolloutBatch,
    policy_loss: Optional[float] = None,
    value_loss: Optional[float] = None,
    reward_loss: Optional[float] = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if rollout_batch.rewards is not None and rollout_batch.rewards.shape[0] > 0:
        metrics["reward_mean"] = float(mx.mean(rollout_batch.rewards).item())
        metrics["reward_std"] = float(mx.std(rollout_batch.rewards).item())
    if rollout_batch.reference_logprobs is not None and rollout_batch.rollout_logprobs.shape[0] > 0:
        kl_values = kl_against_reference(
            rollout_batch.rollout_logprobs.astype(mx.float32),
            rollout_batch.reference_logprobs.astype(mx.float32),
        )
        metrics["kl_to_reference_mean"] = float(mx.mean(kl_values).item())
    if rollout_batch.token_entropies is not None and rollout_batch.completion_lengths.shape[0] > 0:
        entropy_mask = length_mask(
            rollout_batch.completion_lengths,
            rollout_batch.token_entropies.shape[1],
        ).astype(rollout_batch.token_entropies.dtype)
        valid_tokens = float(mx.sum(entropy_mask).item())
        if valid_tokens > 0:
            metrics["entropy_mean"] = float(
                (mx.sum(rollout_batch.token_entropies * entropy_mask) / valid_tokens).item()
            )
    if rollout_batch.completion_lengths.shape[0] > 0:
        metrics["completion_length_mean"] = float(mx.mean(rollout_batch.completion_lengths.astype(mx.float32)).item())
    if rollout_batch.truncation_flags is not None and rollout_batch.truncation_flags.shape[0] > 0:
        metrics["truncation_rate"] = float(mx.mean(rollout_batch.truncation_flags.astype(mx.float32)).item())
    if policy_loss is not None:
        metrics["policy_loss"] = float(policy_loss)
    if value_loss is not None:
        metrics["value_loss"] = float(value_loss)
    if reward_loss is not None:
        metrics["reward_loss"] = float(reward_loss)
    return metrics
