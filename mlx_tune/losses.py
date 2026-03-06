"""
Loss functions for MLX-Tune RL training.

Provides native MLX losses and reference-logprob helpers for:
- DPO (Direct Preference Optimization)
- ORPO (Odds Ratio Preference Optimization)
- GRPO (Group Relative Policy Optimization)
- KTO (Kahneman-Tversky Optimization)
- SimPO (Simple Preference Optimization)
"""

from typing import Any, Callable, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_tune._rl_runtime import (
    PolicyEvalBatch,
    build_token_mask,
    collect_rollouts,
    completion_token_mask,
    compute_advantages,
    evaluate_rewards,
    kl_against_reference,
    length_mask,
    make_policy_eval_batch,
    normalize_logprobs,
    sample_completion,
    score_policy,
    score_policy_in_chunks,
)


def _policy_eval_from_padded(
    input_ids: mx.array,
    lengths: mx.array,
    mode: str = "sequence",
    prompt_lengths: Optional[mx.array] = None,
    completion_lengths: Optional[mx.array] = None,
    rollout_logprobs: Optional[mx.array] = None,
    reference_logprobs: Optional[mx.array] = None,
    advantages: Optional[mx.array] = None,
    labels: Optional[mx.array] = None,
) -> PolicyEvalBatch:
    return PolicyEvalBatch(
        input_ids=input_ids,
        sequence_lengths=lengths,
        prompt_lengths=prompt_lengths,
        completion_lengths=completion_lengths,
        token_mask=build_token_mask(
            input_ids=input_ids,
            sequence_lengths=lengths,
            mode=mode,
            prompt_lengths=prompt_lengths,
            completion_lengths=completion_lengths,
        ),
        rollout_logprobs=rollout_logprobs,
        reference_logprobs=reference_logprobs,
        advantages=advantages,
        labels=labels,
    )


def _token_log_probs(
    model: Any,
    input_ids: mx.array,
    temperature: float = 1.0,
) -> mx.array:
    batch = _policy_eval_from_padded(
        input_ids=input_ids,
        lengths=mx.array([input_ids.shape[1]] * input_ids.shape[0]),
    )
    return score_policy(model, batch, mode="sequence", temperature=temperature).token_logprobs


def compute_log_probs(
    model: Any,
    input_ids: mx.array,
    attention_mask: Optional[mx.array] = None,
) -> mx.array:
    """
    Compute per-sequence log probabilities for a batch of sequences.
    """
    token_log_probs = _token_log_probs(model, input_ids)
    if attention_mask is not None:
        token_log_probs = token_log_probs * attention_mask[:, 1:].astype(token_log_probs.dtype)
    return token_log_probs.sum(axis=-1)


def compute_log_probs_with_lengths(
    model: Any,
    input_ids: mx.array,
    lengths: mx.array,
) -> mx.array:
    """
    Compute per-sequence log probabilities with explicit length masking.
    """
    batch = _policy_eval_from_padded(input_ids=input_ids, lengths=lengths, mode="sequence")
    return score_policy(model, batch, mode="sequence").summed_logprobs


def compute_completion_log_probs(
    model: Any,
    input_ids: mx.array,
    prompt_lengths: mx.array,
    completion_lengths: mx.array,
    temperature: float = 1.0,
) -> mx.array:
    """
    Compute log probabilities over completion tokens only.
    """
    sequence_lengths = prompt_lengths + completion_lengths
    batch = _policy_eval_from_padded(
        input_ids=input_ids,
        lengths=sequence_lengths,
        mode="completion",
        prompt_lengths=prompt_lengths,
        completion_lengths=completion_lengths,
    )
    return score_policy(model, batch, mode="completion", temperature=temperature).summed_logprobs


def _batched_sequence_log_probs(
    model: Any,
    input_ids: mx.array,
    lengths: mx.array,
    batch_size: int = 8,
) -> mx.array:
    batch = _policy_eval_from_padded(input_ids=input_ids, lengths=lengths, mode="sequence")
    return score_policy_in_chunks(model, batch, batch_size=batch_size, mode="sequence").summed_logprobs


def precompute_preference_reference_logprobs(
    model: Any,
    chosen_ids: mx.array,
    rejected_ids: mx.array,
    chosen_lengths: mx.array,
    rejected_lengths: mx.array,
    batch_size: int = 8,
) -> Tuple[mx.array, mx.array]:
    """
    Precompute frozen-reference log probabilities for preference pairs.
    """
    ref_chosen = _batched_sequence_log_probs(model, chosen_ids, chosen_lengths, batch_size)
    ref_rejected = _batched_sequence_log_probs(model, rejected_ids, rejected_lengths, batch_size)
    return mx.stop_gradient(ref_chosen), mx.stop_gradient(ref_rejected)


def precompute_kto_reference_logprobs(
    model: Any,
    input_ids: mx.array,
    lengths: mx.array,
    batch_size: int = 8,
) -> mx.array:
    """
    Precompute frozen-reference log probabilities for KTO samples.
    """
    ref = _batched_sequence_log_probs(model, input_ids, lengths, batch_size)
    return mx.stop_gradient(ref)


def dpo_loss(
    model: Any,
    chosen_ids: mx.array,
    rejected_ids: mx.array,
    chosen_lengths: mx.array,
    rejected_lengths: mx.array,
    beta: float = 0.1,
    reference_chosen_logprobs: Optional[mx.array] = None,
    reference_rejected_logprobs: Optional[mx.array] = None,
    label_smoothing: float = 0.0,
) -> Tuple[mx.array, mx.array]:
    """
    Compute DPO loss.
    """
    chosen_batch = score_policy(
        model,
        _policy_eval_from_padded(
            input_ids=chosen_ids,
            lengths=chosen_lengths,
            mode="sequence",
            reference_logprobs=reference_chosen_logprobs,
        ),
        mode="sequence",
    )
    rejected_batch = score_policy(
        model,
        _policy_eval_from_padded(
            input_ids=rejected_ids,
            lengths=rejected_lengths,
            mode="sequence",
            reference_logprobs=reference_rejected_logprobs,
        ),
        mode="sequence",
    )

    log_pi_chosen = chosen_batch.summed_logprobs
    log_pi_rejected = rejected_batch.summed_logprobs
    log_ref_chosen = (
        mx.stop_gradient(log_pi_chosen)
        if chosen_batch.reference_logprobs is None
        else chosen_batch.reference_logprobs
    )
    log_ref_rejected = (
        mx.stop_gradient(log_pi_rejected)
        if rejected_batch.reference_logprobs is None
        else rejected_batch.reference_logprobs
    )

    logits = beta * ((log_pi_chosen - log_ref_chosen) - (log_pi_rejected - log_ref_rejected))
    if label_smoothing > 0:
        losses = (
            -nn.log_sigmoid(logits) * (1 - label_smoothing)
            - nn.log_sigmoid(-logits) * label_smoothing
        )
    else:
        losses = -nn.log_sigmoid(logits)
    return mx.mean(losses), chosen_lengths.sum() + rejected_lengths.sum()


def orpo_loss(
    model: Any,
    chosen_ids: mx.array,
    rejected_ids: mx.array,
    chosen_lengths: mx.array,
    rejected_lengths: mx.array,
    beta: float = 0.1,
) -> Tuple[mx.array, mx.array]:
    """
    Compute ORPO loss.
    """
    log_pi_chosen = compute_log_probs_with_lengths(model, chosen_ids, chosen_lengths)
    log_pi_rejected = compute_log_probs_with_lengths(model, rejected_ids, rejected_lengths)
    avg_log_pi_chosen = normalize_logprobs(log_pi_chosen, chosen_lengths, mode="mean")

    sft_term = -mx.mean(avg_log_pi_chosen)
    odds_term = -mx.mean(nn.log_sigmoid(log_pi_chosen - log_pi_rejected))
    loss = sft_term + beta * odds_term
    return loss, chosen_lengths.sum() + rejected_lengths.sum()


def kto_loss(
    model: Any,
    input_ids: mx.array,
    lengths: mx.array,
    labels: mx.array,
    beta: float = 0.1,
    reference_logprobs: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Compute KTO loss.
    """
    batch = score_policy(
        model,
        _policy_eval_from_padded(
            input_ids=input_ids,
            lengths=lengths,
            mode="sequence",
            reference_logprobs=reference_logprobs,
            labels=labels,
        ),
        mode="sequence",
    )
    log_pi = batch.summed_logprobs
    log_ref = mx.stop_gradient(log_pi) if batch.reference_logprobs is None else batch.reference_logprobs
    log_ratio = log_pi - log_ref

    positive_mask = labels > 0.5
    negative_mask = ~positive_mask
    weights = mx.where(positive_mask, 1.0, 1.0)
    positive_loss = -nn.log_sigmoid(beta * log_ratio) * positive_mask
    negative_loss = -nn.log_sigmoid(-beta * log_ratio) * negative_mask
    loss = mx.mean(weights * (positive_loss + negative_loss))
    return loss, lengths.sum()


def simpo_loss(
    model: Any,
    chosen_ids: mx.array,
    rejected_ids: mx.array,
    chosen_lengths: mx.array,
    rejected_lengths: mx.array,
    beta: float = 2.0,
    gamma: float = 0.5,
) -> Tuple[mx.array, mx.array]:
    """
    Compute SimPO loss.
    """
    log_pi_chosen = compute_log_probs_with_lengths(model, chosen_ids, chosen_lengths)
    log_pi_rejected = compute_log_probs_with_lengths(model, rejected_ids, rejected_lengths)
    r_chosen = normalize_logprobs(log_pi_chosen, chosen_lengths, mode="mean")
    r_rejected = normalize_logprobs(log_pi_rejected, rejected_lengths, mode="mean")

    logits = beta * (r_chosen - r_rejected - gamma)
    return -mx.mean(nn.log_sigmoid(logits)), chosen_lengths.sum() + rejected_lengths.sum()


def sft_loss(
    model: Any,
    input_ids: mx.array,
    lengths: mx.array,
) -> Tuple[mx.array, mx.array]:
    """
    Standard supervised fine-tuning loss.
    """
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]
    logits = model(inputs)

    mask = length_mask(lengths, targets.shape[1]).astype(logits.dtype)
    ce = nn.losses.cross_entropy(logits, targets, reduction="none")
    masked_ce = ce * mask
    ntoks = mask.sum()
    return masked_ce.sum() / ntoks, ntoks


def generate_with_log_probs(
    model: Any,
    tokenizer: Any,
    prompt_ids: mx.array,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> Tuple[mx.array, mx.array]:
    """
    Generate a sampled completion and return sampled-token log probabilities.
    """
    generated = sample_completion(
        policy=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids.tolist() if hasattr(prompt_ids, "tolist") else list(prompt_ids),
        max_tokens=max_tokens,
        temperature=temperature,
        collect_sample_stats=False,
    )
    token_logprobs = generated["sampled_logprobs"]
    if token_logprobs:
        return mx.array(generated["generated_ids"]), mx.array(token_logprobs, dtype=mx.float32)
    return mx.array(generated["generated_ids"]), mx.zeros((0,), dtype=mx.float32)


def grpo_recompute_loss(
    model: Any,
    reference_model: Any,
    input_ids: mx.array,
    prompt_lengths: mx.array,
    completion_lengths: mx.array,
    rollout_logprobs: mx.array,
    advantages: mx.array,
    beta: float = 0.04,
    clip_epsilon: float = 0.2,
    temperature: float = 1.0,
) -> Tuple[mx.array, mx.array]:
    """
    Recompute GRPO loss on fixed sampled completions.
    """
    ratio_batch = score_policy(
        model,
        _policy_eval_from_padded(
            input_ids=input_ids,
            lengths=prompt_lengths + completion_lengths,
            mode="completion",
            prompt_lengths=prompt_lengths,
            completion_lengths=completion_lengths,
            rollout_logprobs=rollout_logprobs,
            advantages=advantages,
        ),
        mode="completion",
        reference_model=reference_model,
        temperature=temperature,
    )
    kl_batch = score_policy(
        model,
        _policy_eval_from_padded(
            input_ids=input_ids,
            lengths=prompt_lengths + completion_lengths,
            mode="completion",
            prompt_lengths=prompt_lengths,
            completion_lengths=completion_lengths,
        ),
        mode="completion",
        reference_model=reference_model,
        temperature=1.0,
    )

    ratios = mx.exp(ratio_batch.summed_logprobs - rollout_logprobs)
    clipped_ratios = mx.clip(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    unclipped_objective = ratios * advantages
    clipped_objective = clipped_ratios * advantages
    policy_objective = mx.minimum(unclipped_objective, clipped_objective)
    kl_penalty = kl_against_reference(kl_batch.summed_logprobs, kl_batch.reference_logprobs)

    loss = -mx.mean(policy_objective - beta * kl_penalty)
    return loss, completion_lengths.sum()


def grpo_loss(
    model: Any,
    tokenizer: Any,
    prompt_ids: mx.array,
    reward_fn: Callable[[str, str], float],
    prompt_text: str,
    num_generations: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 256,
    beta: float = 0.04,
) -> Tuple[mx.array, int]:
    """
    Legacy log-only GRPO loss retained for compatibility.
    """
    del beta
    rollout_batch = collect_rollouts(
        policy=model,
        tokenizer=tokenizer,
        prompt_samples=[
            {
                "sample_index": 0,
                "prompt": prompt_text,
                "prompt_ids": prompt_ids.tolist() if hasattr(prompt_ids, "tolist") else list(prompt_ids),
                "reward_context": prompt_text,
            }
        ],
        sampling_config={
            "num_generations": num_generations,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )
    reward_batch = evaluate_rewards(rollout_batch, reward_fn)
    advantages = compute_advantages(reward_batch)
    pg_loss = -mx.mean(advantages * rollout_batch.rollout_logprobs)
    return pg_loss, num_generations


def grpo_batch_loss(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    reward_fn: Callable[[str, str], float],
    num_generations: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 256,
    beta: float = 0.04,
) -> Tuple[mx.array, int]:
    """
    Legacy batched GRPO loss retained for compatibility.
    """
    del beta
    prompt_samples = [
        {
            "sample_index": index,
            "prompt": prompt,
            "prompt_ids": tokenizer.encode(prompt),
            "reward_context": prompt,
        }
        for index, prompt in enumerate(prompts)
    ]
    rollout_batch = collect_rollouts(
        policy=model,
        tokenizer=tokenizer,
        prompt_samples=prompt_samples,
        sampling_config={
            "num_generations": num_generations,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )
    reward_batch = evaluate_rewards(rollout_batch, reward_fn)
    advantages = compute_advantages(reward_batch)
    pg_loss = -mx.mean(advantages * rollout_batch.rollout_logprobs)
    return pg_loss, len(prompt_samples) * num_generations


def compute_reference_logprobs(
    model: Any,
    chosen_ids: mx.array,
    rejected_ids: mx.array,
    chosen_lengths: mx.array,
    rejected_lengths: mx.array,
) -> Tuple[mx.array, mx.array]:
    """
    Backwards-compatible alias for batched DPO reference precompute.
    """
    return precompute_preference_reference_logprobs(
        model,
        chosen_ids,
        rejected_ids,
        chosen_lengths,
        rejected_lengths,
    )
