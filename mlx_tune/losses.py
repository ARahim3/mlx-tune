"""
Loss functions for MLX-Tune RL training.

Provides native MLX losses and reference-logprob helpers for:
- DPO (Direct Preference Optimization)
- ORPO (Odds Ratio Preference Optimization)
- GRPO (Group Relative Policy Optimization)
- KTO (Kahneman-Tversky Optimization)
- SimPO (Simple Preference Optimization)
"""

from typing import Optional, Tuple, Callable, List, Any

import mlx.core as mx
import mlx.nn as nn


def _token_log_probs(
    model: Any,
    input_ids: mx.array,
    temperature: float = 1.0,
) -> mx.array:
    """Return token log probabilities aligned to ``input_ids[:, 1:]``."""
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    logits = model(inputs)
    if temperature != 1.0:
        logits = logits / temperature
    log_probs = nn.log_softmax(logits, axis=-1)
    return mx.take_along_axis(
        log_probs,
        targets[:, :, None],
        axis=-1,
    ).squeeze(-1)


def _length_mask(lengths: mx.array, width: int) -> mx.array:
    positions = mx.arange(width)[None, :]
    return positions < lengths[:, None]


def completion_token_mask(
    input_ids: mx.array,
    prompt_lengths: mx.array,
    completion_lengths: mx.array,
) -> mx.array:
    """
    Build a mask over ``input_ids[:, 1:]`` that keeps completion-token losses only.
    """
    width = input_ids.shape[1] - 1
    positions = mx.arange(width)[None, :]
    start = mx.maximum(prompt_lengths - 1, 0)[:, None]
    end = mx.maximum(prompt_lengths + completion_lengths - 1, 0)[:, None]
    return (positions >= start) & (positions < end)


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
    token_log_probs = _token_log_probs(model, input_ids)
    mask = _length_mask(lengths, token_log_probs.shape[1]).astype(token_log_probs.dtype)
    return (token_log_probs * mask).sum(axis=-1)


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
    token_log_probs = _token_log_probs(model, input_ids, temperature=temperature)
    mask = completion_token_mask(input_ids, prompt_lengths, completion_lengths)
    return (token_log_probs * mask.astype(token_log_probs.dtype)).sum(axis=-1)


def _batched_sequence_log_probs(
    model: Any,
    input_ids: mx.array,
    lengths: mx.array,
    batch_size: int = 8,
) -> mx.array:
    if input_ids.shape[0] <= batch_size:
        return compute_log_probs_with_lengths(model, input_ids, lengths)

    chunks = []
    for start in range(0, input_ids.shape[0], batch_size):
        end = start + batch_size
        chunks.append(
            compute_log_probs_with_lengths(
                model,
                input_ids[start:end],
                lengths[start:end],
            )
        )
    return mx.concatenate(chunks, axis=0)


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
    log_pi_chosen = compute_log_probs_with_lengths(model, chosen_ids, chosen_lengths)
    log_pi_rejected = compute_log_probs_with_lengths(model, rejected_ids, rejected_lengths)

    if reference_chosen_logprobs is None or reference_rejected_logprobs is None:
        log_ref_chosen = mx.stop_gradient(log_pi_chosen)
        log_ref_rejected = mx.stop_gradient(log_pi_rejected)
    else:
        log_ref_chosen = reference_chosen_logprobs
        log_ref_rejected = reference_rejected_logprobs

    log_ratio_chosen = log_pi_chosen - log_ref_chosen
    log_ratio_rejected = log_pi_rejected - log_ref_rejected
    logits = beta * (log_ratio_chosen - log_ratio_rejected)

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

    avg_log_pi_chosen = log_pi_chosen / chosen_lengths.astype(log_pi_chosen.dtype)
    sft_term = -mx.mean(avg_log_pi_chosen)
    odds_term = -mx.mean(nn.log_sigmoid(log_pi_chosen - log_pi_rejected))

    loss = sft_term + beta * odds_term
    ntoks = chosen_lengths.sum() + rejected_lengths.sum()
    return loss, ntoks


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
    log_pi = compute_log_probs_with_lengths(model, input_ids, lengths)
    log_ref = mx.stop_gradient(log_pi) if reference_logprobs is None else reference_logprobs
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

    r_chosen = log_pi_chosen / chosen_lengths.astype(log_pi_chosen.dtype)
    r_rejected = log_pi_rejected / rejected_lengths.astype(log_pi_rejected.dtype)

    logits = beta * (r_chosen - r_rejected - gamma)
    loss = -mx.mean(nn.log_sigmoid(logits))
    return loss, chosen_lengths.sum() + rejected_lengths.sum()


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

    mask = _length_mask(lengths, targets.shape[1]).astype(logits.dtype)
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
    generated_ids = list(prompt_ids.tolist()) if hasattr(prompt_ids, "tolist") else list(prompt_ids)
    log_probs = []
    x = mx.array([generated_ids])

    for _ in range(max_tokens):
        logits = model(x)[:, -1, :]
        if temperature > 0:
            scaled = logits / temperature
            probs = mx.softmax(scaled, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
            token_log_prob = nn.log_softmax(scaled, axis=-1)[0, next_token.item()]
        else:
            next_token = mx.argmax(logits, axis=-1)
            token_log_prob = nn.log_softmax(logits, axis=-1)[0, next_token.item()]

        next_token_id = next_token.item()
        generated_ids.append(next_token_id)
        log_probs.append(token_log_prob)
        x = mx.array([generated_ids])

        if hasattr(tokenizer, "eos_token_id") and next_token_id == tokenizer.eos_token_id:
            break

    if log_probs:
        return mx.array(generated_ids), mx.stack(log_probs)
    return mx.array(generated_ids), mx.zeros((0,), dtype=mx.float32)


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
    current_logprobs = compute_completion_log_probs(
        model,
        input_ids,
        prompt_lengths,
        completion_lengths,
        temperature=temperature,
    )
    reference_logprobs = compute_completion_log_probs(
        reference_model,
        input_ids,
        prompt_lengths,
        completion_lengths,
        temperature=temperature,
    )
    reference_logprobs = mx.stop_gradient(reference_logprobs)

    ratios = mx.exp(current_logprobs - rollout_logprobs)
    clipped_ratios = mx.clip(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

    unclipped_objective = ratios * advantages
    clipped_objective = clipped_ratios * advantages
    policy_objective = mx.minimum(unclipped_objective, clipped_objective)

    log_ratio = current_logprobs - reference_logprobs
    kl_penalty = mx.exp(log_ratio) - log_ratio - 1.0

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
    completions = []
    all_log_probs = []

    for _ in range(num_generations):
        generated_ids, log_probs = generate_with_log_probs(
            model,
            tokenizer,
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        completion_ids = generated_ids[len(prompt_ids):]
        completion_text = tokenizer.decode(completion_ids.tolist())
        completions.append(completion_text)
        all_log_probs.append(log_probs.sum() if log_probs.size > 0 else mx.array(0.0))

    rewards = mx.array([reward_fn(completion, prompt_text) for completion in completions])
    rewards_std = mx.std(rewards)
    if rewards_std.item() < 1e-6:
        advantages = rewards - mx.mean(rewards)
    else:
        advantages = (rewards - mx.mean(rewards)) / (rewards_std + 1e-8)

    pg_loss = -mx.mean(advantages * mx.stack(all_log_probs))
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
    losses = []
    total_completions = 0

    for prompt in prompts:
        prompt_ids = mx.array(tokenizer.encode(prompt))
        loss, count = grpo_loss(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            reward_fn=reward_fn,
            prompt_text=prompt,
            num_generations=num_generations,
            temperature=temperature,
            max_tokens=max_tokens,
            beta=beta,
        )
        losses.append(loss)
        total_completions += count

    return mx.mean(mx.stack(losses)), total_completions


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
