"""
Loss functions for MLX-Tune RL training.

Provides native MLX losses and reference-logprob helpers for:
- DPO (Direct Preference Optimization)
- ORPO (Odds Ratio Preference Optimization)
- GRPO (Group Relative Policy Optimization)
- KTO (Kahneman-Tversky Optimization)
- SimPO (Simple Preference Optimization)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

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
    old_logprobs: Optional[mx.array] = None,
    old_token_logprobs: Optional[mx.array] = None,
    reference_logprobs: Optional[mx.array] = None,
    value_predictions: Optional[mx.array] = None,
    returns: Optional[mx.array] = None,
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
        old_logprobs=old_logprobs if old_logprobs is not None else rollout_logprobs,
        old_token_logprobs=old_token_logprobs,
        reference_logprobs=reference_logprobs,
        value_predictions=value_predictions,
        returns=returns,
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


def pairwise_reward_loss(
    chosen_scores: mx.array,
    rejected_scores: mx.array,
    margin: float = 0.0,
) -> mx.array:
    """
    Logistic pairwise preference loss over scalar reward scores.
    """
    return -mx.mean(nn.log_sigmoid(chosen_scores - rejected_scores - margin))


def reward_model_pairwise_loss(
    reward_model: Any,
    chosen_input_ids: mx.array,
    rejected_input_ids: mx.array,
    chosen_sequence_lengths: mx.array,
    rejected_sequence_lengths: mx.array,
    chosen_prompt_lengths: Optional[mx.array] = None,
    rejected_prompt_lengths: Optional[mx.array] = None,
    chosen_completion_lengths: Optional[mx.array] = None,
    rejected_completion_lengths: Optional[mx.array] = None,
    margin: float = 0.0,
) -> Tuple[mx.array, Dict[str, mx.array]]:
    """
    Compute pairwise reward-model loss for chosen/rejected sequences.
    """
    chosen_scores, rejected_scores = reward_model.score_pairs(
        chosen_input_ids=chosen_input_ids,
        rejected_input_ids=rejected_input_ids,
        chosen_sequence_lengths=chosen_sequence_lengths,
        rejected_sequence_lengths=rejected_sequence_lengths,
        chosen_prompt_lengths=chosen_prompt_lengths,
        rejected_prompt_lengths=rejected_prompt_lengths,
        chosen_completion_lengths=chosen_completion_lengths,
        rejected_completion_lengths=rejected_completion_lengths,
    )
    loss = pairwise_reward_loss(chosen_scores, rejected_scores, margin=margin)
    return loss, {
        "chosen_scores": chosen_scores,
        "rejected_scores": rejected_scores,
    }


def value_regression_loss(
    predictions: mx.array,
    targets: mx.array,
    loss_type: str = "mse",
) -> mx.array:
    """
    Compute a pointwise scalar regression loss.
    """
    if loss_type == "mse":
        return mx.mean((predictions - targets) ** 2)
    if loss_type == "mae":
        return mx.mean(mx.abs(predictions - targets))
    raise ValueError(f"Unsupported scalar regression loss: {loss_type}")


def value_model_regression_loss(
    value_model: Any,
    input_ids: mx.array,
    sequence_lengths: mx.array,
    targets: mx.array,
    prompt_lengths: Optional[mx.array] = None,
    completion_lengths: Optional[mx.array] = None,
    loss_type: str = "mse",
) -> Tuple[mx.array, mx.array]:
    """
    Compute pointwise regression loss for a scalar value model.
    """
    predictions = value_model.predict(
        input_ids,
        sequence_lengths=sequence_lengths,
        prompt_lengths=prompt_lengths,
        completion_lengths=completion_lengths,
    )
    return value_regression_loss(predictions, targets, loss_type=loss_type), predictions


def reward_model_regression_loss(
    reward_model: Any,
    input_ids: mx.array,
    sequence_lengths: mx.array,
    targets: mx.array,
    prompt_lengths: Optional[mx.array] = None,
    completion_lengths: Optional[mx.array] = None,
    loss_type: str = "mse",
) -> Tuple[mx.array, mx.array]:
    """
    Compute pointwise regression loss for a scalar reward model.
    """
    predictions = reward_model.score(
        input_ids,
        sequence_lengths=sequence_lengths,
        prompt_lengths=prompt_lengths,
        completion_lengths=completion_lengths,
    )
    return value_regression_loss(predictions, targets, loss_type=loss_type), predictions


def scalar_loss_metrics(loss: mx.array, predictions: mx.array, targets: mx.array) -> Dict[str, float]:
    """
    Compute generic scalar regression metrics.
    """
    mae = mx.mean(mx.abs(predictions - targets))
    mse = mx.mean((predictions - targets) ** 2)
    return {
        "loss": float(loss.item()),
        "mae": float(mae.item()),
        "mse": float(mse.item()),
    }


def pairwise_ranking_accuracy(
    chosen_scores: mx.array,
    rejected_scores: mx.array,
) -> float:
    """
    Compute pairwise ranking accuracy for scalar preference scores.
    """
    return float(mx.mean((chosen_scores > rejected_scores).astype(mx.float32)).item())


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


def ppo_sequence_loss(
    model: Any,
    batch: PolicyEvalBatch,
    beta: float = 0.0,
    clip_epsilon: float = 0.2,
    temperature: float = 1.0,
    reference_model: Optional[Any] = None,
) -> Tuple[mx.array, Dict[str, mx.array]]:
    """
    Compute a clipped PPO objective over full sampled completions.
    """
    scored_batch = score_policy(
        model,
        batch,
        mode="completion",
        reference_model=reference_model if batch.reference_logprobs is None else None,
        temperature=temperature,
    )
    old_logprobs = batch.old_logprobs if batch.old_logprobs is not None else batch.rollout_logprobs
    if old_logprobs is None:
        raise ValueError("PPO loss requires stored old log probabilities.")
    if batch.advantages is None:
        raise ValueError("PPO loss requires advantages.")

    ratios = mx.exp(scored_batch.summed_logprobs - old_logprobs)
    clipped_ratios = mx.clip(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    unclipped_objective = ratios * batch.advantages
    clipped_objective = clipped_ratios * batch.advantages
    policy_objective = mx.minimum(unclipped_objective, clipped_objective)

    kl_penalty = mx.zeros_like(policy_objective)
    if scored_batch.reference_logprobs is not None:
        kl_penalty = kl_against_reference(
            scored_batch.summed_logprobs,
            scored_batch.reference_logprobs,
        )
    loss = -mx.mean(policy_objective - beta * kl_penalty)
    return loss, {
        "policy_logprobs": scored_batch.summed_logprobs,
        "ratios": ratios,
        "kl_penalty": kl_penalty,
    }


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
    loss_type: str = "grpo",
    max_completion_length: Optional[int] = None,
    old_token_logprobs: Optional[mx.array] = None,
    reference_logprobs: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Recompute GRPO-family losses on fixed sampled completions.
    """
    batch = _policy_eval_from_padded(
        input_ids=input_ids,
        lengths=prompt_lengths + completion_lengths,
        mode="completion",
        prompt_lengths=prompt_lengths,
        completion_lengths=completion_lengths,
        rollout_logprobs=rollout_logprobs,
        old_logprobs=rollout_logprobs,
        old_token_logprobs=old_token_logprobs,
        reference_logprobs=reference_logprobs,
        advantages=advantages,
    )
    scored_batch = score_policy(
        model,
        batch,
        mode="completion",
        reference_model=reference_model if reference_logprobs is None else None,
        temperature=temperature,
    )

    sequence_logprobs = scored_batch.summed_logprobs
    normalized_current = normalize_logprobs(sequence_logprobs, completion_lengths, mode="mean")
    kl_scored_batch = scored_batch
    if beta != 0.0 and temperature != 1.0:
        kl_scored_batch = score_policy(
            model,
            batch,
            mode="completion",
            reference_model=reference_model if reference_logprobs is None else None,
            temperature=1.0,
        )
    if kl_scored_batch.reference_logprobs is not None:
        normalized_reference = normalize_logprobs(
            kl_scored_batch.reference_logprobs,
            completion_lengths,
            mode="mean",
        )
        normalized_kl_current = normalize_logprobs(
            kl_scored_batch.summed_logprobs,
            completion_lengths,
            mode="mean",
        )
        sequence_kl_penalty = kl_against_reference(normalized_kl_current, normalized_reference)
    else:
        sequence_kl_penalty = mx.zeros_like(advantages)

    if loss_type == "gspo":
        sequence_ratios = mx.exp(normalized_current - normalize_logprobs(rollout_logprobs, completion_lengths, mode="mean"))
        clipped_sequence_ratios = mx.clip(sequence_ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        policy_objective = mx.minimum(sequence_ratios * advantages, clipped_sequence_ratios * advantages)
        loss = -mx.mean(policy_objective - beta * sequence_kl_penalty)
        return loss, completion_lengths.sum()

    if old_token_logprobs is None:
        sequence_ratios = mx.exp(sequence_logprobs - rollout_logprobs)
        clipped_sequence_ratios = mx.clip(sequence_ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        policy_objective = mx.minimum(
            sequence_ratios * advantages,
            clipped_sequence_ratios * advantages,
        )
        loss = -mx.mean(policy_objective - beta * sequence_kl_penalty)
        return loss, completion_lengths.sum()

    mask = completion_token_mask(input_ids, prompt_lengths, completion_lengths).astype(mx.float32)
    current_token_logprobs = scored_batch.token_logprobs
    token_ratios = mx.exp(current_token_logprobs - old_token_logprobs)
    clipped_token_ratios = mx.clip(token_ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    per_token_objective = mx.minimum(
        token_ratios * advantages[:, None],
        clipped_token_ratios * advantages[:, None],
    )
    masked_objective = per_token_objective * mask

    if loss_type == "grpo":
        policy_objective = masked_objective.sum(axis=-1) / mx.maximum(mask.sum(axis=-1), 1.0)
        loss = -mx.mean(policy_objective - beta * sequence_kl_penalty)
    elif loss_type == "bnpo" or loss_type == "dapo":
        normalizer = mx.maximum(mask.sum(), 1.0)
        loss = -(
            masked_objective.sum() / normalizer
            - beta * mx.mean(sequence_kl_penalty)
        )
    elif loss_type == "dr_grpo":
        denominator = float(max_completion_length or int(mx.max(completion_lengths).item()) or 1)
        loss = -(
            masked_objective.sum() / max(completion_lengths.shape[0] * denominator, 1.0)
            - beta * mx.mean(sequence_kl_penalty)
        )
    else:
        raise ValueError(f"Unsupported GRPO loss_type: {loss_type}")
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
