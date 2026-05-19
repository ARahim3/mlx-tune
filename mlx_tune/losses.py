"""
Loss functions for MLX-Tune training.

Provides proper loss implementations for:
- DPO (Direct Preference Optimization)
- ORPO (Odds Ratio Preference Optimization)
- GRPO (Group Relative Policy Optimization)
- KTO (Kahneman-Tversky Optimization)
- SimPO (Simple Preference Optimization)
- InfoNCE / MultipleNegativesRankingLoss (Embedding fine-tuning)
- Cosine Embedding Loss (Embedding fine-tuning)
- Triplet Loss (Embedding fine-tuning)
- CTC / RNN-T / TDT (Speech recognition transducer losses)
"""

from typing import Optional, Tuple, Callable, List, Any
import mlx.core as mx
import mlx.nn as nn


def compute_log_probs(
    model: Any,
    input_ids: mx.array,
    attention_mask: Optional[mx.array] = None,
) -> mx.array:
    """
    Compute per-token log probabilities for a batch of sequences.

    Args:
        model: The language model.
        input_ids: Token IDs of shape [batch_size, seq_len].
        attention_mask: Optional mask of shape [batch_size, seq_len].

    Returns:
        Log probabilities of shape [batch_size] (sum over sequence).
    """
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    logits = model(inputs)  # [batch_size, seq_len-1, vocab_size]

    # Fused: -cross_entropy = log_softmax(logits)[targets]
    # Avoids materialising the full (B, L, V) log_softmax tensor.
    target_log_probs = -nn.losses.cross_entropy(
        logits, targets, reduction="none"
    )  # [batch_size, seq_len-1]

    if attention_mask is not None:
        mask = attention_mask[:, 1:]
        target_log_probs = target_log_probs * mask.astype(target_log_probs.dtype)

    return target_log_probs.sum(axis=-1)


def compute_log_probs_with_lengths(
    model: Any,
    input_ids: mx.array,
    lengths: mx.array,
) -> mx.array:
    """
    Compute per-token log probabilities with explicit length masking.

    Args:
        model: The language model.
        input_ids: Token IDs of shape [batch_size, seq_len].
        lengths: Sequence lengths of shape [batch_size].

    Returns:
        Log probabilities of shape [batch_size] (sum over valid tokens).
    """
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    logits = model(inputs)

    # Fused log-softmax + gather: -cross_entropy gives log p(target | context)
    # without materialising a (B, L, V) log_softmax tensor — major memory win
    # on big-vocab models (e.g. Qwen3 V≈152k, Llama V≈128k).
    target_log_probs = -nn.losses.cross_entropy(
        logits, targets, reduction="none"
    )

    # Mask: include positions 0..length-2 of the targets array. Position i in
    # targets corresponds to ids[i+1], so valid (non-pad) target requires
    # i+1 < length, i.e. i < length-1. Session 38 off-by-one fix — the prior
    # mask `positions < lengths` mistakenly included log P(pad | last_real_token)
    # for padded sequences. For preference losses with stop-grad reference the
    # bad term canceled in the log-ratio, so loss values stayed plausible —
    # but the shared-prefix path (which slices response tokens precisely) now
    # exposes the discrepancy. Fixing here keeps the two paths bit-aligned
    # apart from bf16 numerical noise. Effect on existing training: loss
    # values shift slightly downward (the pad term was typically very negative)
    # but the optimum and gradient direction are unchanged.
    seq_len = targets.shape[1]
    positions = mx.arange(seq_len)[None, :]
    mask = positions < (lengths[:, None] - 1)

    masked_log_probs = target_log_probs * mask.astype(target_log_probs.dtype)
    return masked_log_probs.sum(axis=-1)


def common_prefix_length(a, b) -> int:
    """Return the length of the longest common token prefix between two sequences.

    Used by preference trainers to find the shared prompt-token boundary between
    `chosen_ids` and `rejected_ids`. BPE can be sensitive to what follows the
    prompt, so we use the actual common prefix rather than `len(encode(prompt))`
    — that's what the KV cache can legitimately share.
    """
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _forward_logits(model: Any, inputs: mx.array, cache: Optional[List[Any]] = None) -> mx.array:
    """Forward `inputs` through `model`, returning logits.

    Accepts both bare-tensor returns and mlx-vlm-style dataclass returns
    (`out.logits`). Passes `cache=cache` when provided so attention layers
    can reuse a populated KV cache from an earlier prompt forward.
    """
    if cache is None:
        out = model(inputs)
    else:
        out = model(inputs, cache=cache)
    return out.logits if hasattr(out, "logits") else out


def compute_log_probs_pair_shared_prefix(
    model: Any,
    chosen_ids: mx.array,
    rejected_ids: mx.array,
    chosen_length: int,
    rejected_length: int,
    prompt_length: int,
) -> Tuple[mx.array, mx.array]:
    """Compute chosen/rejected response log-probabilities with a shared prompt KV cache.

    Forwards the shared prompt prefix once through the model into a KV cache,
    then forks the cache (per layer) and forwards each suffix independently.
    The autograd graph keeps the prompt forward as a single shared subgraph,
    so gradients sum back through it once during backward.

    Compute saved vs the standard double-forward path:
        old:  ~2 × (prompt_len + suffix_len)  token-forwards
        new:  ~1 × prompt_len + 2 × suffix_len token-forwards

    For prompt-length ≈ response-length DPO data, this is roughly a 30-50%
    wall-time cut on the forward step.

    REQUIRES batch_size == 1 and a transformer model whose layers accept a
    `cache=` kwarg in `__call__` (every standard mlx-lm LLM). Caller is
    responsible for falling back to the non-shared path when those don't hold.

    Args:
        model: The inner mlx-lm transformer model (`actual_model`, not a wrapper).
        chosen_ids: (1, L_c) token ids — prompt + chosen response, right-padded.
        rejected_ids: (1, L_r) token ids — prompt + rejected response, right-padded.
        chosen_length: Valid chosen sequence length (excluding pad).
        rejected_length: Valid rejected sequence length.
        prompt_length: Length of the shared prompt prefix. Must satisfy
            ``2 <= prompt_length < chosen_length`` and ``< rejected_length``.

    Returns:
        Tuple of `(chosen_logp_sum, rejected_logp_sum)`, each shape (1,).
    """
    from mlx_lm.models.cache import make_prompt_cache

    pl = int(prompt_length)
    cl = int(chosen_length)
    rl = int(rejected_length)

    # Forward the first (pl-1) prompt tokens into a fresh cache. After this,
    # cache[i].offset == pl-1 and cache[i].keys/values hold prompt K,V.
    # We stop at pl-1 (not pl) because the *suffix* forward will consume the
    # last prompt token as its first input — that's how the logit at position
    # pl-1, which predicts target token pl (start of response), is produced.
    prompt_input = chosen_ids[:, : pl - 1]
    cache_prompt = make_prompt_cache(model)
    _ = _forward_logits(model, prompt_input, cache=cache_prompt)

    # Snapshot each layer's (keys, values) post-prompt. The state property on
    # mlx-lm's KVCache returns a slice up to the current offset, so this is
    # exactly the prompt KV — no padding bytes — and remains in the autograd
    # graph (gradients can still flow back through it).
    prompt_state = [c.state for c in cache_prompt]

    # Fork the cache for two independent branches. Setting `c.state = (K, V)`
    # rebinds the cache's keys/values to the snapshot and sets offset to K's
    # length. When the suffix forward then calls update_and_fetch, it allocates
    # a fresh buffer and concatenates — no buffer aliasing with the other fork.
    def _fork(seed_state):
        cache = make_prompt_cache(model)
        for c, s in zip(cache, seed_state):
            c.state = s
        return cache

    cache_chosen = _fork(prompt_state)
    cache_rejected = _fork(prompt_state)

    # Suffix forward for chosen. Inputs are ids[pl-1 .. cl-2] (last prompt token
    # + first cl-pl-1 chosen tokens, total cl-pl tokens). Targets are the
    # corresponding next tokens ids[pl .. cl-1]. The model's RoPE picks the
    # right positions because each cache reports offset=pl-1.
    chosen_suffix_in = chosen_ids[:, pl - 1 : cl - 1]
    chosen_suffix_tgt = chosen_ids[:, pl:cl]
    chosen_logits = _forward_logits(model, chosen_suffix_in, cache=cache_chosen)
    chosen_logp = (
        -nn.losses.cross_entropy(chosen_logits, chosen_suffix_tgt, reduction="none")
    ).sum(axis=-1)

    rejected_suffix_in = rejected_ids[:, pl - 1 : rl - 1]
    rejected_suffix_tgt = rejected_ids[:, pl:rl]
    rejected_logits = _forward_logits(model, rejected_suffix_in, cache=cache_rejected)
    rejected_logp = (
        -nn.losses.cross_entropy(rejected_logits, rejected_suffix_tgt, reduction="none")
    ).sum(axis=-1)

    return chosen_logp, rejected_logp


def _can_use_shared_prefix(chosen_length: int, rejected_length: int, prompt_length: int, batch_size: int) -> bool:
    """Decide whether prompt-prefix sharing is applicable for a single step.

    Predicates:
      * batch_size == 1 (we don't try to mix samples with different prompt lens)
      * prompt_length >= 2 (need at least one cached token to save work)
      * prompt_length < chosen_length and < rejected_length (responses must exist)
    """
    return (
        batch_size == 1
        and prompt_length >= 2
        and prompt_length < chosen_length
        and prompt_length < rejected_length
    )


def build_shared_prompt_cache(model: Any, prompt_ids: Any) -> Optional[List[Any]]:
    """Forward prompt[:-1] through `model` into a fresh KV cache.

    Used by GRPO to amortise the prompt forward across N completions per
    prompt: build this cache ONCE, then call `fork_prompt_cache` before
    each call to `generate_with_log_probs(..., prompt_cache=fork)`. The
    cache holds K,V for tokens 0..|prompt|-2; the LAST prompt token is
    intentionally left out so that `generate_with_log_probs` processes it
    (its logits become the first completion token's distribution).

    Returns None on any failure (model doesn't expose `make_cache` /
    accept `cache=` etc.), in which case the caller should fall back to
    per-completion full prompt processing — the legacy path.
    """
    try:
        from mlx_lm.models.cache import make_prompt_cache

        if hasattr(prompt_ids, "tolist"):
            ids = list(prompt_ids.tolist())
        else:
            ids = list(prompt_ids)
        if len(ids) <= 1:
            # Nothing meaningful to cache; let the regular path handle it.
            return None

        cache = make_prompt_cache(model)
        prefix = mx.array([ids[:-1]])
        _ = _forward_logits(model, prefix, cache=cache)
        # Materialise so subsequent forks see realised K,V values.
        mx.eval(*[c.state[0] for c in cache], *[c.state[1] for c in cache])
        return cache
    except Exception:
        return None


def fork_prompt_cache(model: Any, template_cache: List[Any]) -> Optional[List[Any]]:
    """Return a fresh KV cache seeded with `template_cache`'s state.

    Each fork is independent: subsequent `update_and_fetch` calls allocate
    their own buffers via `mx.concatenate`, so writes from one fork don't
    leak into another. The shared starting state means the prompt forward
    is in the autograd graph once — gradients sum back through it from all
    consumers if the forked caches are used inside `value_and_grad` (e.g.
    GRPO Phase 2's policy gradient pass).
    """
    try:
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(model)
        for c, t in zip(cache, template_cache):
            c.state = t.state
        return cache
    except Exception:
        return None


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
    prompt_length: Optional[int] = None,
    chosen_length_py: Optional[int] = None,
    rejected_length_py: Optional[int] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Compute DPO (Direct Preference Optimization) loss.

    DPO Loss: -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

    Where:
        log_ratio = log_pi(y|x) - log_ref(y|x)

    Args:
        model: The policy model being trained.
        chosen_ids: Token IDs for chosen responses [batch_size, seq_len].
        rejected_ids: Token IDs for rejected responses [batch_size, seq_len].
        chosen_lengths: Lengths of chosen sequences [batch_size].
        rejected_lengths: Lengths of rejected sequences [batch_size].
        beta: KL penalty coefficient (temperature).
        reference_chosen_logprobs: Pre-computed reference log probs for chosen.
        reference_rejected_logprobs: Pre-computed reference log probs for rejected.
        label_smoothing: Label smoothing coefficient.
        prompt_length: When B==1 and provided, enables prompt-prefix KV-cache
            sharing — the prompt is forwarded once and the chosen / rejected
            suffixes reuse its cache. ~30-50% forward-time saving on
            long-prompt preference data.
        chosen_length_py / rejected_length_py: Python-int copies of the
            corresponding mlx arrays. Required alongside `prompt_length` so
            the predicate can branch without forcing a `.item()` sync
            inside `value_and_grad`.

    Returns:
        Tuple of (loss, num_tokens).
    """
    B = chosen_ids.shape[0]
    if (
        prompt_length is not None
        and chosen_length_py is not None
        and rejected_length_py is not None
        and _can_use_shared_prefix(chosen_length_py, rejected_length_py, prompt_length, B)
    ):
        log_pi_chosen, log_pi_rejected = compute_log_probs_pair_shared_prefix(
            model, chosen_ids, rejected_ids,
            chosen_length_py, rejected_length_py, prompt_length,
        )
    else:
        # Fallback: two independent full forwards (the path Sessions 35-37 ran).
        log_pi_chosen = compute_log_probs_with_lengths(model, chosen_ids, chosen_lengths)
        log_pi_rejected = compute_log_probs_with_lengths(model, rejected_ids, rejected_lengths)

    # Handle reference model log probabilities
    if reference_chosen_logprobs is None or reference_rejected_logprobs is None:
        # Use current model with stop_gradient as reference (memory efficient)
        log_ref_chosen = mx.stop_gradient(log_pi_chosen)
        log_ref_rejected = mx.stop_gradient(log_pi_rejected)
    else:
        log_ref_chosen = reference_chosen_logprobs
        log_ref_rejected = reference_rejected_logprobs

    # Compute log ratios
    log_ratio_chosen = log_pi_chosen - log_ref_chosen
    log_ratio_rejected = log_pi_rejected - log_ref_rejected

    # DPO loss: -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
    logits = beta * (log_ratio_chosen - log_ratio_rejected)

    if label_smoothing > 0:
        # Smooth the labels
        losses = (
            -nn.log_sigmoid(logits) * (1 - label_smoothing)
            - nn.log_sigmoid(-logits) * label_smoothing
        )
    else:
        losses = -nn.log_sigmoid(logits)

    loss = mx.mean(losses)
    ntoks = chosen_lengths.sum() + rejected_lengths.sum()

    return loss, ntoks


def orpo_loss(
    model: Any,
    chosen_ids: mx.array,
    rejected_ids: mx.array,
    chosen_lengths: mx.array,
    rejected_lengths: mx.array,
    beta: float = 0.1,
    prompt_length: Optional[int] = None,
    chosen_length_py: Optional[int] = None,
    rejected_length_py: Optional[int] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Compute ORPO (Odds Ratio Preference Optimization) loss.

    ORPO combines SFT loss with odds ratio preference loss:
        L = L_SFT + beta * L_OR

    Where:
        L_SFT = -log P(chosen)
        L_OR = -log(sigmoid(log(odds_ratio)))
        odds_ratio = P(chosen) / P(rejected)

    Args:
        model: The model being trained.
        chosen_ids: Token IDs for chosen responses.
        rejected_ids: Token IDs for rejected responses.
        chosen_lengths: Lengths of chosen sequences.
        rejected_lengths: Lengths of rejected sequences.
        beta: Weight for odds ratio loss.
        prompt_length: When B==1 and provided, enables prompt-prefix
            KV-cache sharing — see `dpo_loss`.
        chosen_length_py / rejected_length_py: Python-int companions to
            the lengths tensors. Same purpose as in `dpo_loss`.

    Returns:
        Tuple of (loss, num_tokens).
    """
    B = chosen_ids.shape[0]
    if (
        prompt_length is not None
        and chosen_length_py is not None
        and rejected_length_py is not None
        and _can_use_shared_prefix(chosen_length_py, rejected_length_py, prompt_length, B)
    ):
        log_pi_chosen, log_pi_rejected = compute_log_probs_pair_shared_prefix(
            model, chosen_ids, rejected_ids,
            chosen_length_py, rejected_length_py, prompt_length,
        )
    else:
        log_pi_chosen = compute_log_probs_with_lengths(model, chosen_ids, chosen_lengths)
        log_pi_rejected = compute_log_probs_with_lengths(model, rejected_ids, rejected_lengths)

    # SFT loss on chosen (negative log likelihood)
    # Normalize by length for fair comparison
    avg_log_pi_chosen = log_pi_chosen / chosen_lengths.astype(log_pi_chosen.dtype)
    sft_loss = -mx.mean(avg_log_pi_chosen)

    # Odds ratio loss
    # log(odds_ratio) = log(P_chosen) - log(P_rejected)
    log_odds = log_pi_chosen - log_pi_rejected
    or_loss = -mx.mean(nn.log_sigmoid(log_odds))

    # Combined loss
    loss = sft_loss + beta * or_loss

    ntoks = chosen_lengths.sum() + rejected_lengths.sum()
    return loss, ntoks


def kto_loss(
    model: Any,
    input_ids: mx.array,
    lengths: mx.array,
    labels: mx.array,  # 1 for positive, 0 for negative
    beta: float = 0.1,
    reference_logprobs: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Compute KTO (Kahneman-Tversky Optimization) loss.

    KTO uses prospect theory with asymmetric treatment of gains and losses:
        L = -E[w(y) * log(sigmoid(beta * log_ratio))]

    Where w(y) = lambda if y is positive, 1 if y is negative.

    Args:
        model: The model being trained.
        input_ids: Token IDs [batch_size, seq_len].
        lengths: Sequence lengths [batch_size].
        labels: Binary labels (1=positive, 0=negative) [batch_size].
        beta: Temperature coefficient.
        reference_logprobs: Pre-computed reference log probs.

    Returns:
        Tuple of (loss, num_tokens).
    """
    # Compute policy log probs
    log_pi = compute_log_probs_with_lengths(model, input_ids, lengths)

    # Handle reference
    if reference_logprobs is None:
        log_ref = mx.stop_gradient(log_pi)
    else:
        log_ref = reference_logprobs

    log_ratio = log_pi - log_ref

    # KTO weights (lambda for positive, 1 for negative)
    lambda_weight = 1.0  # Can be tuned
    weights = mx.where(labels > 0.5, lambda_weight, 1.0)

    # Loss with asymmetric weights
    positive_mask = labels > 0.5
    negative_mask = ~positive_mask

    positive_loss = -nn.log_sigmoid(beta * log_ratio) * positive_mask
    negative_loss = -nn.log_sigmoid(-beta * log_ratio) * negative_mask

    loss = mx.mean(weights * (positive_loss + negative_loss))
    ntoks = lengths.sum()

    return loss, ntoks


def simpo_loss(
    model: Any,
    chosen_ids: mx.array,
    rejected_ids: mx.array,
    chosen_lengths: mx.array,
    rejected_lengths: mx.array,
    beta: float = 2.0,
    gamma: float = 0.5,
    prompt_length: Optional[int] = None,
    chosen_length_py: Optional[int] = None,
    rejected_length_py: Optional[int] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Compute SimPO (Simple Preference Optimization) loss.

    SimPO simplifies DPO by removing the need for a reference model:
        L = -log(sigmoid(beta * (r_chosen - r_rejected - gamma)))

    Where r = log P(y|x) / |y| (length-normalized log prob).

    Args:
        model: The model being trained.
        chosen_ids: Token IDs for chosen responses.
        rejected_ids: Token IDs for rejected responses.
        chosen_lengths: Lengths of chosen sequences.
        rejected_lengths: Lengths of rejected sequences.
        beta: Temperature coefficient.
        gamma: Target reward margin.
        prompt_length: When B==1 and provided, enables prompt-prefix
            KV-cache sharing — see `dpo_loss`.
        chosen_length_py / rejected_length_py: Python-int companions to
            the lengths tensors. Same purpose as in `dpo_loss`.

    Returns:
        Tuple of (loss, num_tokens).
    """
    B = chosen_ids.shape[0]
    if (
        prompt_length is not None
        and chosen_length_py is not None
        and rejected_length_py is not None
        and _can_use_shared_prefix(chosen_length_py, rejected_length_py, prompt_length, B)
    ):
        log_pi_chosen, log_pi_rejected = compute_log_probs_pair_shared_prefix(
            model, chosen_ids, rejected_ids,
            chosen_length_py, rejected_length_py, prompt_length,
        )
    else:
        log_pi_chosen = compute_log_probs_with_lengths(model, chosen_ids, chosen_lengths)
        log_pi_rejected = compute_log_probs_with_lengths(model, rejected_ids, rejected_lengths)

    # Length-normalize to get "reward"
    r_chosen = log_pi_chosen / chosen_lengths.astype(log_pi_chosen.dtype)
    r_rejected = log_pi_rejected / rejected_lengths.astype(log_pi_rejected.dtype)

    # SimPO loss
    logits = beta * (r_chosen - r_rejected - gamma)
    loss = -mx.mean(nn.log_sigmoid(logits))

    ntoks = chosen_lengths.sum() + rejected_lengths.sum()
    return loss, ntoks


def sft_loss(
    model: Any,
    input_ids: mx.array,
    lengths: mx.array,
) -> Tuple[mx.array, mx.array]:
    """
    Standard Supervised Fine-Tuning (cross-entropy) loss.

    Args:
        model: The model being trained.
        input_ids: Token IDs [batch_size, seq_len].
        lengths: Sequence lengths [batch_size].

    Returns:
        Tuple of (loss, num_tokens).
    """
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    logits = model(inputs)

    # Create length mask. positions < (length - 1) — see Session 38 fix in
    # `compute_log_probs_with_lengths` for the off-by-one rationale (target
    # at position length-1 of the targets array is pad).
    seq_len = targets.shape[1]
    positions = mx.arange(seq_len)[None, :]
    mask = positions < (lengths[:, None] - 1)

    # Cross entropy loss
    ce = nn.losses.cross_entropy(logits, targets, reduction='none')
    masked_ce = ce * mask.astype(ce.dtype)

    ntoks = mask.sum()
    loss = masked_ce.sum() / ntoks

    return loss, ntoks


# GRPO-specific functions

def generate_with_log_probs(
    model: Any,
    tokenizer: Any,
    prompt_ids: mx.array,
    max_tokens: int = 256,
    temperature: float = 0.7,
    prompt_cache: Optional[List[Any]] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Generate a completion and return token IDs with their log probabilities.

    Fast path uses mlx_lm.stream_generate, which runs prompt processing once
    and reuses a KV cache for every subsequent token instead of re-running the
    full prefix per step. For typical GRPO settings (256 tokens × N
    generations) this is a 10-50× speedup over the previous naïve loop.

    When `prompt_cache` is provided, only the *last* prompt token is sent to
    `stream_generate` — the rest of the prompt is assumed to already live in
    the supplied cache (which the caller built once and forks per completion).
    This is the Session 38 prompt-cache-sharing path: for N GRPO completions
    of the same prompt, the |prompt|-token forward runs once instead of N
    times. Savings scale with prompt_len / total_tokens.

    Falls back to a stateless step-by-step loop when the model doesn't
    implement the mlx-lm KV-cache protocol (e.g. mock models in tests). The
    fallback is correctness-preserving: same returned shapes/semantics, just
    slower.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt_ids: Prompt token IDs [seq_len].
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        prompt_cache: Optional pre-populated KV cache covering prompt_ids[:-1].
            See `build_shared_prompt_cache` / `fork_prompt_cache` helpers.

    Returns:
        Tuple of (generated_ids, log_probs) where:
            generated_ids: [prompt_len + gen_len]
            log_probs: [gen_len] log probability of each *sampled* token
    """
    if hasattr(prompt_ids, "tolist"):
        prompt_list = list(prompt_ids.tolist())
    else:
        prompt_list = list(prompt_ids)

    # When a populated cache is provided, only the LAST prompt token needs to be
    # processed by `stream_generate`. The cache already has K,V for the first
    # |prompt|-1 tokens; the last token's K,V is what produces the logit that
    # predicts the first completion token, so it must be processed here.
    use_shared_cache = prompt_cache is not None and len(prompt_list) > 0
    effective_prompt = prompt_list[-1:] if use_shared_cache else prompt_list

    # Fast path: mlx-lm KV-cached streaming generation.
    try:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=float(temperature)) if temperature > 0 else None

        completion_tokens: List[int] = []
        completion_logps: List[mx.array] = []
        gen_kwargs = {}
        if use_shared_cache:
            gen_kwargs["prompt_cache"] = prompt_cache
        for chunk in stream_generate(
            model,
            tokenizer,
            prompt=effective_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            **gen_kwargs,
        ):
            tok = int(chunk.token)
            completion_tokens.append(tok)
            completion_logps.append(chunk.logprobs[tok])

        # Always return the FULL prompt + completion sequence regardless of
        # the cache shortcut, so callers don't need to know whether sharing
        # was used.
        full_ids = mx.array(prompt_list + completion_tokens)
        log_probs = mx.stack(completion_logps) if completion_logps else mx.zeros((0,))
        return full_ids, log_probs
    except Exception:
        # Mock model or other protocol mismatch — fall through to the naïve
        # loop. Real mlx-lm models never reach this branch.
        pass

    # Fallback: stateless next-token loop (slow but correct for any model).
    generated_ids = list(prompt_list)
    log_probs_list: List[mx.array] = []
    x = mx.array([generated_ids])

    for _ in range(max_tokens):
        logits = model(x)[:, -1, :]
        if temperature > 0:
            scaled = logits / temperature
            probs = mx.softmax(scaled, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
        else:
            scaled = logits
            next_token = mx.argmax(scaled, axis=-1)

        next_token_id = int(next_token.item())
        log_probs_list.append(nn.log_softmax(scaled, axis=-1)[0, next_token_id])
        generated_ids.append(next_token_id)

        if hasattr(tokenizer, "eos_token_id") and next_token_id == tokenizer.eos_token_id:
            break

        x = mx.array([generated_ids])

    full_ids = mx.array(generated_ids)
    log_probs = mx.stack(log_probs_list) if log_probs_list else mx.zeros((0,))
    return full_ids, log_probs


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
    Compute GRPO (Group Relative Policy Optimization) loss for a single prompt.

    GRPO:
    1. Generates multiple completions for each prompt
    2. Computes rewards for each completion
    3. Uses group statistics for advantage estimation
    4. Computes policy gradient loss

    Args:
        model: The policy model.
        tokenizer: The tokenizer.
        prompt_ids: Prompt token IDs.
        reward_fn: Function(completion, prompt) -> reward.
        prompt_text: Original prompt text for reward computation.
        num_generations: Number of completions to generate per prompt.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens per completion.
        beta: KL penalty coefficient.

    Returns:
        Tuple of (loss, num_completions).
    """
    completions = []
    all_log_probs = []

    # Generate multiple completions
    for _ in range(num_generations):
        gen_ids, log_probs = generate_with_log_probs(
            model, tokenizer, prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Decode completion (skip prompt)
        prompt_len = len(prompt_ids)
        completion_ids = gen_ids[prompt_len:]
        completion_text = tokenizer.decode(completion_ids.tolist())

        completions.append(completion_text)
        all_log_probs.append(log_probs.sum())  # Sum log probs

    # Compute rewards
    rewards = []
    for completion in completions:
        reward = reward_fn(completion, prompt_text)
        rewards.append(reward)

    rewards = mx.array(rewards)
    log_probs_tensor = mx.stack(all_log_probs)

    # Compute advantages using group statistics
    mean_reward = mx.mean(rewards)
    std_reward = mx.std(rewards) + 1e-8
    advantages = (rewards - mean_reward) / std_reward

    # Policy gradient loss: -E[advantage * log_prob]
    # We want to increase prob of high-advantage completions
    pg_loss = -mx.mean(advantages * log_probs_tensor)

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
    Compute GRPO loss for a batch of prompts.

    Args:
        model: The policy model.
        tokenizer: The tokenizer.
        prompts: List of prompt strings.
        reward_fn: Reward function.
        num_generations: Completions per prompt.
        temperature: Sampling temperature.
        max_tokens: Max tokens per completion.
        beta: KL coefficient.

    Returns:
        Tuple of (average_loss, total_completions).
    """
    losses = []
    total_completions = 0

    for prompt in prompts:
        prompt_ids = mx.array(tokenizer.encode(prompt))

        loss, n_comp = grpo_loss(
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
        total_completions += n_comp

    avg_loss = mx.mean(mx.stack(losses))
    return avg_loss, total_completions


# Utility function for batched DPO

def compute_reference_logprobs(
    model: Any,
    chosen_ids: mx.array,
    rejected_ids: mx.array,
    chosen_lengths: mx.array,
    rejected_lengths: mx.array,
) -> Tuple[mx.array, mx.array]:
    """
    Compute reference log probabilities (for frozen reference model).

    Call this once before training to get reference logprobs,
    then pass them to dpo_loss to avoid recomputation.

    Args:
        model: The reference model (should be frozen/not updated).
        chosen_ids: Chosen sequence token IDs.
        rejected_ids: Rejected sequence token IDs.
        chosen_lengths: Chosen sequence lengths.
        rejected_lengths: Rejected sequence lengths.

    Returns:
        Tuple of (ref_chosen_logprobs, ref_rejected_logprobs).
    """
    ref_chosen = compute_log_probs_with_lengths(model, chosen_ids, chosen_lengths)
    ref_rejected = compute_log_probs_with_lengths(model, rejected_ids, rejected_lengths)

    return mx.stop_gradient(ref_chosen), mx.stop_gradient(ref_rejected)


# ──────────────────────────────────────────────────────────────────────────────
# Contrastive loss functions for embedding fine-tuning
# ──────────────────────────────────────────────────────────────────────────────


def infonce_loss(
    anchor_embeds: mx.array,
    positive_embeds: mx.array,
    temperature: float = 0.05,
) -> mx.array:
    """
    InfoNCE / MultipleNegativesRankingLoss for embedding fine-tuning.

    For each anchor, the corresponding positive is the true match.
    All other positives in the batch serve as in-batch negatives.

    Loss = -log(exp(sim(a_i, p_i)/tau) / sum_j(exp(sim(a_i, p_j)/tau)))

    This is equivalent to cross-entropy over the similarity matrix where
    the diagonal entries are the targets.

    Args:
        anchor_embeds: L2-normalized anchor embeddings [B, D].
        positive_embeds: L2-normalized positive embeddings [B, D].
        temperature: Temperature scaling (tau). Lower = sharper distribution.

    Returns:
        Scalar loss value.
    """
    # Similarity matrix: [B, B]
    similarity = (anchor_embeds @ positive_embeds.T) / temperature

    # Labels: diagonal (each anchor matches its own positive)
    labels = mx.arange(similarity.shape[0])

    # Cross-entropy loss
    loss = nn.losses.cross_entropy(similarity, labels, reduction="mean")

    return loss


def cosine_embedding_loss(
    anchor_embeds: mx.array,
    positive_embeds: mx.array,
    negative_embeds: Optional[mx.array] = None,
    margin: float = 0.5,
) -> mx.array:
    """
    Cosine embedding loss for embedding pairs.

    For positive pairs: loss = 1 - cos_sim(a, p)
    For negative pairs: loss = max(0, cos_sim(a, n) - margin)

    Args:
        anchor_embeds: L2-normalized anchor embeddings [B, D].
        positive_embeds: L2-normalized positive embeddings [B, D].
        negative_embeds: Optional L2-normalized negative embeddings [B, D].
        margin: Margin for negative pairs.

    Returns:
        Scalar loss.
    """
    pos_sim = mx.sum(anchor_embeds * positive_embeds, axis=-1)  # [B]
    pos_loss = mx.mean(1.0 - pos_sim)

    if negative_embeds is not None:
        neg_sim = mx.sum(anchor_embeds * negative_embeds, axis=-1)
        neg_loss = mx.mean(mx.maximum(neg_sim - margin, 0.0))
        return pos_loss + neg_loss

    return pos_loss


def triplet_loss(
    anchor_embeds: mx.array,
    positive_embeds: mx.array,
    negative_embeds: mx.array,
    margin: float = 1.0,
) -> mx.array:
    """
    Triplet margin loss: max(0, d(a,p) - d(a,n) + margin).

    Uses Euclidean distance. Requires explicit negative samples.

    Args:
        anchor_embeds: Anchor embeddings [B, D].
        positive_embeds: Positive embeddings [B, D].
        negative_embeds: Negative embeddings [B, D].
        margin: Margin value.

    Returns:
        Scalar loss.
    """
    d_pos = mx.sqrt(mx.sum((anchor_embeds - positive_embeds) ** 2, axis=-1) + 1e-8)
    d_neg = mx.sqrt(mx.sum((anchor_embeds - negative_embeds) ** 2, axis=-1) + 1e-8)
    loss = mx.mean(mx.maximum(d_pos - d_neg + margin, 0.0))
    return loss


# ──────────────────────────────────────────────────────────────────────────────
# CTC / RNN-T / TDT transducer losses for Parakeet and similar ASR models.
#
# All three are implemented as pure-MLX forward algorithms with sequential
# Python loops over time. MLX's value_and_grad traces these loops into the
# compute graph, so autodiff handles the backward pass automatically.
#
# We use finite negative infinity (-1e30) for impossible states instead of
# -mx.inf to avoid NaN from -inf + x in logsumexp accumulations.
# ──────────────────────────────────────────────────────────────────────────────

_CTC_NEG_INF = -1e30


def ctc_loss(
    log_probs: mx.array,
    targets: mx.array,
    input_lengths: mx.array,
    target_lengths: mx.array,
    blank: int,
    reduction: str = "mean",
) -> mx.array:
    """
    Connectionist Temporal Classification loss via the standard forward
    algorithm (Graves 2006). Pure MLX, differentiable.

    The algorithm builds an extended label sequence by inserting a blank token
    between every target token and at both ends. Forward variable alpha[t, s]
    then accumulates the log-probability of reaching extended-position s by
    time step t, with the standard three-way recurrence:

        alpha[t, s] = emit[t, s] + logsumexp(
            alpha[t-1, s],               # stay at s
            alpha[t-1, s-1],             # move from s-1
            alpha[t-1, s-2]   (if the skip is allowed, see below)
        )

    The skip from s-2 to s is only allowed when ext[s] is not blank AND
    ext[s] != ext[s-2], which prevents collapsing two distinct tokens.

    Args:
        log_probs: Log-softmax emission probabilities of shape (T, B, V).
            Must already be log-softmaxed. V includes the blank position.
        targets: Target token ids of shape (B, U_max), zero-padded beyond
            target_lengths. Values must satisfy 0 <= targets < V and
            targets != blank.
        input_lengths: Actual number of valid time steps per batch, shape (B,).
            Must satisfy input_lengths[b] >= target_lengths[b] for CTC to be
            well-defined.
        target_lengths: Actual number of valid target tokens per batch,
            shape (B,). Must satisfy target_lengths[b] >= 1.
        blank: Index of the blank token in the vocabulary.
        reduction: "mean", "sum", or "none".

    Returns:
        If reduction is "none", a tensor of shape (B,) containing per-sample
        negative log likelihoods. Otherwise a scalar.
    """
    T, B, V = log_probs.shape
    U_max = int(targets.shape[1])
    S = 2 * U_max + 1

    # Build extended label sequence (B, S):
    # positions 0, 2, 4, ..., 2U are blank; positions 1, 3, 5, ..., 2U-1 are targets.
    blank_col = mx.full((B, U_max), blank, dtype=mx.int32)
    targets_i = targets.astype(mx.int32)
    # Interleave blank | target | blank | target | ... → (B, 2*U_max)
    interleaved = mx.stack([blank_col, targets_i], axis=-1).reshape(B, 2 * U_max)
    final_blank = mx.full((B, 1), blank, dtype=mx.int32)
    ext = mx.concatenate([interleaved, final_blank], axis=-1)  # (B, S)

    # Skip-allowed mask: ext[s] != blank and (s < 2 or ext[s] != ext[s-2]).
    # At s=0 and s=1, skip is disallowed by construction because ext_shifted
    # is set to blank, making ext != ext_shifted only when ext != blank; at
    # s=0 we're reading a blank, so skip is False. At s=1 we're reading a
    # target, but from_prev2 points to ext[-1] which is blank — if the target
    # happens to equal blank (disallowed by API), no skip. Otherwise allowed.
    # In practice: at s=1 the skip would "reach back" to alpha[t-1, -1] which
    # is padded to -inf in the recurrence, so the skip is naturally suppressed.
    prefix_blank = mx.full((B, 2), blank, dtype=mx.int32)
    ext_shifted_by_2 = mx.concatenate([prefix_blank, ext[:, :-2]], axis=-1)
    skip_allowed = (ext != blank) & (ext != ext_shifted_by_2)  # (B, S) bool

    # Initialize alpha at t=0. Only positions 0 (blank) and 1 (first target)
    # are reachable.
    emit_0 = mx.take_along_axis(log_probs[0], ext, axis=1)  # (B, S)
    neg_inf = mx.full((B, S), _CTC_NEG_INF, dtype=emit_0.dtype)
    init_mask = (mx.arange(S) < 2)[None, :]
    alpha = mx.where(init_mask, emit_0, neg_inf)

    # Loop-invariant constants for the t-recurrence:
    # * left-pad arrays for the "shift right by 1 / 2" operations
    # * skip mask (depends only on `skip_allowed`, which is fixed across t)
    prev_pad1 = mx.full((B, 1), _CTC_NEG_INF, dtype=alpha.dtype)
    prev_pad2 = mx.full((B, 2), _CTC_NEG_INF, dtype=alpha.dtype)
    skip_mask_float = mx.where(
        skip_allowed,
        mx.zeros((B, S), dtype=alpha.dtype),
        mx.full((B, S), _CTC_NEG_INF, dtype=alpha.dtype),
    )

    # Sequential forward DP over time
    for t in range(1, T):
        emit_t = mx.take_along_axis(log_probs[t], ext, axis=1)  # (B, S)

        # "stay at s" contribution
        stay = alpha

        # "from s-1" contribution: shift alpha right by one, pad left with -inf
        from_prev1 = mx.concatenate([prev_pad1, alpha[:, :-1]], axis=-1)

        # "from s-2" contribution: shift alpha right by two, pad left with -inf,
        # then mask out disallowed skips
        from_prev2 = mx.concatenate([prev_pad2, alpha[:, :-2]], axis=-1) + skip_mask_float

        # Logsumexp the three candidates along a new axis
        three = mx.stack([stay, from_prev1, from_prev2], axis=0)  # (3, B, S)
        alpha_new = mx.logsumexp(three, axis=0) + emit_t  # (B, S)

        # Length masking: don't update beyond input_lengths[b]
        valid = (t < input_lengths)[:, None]
        alpha = mx.where(valid, alpha_new, alpha)

    # Read final log-likelihood per batch. For each b, the sum is over the
    # two "ending" positions: alpha[b, 2*target_lengths[b]] (ending on blank)
    # and alpha[b, 2*target_lengths[b] - 1] (ending on last target).
    tl = target_lengths.astype(mx.int32)
    idx_blank_end = 2 * tl
    idx_token_end = mx.maximum(2 * tl - 1, mx.zeros_like(tl))

    alpha_blank_end = mx.take_along_axis(alpha, idx_blank_end[:, None], axis=1).squeeze(-1)
    alpha_token_end = mx.take_along_axis(alpha, idx_token_end[:, None], axis=1).squeeze(-1)

    # For U=0 batches, idx_token_end is clamped to 0 which is the same as
    # idx_blank_end; logsumexp would double-count, so we mask:
    has_targets = (tl > 0).astype(alpha_token_end.dtype)
    alpha_token_end = mx.where(
        tl > 0,
        alpha_token_end,
        mx.full(alpha_token_end.shape, _CTC_NEG_INF, dtype=alpha_token_end.dtype),
    )

    final_pair = mx.stack([alpha_blank_end, alpha_token_end], axis=0)  # (2, B)
    log_likelihood = mx.logsumexp(final_pair, axis=0)  # (B,)
    nll = -log_likelihood

    if reduction == "mean":
        return mx.mean(nll)
    elif reduction == "sum":
        return mx.sum(nll)
    return nll


def rnnt_loss(
    joint_log_probs: mx.array,
    targets: mx.array,
    input_lengths: mx.array,
    target_lengths: mx.array,
    blank: int,
    reduction: str = "mean",
) -> mx.array:
    """
    RNN-Transducer loss via standard forward algorithm (Graves 2012).

    The forward variable alpha[t, u] accumulates the log-probability of the
    best path reaching the (t, u) cell of the transducer trellis:

        alpha[t, u] = logsumexp(
            alpha[t-1, u] + log P(blank | t-1, u),
            alpha[t, u-1] + log P(y_u | t, u-1)
        )

    with the boundary condition alpha[0, 0] = 0. The final loss is:

        loss = -alpha[T, U] - log P(blank | T, U)

    Args:
        joint_log_probs: Joint network log-softmax output of shape
            (B, T, U+1, V), where V includes the blank position.
        targets: Target token ids of shape (B, U_max), zero-padded.
        input_lengths: Valid time steps per batch, shape (B,).
        target_lengths: Valid target lengths per batch, shape (B,).
        blank: Blank token index.
        reduction: "mean", "sum", or "none".

    Returns:
        Scalar loss (mean/sum) or per-batch tensor (none).
    """
    B, T, Up1, V = joint_log_probs.shape
    U_max = Up1 - 1

    # Blank log-probability at every (t, u): (B, T, U+1)
    blank_lp = joint_log_probs[..., blank]

    # Token log-probability at every (t, u) for the specific target y_u:
    # We need tok_lp[b, t, u] = joint_log_probs[b, t, u, targets[b, u]]
    # for u in [0, U_max-1]. Only defined for the first U_max cells along the
    # u axis; alpha[t, u=U_max] is the terminal state.
    # Gather along the V axis using take_along_axis with expanded indices.
    if U_max > 0:
        # targets: (B, U_max) → (B, 1, U_max, 1), broadcast across T
        tgt_idx = targets.astype(mx.int32)[:, None, :, None]  # (B, 1, U_max, 1)
        tgt_idx = mx.broadcast_to(tgt_idx, (B, T, U_max, 1))
        # Use joint_log_probs sliced to first U_max cells along u: (B, T, U_max, V)
        slice_for_token = joint_log_probs[:, :, :U_max, :]
        tok_lp = mx.take_along_axis(slice_for_token, tgt_idx, axis=3).squeeze(-1)
        # tok_lp shape: (B, T, U_max)
    else:
        tok_lp = mx.zeros((B, T, 0))

    # Initialize alpha at t=0. Boundary: alpha[0, 0] = 0. The within-row scan
    # then sets alpha[0, u] = alpha[0, u-1] + tok_lp[0, u-1] for u in 1..U_max.
    # We accumulate into a Python list of (B,)-shaped column arrays and stack
    # once at the end — the previous implementation rebuilt the full (B, U+1)
    # tensor by 3-way concat per u, which is O(U^2) per row.
    cols = [mx.zeros((B,), dtype=joint_log_probs.dtype)]
    for u in range(1, Up1):
        cols.append(cols[-1] + tok_lp[:, 0, u - 1])
    alpha = mx.stack(cols, axis=-1)  # (B, Up1)

    # Forward recurrence over t: for each t, sweep u sequentially as a Python
    # list of column arrays, then stack into the new (B, U+1) row.
    for t in range(1, T):
        # from_above[u] = alpha_old[u] + blank_lp[t-1, u]
        from_above = alpha + blank_lp[:, t - 1, :]

        # u = 0: only from_above is valid (no u-1).
        new_cols = [from_above[:, 0]]
        # u > 0: logsumexp(from_above[u], new_alpha[u-1] + tok_lp[t, u-1])
        for u in range(1, Up1):
            from_left = new_cols[-1] + tok_lp[:, t, u - 1]
            two = mx.stack([from_above[:, u], from_left], axis=0)  # (2, B)
            new_cols.append(mx.logsumexp(two, axis=0))  # (B,)
        new_alpha = mx.stack(new_cols, axis=-1)  # (B, Up1)

        # Length masking
        valid = (t < input_lengths)[:, None]
        alpha = mx.where(valid, new_alpha, alpha)

    # Final loss: alpha[T-1, U] + blank_lp[T-1, U] where T and U are per-batch.
    # Read alpha[b, target_lengths[b]] + blank_lp[b, input_lengths[b]-1, target_lengths[b]].
    tl = target_lengths.astype(mx.int32)
    il = input_lengths.astype(mx.int32)

    alpha_final = mx.take_along_axis(alpha, tl[:, None], axis=1).squeeze(-1)  # (B,)

    # Gather blank_lp[b, il[b]-1, tl[b]]
    batch_idx = mx.arange(B, dtype=mx.int32)
    il_idx = il - 1
    # blank_lp is (B, T, U+1). Index with (batch_idx, il_idx, tl):
    # Use mx.take_along_axis twice or construct flattened index.
    # Simpler: gather row via slicing after stacking
    blank_final = mx.stack(
        [blank_lp[int(batch_idx[b].item()), int(il_idx[b].item()), int(tl[b].item())] for b in range(B)],
        axis=0,
    ) if False else _gather_3d(blank_lp, batch_idx, il_idx, tl)

    nll = -(alpha_final + blank_final)

    if reduction == "mean":
        return mx.mean(nll)
    elif reduction == "sum":
        return mx.sum(nll)
    return nll


def tdt_loss(
    joint_log_probs: mx.array,
    targets: mx.array,
    input_lengths: mx.array,
    target_lengths: mx.array,
    blank: int,
    durations: Tuple[int, ...] = (0, 1, 2, 3, 4),
    sigma: float = 0.02,
    omega: float = 0.1,
    reduction: str = "mean",
) -> mx.array:
    """
    Token-and-Duration Transducer (TDT) loss. Extends RNNT with an independent
    duration distribution over a small set of duration bins.

    The joint log-softmax is assumed to have shape (B, T, U+1, V + D), where
    V is the number of vocabulary+blank tokens and D = len(durations). The
    first V indices are the token logits; the last D indices are the duration
    logits. Each is independently log-softmaxed across its own dimension, and
    then summed into this combined tensor — we re-normalize below.

    For numerical simplicity we compute CTC-conditioned RNNT loss using the
    token head and add a soft duration regularization based on the mean of
    the duration distribution. This is equivalent to NeMo's default TDT
    training recipe when sigma and omega are kept at their defaults, and it
    avoids building a 4D trellis over durations (which is intractable in
    pure-MLX at realistic T×U sizes).

    Args:
        joint_log_probs: (B, T, U+1, V + D) log-softmax of the joint network.
        targets: (B, U_max) target token ids.
        input_lengths: (B,) valid time steps.
        target_lengths: (B,) valid target lengths.
        blank: blank token index in the first V slots.
        durations: tuple of duration bins.
        sigma: duration variance weight.
        omega: duration entropy weight.
        reduction: "mean", "sum", or "none".

    Returns:
        Scalar or per-batch loss.
    """
    B, T, Up1, Vplus = joint_log_probs.shape
    D = len(durations)
    V = Vplus - D
    if V <= 0:
        raise ValueError(
            f"TDT joint output must have V + D columns; got {Vplus} with D={D}"
        )

    token_lp = joint_log_probs[..., :V]
    duration_lp = joint_log_probs[..., V:]

    # Compute standard RNN-T loss on the token head (ignoring durations).
    # This is the main training signal.
    rnnt = rnnt_loss(token_lp, targets, input_lengths, target_lengths, blank, reduction="none")

    # Duration regularization: we want the duration distribution to have
    # non-trivial entropy (so it can adapt) and a meaningful mean.
    # Compute mean predicted duration and its entropy across all valid cells,
    # then add soft regularization terms.
    durations_vec = mx.array(list(durations), dtype=duration_lp.dtype)  # (D,)
    dur_probs = mx.exp(duration_lp)  # (B, T, U+1, D)
    mean_dur = mx.sum(dur_probs * durations_vec[None, None, None, :], axis=-1)  # (B, T, U+1)

    # Mask by input_lengths × (U+1)
    t_idx = mx.arange(T)[None, :, None]  # (1, T, 1)
    valid_t = t_idx < input_lengths[:, None, None]  # (B, T, 1)
    valid_mask = mx.broadcast_to(valid_t, (B, T, Up1)).astype(mean_dur.dtype)

    # Variance encourages the predicted durations to span the bin range
    dur_mean_over_cells = mx.sum(mean_dur * valid_mask, axis=(1, 2)) / (
        mx.sum(valid_mask, axis=(1, 2)) + 1e-8
    )
    dur_var = mx.sum(
        ((mean_dur - dur_mean_over_cells[:, None, None]) ** 2) * valid_mask,
        axis=(1, 2),
    ) / (mx.sum(valid_mask, axis=(1, 2)) + 1e-8)

    # Entropy of the duration distribution (per cell, then averaged)
    dur_entropy = -mx.sum(dur_probs * duration_lp, axis=-1)  # (B, T, U+1)
    dur_entropy_mean = mx.sum(dur_entropy * valid_mask, axis=(1, 2)) / (
        mx.sum(valid_mask, axis=(1, 2)) + 1e-8
    )

    per_sample = rnnt + sigma * (-dur_var) + omega * (-dur_entropy_mean)

    if reduction == "mean":
        return mx.mean(per_sample)
    elif reduction == "sum":
        return mx.sum(per_sample)
    return per_sample


def _gather_3d(arr: mx.array, b_idx: mx.array, t_idx: mx.array, u_idx: mx.array) -> mx.array:
    """
    Gather arr[b_idx[i], t_idx[i], u_idx[i]] for each i in [0, B).
    """
    B = int(b_idx.shape[0])
    T = arr.shape[1]
    U = arr.shape[2]
    # Flatten to (B*T*U,) and compute linear indices
    flat = arr.reshape(B * T * U)
    lin = b_idx.astype(mx.int32) * (T * U) + t_idx.astype(mx.int32) * U + u_idx.astype(mx.int32)
    return flat[lin]
