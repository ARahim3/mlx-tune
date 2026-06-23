"""
LLM-JEPA: a Joint-Embedding Predictive objective for LLM fine-tuning.

Implements the training objective from *LLM-JEPA: Large Language Models Meet
Joint Embedding Predictive Architectures* (Huang, LeCun & Balestriero,
arXiv 2509.14252; repo ``rbalestr-lab/llm-jepa``).

LLM-JEPA is **not a model** — it is an extra loss term bolted onto standard
next-token prediction (NTP). Each training item has two "views" that express the
same content in different surface forms — e.g. a natural-language description
(``text``) and its regex / SQL / code (``code``). The objective is::

    L = L_NTP  +  lambda * d( Pred(Enc(text)), Enc(code) )

where ``Enc(x)`` is the last-token, last-layer hidden state of the LLM for view
``x``, ``Pred`` is an (optional) predictor formed by appending ``k`` learnable
``[PRED]`` slots, and ``d`` is a cosine dissimilarity (default) / l2 / mse /
infonce distance. The NTP term is the usual causal-LM loss, computed over the
concatenation of the combined sequence and the two views (matching the
reference). With the default ``num_predictors=0`` the predictor is the identity,
so the JEPA term simply pulls the two views' representations together.

The fine-tuned artifact is a **normal LoRA-fine-tuned LLM** — save / merge /
generate all work exactly as for SFT. The predictor slots only shape the
representation during training and are not needed for downstream generation.

Faithful reference defaults (read from the repo's ``finetune.py``):
    * cosine dissimilarity ``1 - mean(cos(pred, target))`` (variants: l2/mse/infonce)
    * **no** stop-gradient on the target view (symmetric, both encoded with grad)
    * NTP over the 3-way concat ``[combined, text, code]``
    * ``num_predictors=0`` (predictor = identity); ``lambda = 0.1``
    * ``Enc`` = last non-pad token, final layer

Deviations (documented):
    * ``k>0`` predictors use appended *trainable embeddings* via the model's
      ``input_embeddings=`` forward path instead of resizing the vocabulary with
      ``<|predictor_i|>`` tokens — behaviourally equivalent (the LLM's transform
      of ``view + pred slots``), and avoids a quantized-embedding resize. The
      ``k>0`` predictor forward runs per-sample to dodge right-padding
      contamination; the default ``k=0`` path is fully batched.

This module is intentionally self-contained (like ``jepa.py``): the loss helpers
live here rather than in ``losses.py``.
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

try:
    import mlx.optimizers as optim
    HAS_NATIVE_TRAINING = True
except ImportError:  # pragma: no cover - mlx always ships optimizers
    HAS_NATIVE_TRAINING = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class LLMJEPAConfig:
    """Configuration for :class:`LLMJEPATrainer`.

    Mirrors the SFT/DPO config style used elsewhere in mlx-tune, plus the
    JEPA-specific knobs.

    Args:
        jepa_lambda: Weight on the JEPA term (``L = L_NTP + lambda * L_JEPA``).
            Reference default ``0.1``.
        jepa_distance: Distance for the JEPA term — ``"cosine"`` (default),
            ``"l2"``, ``"mse"`` or ``"infonce"``.
        jepa_ratio: JEPA-loss dropout. If ``> 0`` it is the *probability of
            keeping* the JEPA term on a given step (so ``0.25`` skips it 75% of
            the time, amortising the extra forward). ``-1`` disables dropout
            (always on). Matches the reference's amortisation knob.
        num_predictors: ``k`` learnable ``[PRED]`` slots appended to the ``text``
            view to form the predictor. ``0`` (default) → predictor is identity.
        front_pred: Prepend the predictor slots instead of appending.
        text_field / code_field: Dataset keys for the two views. Falls back to
            ``prompt`` / ``completion`` when these are absent.
        ntp_on: ``"all"`` → NTP over ``[combined, text, code]`` (reference);
            ``"combined"`` → NTP over the combined sequence only.
        response_only: Mask the ``text`` prefix in the *combined* sequence so NTP
            trains only on the ``code`` continuation (standard SFT prompt
            masking). The two view rows are always trained in full.
        temperature: InfoNCE temperature (only used when ``jepa_distance="infonce"``).
    """

    def __init__(
        self,
        # JEPA knobs
        jepa_lambda: float = 0.1,
        jepa_distance: str = "cosine",
        jepa_ratio: float = -1.0,
        num_predictors: int = 0,
        front_pred: bool = False,
        text_field: str = "text",
        code_field: str = "code",
        ntp_on: str = "all",
        response_only: bool = False,
        temperature: float = 0.07,
        # Training args (SFT-style)
        output_dir: str = "./llm_jepa_outputs",
        learning_rate: float = 2e-4,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 1,
        num_train_epochs: int = 3,
        max_steps: int = -1,
        warmup_ratio: float = 0.03,
        warmup_steps: int = 0,
        logging_steps: int = 10,
        save_steps: int = 100,
        max_seq_length: int = 1024,
        weight_decay: float = 0.01,
        grad_checkpoint: bool = False,
        **kwargs,
    ):
        self.jepa_lambda = jepa_lambda
        self.jepa_distance = jepa_distance
        self.jepa_ratio = jepa_ratio
        self.num_predictors = num_predictors
        self.front_pred = front_pred
        self.text_field = text_field
        self.code_field = code_field
        self.ntp_on = ntp_on
        self.response_only = response_only
        self.temperature = temperature

        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_seq_length = max_seq_length
        self.weight_decay = weight_decay
        self.grad_checkpoint = grad_checkpoint

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Loss core (unit-testable, no trainer state)
# ---------------------------------------------------------------------------
def _resolve_decoder(base: Any) -> Tuple[Any, Any]:
    """Locate the inner decoder (returns hidden states) and the LM head.

    mlx-lm models nest at different depths: plain ``llama``/``qwen3`` are
    ``Model.model`` (2-level), while ``qwen3_5`` is
    ``Model.language_model.model`` (3-level) and VLMs use ``.language_model``.
    Walk ``.model`` / ``.language_model`` down to the module that owns
    ``embed_tokens`` — calling that module returns post-norm hidden states.

    The head is the first ``lm_head`` found along the chain (untied models) or
    ``inner.embed_tokens.as_linear`` (tied). ``inner`` is *not* stored on any
    nn.Module here — callers re-resolve it from ``base`` so its parameters are
    never double-registered.

    Returns ``(inner_decoder, head_callable)``.
    """
    chain: List[Any] = []
    node = base
    for _ in range(6):
        chain.append(node)
        if hasattr(node, "embed_tokens"):
            break
        nxt = getattr(node, "model", None)
        if nxt is None:
            nxt = getattr(node, "language_model", None)
        if nxt is None:
            break
        node = nxt
    inner = chain[-1]
    if not hasattr(inner, "embed_tokens"):
        raise AttributeError(
            f"LLM-JEPA: could not locate a decoder with embed_tokens on "
            f"{type(base).__name__}"
        )
    head = None
    for n in chain:
        lm = getattr(n, "lm_head", None)
        if lm is not None:
            head = lm
            break
    if head is None:
        head = inner.embed_tokens.as_linear
    return inner, head


def _logits_from_hidden(base: Any, h: mx.array) -> mx.array:
    """Project final hidden states to vocab logits via the resolved LM head."""
    _, head = _resolve_decoder(base)
    return head(h)


def _last_token(h: mx.array, lengths: mx.array) -> mx.array:
    """Gather the final non-pad token's hidden state per row.

    Args:
        h: ``[B, L, H]`` hidden states.
        lengths: ``[B]`` token counts (right-padded sequences).

    Returns:
        ``[B, H]`` last-token representations.
    """
    idx = mx.clip(lengths - 1, 0, h.shape[1] - 1)
    return h[mx.arange(h.shape[0]), idx, :]


def _l2_normalize(x: mx.array, eps: float = 1e-8) -> mx.array:
    return x / (mx.linalg.norm(x, axis=-1, keepdims=True) + eps)


def _jepa_distance(
    pred: mx.array,
    target: mx.array,
    kind: str = "cosine",
    temperature: float = 0.07,
) -> mx.array:
    """JEPA distance between predicted and target view embeddings.

    ``pred`` / ``target`` are ``[B, H]``. Returns a scalar. Formulas match the
    reference ``finetune.py``:

    * ``cosine`` → ``1 - mean(cos(pred, target))``
    * ``l2``     → ``mean(||pred - target||_2)``
    * ``mse``    → ``mean((pred - target)^2)``
    * ``infonce``→ cross-entropy on the normalised similarity matrix (in-batch
      negatives, diagonal positives)
    """
    if kind == "cosine":
        cos = (_l2_normalize(pred) * _l2_normalize(target)).sum(axis=-1)
        return 1.0 - cos.mean()
    if kind == "l2":
        return mx.linalg.norm(pred - target, axis=-1).mean()
    if kind == "mse":
        return ((pred - target) ** 2).mean()
    if kind == "infonce":
        p = _l2_normalize(pred)
        t = _l2_normalize(target)
        logits = (p @ t.T) / temperature
        labels = mx.arange(p.shape[0])
        return nn.losses.cross_entropy(logits, labels, reduction="mean")
    raise ValueError(f"Unknown jepa_distance: {kind!r}")


def _ntp_loss(
    logits: mx.array,
    input_ids: mx.array,
    lengths: mx.array,
    mask_until: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """Masked causal-LM (next-token) loss.

    Shares the masking convention of ``losses.sft_loss`` (positions
    ``< length - 1`` are valid targets; the token at ``length-1`` is pad).
    ``mask_until`` optionally masks a leading prefix per row (response-only).

    Returns ``(loss, n_tokens)``.
    """
    shift_logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    seq_len = targets.shape[1]
    positions = mx.arange(seq_len)[None, :]

    valid = positions < (lengths[:, None] - 1)
    if mask_until is not None:
        valid = valid & (positions >= mask_until[:, None])

    ce = nn.losses.cross_entropy(shift_logits, targets, reduction="none")
    mask = valid.astype(ce.dtype)
    ntoks = mask.sum()
    loss = (ce * mask).sum() / mx.maximum(ntoks, 1.0)
    return loss, ntoks


class _JEPAModule(nn.Module):
    """Differentiation target: the base LLM plus optional ``[PRED]`` slots.

    Wrapping lets ``nn.value_and_grad`` reach both the LoRA adapters (inside
    ``base``) and the predictor-slot embeddings in one call. The base is frozen
    except for its LoRA layers, so ``trainable_parameters()`` is
    ``{LoRA, pred_tokens}``.
    """

    def __init__(self, base: nn.Module, num_predictors: int, hidden_size: int):
        super().__init__()
        self.base = base
        self.num_predictors = num_predictors
        if num_predictors > 0:
            self.pred_tokens = mx.random.normal((num_predictors, hidden_size)) * 0.02

    def _predict(self, text_ids: mx.array, text_lengths: mx.array, front: bool) -> mx.array:
        """Run the predictor over each ``text`` view + ``k`` learnable slots.

        Per-sample (un-padded) forward via the ``input_embeddings=`` path so the
        appended slots never attend to right-padding. Returns ``[B, H]`` — the
        final slot's last-layer hidden state.
        """
        inner, _ = _resolve_decoder(self.base)  # inner decoder (returns hidden states)
        embed = inner.embed_tokens
        preds = []
        B = text_ids.shape[0]
        for i in range(B):
            Ti = int(text_lengths[i])
            ids_i = text_ids[i : i + 1, :Ti]            # [1, Ti]
            tok_emb = embed(ids_i)                       # [1, Ti, H]
            slots = self.pred_tokens[None]               # [1, k, H]
            if front:
                full = mx.concatenate([slots, tok_emb], axis=1)
            else:
                full = mx.concatenate([tok_emb, slots], axis=1)
            h = inner(None, input_embeddings=full)       # token ids unused when emb given
            preds.append(h[:, -1, :])                     # final slot (appended) / last tok
        return mx.concatenate(preds, axis=0)


def llm_jepa_loss(
    jmod: _JEPAModule,
    input_ids: mx.array,
    lengths: mx.array,
    group_size: int,
    do_jepa: bool,
    *,
    jepa_lambda: float = 0.1,
    jepa_distance: str = "cosine",
    num_predictors: int = 0,
    front_pred: bool = False,
    ntp_on: str = "all",
    mask_until: Optional[mx.array] = None,
    temperature: float = 0.07,
) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
    """Combined LLM-JEPA loss over a ``[3B, L]`` batch.

    The batch is three stacked groups of ``B`` rows: ``[combined, text, code]``.
    A single ``base.model`` forward yields the hidden states used for both the
    NTP logits (via the model head) and the two view encodings.

    Returns ``(total_loss, (ntp_loss, jepa_loss))`` — the aux tuple is for
    logging; gradients flow through ``total_loss``.
    """
    base = jmod.base
    B = group_size

    inner, head = _resolve_decoder(base)
    h = inner(input_ids)                       # [3B, L, H] post-norm hidden states
    logits = head(h)                           # [3B, L, V]

    # --- NTP term -----------------------------------------------------------
    if ntp_on == "combined":
        ntp, _ = _ntp_loss(
            logits[:B], input_ids[:B], lengths[:B],
            None if mask_until is None else mask_until[:B],
        )
    else:  # "all" — reference: causal LM over [combined, text, code]
        ntp, _ = _ntp_loss(logits, input_ids, lengths, mask_until)

    # --- JEPA term ----------------------------------------------------------
    if do_jepa:
        enc = _last_token(h, lengths)         # [3B, H]
        enc_code = enc[2 * B : 3 * B]
        if num_predictors > 0:
            pred_text = jmod._predict(
                input_ids[B : 2 * B], lengths[B : 2 * B], front_pred
            )
        else:
            pred_text = enc[B : 2 * B]        # predictor = identity
        jepa = _jepa_distance(pred_text, enc_code, jepa_distance, temperature)
    else:
        jepa = mx.zeros((), dtype=ntp.dtype)

    total = ntp + jepa_lambda * jepa
    return total, (ntp, jepa)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def _make_lr_schedule(lr: float, warmup: int, total: int):
    """Linear warmup → cosine decay (falls back to plain cosine if no warmup)."""
    total = max(1, total)
    warmup = max(0, min(warmup, total - 1)) if total > 1 else 0
    if warmup > 0:
        w = optim.linear_schedule(0.0, lr, warmup)
        c = optim.cosine_decay(lr, max(1, total - warmup))
        return optim.join_schedules([w, c], [warmup])
    return optim.cosine_decay(lr, total)


class LLMJEPATrainer:
    """Fine-tune an LLM with the LLM-JEPA objective (NTP + JEPA view alignment).

    Usage mirrors the RL trainers::

        >>> from mlx_tune import FastLanguageModel, LLMJEPATrainer, LLMJEPAConfig
        >>> model, tok = FastLanguageModel.from_pretrained("...")
        >>> model = FastLanguageModel.get_peft_model(model, r=16)
        >>> data = [{"text": "match digits", "code": r"\\d+"}, ...]
        >>> LLMJEPATrainer(model, data, tokenizer=tok,
        ...                args=LLMJEPAConfig(jepa_lambda=0.1)).train()

    The dataset is a list of dicts with the two views under
    ``args.text_field`` / ``args.code_field`` (default ``text`` / ``code``),
    falling back to ``prompt`` / ``completion``.
    """

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        tokenizer: Optional[Any] = None,
        args: Optional[LLMJEPAConfig] = None,
        **kwargs,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError("LLMJEPATrainer needs a tokenizer (pass tokenizer=...).")

        self.config = args or LLMJEPAConfig()
        c = self.config
        self.jepa_lambda = c.jepa_lambda
        self.jepa_distance = c.jepa_distance
        self.jepa_ratio = c.jepa_ratio
        self.num_predictors = int(c.num_predictors)
        self.front_pred = c.front_pred
        self.text_field = c.text_field
        self.code_field = c.code_field
        self.ntp_on = c.ntp_on
        self.response_only = c.response_only
        self.temperature = c.temperature

        self.learning_rate = c.learning_rate
        self.batch_size = max(1, c.per_device_train_batch_size)
        self.max_steps = c.max_steps
        self.max_seq_length = c.max_seq_length
        self.warmup_ratio = c.warmup_ratio
        self.warmup_steps = c.warmup_steps
        self.logging_steps = c.logging_steps
        self.save_steps = c.save_steps
        self.weight_decay = c.weight_decay

        self.output_dir = Path(c.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_path = self.output_dir / "adapters"
        self.adapter_path.mkdir(parents=True, exist_ok=True)

        # iters: max_steps wins, else epochs over the dataset
        if self.max_steps > 0:
            self.iters = self.max_steps
        else:
            n = len(train_dataset) if hasattr(train_dataset, "__len__") else 100
            self.iters = max(1, (n // self.batch_size) * c.num_train_epochs)

        print("LLMJEPATrainer initialized:")
        print(f"  JEPA: lambda={self.jepa_lambda} distance={self.jepa_distance} "
              f"k={self.num_predictors} ratio={self.jepa_ratio}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Iterations: {self.iters}")
        print(f"  Batch size: {self.batch_size}")

    # -- data ----------------------------------------------------------------
    def _views(self, sample: Dict) -> Tuple[str, str]:
        text = sample.get(self.text_field, sample.get("prompt", ""))
        code = sample.get(self.code_field, sample.get("completion", ""))
        return str(text), str(code)

    def _encode(self, s: str) -> List[int]:
        ids = self.tokenizer.encode(s)
        if not ids:
            ids = [self.tokenizer.eos_token_id or 0]
        if len(ids) > self.max_seq_length:
            ids = ids[: self.max_seq_length]
        return ids

    def _pad_id(self) -> int:
        return (
            getattr(self.tokenizer, "pad_token_id", None)
            or getattr(self.tokenizer, "eos_token_id", None)
            or 0
        )

    def _prepare_batch(self, samples: List[Dict]):
        """Build the ``[3B, L]`` padded batch of ``[combined, text, code]``.

        Returns ``(input_ids, lengths, B, mask_until)`` where ``mask_until`` is
        ``None`` unless response-only masking is on.
        """
        B = len(samples)
        text_rows, code_rows, comb_rows, comb_prefix = [], [], [], []
        for s in samples:
            text, code = self._views(s)
            t_ids = self._encode(text)
            c_ids = self._encode(code)
            combined = (t_ids + c_ids)[: self.max_seq_length]
            text_rows.append(t_ids)
            code_rows.append(c_ids)
            comb_rows.append(combined)
            comb_prefix.append(min(len(t_ids), len(combined)))

        rows = comb_rows + text_rows + code_rows           # groups: combined, text, code
        lengths = [len(r) for r in rows]
        max_len = max(lengths)
        pad = self._pad_id()
        padded = [r + [pad] * (max_len - len(r)) for r in rows]

        input_ids = mx.array(padded)
        lengths_arr = mx.array(lengths)

        mask_until = None
        if self.response_only:
            # mask the text prefix only in the combined rows; views train fully
            mu = comb_prefix + [0] * B + [0] * B
            mask_until = mx.array(mu)
        return input_ids, lengths_arr, B, mask_until

    # -- training ------------------------------------------------------------
    def train(self):
        from mlx_tune._perf import configure_wired_limit, enable_grad_checkpoint
        from mlx_tune.rl_trainers import _save_adapters_and_config

        print("=" * 70)
        print("Starting LLM-JEPA Training")
        print("=" * 70)
        configure_wired_limit()

        # LoRA + grad checkpointing + train mode
        if hasattr(self.model, "_apply_lora") and not getattr(self.model, "_lora_applied", False):
            print("Applying LoRA adapters...")
            self.model._apply_lora()
        enable_grad_checkpoint(self.model)

        base = self.model.model if hasattr(self.model, "model") else self.model
        if hasattr(self.model, "train"):
            self.model.train()
        if hasattr(base, "train"):
            base.train()

        hidden_size = self._infer_hidden_size(base) if self.num_predictors > 0 else 0
        jmod = _JEPAModule(base, self.num_predictors, hidden_size)

        lr_schedule = _make_lr_schedule(
            self.learning_rate, self._warmup_iters(), self.iters
        )
        optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=self.weight_decay)

        def loss_fn(m, input_ids, lengths, B, do_jepa, mask_until):
            return llm_jepa_loss(
                m, input_ids, lengths, B, do_jepa,
                jepa_lambda=self.jepa_lambda,
                jepa_distance=self.jepa_distance,
                num_predictors=self.num_predictors,
                front_pred=self.front_pred,
                ntp_on=self.ntp_on,
                mask_until=mask_until,
                temperature=self.temperature,
            )

        loss_and_grad = nn.value_and_grad(jmod, loss_fn)

        data = list(self.train_dataset)
        n = len(data)
        bs = self.batch_size

        print(f"\nStarting training for {self.iters} iterations...")
        run_ntp = mx.zeros((), dtype=mx.float32)
        run_jepa = mx.zeros((), dtype=mx.float32)
        for step in range(self.iters):
            batch = [data[(step * bs + i) % n] for i in range(bs)]
            input_ids, lengths, B, mask_until = self._prepare_batch(batch)
            do_jepa = self._jepa_active(step)

            (total, (ntp, jepa)), grads = loss_and_grad(
                jmod, input_ids, lengths, B, do_jepa, mask_until
            )
            optimizer.update(jmod, grads)
            mx.eval(jmod.parameters(), optimizer.state, total)

            run_ntp = run_ntp + ntp.astype(mx.float32)
            run_jepa = run_jepa + jepa.astype(mx.float32)

            if (step + 1) % self.logging_steps == 0:
                d = self.logging_steps
                print(
                    f"  Step {step + 1}/{self.iters} | "
                    f"NTP: {float(run_ntp.item()) / d:.4f} | "
                    f"JEPA: {float(run_jepa.item()) / d:.4f} | "
                    f"lr: {float(lr_schedule(step)) if callable(lr_schedule) else lr_schedule:.2e}"
                )
                run_ntp = mx.zeros((), dtype=mx.float32)
                run_jepa = mx.zeros((), dtype=mx.float32)

            if (step + 1) % self.save_steps == 0:
                self._save(_save_adapters_and_config, jmod)

        self._save(_save_adapters_and_config, jmod)
        print("\n" + "=" * 70)
        print("LLM-JEPA Training Complete!")
        print("=" * 70)
        print(f"  Adapters saved to: {self.adapter_path}")
        return {"status": "success", "adapter_path": str(self.adapter_path)}

    # -- helpers -------------------------------------------------------------
    def _warmup_iters(self) -> int:
        if self.warmup_steps and self.warmup_steps > 0:
            return int(self.warmup_steps)
        return int(self.warmup_ratio * self.iters)

    def _jepa_active(self, step: int) -> bool:
        """Whether the JEPA term is on this step (``jepa_ratio`` loss dropout)."""
        if self.jepa_lambda == 0:
            return False
        if self.jepa_ratio is None or self.jepa_ratio < 0:
            return True
        return bool(mx.random.uniform().item() < self.jepa_ratio)

    @staticmethod
    def _infer_hidden_size(base: Any) -> int:
        # Robust across quantized embeddings and nesting depth: one 1-token
        # forward through the resolved decoder gives the hidden dim directly.
        inner, _ = _resolve_decoder(base)
        h = inner(mx.array([[0]]))
        mx.eval(h)
        return int(h.shape[-1])

    def _save(self, save_fn, jmod: _JEPAModule):
        """Save the LoRA adapter (standard) + the predictor slots (if any)."""
        ok = save_fn(self.model, self.adapter_path)
        if self.num_predictors > 0:
            mx.save_safetensors(
                str(self.adapter_path / "jepa_pred_tokens.safetensors"),
                {"pred_tokens": jmod.pred_tokens},
            )
        if ok:
            print(f"  ✓ Saved checkpoint to {self.adapter_path}")
