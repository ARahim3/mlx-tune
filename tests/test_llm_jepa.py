"""Tests for LLM-JEPA (mlx_tune.llm_jepa)."""

import os

import pytest

import mlx.core as mx
import mlx.nn as nn

from mlx_tune import LLMJEPATrainer, LLMJEPAConfig, llm_jepa_loss
from mlx_tune.llm_jepa import (
    _jepa_distance,
    _last_token,
    _ntp_loss,
    _resolve_decoder,
    _make_lr_schedule,
    _JEPAModule,
)


# ---------------------------------------------------------------------------
# Tiny stub LLM (no download) — mimics mlx-lm's 2-level Model.model nesting
# ---------------------------------------------------------------------------
class _Inner(nn.Module):
    def __init__(self, vocab, H):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, H)
        self.norm = nn.LayerNorm(H)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        return self.norm(h)


class _StubLLM(nn.Module):
    """`Model.model` (inner decoder) + optional untied `lm_head`."""

    def __init__(self, vocab=32, H=16, tied=False):
        super().__init__()
        self.model = _Inner(vocab, H)
        if not tied:
            self.lm_head = nn.Linear(H, vocab, bias=False)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_config_defaults_and_to_dict():
    c = LLMJEPAConfig()
    assert c.jepa_lambda == 0.1
    assert c.jepa_distance == "cosine"
    assert c.num_predictors == 0
    d = c.to_dict()
    assert d["jepa_lambda"] == 0.1 and "output_dir" in d


def test_config_overrides():
    c = LLMJEPAConfig(jepa_lambda=0.25, jepa_distance="l2", num_predictors=3, max_steps=7)
    assert (c.jepa_lambda, c.jepa_distance, c.num_predictors, c.max_steps) == (0.25, "l2", 3, 7)


# ---------------------------------------------------------------------------
# Distance variants
# ---------------------------------------------------------------------------
def test_jepa_distance_cosine_identical_and_orthogonal():
    a = mx.array([[1.0, 0.0], [0.0, 1.0]])
    assert float(_jepa_distance(a, a, "cosine")) == pytest.approx(0.0, abs=1e-5)
    orth = mx.array([[0.0, 1.0], [1.0, 0.0]])
    assert float(_jepa_distance(a, orth, "cosine")) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("kind", ["cosine", "l2", "mse", "infonce"])
def test_jepa_distance_nonneg_and_identical_minimal(kind):
    mx.random.seed(0)
    a = mx.random.normal((6, 8))
    b = mx.random.normal((6, 8))
    d_same = float(_jepa_distance(a, a, kind))
    d_diff = float(_jepa_distance(a, b, kind))
    assert d_same >= -1e-6
    # cosine/l2/mse collapse to 0 on identical inputs; infonce is small but > 0
    if kind in ("cosine", "l2", "mse"):
        assert d_same == pytest.approx(0.0, abs=1e-5)
        assert d_diff > d_same
    else:
        assert d_same < d_diff


def test_jepa_distance_unknown_raises():
    a = mx.zeros((2, 3))
    with pytest.raises(ValueError):
        _jepa_distance(a, a, "bogus")


# ---------------------------------------------------------------------------
# Helpers: last-token gather, NTP masking, decoder resolution, lr schedule
# ---------------------------------------------------------------------------
def test_last_token_gather_with_padding():
    h = mx.arange(2 * 3 * 2).reshape(2, 3, 2).astype(mx.float32)
    lt = _last_token(h, mx.array([2, 3]))  # row0 -> idx1, row1 -> idx2
    assert lt.tolist() == [[2.0, 3.0], [10.0, 11.0]]


def test_last_token_clips_zero_length():
    h = mx.arange(1 * 3 * 2).reshape(1, 3, 2).astype(mx.float32)
    lt = _last_token(h, mx.array([0]))  # clip -> idx0
    assert lt.tolist() == [[0.0, 1.0]]


def test_ntp_loss_token_count_and_mask_until():
    logits = mx.zeros((2, 4, 10))
    ids = mx.array([[1, 2, 3, 0], [4, 5, 6, 7]])
    _, ntok = _ntp_loss(logits, ids, mx.array([3, 4]))
    assert float(ntok) == 5.0  # row0: positions<2 ->2 ; row1: positions<3 ->3
    # mask_until=2 on row0 removes its 2 valid targets -> only row1's 3 remain
    _, ntok2 = _ntp_loss(logits, ids, mx.array([3, 4]), mask_until=mx.array([2, 0]))
    assert float(ntok2) == 3.0


@pytest.mark.parametrize("tied", [True, False])
def test_resolve_decoder_tied_and_untied(tied):
    base = _StubLLM(tied=tied)
    inner, head = _resolve_decoder(base)
    assert hasattr(inner, "embed_tokens")
    h = inner(mx.array([[1, 2, 3]]))
    logits = head(h)
    assert logits.shape == (1, 3, 32)
    if not tied:
        assert head is base.lm_head


def test_make_lr_schedule_warmup_then_decay():
    sch = _make_lr_schedule(1e-3, warmup=4, total=20)
    assert float(sch(0)) == pytest.approx(0.0, abs=1e-9)   # warmup starts at 0
    assert float(sch(4)) > float(sch(0))                    # ramps up
    assert float(sch(19)) < float(sch(4))                   # then decays


# ---------------------------------------------------------------------------
# Predictor module param wiring
# ---------------------------------------------------------------------------
def test_jepa_module_pred_tokens_present_only_for_k_gt_0():
    base = _StubLLM()
    j0 = _JEPAModule(base, num_predictors=0, hidden_size=16)
    assert not hasattr(j0, "pred_tokens")
    j2 = _JEPAModule(base, num_predictors=2, hidden_size=16)
    assert j2.pred_tokens.shape == (2, 16)


# ---------------------------------------------------------------------------
# Full loss on the stub (fast, no model download) + gradient flow
# ---------------------------------------------------------------------------
def _stub_batch(B=2, L=4, vocab=32):
    # groups: [combined, text, code]
    ids = mx.random.randint(1, vocab, (3 * B, L))
    lengths = mx.array([L] * (3 * B))
    return ids, lengths, B


@pytest.mark.parametrize("k", [0, 2])
def test_llm_jepa_loss_runs_and_grads(k):
    mx.random.seed(0)
    base = _StubLLM()
    jmod = _JEPAModule(base, num_predictors=k, hidden_size=16)
    ids, lengths, B = _stub_batch()

    def loss_fn(m, ids, lengths, B, do):
        total, _ = llm_jepa_loss(m, ids, lengths, B, do, num_predictors=k)
        return total

    val, grads = nn.value_and_grad(jmod, loss_fn)(jmod, ids, lengths, B, True)
    assert val.ndim == 0 and float(val) >= 0.0
    flat = dict(__import__("mlx").utils.tree_flatten(grads))
    # gradients reach the base LM head/embeddings
    assert any(g is not None for g in flat.values())
    if k > 0:
        assert "pred_tokens" in flat  # predictor slots receive gradient


def test_llm_jepa_loss_aux_components_and_dropout():
    base = _StubLLM()
    jmod = _JEPAModule(base, num_predictors=0, hidden_size=16)
    ids, lengths, B = _stub_batch()
    total, (ntp, jepa) = llm_jepa_loss(jmod, ids, lengths, B, True, jepa_lambda=0.1)
    assert float(jepa) != 0.0
    # do_jepa=False zeroes the JEPA term
    total0, (ntp0, jepa0) = llm_jepa_loss(jmod, ids, lengths, B, False, jepa_lambda=0.1)
    assert float(jepa0) == 0.0


def test_stub_overfit_loss_decreases():
    """A few optimizer steps on the stub reduce total loss (mechanics check)."""
    import mlx.optimizers as optim
    mx.random.seed(0)
    base = _StubLLM()
    jmod = _JEPAModule(base, num_predictors=0, hidden_size=16)
    ids, lengths, B = _stub_batch()
    opt = optim.Adam(learning_rate=1e-2)

    def loss_fn(m, ids, lengths, B):
        total, _ = llm_jepa_loss(m, ids, lengths, B, True)
        return total

    lag = nn.value_and_grad(jmod, loss_fn)
    first = float(lag(jmod, ids, lengths, B)[0])
    for _ in range(20):
        val, grads = lag(jmod, ids, lengths, B)
        opt.update(jmod, grads)
        mx.eval(jmod.parameters(), opt.state)
    last = float(val)
    assert last < first


# ---------------------------------------------------------------------------
# Trainer data prep
# ---------------------------------------------------------------------------
class _FakeTok:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, s):
        # deterministic: 2 + (len of each whitespace-split token), bounded
        return [2 + (len(w) % 20) for w in s.split()] or [1]


def test_trainer_prepare_batch_groups_and_padding():
    tok = _FakeTok()
    data = [{"text": "aa bb", "code": "x"}, {"text": "ccc", "code": "yy zz"}]
    tr = LLMJEPATrainer(
        object(),  # model unused for _prepare_batch
        data,
        tokenizer=tok,
        args=LLMJEPAConfig(per_device_train_batch_size=2, max_steps=1),
    )
    ids, lengths, B, mask_until = tr._prepare_batch(data)
    assert B == 2
    assert ids.shape[0] == 3 * B            # [combined, text, code] x B
    assert mask_until is None               # response_only off by default
    assert lengths.shape == (3 * B,)


def test_trainer_response_only_mask_until():
    tok = _FakeTok()
    data = [{"text": "aa bb", "code": "x"}]
    tr = LLMJEPATrainer(
        object(), data, tokenizer=tok,
        args=LLMJEPAConfig(per_device_train_batch_size=1, response_only=True, max_steps=1),
    )
    _, _, B, mask_until = tr._prepare_batch(data)
    assert mask_until is not None
    # combined row masks the text prefix (2 tokens), the two view rows mask nothing
    assert mask_until.tolist() == [2, 0, 0]


def test_trainer_requires_tokenizer():
    with pytest.raises(ValueError):
        LLMJEPATrainer(object(), [{"text": "a", "code": "b"}], tokenizer=None)


# ---------------------------------------------------------------------------
# Slow E2E — real model
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_llm_jepa_e2e_real_model(tmp_path):
    from mlx_tune import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        "mlx-community/Qwen3.5-0.8B-MLX-4bit", max_seq_length=256
    )
    model = FastLanguageModel.get_peft_model(
        model, r=8, target_modules=["q_proj", "v_proj"], lora_alpha=16
    )
    data = [
        {"text": "match one or more digits", "code": r"\d+"},
        {"text": "match whitespace", "code": r"\s"},
        {"text": "a lowercase letter", "code": r"[a-z]"},
        {"text": "exactly three digits", "code": r"\d{3}"},
    ] * 3

    losses = []

    class _Capture(LLMJEPATrainer):
        pass

    cfg = LLMJEPAConfig(
        jepa_lambda=0.1, max_steps=12, logging_steps=2,
        per_device_train_batch_size=2, max_seq_length=128,
        output_dir=str(tmp_path / "out"),
    )
    trainer = LLMJEPATrainer(model, data, tokenizer=tok, args=cfg)
    result = trainer.train()
    assert result["status"] == "success"

    adir = tmp_path / "out" / "adapters"
    assert (adir / "adapters.safetensors").exists()
    assert (adir / "adapter_config.json").exists()

    # the fine-tuned artifact is a normal LLM that generates
    out = model.generate(prompt="match one or more digits", max_tokens=8, verbose=False)
    assert isinstance(out, str) and len(out) > 0
