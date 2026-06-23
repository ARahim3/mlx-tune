"""Tests for V-JEPA 2 (mlx_tune.vjepa2)."""

import numpy as np
import pytest

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_tune import (
    FastVideoJEPAModel,
    VJEPA2ModelWrapper,
    VideoViTEncoder,
    VJEPA2ForVideoClassification,
    VJEPA2PretrainedVideoClassifier,
    VideoClassifierTrainer,
    VideoClassifierConfig,
    apply_lora_to_vjepa2_encoder,
    latent_energy,
    video_linear_probe,
    video_knn_probe,
    video_attentive_probe,
)
from mlx_tune.vjepa2 import (
    _build_vjepa2,
    _build_vjepa2_classifier,
    _convert_vjepa2_classifier_weights,
    _rotate_queries_or_keys,
    _vjepa2_config_dict,
    _prep_video,
)


def _tiny_hf_encoder_weights(hidden, depth, patch, tubelet, prefix="encoder."):
    inner = int(hidden * 4.0)

    def L(o, i):
        return mx.random.normal((o, i)) * 0.02

    hf = {
        # Conv3d weight: HF (out, in, kT, kH, kW)
        prefix + "embeddings.patch_embeddings.proj.weight":
            mx.random.normal((hidden, 3, tubelet, patch, patch)) * 0.02,
        prefix + "embeddings.patch_embeddings.proj.bias": mx.zeros((hidden,)),
        prefix + "layernorm.weight": mx.ones((hidden,)),
        prefix + "layernorm.bias": mx.zeros((hidden,)),
    }
    for i in range(depth):
        h = f"{prefix}layer.{i}."
        for nm in ("norm1", "norm2"):
            hf[h + nm + ".weight"] = mx.ones((hidden,))
            hf[h + nm + ".bias"] = mx.zeros((hidden,))
        for p in ("query", "key", "value", "proj"):
            hf[h + f"attention.{p}.weight"] = L(hidden, hidden)
            hf[h + f"attention.{p}.bias"] = mx.zeros((hidden,))
        hf[h + "mlp.fc1.weight"] = L(inner, hidden)
        hf[h + "mlp.fc1.bias"] = mx.zeros((inner,))
        hf[h + "mlp.fc2.weight"] = L(hidden, inner)
        hf[h + "mlp.fc2.bias"] = mx.zeros((hidden,))
    return hf


def _tiny_hf_predictor_weights(hidden, pred_hidden, depth, num_mask_tokens=10):
    inner = int(pred_hidden * 4.0)

    def L(o, i):
        return mx.random.normal((o, i)) * 0.02

    hf = {
        "predictor.embeddings.predictor_embeddings.weight": L(pred_hidden, hidden),
        "predictor.embeddings.predictor_embeddings.bias": mx.zeros((pred_hidden,)),
        "predictor.embeddings.mask_tokens":
            mx.random.normal((num_mask_tokens, 1, 1, pred_hidden)) * 0.02,
        "predictor.layernorm.weight": mx.ones((pred_hidden,)),
        "predictor.layernorm.bias": mx.zeros((pred_hidden,)),
        "predictor.proj.weight": L(hidden, pred_hidden),
        "predictor.proj.bias": mx.zeros((hidden,)),
    }
    for i in range(depth):
        h = f"predictor.layer.{i}."
        for nm in ("norm1", "norm2"):
            hf[h + nm + ".weight"] = mx.ones((pred_hidden,))
            hf[h + nm + ".bias"] = mx.zeros((pred_hidden,))
        for p in ("query", "key", "value", "proj"):
            hf[h + f"attention.{p}.weight"] = L(pred_hidden, pred_hidden)
            hf[h + f"attention.{p}.bias"] = mx.zeros((pred_hidden,))
        hf[h + "mlp.fc1.weight"] = L(inner, pred_hidden)
        hf[h + "mlp.fc1.bias"] = mx.zeros((inner,))
        hf[h + "mlp.fc2.weight"] = L(pred_hidden, inner)
        hf[h + "mlp.fc2.bias"] = mx.zeros((pred_hidden,))
    return hf


def _tiny_vjepa2_hf(hidden=32, depth=2, heads=4, patch=8, tubelet=2, crop=16, frames=4,
                    with_predictor=False, pred_hidden=24, pred_depth=2):
    """Synthetic HF-format V-JEPA 2 weight dict + config for offline converter tests."""
    hfcfg = dict(model_type="vjepa2", hidden_size=hidden, num_hidden_layers=depth,
                 num_attention_heads=heads, patch_size=patch, tubelet_size=tubelet,
                 crop_size=crop, frames_per_clip=frames, in_chans=3, mlp_ratio=4.0,
                 layer_norm_eps=1e-6, qkv_bias=True)
    if with_predictor:
        hfcfg.update(pred_hidden_size=pred_hidden, pred_num_hidden_layers=pred_depth,
                     pred_num_attention_heads=4, pred_mlp_ratio=4.0,
                     pred_num_mask_tokens=10)
    cfg = _vjepa2_config_dict(hfcfg)
    hf = _tiny_hf_encoder_weights(hidden, depth, patch, tubelet)
    if with_predictor:
        hf.update(_tiny_hf_predictor_weights(hidden, pred_hidden, pred_depth))
    return _build_vjepa2(cfg, hf), cfg


# ── structure ─────────────────────────────────────────────────────────────────

def test_vjepa2_converter_offline():
    model, cfg = _tiny_vjepa2_hf()
    assert cfg["arch"] == "vjepa2"
    # T'=frames//tubelet=2, grid=crop//patch=2 → tokens = 2*2*2 = 8
    vid = mx.zeros((1, 4, 16, 16, 3))
    last = model.encoder(vid)
    assert last.shape == (1, 8, 32)
    assert model.encode([np.zeros((4, 16, 16, 3), np.uint8)]).shape == (1, 32)


def test_vjepa2_lora_target_count():
    model, _ = _tiny_vjepa2_hf()
    n = apply_lora_to_vjepa2_encoder(model.encoder, r=4)
    assert n == 2 * 6  # 2 layers × (query,key,value,proj,fc1,fc2)


@pytest.mark.parametrize("mode", ["frozen", "lora", "full"])
def test_vjepa2_classifier_trainable_surface(mode):
    model, _ = _tiny_vjepa2_hf()
    clf = FastVideoJEPAModel.for_video_classification(model, num_classes=2, finetune=mode, r=4)
    out = clf(mx.zeros((1, 4, 16, 16, 3)))
    assert out.shape == (1, 2)
    tot = sum(v.size for _, v in tree_flatten(clf.parameters()))
    tr = sum(v.size for _, v in tree_flatten(clf.trainable_parameters()))
    pct = 100.0 * tr / tot
    if mode == "full":
        assert pct == 100.0
    elif mode == "frozen":
        assert pct < 5.0
    else:
        assert 0.0 < pct < 60.0


def test_vjepa2_classifier_unknown_mode_raises():
    model, _ = _tiny_vjepa2_hf()
    with pytest.raises(ValueError):
        FastVideoJEPAModel.for_video_classification(model, num_classes=2, finetune="bogus")


def test_vjepa2_classifier_trainer_smoke():
    model, _ = _tiny_vjepa2_hf()
    clf = FastVideoJEPAModel.for_video_classification(model, num_classes=2, finetune="lora", r=4)
    rng = np.random.default_rng(0)
    vids = [(rng.random((4, 16, 16, 3)) * 255).astype(np.uint8) for _ in range(6)]
    labels = [i % 2 for i in range(6)]
    cfg = VideoClassifierConfig(batch_size=2, max_steps=2, log_every=100)
    tr = VideoClassifierTrainer(clf, cfg, vids, labels, eval_videos=vids, eval_labels=labels)
    hist = tr.train()
    assert len(hist) == 2 and all(np.isfinite(h) for h in hist)
    assert 0.0 <= tr.evaluate() <= 1.0


def test_vjepa2_save_load_roundtrip(tmp_path):
    model, _ = _tiny_vjepa2_hf()
    vid = [np.zeros((4, 16, 16, 3), np.uint8)]
    f1 = model.encode(vid)
    model.save_pretrained(tmp_path)
    reloaded, _ = FastVideoJEPAModel.from_pretrained(str(tmp_path))
    f2 = reloaded.encode(vid)
    assert float(mx.max(mx.abs(f1 - f2))) < 1e-5


def test_vjepa2_encode_tokens_and_probes():
    model, _ = _tiny_vjepa2_hf()
    rng = np.random.default_rng(0)
    trx = [(rng.random((4, 16, 16, 3)) * 255).astype(np.uint8) for _ in range(8)]
    tr_y = [i % 2 for i in range(8)]
    tex = [(rng.random((4, 16, 16, 3)) * 255).astype(np.uint8) for _ in range(4)]
    te_y = [i % 2 for i in range(4)]
    toks = model.encode_tokens(trx[:2])
    assert isinstance(toks, list) and toks[0].ndim == 2 and toks[0].shape[-1] == 32
    assert 0.0 <= video_knn_probe(model, trx, tr_y, tex, te_y, k=3) <= 1.0
    assert 0.0 <= video_attentive_probe(model, trx, tr_y, tex, te_y, epochs=3, heads=2) <= 1.0


def test_vjepa2_classifier_predict_shapes():
    model, _ = _tiny_vjepa2_hf()
    clf = FastVideoJEPAModel.for_video_classification(model, num_classes=2, finetune="frozen")
    rng = np.random.default_rng(0)
    vids = [(rng.random((4, 16, 16, 3)) * 255).astype(np.uint8) for _ in range(5)]
    preds = clf.predict(vids)
    assert preds.shape == (5,)
    probs = clf.predict(vids, return_probs=True)
    assert probs.shape == (5, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


@pytest.mark.parametrize("mode", ["frozen", "lora", "full"])
def test_vjepa2_classifier_save_load_predict_roundtrip(mode, tmp_path):
    model, _ = _tiny_vjepa2_hf()
    clf = FastVideoJEPAModel.for_video_classification(model, num_classes=2, finetune=mode, r=4)
    rng = np.random.default_rng(1)
    vids = [(rng.random((4, 16, 16, 3)) * 255).astype(np.uint8) for _ in range(6)]
    preds_before = clf.predict(vids)
    clf.save_pretrained(tmp_path)
    reloaded = FastVideoJEPAModel.load_classifier(str(tmp_path))
    preds_after = reloaded.predict(vids)
    assert np.array_equal(preds_before, preds_after)


# ── predictor (masked latent prediction) ─────────────────────────────────────

def test_vjepa2_predictor_offline():
    model, cfg = _tiny_vjepa2_hf(with_predictor=True)
    assert model.has_predictor
    vid = (np.random.default_rng(0).random((4, 16, 16, 3)) * 255).astype(np.uint8)
    # clip → 2 frame-slabs × 2×2 grid = 8 tokens; first 2 frames = 4 context tokens
    pred, target = model.predict_latents(vid, context_frames=2)
    assert pred.shape == (1, 4, 32) and target.shape == (1, 4, 32)
    # explicit ids path
    pred2, target2 = model.predict_latents(
        vid, context_ids=np.arange(4), target_ids=np.arange(4, 8))
    assert float(mx.max(mx.abs(pred - pred2))) < 1e-5
    assert float(mx.max(mx.abs(target - target2))) < 1e-5


def test_vjepa2_predictor_context_order_invariance():
    """Sort/unsort + explicit position-id RoPE → context order must not matter."""
    model, _ = _tiny_vjepa2_hf(with_predictor=True)
    tok = mx.random.normal((1, 8, 32))
    out1 = model.predictor(tok, mx.array([0, 1, 2, 3]), mx.array([4, 5, 6, 7]))
    out2 = model.predictor(tok, mx.array([2, 0, 3, 1]), mx.array([4, 5, 6, 7]))
    assert float(mx.max(mx.abs(out1 - out2))) < 1e-5


def test_vjepa2_no_predictor_raises():
    model, _ = _tiny_vjepa2_hf(with_predictor=False)
    assert not model.has_predictor
    vid = np.zeros((4, 16, 16, 3), np.uint8)
    with pytest.raises(ValueError, match="predictor"):
        model.predict_latents(vid, context_frames=2)


def test_vjepa2_predictor_bad_context_frames():
    model, _ = _tiny_vjepa2_hf(with_predictor=True)
    vid = np.zeros((4, 16, 16, 3), np.uint8)
    with pytest.raises(ValueError):
        model.predict_latents(vid, context_frames=3)   # not a tubelet multiple
    with pytest.raises(ValueError):
        model.predict_latents(vid, context_frames=4)   # no target tokens left


def test_vjepa2_predictor_save_load_roundtrip(tmp_path):
    model, _ = _tiny_vjepa2_hf(with_predictor=True)
    vid = (np.random.default_rng(1).random((4, 16, 16, 3)) * 255).astype(np.uint8)
    p1, _ = model.predict_latents(vid, context_frames=2)
    model.save_pretrained(tmp_path)
    reloaded, _ = FastVideoJEPAModel.from_pretrained(str(tmp_path))
    assert reloaded.has_predictor
    p2, _ = reloaded.predict_latents(vid, context_frames=2)
    assert float(mx.max(mx.abs(p1 - p2))) < 1e-5


def test_latent_energy():
    a = mx.random.normal((1, 4, 8))
    assert float(latent_energy(a, a)) < 1e-6
    assert float(latent_energy(a, a, kind="cosine")) < 1e-5
    assert latent_energy(a, a + 1.0, per_token=True).shape == (1, 4)
    assert float(latent_energy(a, a + 1.0)) > 0.0
    with pytest.raises(ValueError):
        latent_energy(a, a, kind="bogus")


def test_rope_2d_pos_matches_1d():
    x = mx.random.normal((2, 4, 6, 10))
    pos1 = mx.arange(6)
    pos2 = mx.broadcast_to(pos1[None], (2, 6))
    out1 = _rotate_queries_or_keys(x, pos1, 10000.0)
    out2 = _rotate_queries_or_keys(x, pos2, 10000.0)
    assert float(mx.max(mx.abs(out1 - out2))) < 1e-6


# ── pretrained classification checkpoints (attentive pooler + head) ──────────

def _tiny_classifier(hidden=32, depth=2, heads=4, patch=8, tubelet=2, crop=16,
                     num_classes=3, num_pooler_layers=2):
    def L(o, i):
        return mx.random.normal((o, i)) * 0.02

    config = {
        "arch": "vjepa2_classifier", "hidden": hidden, "depth": depth, "heads": heads,
        "patch_size": patch, "tubelet_size": tubelet, "crop_size": crop,
        "mlp_ratio": 4.0, "eps": 1e-6, "qkv_bias": True, "theta": 10000.0,
        "in_chans": 3, "frames_per_clip": 4, "num_pooler_layers": num_pooler_layers,
        "shortest_edge": 18,
    }
    raw = _tiny_hf_encoder_weights(hidden, depth, patch, tubelet, prefix="vjepa2.encoder.")
    inner = int(hidden * 4.0)
    raw["pooler.query_tokens"] = mx.random.normal((1, 1, hidden)) * 0.02
    cl = "pooler.cross_attention_layer."
    for nm in ("layer_norm1", "layer_norm2"):
        raw[cl + nm + ".weight"] = mx.ones((hidden,))
        raw[cl + nm + ".bias"] = mx.zeros((hidden,))
    for p in ("q_proj", "k_proj", "v_proj"):
        raw[cl + f"cross_attn.{p}.weight"] = L(hidden, hidden)
        raw[cl + f"cross_attn.{p}.bias"] = mx.zeros((hidden,))
    raw[cl + "mlp.fc1.weight"] = L(inner, hidden)
    raw[cl + "mlp.fc1.bias"] = mx.zeros((inner,))
    raw[cl + "mlp.fc2.weight"] = L(hidden, inner)
    raw[cl + "mlp.fc2.bias"] = mx.zeros((hidden,))
    for i in range(num_pooler_layers):
        sl = f"pooler.self_attention_layers.{i}."
        for nm in ("layer_norm1", "layer_norm2"):
            raw[sl + nm + ".weight"] = mx.ones((hidden,))
            raw[sl + nm + ".bias"] = mx.zeros((hidden,))
        for p in ("q_proj", "k_proj", "v_proj", "out_proj"):
            raw[sl + f"self_attn.{p}.weight"] = L(hidden, hidden)
            raw[sl + f"self_attn.{p}.bias"] = mx.zeros((hidden,))
        raw[sl + "mlp.fc1.weight"] = L(inner, hidden)
        raw[sl + "mlp.fc1.bias"] = mx.zeros((inner,))
        raw[sl + "mlp.fc2.weight"] = L(hidden, inner)
        raw[sl + "mlp.fc2.bias"] = mx.zeros((hidden,))
    raw["classifier.weight"] = L(num_classes, hidden)
    raw["classifier.bias"] = mx.zeros((num_classes,))

    id2label = {str(i): f"class_{i}" for i in range(num_classes)}
    clf = _build_vjepa2_classifier(config, num_classes, id2label)
    clf.load_weights(list(_convert_vjepa2_classifier_weights(raw, depth).items()))
    mx.eval(clf.parameters())
    clf.eval()
    return clf


def test_vjepa2_pretrained_classifier_offline():
    clf = _tiny_classifier()
    assert isinstance(clf, VJEPA2PretrainedVideoClassifier)
    out = clf(mx.zeros((1, 4, 16, 16, 3)))
    assert out.shape == (1, 3)
    rng = np.random.default_rng(0)
    vids = [(rng.random((6, 20, 24, 3)) * 255).astype(np.uint8) for _ in range(2)]
    res = clf.predict(vids, top_k=2)
    assert len(res) == 2 and len(res[0]) == 2
    assert set(res[0][0]) == {"id", "label", "prob"}
    assert res[0][0]["label"].startswith("class_")
    assert res[0][0]["prob"] >= res[0][1]["prob"]


def test_vjepa2_pretrained_classifier_save_load(tmp_path):
    clf = _tiny_classifier()
    rng = np.random.default_rng(2)
    vids = [(rng.random((4, 16, 16, 3)) * 255).astype(np.uint8) for _ in range(3)]
    before = clf.predict(vids)
    clf.save_pretrained(tmp_path)
    reloaded, _ = FastVideoJEPAModel.from_pretrained(str(tmp_path))
    assert isinstance(reloaded, VJEPA2PretrainedVideoClassifier)
    assert reloaded.id2label == clf.id2label
    after = reloaded.predict(vids)
    assert [r[0]["id"] for r in before] == [r[0]["id"] for r in after]
    assert np.allclose([r[0]["prob"] for r in before], [r[0]["prob"] for r in after],
                       atol=1e-5)


def test_prep_video_modes():
    vid = (np.random.default_rng(0).random((6, 40, 30, 3)) * 255).astype(np.uint8)
    a = _prep_video(vid, 16)
    assert a.shape == (6, 16, 16, 3)
    b = _prep_video(vid, 16, shortest_edge=18, num_frames=4)
    assert b.shape == (4, 16, 16, 3)


# ── slow: parity against HF + real LoRA ──────────────────────────────────────

@pytest.mark.slow
def test_vjepa2_parity_vs_hf():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoModel

    repo = "facebook/vjepa2-vitl-fpc64-256"
    model, _ = FastVideoJEPAModel.from_pretrained(repo)
    rng = np.random.default_rng(0)
    vid = (rng.random((4, 256, 256, 3)) * 255).astype(np.uint8)
    arr = _prep_video(vid, model.crop_size)[None]
    last_mlx = np.array(model.encoder(mx.array(arr)))

    hf = AutoModel.from_pretrained(repo, attn_implementation="eager").eval()
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    x = (vid.astype(np.float32) / 255.0 - mean) / std
    xt = torch.from_numpy(x.transpose(0, 3, 1, 2))[None]
    with torch.no_grad():
        last_hf = hf(pixel_values_videos=xt, skip_predictor=True).last_hidden_state.numpy()

    cos = float((last_mlx.ravel() * last_hf.ravel()).sum() /
                (np.linalg.norm(last_mlx) * np.linalg.norm(last_hf)))
    assert cos > 0.999
    assert np.abs(last_mlx - last_hf).max() < 0.1


@pytest.mark.slow
def test_vjepa2_predictor_parity_vs_hf():
    """The ported predictor must match HF's masked latent predictions (cos ~1)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoModel

    repo = "facebook/vjepa2-vitl-fpc64-256"
    model, _ = FastVideoJEPAModel.from_pretrained(repo)
    assert model.has_predictor
    rng = np.random.default_rng(0)
    vid = (rng.random((4, 256, 256, 3)) * 255).astype(np.uint8)
    arr = _prep_video(vid, model.crop_size)[None]
    tokens = model.encoder(mx.array(arr))
    n = tokens.shape[1]
    n_ctx = n // 2
    pred_mlx = np.array(model.predictor(
        tokens, mx.arange(n_ctx), mx.arange(n_ctx, n)))

    hf = AutoModel.from_pretrained(repo, attn_implementation="eager").eval()
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    x = (vid.astype(np.float32) / 255.0 - mean) / std
    xt = torch.from_numpy(x.transpose(0, 3, 1, 2))[None]
    with torch.no_grad():
        out = hf(pixel_values_videos=xt,
                 context_mask=[torch.arange(n_ctx).unsqueeze(0)],
                 target_mask=[torch.arange(n_ctx, n).unsqueeze(0)])
    pred_hf = out.predictor_output.last_hidden_state.numpy()
    tgt_hf = out.predictor_output.target_hidden_state.numpy()

    cos = float((pred_mlx.ravel() * pred_hf.ravel()).sum() /
                (np.linalg.norm(pred_mlx) * np.linalg.norm(pred_hf)))
    assert cos > 0.999
    assert np.abs(pred_mlx - pred_hf).max() < 0.01

    # predict_latents target must equal HF's target_hidden_state
    _, target = model.predict_latents(vid, context_frames=2)
    tcos = float((np.array(target).ravel() * tgt_hf.ravel()).sum() /
                 (np.linalg.norm(np.array(target)) * np.linalg.norm(tgt_hf)))
    assert tcos > 0.999


@pytest.mark.slow
def test_vjepa2_ssv2_classifier_parity_vs_hf():
    """Meta's SSv2 classification checkpoint: logits parity + same top-1 label."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import VJEPA2ForVideoClassification as HFClf

    repo = "facebook/vjepa2-vitl-fpc16-256-ssv2"
    clf, _ = FastVideoJEPAModel.from_pretrained(repo)
    assert isinstance(clf, VJEPA2PretrainedVideoClassifier)
    assert clf.num_classes == 174 and len(clf.id2label) == 174

    rng = np.random.default_rng(0)
    vid = (rng.random((16, 256, 256, 3)) * 255).astype(np.uint8)
    arr = clf._prep(vid)[None].astype(np.float32)
    logits_mlx = np.array(clf(mx.array(arr)))

    hf = HFClf.from_pretrained(repo, attn_implementation="eager").eval()
    xt = torch.from_numpy(arr.transpose(0, 1, 4, 2, 3))
    with torch.no_grad():
        logits_hf = hf(pixel_values_videos=xt).logits.numpy()

    cos = float((logits_mlx.ravel() * logits_hf.ravel()).sum() /
                (np.linalg.norm(logits_mlx) * np.linalg.norm(logits_hf)))
    assert cos > 0.999
    assert int(logits_mlx.argmax()) == int(logits_hf.argmax())

    res = clf.predict([vid], top_k=3)
    assert res[0][0]["id"] == int(logits_hf.argmax())
    assert isinstance(res[0][0]["label"], str) and len(res[0]) == 3


@pytest.mark.slow
def test_vjepa2_video_probe_above_chance():
    model, _ = FastVideoJEPAModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    rng = np.random.default_rng(0)
    centers = [(220, 40, 40), (40, 40, 220)]

    def make(n, seed):
        r = np.random.default_rng(seed)
        xs, ys = [], []
        for i in range(n):
            c = i % 2
            v = (r.random((4, 256, 256, 3)) * 40).astype(np.uint8)
            v[:, 60:180, 60:180] = np.array(centers[c], np.uint8)
            xs.append(v)
            ys.append(c)
        return xs, ys

    trx, tr_y = make(8, 0)
    tex, te_y = make(6, 1)
    acc = video_linear_probe(model, trx, tr_y, tex, te_y, epochs=80, lr=5e-3)
    assert acc > 0.66  # chance 0.5
