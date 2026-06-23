"""Tests for JEPA / LeJEPA (mlx_tune.jepa)."""

import numpy as np
import pytest

import mlx.core as mx

from mlx_tune import (
    FastJEPAModel,
    JEPAConfig,
    JEPATrainer,
    JEPAAugment,
    JEPADataCollator,
    ViTEncoder,
    sigreg_loss,
    lejepa_loss,
    lejepa_prediction_loss,
    sample_directions,
    linear_probe,
    knn_probe,
    attentive_probe,
    JEPAForImageClassification,
    JEPAClassifierTrainer,
    JEPAClassifierConfig,
    apply_lora_to_encoder,
)
from mlx.utils import tree_flatten

from mlx_tune import (
    JEPARegressionTrainer,
    JEPADenseTrainer,
    ImageFolderDataset,
)


def _write_image_folder(root, n=16, size=48, seed=0):
    """Write `n` PNGs (half in a subdir) and return the folder path."""
    from PIL import Image
    import os
    rng = np.random.default_rng(seed)
    sub = os.path.join(root, "sub")
    os.makedirs(sub, exist_ok=True)
    for i in range(n):
        arr = (rng.random((size, size, 3)) * 255).astype(np.uint8)
        dest = root if i % 2 else sub
        Image.fromarray(arr).save(os.path.join(dest, f"img{i}.png"))
    return root


# ---------------------------------------------------------------------------
# Streaming dataset + resumable checkpoints
# ---------------------------------------------------------------------------
def test_image_folder_dataset(tmp_path):
    _write_image_folder(str(tmp_path), n=16)
    ds = ImageFolderDataset(str(tmp_path))         # recursive by default
    assert len(ds) == 16
    from PIL import Image
    assert isinstance(ds[0], Image.Image) and ds[0].mode == "RGB"
    flat = ImageFolderDataset(str(tmp_path), recursive=False)
    assert len(flat) == 8                          # only the top-level half


def test_image_folder_dataset_errors(tmp_path):
    with pytest.raises(ValueError):
        ImageFolderDataset(str(tmp_path / "nope"))  # missing
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        ImageFolderDataset(str(tmp_path / "empty"))  # no images


def test_jepa_streaming_and_resume(tmp_path):
    _write_image_folder(str(tmp_path / "imgs"), n=16)
    ds = ImageFolderDataset(str(tmp_path / "imgs"))
    ck = str(tmp_path / "ckpt")

    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=48, patch_size=16)
    cfg = JEPAConfig(img_size=48, batch_size=4, max_steps=6, num_slices=32,
                     log_every=100, save_steps=3, output_dir=str(tmp_path), checkpoint_dir=ck)
    h1 = JEPATrainer(model, cfg, ds).train()
    assert len(h1) == 6
    import json
    assert json.loads((tmp_path / "ckpt" / "trainer_state.json").read_text())["step"] == 6

    # fresh trainer resumes from the checkpoint and runs to 10 total
    model2, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=48, patch_size=16)
    cfg2 = JEPAConfig(img_size=48, batch_size=4, max_steps=10, num_slices=32,
                      log_every=100, output_dir=str(tmp_path), checkpoint_dir=ck, resume=True)
    h2 = JEPATrainer(model2, cfg2, ds).train()
    assert len(h2) == 10                            # resumed history extended to total
from mlx_tune.jepa import (
    _interpolate_pos_embed,
    _perfect_square,
    _convert_ijepa_weights,
    ViTEncoder,
)


def _rand_imgs(n, size=64, seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.random((size, size, 3)) * 255).astype(np.uint8) for _ in range(n)]


# ---------------------------------------------------------------------------
# Dense / regression heads
# ---------------------------------------------------------------------------
def test_regression_head_train_predict_saveload(tmp_path):
    imgs = _rand_imgs(24)
    targets = [float(np.asarray(im).mean() / 255.0) for im in imgs]  # learnable scalar
    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    reg = FastJEPAModel.for_image_regression(model, out_dim=1, finetune="frozen")
    cfg = JEPAClassifierConfig(img_size=64, batch_size=8, max_steps=15,
                               learning_rate=2e-3, log_every=100)
    tr = JEPARegressionTrainer(reg, cfg, imgs, targets, eval_images=imgs, eval_targets=targets)
    hist = tr.train()
    assert hist[-1] < hist[0]                          # MSE decreases
    metrics = tr.evaluate()
    assert set(metrics) == {"mae", "rmse"} and metrics["mae"] >= 0
    assert reg.predict(imgs[:3]).shape == (3,)
    reg.save_pretrained(tmp_path / "reg")
    reg2 = FastJEPAModel.load_regressor(tmp_path / "reg")
    assert float(np.max(np.abs(reg.predict(imgs[:3]) - reg2.predict(imgs[:3])))) < 1e-5


def test_dense_regression_head_shapes_and_saveload(tmp_path):
    imgs = _rand_imgs(12)
    targets = [(np.asarray(im).mean(-1) / 255.0).astype(np.float32) for im in imgs]  # (64,64)
    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    dense = FastJEPAModel.for_dense_prediction(model, out_channels=1, task="regression",
                                               finetune="frozen")
    out = dense(mx.array(np.stack([(im / 255.0 - 0.5) / 0.5 for im in imgs[:2]]).astype(np.float32)))
    assert out.shape == (2, 64, 64, 1)
    cfg = JEPAClassifierConfig(img_size=64, batch_size=4, max_steps=6, log_every=100)
    dt = JEPADenseTrainer(dense, cfg, imgs, targets, eval_images=imgs, eval_targets=targets)
    dt.train()
    assert dense.predict(imgs[:2]).shape == (2, 64, 64)
    assert "mae" in dt.evaluate()
    dense.save_pretrained(tmp_path / "dense")
    d2 = FastJEPAModel.load_dense(tmp_path / "dense")
    assert d2.task == "regression" and d2.out_channels == 1


def test_segmentation_head_shapes_and_metric():
    imgs = _rand_imgs(12)
    seg_t = [(np.asarray(im).mean(-1) // 90).clip(0, 2).astype(np.int32) for im in imgs]
    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    seg = FastJEPAModel.for_dense_prediction(model, out_channels=3, task="segmentation",
                                             finetune="frozen")
    cfg = JEPAClassifierConfig(img_size=64, batch_size=4, max_steps=6, log_every=100)
    st = JEPADenseTrainer(seg, cfg, imgs, seg_t, eval_images=imgs, eval_targets=seg_t)
    st.train()
    preds = seg.predict(imgs[:2])
    assert preds.shape == (2, 64, 64) and preds.max() <= 2
    acc = st.evaluate()["pixel_acc"]
    assert 0.0 <= acc <= 1.0


def test_dense_prediction_rejects_bad_task():
    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    with pytest.raises(ValueError):
        FastJEPAModel.for_dense_prediction(model, out_channels=2, task="bogus")


@pytest.mark.parametrize("mode", ["frozen", "lora", "full"])
def test_regression_finetune_modes_build(mode):
    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    reg = FastJEPAModel.for_image_regression(model, out_dim=2, finetune=mode, r=4)
    n_train = sum(v.size for _, v in tree_flatten(reg.trainable_parameters()))
    assert n_train > 0
    assert reg(mx.zeros((1, 64, 64, 3))).shape == (1, 2)


# ---------------------------------------------------------------------------
# Positional-embedding interpolation (non-native input sizes)
# ---------------------------------------------------------------------------
def test_perfect_square():
    assert _perfect_square(196) == 14
    assert _perfect_square(256) == 16
    assert _perfect_square(197) is None


@pytest.mark.parametrize("has_cls", [False, True])
def test_interpolate_pos_embed_shapes(has_cls):
    n = 14 * 14 + (1 if has_cls else 0)
    pe = mx.random.normal((1, n, 8))
    out = _interpolate_pos_embed(pe, 16, has_cls=has_cls)
    assert out.shape == (1, 16 * 16 + (1 if has_cls else 0), 8)


def test_interpolate_pos_embed_noop_when_equal():
    pe = mx.random.normal((1, 14 * 14, 8))
    out = _interpolate_pos_embed(pe, 14, has_cls=False)
    assert bool((out == pe).all().item())  # bit-identical: native res unaffected


def test_encoder_accepts_non_native_input_via_interpolation():
    # encoder built for 64px (grid 4) must still run on 96px (grid 6)
    enc = ViTEncoder(img_size=64, patch_size=16, dim=32, depth=2, heads=2)
    x = mx.random.normal((2, 96, 96, 3))
    toks = enc.forward_tokens(x)
    assert toks.shape == (2, 6 * 6 + 1, 32)  # +1 CLS
    assert enc(x).shape == (2, 32)


def test_ijepa_converter_interpolates_to_target_grid():
    hf = {
        "embeddings.patch_embeddings.projection.weight": mx.zeros((8, 3, 16, 16)),
        "embeddings.patch_embeddings.projection.bias": mx.zeros((8,)),
        "embeddings.position_embeddings": mx.random.normal((1, 8 * 8, 8)),
        "layernorm.weight": mx.ones((8,)),
        "layernorm.bias": mx.zeros((8,)),
    }
    conv = _convert_ijepa_weights(hf, depth=0, n_patches=10 * 10)
    assert conv["encoder.pos_embed"].shape == (1, 100, 8)


def _synthetic(n, size=64, seed=0):
    """Three-class images: distinct color blob + noise (learnable structure)."""
    rng = np.random.default_rng(seed)
    centers = [(255, 40, 40), (40, 255, 40), (40, 40, 255)]
    imgs, labels = [], []
    for i in range(n):
        c = i % 3
        img = (rng.random((size, size, 3)) * 60).astype(np.uint8)
        y, x = int(rng.integers(8, size - 24)), int(rng.integers(8, size - 24))
        img[y:y + 20, x:x + 20] = np.array(centers[c], dtype=np.uint8)
        imgs.append(img)
        labels.append(c)
    return imgs, labels


# ── SIGReg + losses ──────────────────────────────────────────────────────────

def test_sigreg_near_zero_for_gaussian():
    mx.random.seed(0)
    d = sample_directions(32, 256)
    z = mx.random.normal((2048, 32))
    assert float(sigreg_loss(z, d)) < 0.01


def test_sigreg_penalizes_collapse():
    mx.random.seed(0)
    d = sample_directions(32, 256)
    gauss = sigreg_loss(mx.random.normal((2048, 32)), d)
    collapsed = sigreg_loss(mx.ones((2048, 32)) * 0.3, d)
    assert float(collapsed) > float(gauss) * 10


def test_sigreg_penalizes_wrong_variance_and_mean():
    """SIGReg must penalise under-/over-variance and a non-zero mean — these are
    the moments whose neglect would let an encoder collapse silently."""
    mx.random.seed(0)
    d = sample_directions(32, 512)
    gauss = float(sigreg_loss(mx.random.normal((2048, 32)), d))
    compressed = float(sigreg_loss(0.1 * mx.random.normal((2048, 32)), d))
    spread = float(sigreg_loss(3.0 * mx.random.normal((2048, 32)), d))
    shifted = float(sigreg_loss(mx.random.normal((2048, 32)) + 5.0, d))
    assert compressed > gauss * 10      # low variance penalised (collapse direction)
    assert spread > gauss * 10          # high variance penalised
    assert shifted > gauss * 10         # non-zero mean penalised


def test_sigreg_gradient_pulls_variance_to_one():
    """Minimising SIGReg alone on free embeddings must drive the per-dim std to ~1
    from *both* sides — the core anti-collapse mechanism, checked without an encoder."""
    import mlx.nn as nn
    import mlx.optimizers as optim

    class Emb(nn.Module):
        def __init__(self, z0):
            super().__init__(); self.z = z0

    for init_std in (0.1, 3.0):
        mx.random.seed(0)
        m = Emb(init_std * mx.random.normal((1024, 32)))
        lag = nn.value_and_grad(m, lambda mm: sigreg_loss(mm.z, sample_directions(32, 256)))
        opt = optim.Adam(learning_rate=0.05)
        for _ in range(400):
            _, g = lag(m); opt.update(m, g); mx.eval(m.parameters())
        std = float(mx.mean(mx.std(m.z, axis=0)))
        assert 0.85 < std < 1.15, f"init_std={init_std} -> std={std}"


def test_lejepa_loss_no_collapse_on_free_embeddings():
    """The decisive correctness check: optimising the *full* LeJEPA objective on free
    embeddings must align the views (pred -> 0) while keeping the batch spread out
    (std ~ 1) — i.e. SIGReg counteracts the prediction term's collapse pull. A
    regression here would mean the loss itself permits collapse (vs. an encoder /
    scale issue)."""
    import mlx.nn as nn
    import mlx.optimizers as optim

    class Emb(nn.Module):
        def __init__(self, z0):
            super().__init__(); self.z = z0

    mx.random.seed(0)
    B, K = 96, 32
    m = Emb(mx.random.normal((3, B, K)))
    lag = nn.value_and_grad(
        m, lambda mm: lejepa_loss(mm.z, 2, sample_directions(K, 256), lam=0.05))
    opt = optim.Adam(learning_rate=0.05)
    for _ in range(600):
        _, g = lag(m); opt.update(m, g); mx.eval(m.parameters())
    pred = float(lejepa_prediction_loss(m.z, 2))
    batch_std = float(mx.mean(mx.std(m.z, axis=1)))   # std across the batch axis
    assert pred < 0.05, f"views did not align (pred={pred})"
    assert batch_std > 0.5, f"embeddings collapsed under full loss (batch_std={batch_std})"


def test_sample_directions_unit_norm():
    mx.random.seed(0)
    d = sample_directions(48, 100)
    norms = mx.linalg.norm(d, axis=0)
    assert float(mx.max(mx.abs(norms - 1.0))) < 1e-4


def test_prediction_loss_zero_when_views_identical():
    z = mx.random.normal((1, 8, 16))
    z = mx.broadcast_to(z, (5, 8, 16))  # all 5 views identical
    assert float(lejepa_prediction_loss(z, n_global=2)) < 1e-6


def test_lejepa_loss_components():
    mx.random.seed(0)
    z = mx.random.normal((4, 16, 32))
    d = sample_directions(32, 128)
    total, pred, sig = lejepa_loss(z, n_global=2, directions=d, lam=0.05, return_components=True)
    assert all(np.isfinite(float(x)) for x in (total, pred, sig))
    assert abs(float(total) - (0.95 * float(pred) + 0.05 * float(sig))) < 1e-4


# ── Encoder / model ──────────────────────────────────────────────────────────

def test_vit_encoder_output_shape():
    enc = ViTEncoder(img_size=64, patch_size=16, dim=64, depth=2, heads=2)
    out = enc(mx.zeros((3, 64, 64, 3)))
    assert out.shape == (3, 64)


def test_fastjepa_build_debug_preset():
    model, tok = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    assert tok is None
    assert model.embed_dim == 64
    assert model.img_size == 64


def test_fastjepa_unknown_preset_raises():
    with pytest.raises(ValueError):
        FastJEPAModel.from_pretrained("not-a-real-preset")


# ── Augmentation / collator / encode ─────────────────────────────────────────

def test_augment_view_count_and_shape():
    aug = JEPAAugment(img_size=32, n_global=2, n_local=4)
    views = aug((np.random.rand(50, 50, 3) * 255).astype(np.uint8))
    assert len(views) == 6
    assert all(v.shape == (32, 32, 3) for v in views)


def test_collator_shape():
    aug = JEPAAugment(img_size=32, n_global=2, n_local=3)
    coll = JEPADataCollator(aug)
    batch = [(np.random.rand(40, 40, 3) * 255).astype(np.uint8) for _ in range(5)]
    out = coll(batch)
    assert out.shape == (5, 5, 32, 32, 3)  # (V=5, B=5, H, W, C)


def test_encode_shape():
    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    imgs, _ = _synthetic(10)
    feats = model.encode(imgs, batch_size=4)
    assert feats.shape == (10, 64)


def test_encode_tokens_shape():
    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    imgs, _ = _synthetic(6)
    toks = model.encode_tokens(imgs, batch_size=4)
    # grid 64/16 = 4 → 16 patches + 1 CLS = 17 tokens, dim 64
    assert toks.shape == (6, 17, 64)


def test_knn_probe_runs():
    mx.random.seed(0)
    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    tr_x, tr_y = _synthetic(30, seed=0)
    te_x, te_y = _synthetic(15, seed=1)
    acc = knn_probe(model, tr_x, tr_y, te_x, te_y, k=5)
    assert 0.0 <= acc <= 1.0


def test_attentive_probe_runs():
    mx.random.seed(0)
    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    tr_x, tr_y = _synthetic(30, seed=0)
    te_x, te_y = _synthetic(15, seed=1)
    acc = attentive_probe(model, tr_x, tr_y, te_x, te_y, epochs=5, heads=2)
    assert 0.0 <= acc <= 1.0


def test_warm_start_continues_ssl():
    """A loaded/trained encoder can be fed straight back into JEPATrainer."""
    mx.random.seed(0)
    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    imgs, _ = _synthetic(16)
    cfg = JEPAConfig(img_size=64, batch_size=8, n_global=2, n_local=1,
                     num_slices=32, max_steps=2, log_every=100)
    h1 = JEPATrainer(model, args=cfg, train_dataset=imgs).train()
    # warm-start: continue SSL from the same (now-trained) encoder
    h2 = JEPATrainer(model, args=cfg, train_dataset=imgs).train()
    assert len(h1) == 2 and len(h2) == 2 and all(np.isfinite(h2))


def test_save_load_roundtrip(tmp_path):
    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    imgs, _ = _synthetic(6)
    f1 = model.encode(imgs)
    model.save_pretrained(tmp_path)
    reloaded, _ = FastJEPAModel.from_pretrained(str(tmp_path))
    f2 = reloaded.encode(imgs)
    assert float(mx.max(mx.abs(f1 - f2))) < 1e-5


def test_single_training_step_finite():
    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    imgs, _ = _synthetic(16)
    cfg = JEPAConfig(img_size=64, batch_size=8, n_global=2, n_local=2,
                     num_slices=64, max_steps=2, log_every=100)
    hist = JEPATrainer(model, args=cfg, train_dataset=imgs).train()
    assert len(hist) == 2 and all(np.isfinite(h) for h in hist)


# ── Slow E2E (deselected by default) ─────────────────────────────────────────

@pytest.mark.slow
def test_training_loss_decreases():
    mx.random.seed(0)
    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    imgs, _ = _synthetic(240)
    cfg = JEPAConfig(img_size=64, batch_size=32, n_global=2, n_local=4,
                     num_slices=256, max_steps=40, learning_rate=1e-3, log_every=100)
    hist = JEPATrainer(model, args=cfg, train_dataset=imgs).train()
    assert np.mean(hist[-5:]) < np.mean(hist[:5])


@pytest.mark.slow
def test_linear_probe_above_chance():
    mx.random.seed(0)
    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    tr_x, tr_y = _synthetic(240, seed=0)
    te_x, te_y = _synthetic(90, seed=1)
    cfg = JEPAConfig(img_size=64, batch_size=32, n_global=2, n_local=4,
                     num_slices=256, max_steps=40, learning_rate=1e-3, log_every=100)
    JEPATrainer(model, args=cfg, train_dataset=tr_x).train()
    acc = linear_probe(model, tr_x, tr_y, te_x, te_y, epochs=60, lr=5e-3)
    assert acc > 0.5  # chance = 0.33


# ── I-JEPA: pretrained loader + classification (offline structural) ───────────

def _tiny_ijepa_hf(dim=32, depth=2, heads=4, patch=8, img=16):
    """Build a tiny synthetic HF-format I-JEPA weight dict + config for the converter."""
    from mlx_tune.jepa import _ijepa_config_dict, _build_ijepa

    hfcfg = dict(model_type="ijepa", hidden_size=dim, num_hidden_layers=depth,
                 num_attention_heads=heads, intermediate_size=dim * 2, patch_size=patch,
                 image_size=img, num_channels=3, layer_norm_eps=1e-6, qkv_bias=True)
    cfg = _ijepa_config_dict(hfcfg, None)
    grid = img // patch
    n_patches = grid * grid
    inner = dim * 2

    def L(o, i):
        return mx.random.normal((o, i)) * 0.02

    hf = {
        "embeddings.patch_embeddings.projection.weight": mx.random.normal((dim, 3, patch, patch)) * 0.02,
        "embeddings.patch_embeddings.projection.bias": mx.zeros((dim,)),
        "embeddings.position_embeddings": mx.random.normal((1, n_patches, dim)) * 0.02,
        "layernorm.weight": mx.ones((dim,)),
        "layernorm.bias": mx.zeros((dim,)),
    }
    for i in range(depth):
        h = f"encoder.layer.{i}."
        for nm in ("layernorm_before", "layernorm_after"):
            hf[h + nm + ".weight"] = mx.ones((dim,))
            hf[h + nm + ".bias"] = mx.zeros((dim,))
        for q in ("query", "key", "value"):
            hf[h + f"attention.attention.{q}.weight"] = L(dim, dim)
            hf[h + f"attention.attention.{q}.bias"] = mx.zeros((dim,))
        hf[h + "attention.output.dense.weight"] = L(dim, dim)
        hf[h + "attention.output.dense.bias"] = mx.zeros((dim,))
        hf[h + "intermediate.dense.weight"] = L(inner, dim)
        hf[h + "intermediate.dense.bias"] = mx.zeros((inner,))
        hf[h + "output.dense.weight"] = L(dim, inner)
        hf[h + "output.dense.bias"] = mx.zeros((dim,))
    return _build_ijepa(cfg, hf), cfg


def test_ijepa_converter_offline():
    model, cfg = _tiny_ijepa_hf()
    assert cfg["use_cls_token"] is False
    assert cfg["arch"] == "ijepa"
    out = model(mx.zeros((2, 16, 16, 3)))
    assert out.shape == (2, 32)


def test_apply_lora_to_encoder_counts():
    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    n = apply_lora_to_encoder(model.encoder, r=4)
    assert n == 2 * 6  # 2 blocks × 6 target linears


@pytest.mark.parametrize("mode,expect_pct", [("frozen", 0.5), ("lora", 5.0), ("full", 100.0)])
def test_classifier_trainable_surface(mode, expect_pct):
    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    clf = FastJEPAModel.for_image_classification(model, num_classes=3, finetune=mode, r=4)
    out = clf(mx.zeros((2, 64, 64, 3)))
    assert out.shape == (2, 3)
    tot = sum(v.size for _, v in tree_flatten(clf.parameters()))
    tr = sum(v.size for _, v in tree_flatten(clf.trainable_parameters()))
    pct = 100.0 * tr / tot
    if mode == "full":
        assert pct == 100.0
    elif mode == "frozen":
        assert pct < 1.0
    else:  # lora
        assert 0.1 < pct < 10.0


def test_classifier_unknown_mode_raises():
    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    with pytest.raises(ValueError):
        FastJEPAModel.for_image_classification(model, num_classes=3, finetune="bogus")


def test_classifier_trainer_smoke():
    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    clf = FastJEPAModel.for_image_classification(model, num_classes=3, finetune="lora", r=4)
    imgs, labels = _synthetic(24)
    cfg = JEPAClassifierConfig(img_size=64, batch_size=8, max_steps=3, log_every=100)
    tr = JEPAClassifierTrainer(clf, cfg, imgs, labels, eval_images=imgs, eval_labels=labels)
    hist = tr.train()
    assert len(hist) == 3 and all(np.isfinite(h) for h in hist)
    acc = tr.evaluate()
    assert 0.0 <= acc <= 1.0


def test_classifier_predict_shapes():
    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    clf = FastJEPAModel.for_image_classification(model, num_classes=3, finetune="frozen")
    imgs, _ = _synthetic(10)
    preds = clf.predict(imgs, batch_size=4)
    assert preds.shape == (10,) and preds.dtype != object
    probs = clf.predict(imgs, batch_size=4, return_probs=True)
    assert probs.shape == (10, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


@pytest.mark.parametrize("mode", ["frozen", "lora", "full"])
def test_classifier_save_load_predict_roundtrip(mode, tmp_path):
    """Train-free lifecycle check: reloaded classifier predicts identically."""
    model, _ = FastJEPAModel.from_pretrained("vit-debug", img_size=64, patch_size=16)
    clf = FastJEPAModel.for_image_classification(model, num_classes=3, finetune=mode, r=4)
    imgs, _ = _synthetic(12)
    preds_before = clf.predict(imgs)
    clf.save_pretrained(tmp_path)
    reloaded = FastJEPAModel.load_classifier(str(tmp_path))
    preds_after = reloaded.predict(imgs)
    assert np.array_equal(preds_before, preds_after)


@pytest.mark.slow
def test_ijepa_parity_vs_hf():
    """The MLX I-JEPA port must match the HuggingFace torch model."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import AutoModel

    repo = "facebook/ijepa_vith14_1k"
    model, _ = FastJEPAModel.from_pretrained(repo)
    rng = np.random.default_rng(0)
    img = (rng.random((224, 224, 3)) * 255).astype(np.uint8)
    feat_mlx = np.array(model.encode([img]))

    hf = AutoModel.from_pretrained(repo, attn_implementation="eager").eval()
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    x = (img.astype(np.float32) / 255.0 - mean) / std
    xt = torch.from_numpy(x.transpose(2, 0, 1))[None]
    with torch.no_grad():
        feat_hf = hf(pixel_values=xt).last_hidden_state.mean(dim=1).numpy()

    cos = float((feat_mlx * feat_hf).sum() /
                (np.linalg.norm(feat_mlx) * np.linalg.norm(feat_hf)))
    assert cos > 0.999
    assert np.abs(feat_mlx - feat_hf).max() < 0.05


@pytest.mark.slow
def test_ijepa_lora_finetune_decreases():
    """Real I-JEPA LoRA fine-tune should learn a trivial 3-class task to high acc."""
    model, _ = FastJEPAModel.from_pretrained("facebook/ijepa_vith14_1k")
    clf = FastJEPAModel.for_image_classification(model, num_classes=3, finetune="lora", r=8)
    rng = np.random.default_rng(0)
    centers = [(230, 30, 30), (30, 230, 30), (30, 30, 230)]

    def make(n, seed):
        r = np.random.default_rng(seed)
        xs, ys = [], []
        for i in range(n):
            c = i % 3
            im = (r.random((224, 224, 3)) * 60).astype(np.uint8)
            y, x = int(r.integers(20, 150)), int(r.integers(20, 150))
            im[y:y + 60, x:x + 60] = np.array(centers[c], np.uint8)
            xs.append(im)
            ys.append(c)
        return xs, ys

    tr_x, tr_y = make(60, 0)
    te_x, te_y = make(30, 1)
    # warmup matters for LoRA on a deep ViT (see JEPAClassifierConfig docstring).
    cfg = JEPAClassifierConfig(img_size=224, batch_size=6, num_epochs=5,
                               learning_rate=3e-4, warmup_ratio=0.2, log_every=100)
    trainer = JEPAClassifierTrainer(clf, cfg, tr_x, tr_y, eval_images=te_x, eval_labels=te_y)
    hist = trainer.train()
    assert all(np.isfinite(h) for h in hist)
    assert hist[-1] < hist[0]
    assert trainer.evaluate() > 0.66  # chance = 0.33
