"""
JEPA (Joint-Embedding Predictive Architecture) for MLX-Tune.

Phase 1 implements **LeJEPA** (Balestriero & LeCun, arXiv 2511.08544) — a
heuristics-free self-supervised objective: a single encoder trained with a
multi-view prediction loss plus **SIGReg** (Sketched Isotropic Gaussian
Regularization). No EMA teacher, no predictor network, no stop-gradient.

This lets you *pretrain a Vision Transformer from scratch* on Apple Silicon with
an Unsloth-flavoured API:

    from mlx_tune import FastJEPAModel, JEPATrainer, JEPAConfig

    model, _ = FastJEPAModel.from_pretrained("vit-tiny", img_size=128)
    trainer = JEPATrainer(model, args=JEPAConfig(num_epochs=5), train_dataset=images)
    trainer.train()
    feats = model.encode(images)          # frozen features for downstream tasks

See `jepa.md` at the repo root for the research, math, and roadmap (I-JEPA /
V-JEPA 2 phases).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from ._perf import configure_wired_limit


def _threaded_prefetch(batch_iter, collate_fn, depth: int = 2):
    """Yield ``collate_fn(batch)`` results, prepared on a background thread.

    Overlaps CPU-bound multi-crop augmentation with GPU compute so the encoder
    isn't stalled waiting on PIL. ``depth`` controls how many batches run ahead.
    """
    import queue
    import threading

    q: "queue.Queue" = queue.Queue(maxsize=depth)
    sentinel = object()

    def worker():
        try:
            for batch in batch_iter:
                q.put(collate_fn(batch))
        except Exception as exc:  # surface worker errors to the consumer
            q.put(exc)
        finally:
            q.put(sentinel)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item

# ImageNet normalisation (matches the LeJEPA reference recipe).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# LeJEPA losses (SIGReg + multi-view prediction)
# ──────────────────────────────────────────────────────────────────────────────

def _trapz(y: mx.array, t: mx.array) -> mx.array:
    """Trapezoidal integration over the last axis for a uniform grid ``t``."""
    dx = t[1] - t[0]
    edges = 0.5 * (y[..., 0] + y[..., -1])
    return (mx.sum(y, axis=-1) - edges) * dx


def sample_directions(dim: int, num_slices: int, key: Optional[mx.array] = None) -> mx.array:
    """Sample ``num_slices`` random unit directions in ``R^dim`` (columns)."""
    if key is None:
        a = mx.random.normal((dim, num_slices))
    else:
        a = mx.random.normal((dim, num_slices), key=key)
    return a / (mx.linalg.norm(a, axis=0, keepdims=True) + 1e-8)


def sigreg_loss(
    z: mx.array,
    directions: mx.array,
    n_points: int = 17,
    t_max: float = 5.0,
    sigma: float = 1.0,
) -> mx.array:
    """Sketched Isotropic Gaussian Regularization (Epps-Pulley flavour).

    Pushes the batch of embeddings ``z`` toward an isotropic Gaussian ``N(0, I)``
    by testing, along many random 1-D projections, whether the projected samples
    look like a standard normal. Uses the empirical characteristic function
    (``exp(i t x) = cos(t x) + i sin(t x)``) so everything is real arithmetic.

    Args:
        z: Embeddings ``(N, K)`` (N samples, K dims). Not normalised.
        directions: Unit projection directions ``(K, S)`` from :func:`sample_directions`.
        n_points: Quadrature points for the characteristic-function integral.
        t_max: Integration domain is ``[-t_max, t_max]``.
        sigma: Bandwidth of the Gaussian weight ``w(t) = exp(-0.5 t^2 / sigma^2)``.

    Returns:
        Scalar regularization loss (0 when ``z`` is exactly standard normal).
    """
    proj = z @ directions                              # (N, S)
    t = mx.linspace(-t_max, t_max, n_points)           # (P,)
    xt = proj[:, :, None] * t[None, None, :]           # (N, S, P)
    cos_mean = mx.mean(mx.cos(xt), axis=0)             # (S, P)  Re[ECF]
    sin_mean = mx.mean(mx.sin(xt), axis=0)             # (S, P)  Im[ECF]
    phi = mx.exp(-0.5 * t * t)                         # (P,)    CF of N(0,1)
    weight = mx.exp(-0.5 * t * t / (sigma * sigma))    # (P,)    Gaussian window
    err = ((cos_mean - phi[None, :]) ** 2 + sin_mean ** 2) * weight[None, :]
    integ = _trapz(err, t)                              # (S,)
    return mx.mean(integ)


def lejepa_prediction_loss(z: mx.array, n_global: int) -> mx.array:
    """Multi-view prediction loss: every view predicts the mean of global views.

    Args:
        z: View embeddings ``(V, B, K)`` — first ``n_global`` are the global views.
        n_global: Number of global views.

    Returns:
        Scalar squared-error prediction loss.
    """
    center = mx.mean(z[:n_global], axis=0)             # (B, K)
    diff = z - center[None]                             # (V, B, K)
    return mx.mean(diff * diff)


def lejepa_loss(
    z: mx.array,
    n_global: int,
    directions: mx.array,
    lam: float = 0.05,
    n_points: int = 17,
    t_max: float = 5.0,
    sigma: float = 1.0,
    return_components: bool = False,
):
    """Full LeJEPA objective ``(1 - lam) * L_pred + lam * SIGReg``.

    Args:
        z: View embeddings ``(V, B, K)``.
        n_global: Number of global views (first along axis 0).
        directions: Unit projection directions ``(K, S)``.
        lam: Weight on the SIGReg term (single LeJEPA hyperparameter, ~0.05).
        return_components: If True, also return ``(pred, sigreg)`` for logging.

    Returns:
        Scalar total loss, or ``(total, pred, sigreg)`` if ``return_components``.
    """
    pred = lejepa_prediction_loss(z, n_global)
    V = z.shape[0]
    sig_terms = [
        sigreg_loss(z[v], directions, n_points=n_points, t_max=t_max, sigma=sigma)
        for v in range(V)
    ]
    sig = mx.mean(mx.stack(sig_terms))
    total = (1.0 - lam) * pred + lam * sig
    if return_components:
        return total, pred, sig
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Compact Vision Transformer encoder (pure MLX)
# ──────────────────────────────────────────────────────────────────────────────

class _MHA(nn.Module):
    """Multi-head self-attention using MLX's fused scaled-dot-product kernel.

    Separate q/k/v/out projections (named to line up with HF I-JEPA weights:
    ``query``/``key``/``value`` → ``q_proj``/``k_proj``/``v_proj``,
    ``attention.output.dense`` → ``out_proj``). Uses
    ``mx.fast.scaled_dot_product_attention`` — much faster and lower-memory than
    a hand-rolled QKᵀ/softmax/V.
    """

    def __init__(self, dim: int, heads: int, qkv_bias: bool = True):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, D = x.shape
        H, hd = self.heads, self.head_dim
        q = self.q_proj(x).reshape(B, N, H, hd).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, H, hd).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, H, hd).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        o = o.transpose(0, 2, 1, 3).reshape(B, N, D)
        return self.out_proj(o)


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block (attr names line up with HF ViT/I-JEPA)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0,
                 eps: float = 1e-5, qkv_bias: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)        # HF: layernorm_before
        self.attn = _MHA(dim, heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=eps)        # HF: layernorm_after
        hidden = int(round(dim * mlp_ratio))
        self.fc1 = nn.Linear(dim, hidden)              # HF: intermediate.dense
        self.fc2 = nn.Linear(hidden, dim)              # HF: output.dense

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.fc2(nn.gelu(self.fc1(self.norm2(x))))
        return x


def _perfect_square(n: int) -> Optional[int]:
    """Return ``isqrt(n)`` if ``n`` is a perfect square, else ``None``."""
    r = int(round(n ** 0.5))
    return r if r * r == n else None


def _interpolate_pos_embed(pos_embed: mx.array, new_grid: int, has_cls: bool) -> mx.array:
    """Bicubically resize a learned positional embedding to a new patch grid.

    ``pos_embed`` is ``(1, N(+1), dim)`` where the ``N`` patch tokens form a
    square ``old_grid × old_grid``. The grid is interpolated to
    ``new_grid × new_grid`` (the standard ViT trick for running a model at a
    resolution it wasn't trained at — DINO/timm use bicubic). A leading CLS slot
    is preserved untouched. No-op (returns the input) when the grids already
    match, so the native-resolution path stays bit-identical.
    """
    cls = None
    pe = pos_embed
    if has_cls:
        cls, pe = pe[:, :1, :], pe[:, 1:, :]
    n, dim = pe.shape[1], pe.shape[2]
    old = _perfect_square(n)
    if old is None:
        raise ValueError(f"pos-embed has {n} patch tokens, not a square grid")
    if old == new_grid:
        return pos_embed
    from mlx.nn.layers.upsample import upsample_cubic
    g = pe.astype(mx.float32).reshape(1, old, old, dim)
    # +0.1 guards against float floor in the output-size computation int(scale*N).
    scale = (new_grid + 0.1) / old
    g = upsample_cubic(g, (scale, scale), align_corners=False)
    g = g[:, :new_grid, :new_grid, :].reshape(1, new_grid * new_grid, dim)
    g = g.astype(pos_embed.dtype)
    return g if cls is None else mx.concatenate([cls, g], axis=1)


class ViTEncoder(nn.Module):
    """Configurable ViT: patch embed → (CLS) + pos embed → transformer → pool.

    Input is ``(B, H, W, 3)`` (NHWC, MLX convention); output is a pooled
    backbone embedding ``(B, dim)`` (mean over tokens). No L2 normalisation
    (would fight the isotropic-Gaussian target). The same class serves LeJEPA
    from-scratch (CLS token, learned pos embed) and pretrained **I-JEPA**
    (no CLS, pos embed loaded from the checkpoint).
    """

    def __init__(self, img_size: int, patch_size: int, dim: int, depth: int, heads: int,
                 in_chans: int = 3, mlp_ratio: float = 4.0, eps: float = 1e-5,
                 qkv_bias: bool = True, use_cls_token: bool = True,
                 pos_embed_type: str = "learned"):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size})")
        self.dim = dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.use_cls_token = use_cls_token
        grid = img_size // patch_size
        n_patches = grid * grid
        n_tokens = n_patches + (1 if use_cls_token else 0)
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        if use_cls_token:
            self.cls_token = mx.random.normal((1, 1, dim)) * 0.02
        if pos_embed_type == "learned":
            self.pos_embed = mx.random.normal((1, n_tokens, dim)) * 0.02
        elif pos_embed_type == "none":
            self.pos_embed = mx.zeros((1, n_tokens, dim))
        else:
            raise ValueError(f"Unknown pos_embed_type '{pos_embed_type}'")
        self.blocks = [
            _TransformerBlock(dim, heads, mlp_ratio=mlp_ratio, eps=eps, qkv_bias=qkv_bias)
            for _ in range(depth)
        ]
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward_tokens(self, x: mx.array) -> mx.array:
        """Return the full token sequence ``(B, N, dim)`` (after final LayerNorm).

        This is what an attentive-pooling probe consumes; ``__call__`` mean-pools
        the result for the default pooled embedding.
        """
        x = self.patch_embed(x)                         # (B, g, g, dim)
        B = x.shape[0]
        x = x.reshape(B, -1, self.dim)                  # (B, N, dim)
        if self.use_cls_token:
            cls = mx.broadcast_to(self.cls_token, (B, 1, self.dim))
            x = mx.concatenate([cls, x], axis=1)        # (B, N+1, dim)
        pos = self.pos_embed
        if pos.shape[1] != x.shape[1]:
            # Input resolution differs from the stored pos-embed grid — interpolate
            # on the fly (no-op when they match, so native res stays bit-identical).
            n_patch = x.shape[1] - (1 if self.use_cls_token else 0)
            new_grid = _perfect_square(n_patch)
            if new_grid is None:
                raise ValueError(f"non-square patch count {n_patch} (input not square?)")
            pos = _interpolate_pos_embed(pos, new_grid, self.use_cls_token)
        x = x + pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)                              # (B, N, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.mean(self.forward_tokens(x), axis=1)   # (B, dim)


_PRESETS = {
    "vit-debug": dict(dim=64, depth=2, heads=2),     # tiny, for tests
    "vit-tiny": dict(dim=192, depth=12, heads=3),
    "vit-small": dict(dim=384, depth=12, heads=6),
    "vit-base": dict(dim=768, depth=12, heads=12),
}


# ──────────────────────────────────────────────────────────────────────────────
# Image helpers (PIL/numpy multi-crop augmentation)
# ──────────────────────────────────────────────────────────────────────────────

def _to_pil(img: Any):
    from PIL import Image

    if isinstance(img, dict):
        for k in ("image", "img", "jpg", "png", "pixel_values"):
            if k in img:
                img = img[k]
                break
        else:
            raise ValueError(f"Could not find an image in dict keys {list(img.keys())}")
    from PIL import Image as _Image
    if isinstance(img, _Image.Image):
        return img.convert("RGB")
    if isinstance(img, mx.array):
        img = np.array(img)
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if float(arr.max(initial=0.0)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr).convert("RGB")


def _normalize(pil_img, size: int) -> np.ndarray:
    a = np.asarray(pil_img, dtype=np.float32) / 255.0   # (H, W, 3)
    return (a - _IMAGENET_MEAN) / _IMAGENET_STD


def _eval_transform(img: Any, size: int) -> np.ndarray:
    from PIL import Image

    pil = _to_pil(img).resize((size, size), Image.BILINEAR)
    return _normalize(pil, size)


class JEPAAugment:
    """DINO-style multi-crop augmentation producing global + local views.

    ``__call__(image)`` returns a list of ``n_global + n_local`` normalised
    ``(img_size, img_size, 3)`` float32 arrays. Multi-scale comes from the crop
    *area* (global vs local scale ranges); all views are rendered at the same
    pixel resolution to keep a single positional-embedding grid (v1 design).
    """

    def __init__(self, img_size: int = 128, n_global: int = 2, n_local: int = 6,
                 global_scale: Tuple[float, float] = (0.4, 1.0),
                 local_scale: Tuple[float, float] = (0.05, 0.4),
                 hflip: bool = True, color_jitter: float = 0.4,
                 grayscale_p: float = 0.2, blur_p: float = 0.1, seed: int = 42):
        self.img_size = img_size
        self.n_global = n_global
        self.n_local = n_local
        self.global_scale = global_scale
        self.local_scale = local_scale
        self.hflip = hflip
        self.color_jitter = color_jitter
        self.grayscale_p = grayscale_p
        self.blur_p = blur_p
        self._rng = random.Random(seed)

    def _rrc(self, pil, scale):
        from PIL import Image

        W, H = pil.size
        area = W * H
        ratio = (3.0 / 4.0, 4.0 / 3.0)
        for _ in range(10):
            target = area * self._rng.uniform(*scale)
            ar = math.exp(self._rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
            w = int(round(math.sqrt(target * ar)))
            h = int(round(math.sqrt(target / ar)))
            if 0 < w <= W and 0 < h <= H:
                x = self._rng.randint(0, W - w)
                y = self._rng.randint(0, H - h)
                crop = pil.crop((x, y, x + w, y + h))
                return crop.resize((self.img_size, self.img_size), Image.BILINEAR)
        s = min(W, H)
        x, y = (W - s) // 2, (H - s) // 2
        return pil.crop((x, y, x + s, y + s)).resize((self.img_size, self.img_size), Image.BILINEAR)

    def _photometric(self, pil):
        from PIL import ImageEnhance, ImageFilter, ImageOps

        if self.hflip and self._rng.random() < 0.5:
            from PIL import Image
            pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
        if self.color_jitter > 0:
            for Enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                f = 1.0 + self._rng.uniform(-self.color_jitter, self.color_jitter)
                pil = Enh(pil).enhance(max(0.1, f))
        if self._rng.random() < self.grayscale_p:
            pil = ImageOps.grayscale(pil).convert("RGB")
        if self._rng.random() < self.blur_p:
            pil = pil.filter(ImageFilter.GaussianBlur(radius=self._rng.uniform(0.1, 2.0)))
        return pil

    def __call__(self, image: Any) -> List[np.ndarray]:
        pil = _to_pil(image)
        views: List[np.ndarray] = []
        for _ in range(self.n_global):
            v = self._photometric(self._rrc(pil, self.global_scale))
            views.append(_normalize(v, self.img_size))
        for _ in range(self.n_local):
            v = self._photometric(self._rrc(pil, self.local_scale))
            views.append(_normalize(v, self.img_size))
        return views


class JEPADataCollator:
    """Turns a list of images into a ``(V, B, H, W, 3)`` float32 view tensor."""

    def __init__(self, augment: JEPAAugment):
        self.augment = augment

    def __call__(self, batch: Sequence[Any]) -> np.ndarray:
        per_sample = [self.augment(s) for s in batch]   # list[B] of list[V]
        B = len(per_sample)
        V = len(per_sample[0])
        views = np.stack([
            np.stack([per_sample[b][v] for b in range(B)]) for v in range(V)
        ])                                               # (V, B, H, W, 3)
        return views.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Model wrapper + FastJEPAModel
# ──────────────────────────────────────────────────────────────────────────────

class JEPAModelWrapper(nn.Module):
    """Holds the encoder; provides ``encode()`` and save/load."""

    def __init__(self, encoder: ViTEncoder, config: dict):
        super().__init__()
        self.encoder = encoder
        self.config = dict(config)
        self.img_size = int(config["img_size"])
        self.embed_dim = encoder.dim

    def __call__(self, x: mx.array) -> mx.array:
        return self.encoder(x)

    def encode(self, images: Sequence[Any], batch_size: int = 64,
               img_size: Optional[int] = None) -> mx.array:
        """Extract frozen pooled backbone features ``(N, dim)`` (eval transform)."""
        was_training = self.training
        self.eval()
        size = img_size or self.img_size
        out: List[np.ndarray] = []
        n = len(images)
        for i in range(0, n, batch_size):
            chunk = [images[j] for j in range(i, min(i + batch_size, n))]
            arr = np.stack([_eval_transform(im, size) for im in chunk]).astype(np.float32)
            z = self.__call__(mx.array(arr))
            mx.eval(z)
            out.append(np.array(z))
        if was_training:
            self.train()
        return mx.array(np.concatenate(out, axis=0))

    def encode_tokens(self, images: Sequence[Any], batch_size: int = 64,
                      img_size: Optional[int] = None) -> mx.array:
        """Extract frozen *token-level* features ``(N, T, dim)`` (eval transform).

        Unlike :meth:`encode` (which mean-pools), this returns the full token
        sequence per image — needed by :func:`attentive_probe`, which learns an
        attention-pooling head over the tokens (the canonical I-JEPA / V-JEPA 2
        evaluation, which mean-pooling under-reads).
        """
        was_training = self.training
        self.eval()
        size = img_size or self.img_size
        out: List[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            chunk = [images[j] for j in range(i, min(i + batch_size, len(images)))]
            arr = np.stack([_eval_transform(im, size) for im in chunk]).astype(np.float32)
            z = self.encoder.forward_tokens(mx.array(arr))
            mx.eval(z)
            out.append(np.array(z))
        if was_training:
            self.train()
        return mx.array(np.concatenate(out, axis=0))

    def save_pretrained(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        weights = dict(tree_flatten(self.parameters()))
        mx.save_safetensors(str(p / "model.safetensors"), weights)
        (p / "jepa_config.json").write_text(json.dumps(self.config, indent=2))


def _encoder_from_config(config: dict) -> ViTEncoder:
    return ViTEncoder(
        img_size=int(config["img_size"]),
        patch_size=int(config["patch_size"]),
        dim=int(config["dim"]),
        depth=int(config["depth"]),
        heads=int(config["heads"]),
        mlp_ratio=float(config.get("mlp_ratio", 4.0)),
        eps=float(config.get("eps", 1e-5)),
        qkv_bias=bool(config.get("qkv_bias", True)),
        use_cls_token=bool(config.get("use_cls_token", True)),
        pos_embed_type=str(config.get("pos_embed_type", "learned")),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pretrained I-JEPA (Meta) loader — HF safetensors → MLX (no torch needed)
# ──────────────────────────────────────────────────────────────────────────────

def _convert_ijepa_weights(hf: dict, depth: int, n_patches: int) -> dict:
    """Map HuggingFace I-JEPA weight names → our ``ViTEncoder`` param names.

    PyTorch ``nn.Linear`` weight is ``(out, in)`` — identical to MLX, so no
    transpose. PyTorch ``Conv2d`` weight is ``(out, in, kH, kW)``; MLX wants
    ``(out, kH, kW, in)`` → transpose ``(0, 2, 3, 1)``. All params are prefixed
    ``encoder.`` because the wrapper holds the ViT as ``self.encoder``.
    """
    out = {}
    out["encoder.patch_embed.weight"] = mx.transpose(
        hf["embeddings.patch_embeddings.projection.weight"], (0, 2, 3, 1)
    )
    out["encoder.patch_embed.bias"] = hf["embeddings.patch_embeddings.projection.bias"]

    pe = hf["embeddings.position_embeddings"]   # (1, native(+cls), dim)
    target_grid = _perfect_square(n_patches)
    native = pe.shape[1]
    if _perfect_square(native) is not None:
        native_grid = _perfect_square(native)            # no CLS slot
    elif _perfect_square(native - 1) is not None:
        pe = pe[:, 1:, :]                                 # strip leading CLS slot
        native_grid = _perfect_square(native - 1)
    else:
        raise ValueError(f"I-JEPA pos-embed length {native} is not a (CLS+)square grid")
    if target_grid is not None and native_grid != target_grid:
        # Loading at a non-native resolution → interpolate the grid (bicubic).
        pe = _interpolate_pos_embed(pe, target_grid, has_cls=False)
    out["encoder.pos_embed"] = pe

    for i in range(depth):
        h = f"encoder.layer.{i}."
        m = f"encoder.blocks.{i}."
        out[m + "norm1.weight"] = hf[h + "layernorm_before.weight"]
        out[m + "norm1.bias"] = hf[h + "layernorm_before.bias"]
        out[m + "attn.q_proj.weight"] = hf[h + "attention.attention.query.weight"]
        out[m + "attn.q_proj.bias"] = hf[h + "attention.attention.query.bias"]
        out[m + "attn.k_proj.weight"] = hf[h + "attention.attention.key.weight"]
        out[m + "attn.k_proj.bias"] = hf[h + "attention.attention.key.bias"]
        out[m + "attn.v_proj.weight"] = hf[h + "attention.attention.value.weight"]
        out[m + "attn.v_proj.bias"] = hf[h + "attention.attention.value.bias"]
        out[m + "attn.out_proj.weight"] = hf[h + "attention.output.dense.weight"]
        out[m + "attn.out_proj.bias"] = hf[h + "attention.output.dense.bias"]
        out[m + "norm2.weight"] = hf[h + "layernorm_after.weight"]
        out[m + "norm2.bias"] = hf[h + "layernorm_after.bias"]
        out[m + "fc1.weight"] = hf[h + "intermediate.dense.weight"]
        out[m + "fc1.bias"] = hf[h + "intermediate.dense.bias"]
        out[m + "fc2.weight"] = hf[h + "output.dense.weight"]
        out[m + "fc2.bias"] = hf[h + "output.dense.bias"]

    out["encoder.norm.weight"] = hf["layernorm.weight"]
    out["encoder.norm.bias"] = hf["layernorm.bias"]
    return out


def _ijepa_config_dict(cfg: dict, img_size: Optional[int]) -> dict:
    """Build our encoder config from a HuggingFace I-JEPA ``config.json`` dict."""
    dim = int(cfg["hidden_size"])
    native = int(cfg["image_size"])
    patch = int(cfg["patch_size"])
    size = int(img_size) if img_size else native
    if size % patch != 0:
        raise ValueError(
            f"img_size ({size}) must be divisible by the model's patch_size ({patch})."
        )
    # size != native is now supported: the learned pos-embed is bicubically
    # interpolated to the new grid at load time (see _convert_ijepa_weights).
    return {
        "arch": "ijepa",
        "img_size": size,
        "patch_size": int(cfg["patch_size"]),
        "dim": dim,
        "depth": int(cfg["num_hidden_layers"]),
        "heads": int(cfg["num_attention_heads"]),
        "mlp_ratio": float(cfg["intermediate_size"]) / dim,
        "eps": float(cfg.get("layer_norm_eps", 1e-6)),
        "qkv_bias": bool(cfg.get("qkv_bias", True)),
        "use_cls_token": False,
        "pos_embed_type": "learned",
    }


def _build_ijepa(config: dict, hf_weights: dict) -> "JEPAModelWrapper":
    """Construct a wrapped MLX I-JEPA from our config + raw HF weights.

    Split out from the hub download so it can be unit-tested offline with a
    synthetic HF-format checkpoint.
    """
    encoder = _encoder_from_config(config)
    model = JEPAModelWrapper(encoder, config)
    grid = config["img_size"] // config["patch_size"]
    converted = _convert_ijepa_weights(hf_weights, config["depth"], grid * grid)
    model.load_weights(list(converted.items()))
    mx.eval(model.parameters())
    return model


def _load_pretrained_ijepa(repo: str, img_size: Optional[int] = None) -> Tuple["JEPAModelWrapper", None]:
    """Download a Meta I-JEPA checkpoint from the HF Hub and convert it to MLX."""
    import glob as _glob
    from huggingface_hub import hf_hub_download, snapshot_download

    cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    if cfg.get("model_type") != "ijepa":
        raise ValueError(
            f"'{repo}' is not an I-JEPA checkpoint (model_type={cfg.get('model_type')!r}). "
            f"Use a 'facebook/ijepa_*' repo, a preset ({list(_PRESETS)}), or a saved dir."
        )
    config = _ijepa_config_dict(cfg, img_size)

    local = snapshot_download(repo, allow_patterns=["*.safetensors"])
    files = sorted(_glob.glob(str(Path(local) / "*.safetensors")))
    hf: dict = {}
    for f in files:
        hf.update(mx.load(f))
    return _build_ijepa(config, hf), None


def _set_finetune_mode(head_model: nn.Module, base_config: dict, finetune: str,
                       r: int, lora_alpha: int, lora_dropout: float,
                       target_modules: Optional[List[str]], extra: Optional[dict] = None) -> None:
    """Set a downstream head-model's trainable surface (frozen / lora / full).

    Shared by the classification / regression / dense builders. ``head_model``
    must expose ``.encoder`` and ``.head``. Stashes the arch config on the
    encoder (``_jepa_config``) so the model can be rebuilt exactly on reload.
    """
    head_model.encoder._jepa_config = dict(base_config)
    head_model.encoder._jepa_config.update(
        {"finetune": finetune, "lora_r": r, "lora_alpha": lora_alpha,
         "lora_dropout": lora_dropout, "lora_targets": target_modules}
    )
    if extra:
        head_model.encoder._jepa_config.update(extra)
    if finetune == "full":
        head_model.unfreeze()
    elif finetune == "frozen":
        head_model.encoder.freeze()
        head_model.head.unfreeze()
    elif finetune == "lora":
        # Freeze the base FIRST, then wrap with LoRA so only adapters + head train.
        head_model.encoder.freeze()
        n = apply_lora_to_encoder(
            head_model.encoder, r=r, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout, target_modules=target_modules,
        )
        if n == 0:
            raise ValueError("No LoRA targets matched the encoder blocks")
        head_model.head.unfreeze()
    else:
        raise ValueError(f"finetune must be 'frozen', 'lora', or 'full'; got {finetune!r}")
    mx.eval(head_model.parameters())


class FastJEPAModel:
    """Unsloth-style entry point for JEPA models.

    ``from_pretrained`` accepts three kinds of ``model_name``:

    * a **preset** (``vit-debug``/``vit-tiny``/``vit-small``/``vit-base``) →
      randomly-initialised ViT for LeJEPA from-scratch pretraining;
    * a **saved JEPA directory** (created by ``save_pretrained``) → reload;
    * a **pretrained Meta I-JEPA repo** (``facebook/ijepa_*``) → download +
      convert HF weights to MLX for feature extraction / downstream tuning.

    Returns ``(model, None)`` — there is no tokenizer for vision SSL.
    """

    @staticmethod
    def from_pretrained(
        model_name: str = "vit-tiny",
        img_size: Optional[int] = None,
        patch_size: int = 16,
        **overrides,
    ) -> Tuple[JEPAModelWrapper, None]:
        p = Path(model_name)
        if p.exists() and (p / "jepa_config.json").exists():
            config = json.loads((p / "jepa_config.json").read_text())
            config.update(overrides)
            encoder = _encoder_from_config(config)
            model = JEPAModelWrapper(encoder, config)
            weights = mx.load(str(p / "model.safetensors"))
            model.load_weights(list(weights.items()))
            mx.eval(model.parameters())
            return model, None

        if model_name in _PRESETS:
            preset = dict(_PRESETS[model_name])
            config = {
                "arch": "lejepa",
                "preset": model_name,
                "img_size": int(img_size) if img_size else 128,
                "patch_size": patch_size,
                "dim": preset["dim"],
                "depth": preset["depth"],
                "heads": preset["heads"],
                "mlp_ratio": 4.0,
                "eps": 1e-5,
                "qkv_bias": True,
                "use_cls_token": True,
                "pos_embed_type": "learned",
            }
            config.update({k: v for k, v in overrides.items() if k in config})
            encoder = _encoder_from_config(config)
            model = JEPAModelWrapper(encoder, config)
            mx.eval(model.parameters())
            return model, None

        # Otherwise: treat as a Hugging Face pretrained checkpoint (I-JEPA).
        if "/" in model_name:
            return _load_pretrained_ijepa(model_name, img_size=img_size)

        raise ValueError(
            f"Unknown JEPA model '{model_name}'. Use a preset ({list(_PRESETS)}), a saved "
            f"checkpoint directory, or a Hugging Face 'facebook/ijepa_*' repo."
        )

    @staticmethod
    def for_image_classification(
        model: JEPAModelWrapper,
        num_classes: int,
        finetune: str = "lora",
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ) -> JEPAForImageClassification:
        """Attach a classification head and set the encoder's trainable surface.

        ``finetune``:
          * ``"frozen"`` — encoder frozen, train only the head (linear probe).
          * ``"lora"``   — encoder frozen except LoRA adapters + head (default).
          * ``"full"``   — everything trainable.
        """
        clf = JEPAForImageClassification(model.encoder, num_classes, model.img_size)
        _set_finetune_mode(clf, model.config, finetune, r, lora_alpha,
                           lora_dropout, target_modules)
        return clf

    @staticmethod
    def for_image_regression(
        model: JEPAModelWrapper,
        out_dim: int = 1,
        finetune: str = "lora",
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ) -> "JEPAForImageRegression":
        """Attach a regression head (counting, scalar/vector value prediction).

        ``out_dim=1`` → scalar regression (e.g. object counting — I-JEPA's
        headline task). ``finetune`` matches :meth:`for_image_classification`.
        """
        reg = JEPAForImageRegression(model.encoder, out_dim, model.img_size)
        _set_finetune_mode(reg, model.config, finetune, r, lora_alpha,
                           lora_dropout, target_modules, extra={"out_dim": out_dim})
        return reg

    @staticmethod
    def for_dense_prediction(
        model: JEPAModelWrapper,
        out_channels: int,
        task: str = "regression",
        finetune: str = "lora",
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ) -> "JEPAForDensePrediction":
        """Attach a dense per-patch head (depth maps, segmentation).

        ``task="regression"`` → per-pixel value map (e.g. depth), MSE training.
        ``task="segmentation"`` → per-pixel class logits, cross-entropy training.
        Output is upsampled from the patch grid to ``img_size``.
        """
        dense = JEPAForDensePrediction(model.encoder, out_channels, model.img_size, task=task)
        _set_finetune_mode(dense, model.config, finetune, r, lora_alpha,
                           lora_dropout, target_modules,
                           extra={"out_channels": out_channels, "dense_task": task})
        return dense

    @staticmethod
    def load_classifier(path: Union[str, Path]):
        """Reload a downstream head-model saved by its ``save_pretrained``.

        Dispatches on the saved ``task`` to rebuild the right head
        (classification / regression / dense prediction), re-applying any LoRA
        structure, then loads all weights — so the reloaded model produces
        identical predictions to the trained one. ``load_regressor`` /
        ``load_dense`` are aliases for discoverability.
        """
        p = Path(path)
        cfg = json.loads((p / "jepa_classifier_config.json").read_text())
        encoder = _encoder_from_config(cfg)
        task = cfg.get("task", "image_classification")
        img = int(cfg["img_size"])
        if task == "image_regression":
            head_model = JEPAForImageRegression(encoder, int(cfg["out_dim"]), img)
        elif task == "dense_prediction":
            head_model = JEPAForDensePrediction(
                encoder, int(cfg["out_channels"]), img,
                task=cfg.get("dense_task", "regression"))
        else:
            head_model = JEPAForImageClassification(encoder, int(cfg["num_classes"]), img)
        if cfg.get("finetune") == "lora" or cfg.get("has_lora"):
            apply_lora_to_encoder(
                head_model.encoder, r=int(cfg.get("lora_r", 16)),
                lora_alpha=int(cfg.get("lora_alpha", 16)),
                lora_dropout=float(cfg.get("lora_dropout", 0.0)),
                target_modules=cfg.get("lora_targets"),
            )
        weights = mx.load(str(p / "model.safetensors"))
        head_model.load_weights(list(weights.items()))
        mx.eval(head_model.parameters())
        return head_model

    @staticmethod
    def load_regressor(path: Union[str, Path]) -> "JEPAForImageRegression":
        """Alias of :meth:`load_classifier` for regression heads."""
        return FastJEPAModel.load_classifier(path)

    @staticmethod
    def load_dense(path: Union[str, Path]) -> "JEPAForDensePrediction":
        """Alias of :meth:`load_classifier` for dense-prediction heads."""
        return FastJEPAModel.load_classifier(path)


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class JEPAConfig:
    """Configuration for :class:`JEPATrainer` (LeJEPA pretraining)."""

    output_dir: str = "jepa_output"
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    batch_size: int = 64
    num_epochs: int = 1
    max_steps: int = -1
    warmup_ratio: float = 0.05
    lam: float = 0.05
    num_slices: int = 1024
    n_points: int = 17
    t_max: float = 5.0
    sigma: float = 1.0
    n_global: int = 2
    n_local: int = 6
    img_size: int = 128
    global_scale: Tuple[float, float] = (0.4, 1.0)
    local_scale: Tuple[float, float] = (0.05, 0.4)
    color_jitter: float = 0.4
    grayscale_p: float = 0.2
    blur_p: float = 0.1
    hflip: bool = True
    shuffle: bool = True
    log_every: int = 10
    seed: int = 42
    # Resumable checkpointing (for long on-device runs over a streaming corpus):
    # every `save_steps` steps, write the encoder + optimizer state + step to
    # `checkpoint_dir` (defaults to <output_dir>/checkpoint). With `resume=True`
    # training continues from the latest checkpoint if one exists.
    save_steps: int = 0
    resume: bool = False
    checkpoint_dir: Optional[str] = None


class ImageFolderDataset:
    """Lazy, indexable image-folder dataset for streaming SSL pretraining.

    Scans ``root`` for image files (recursively by default) and decodes each
    image on access — so the corpus never has to fit in memory. Plugs straight
    into :class:`JEPATrainer` (and the classifier trainers) because it supports
    ``len()`` and ``[i]``; decoding happens on the background prefetch thread.

    >>> ds = ImageFolderDataset("/path/to/images")
    >>> JEPATrainer(model, JEPAConfig(...), ds).train()
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".ppm"}

    def __init__(self, root: Union[str, Path], recursive: bool = True,
                 extensions: Optional[Sequence[str]] = None):
        self.root = Path(root)
        if not self.root.exists():
            raise ValueError(f"ImageFolderDataset: {root} does not exist")
        exts = {e.lower() for e in (extensions or self.IMG_EXTS)}
        it = self.root.rglob("*") if recursive else self.root.glob("*")
        self.paths = sorted(p for p in it if p.suffix.lower() in exts)
        if not self.paths:
            raise ValueError(f"ImageFolderDataset: no images found under {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        from PIL import Image
        return Image.open(self.paths[i]).convert("RGB")


def _save_jepa_checkpoint(ckpt_dir: Path, model: nn.Module, opt, step: int,
                          history: List[float]) -> None:
    """Save encoder + optimizer state + step/history for resumable training."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(ckpt_dir / "model.safetensors"),
                        dict(tree_flatten(model.parameters())))
    mx.save_safetensors(str(ckpt_dir / "optimizer.safetensors"),
                        dict(tree_flatten(opt.state)))
    (ckpt_dir / "trainer_state.json").write_text(
        json.dumps({"step": step, "history": history}))


def _load_jepa_checkpoint(ckpt_dir: Path, model: nn.Module, opt) -> Tuple[int, List[float]]:
    """Restore a checkpoint in place; return ``(start_step, history)``."""
    from mlx.utils import tree_unflatten
    model.load_weights(str(ckpt_dir / "model.safetensors"))
    opt.init(model.trainable_parameters())
    opt.state = tree_unflatten(list(mx.load(str(ckpt_dir / "optimizer.safetensors")).items()))
    state = json.loads((ckpt_dir / "trainer_state.json").read_text())
    return int(state["step"]), list(state.get("history", []))


class JEPATrainer:
    """Trains a :class:`JEPAModelWrapper` with the LeJEPA objective."""

    def __init__(self, model: JEPAModelWrapper, args: JEPAConfig,
                 train_dataset: Sequence[Any], data_collator: Optional[JEPADataCollator] = None):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        if data_collator is None:
            augment = JEPAAugment(
                img_size=args.img_size, n_global=args.n_global, n_local=args.n_local,
                global_scale=args.global_scale, local_scale=args.local_scale,
                hflip=args.hflip, color_jitter=args.color_jitter,
                grayscale_p=args.grayscale_p, blur_p=args.blur_p, seed=args.seed,
            )
            data_collator = JEPADataCollator(augment)
        self.data_collator = data_collator
        self.history: List[float] = []

    def _make_schedule(self, total_steps: int):
        lr = self.args.learning_rate
        warmup_steps = int(self.args.warmup_ratio * total_steps)
        if warmup_steps > 0:
            return optim.join_schedules(
                [optim.linear_schedule(0.0, lr, warmup_steps),
                 optim.cosine_decay(lr, max(1, total_steps - warmup_steps))],
                [warmup_steps],
            )
        return optim.cosine_decay(lr, max(1, total_steps))

    def train(self) -> List[float]:
        args = self.args
        model = self.model
        n = len(self.train_dataset)
        if n == 0:
            raise ValueError("train_dataset is empty")
        steps_per_epoch = max(1, math.ceil(n / args.batch_size))
        if args.max_steps and args.max_steps > 0:
            total_steps = args.max_steps
        else:
            total_steps = args.num_epochs * steps_per_epoch

        configure_wired_limit()
        schedule = self._make_schedule(total_steps)
        opt = optim.AdamW(learning_rate=schedule, weight_decay=args.weight_decay)
        model.train()
        K = model.embed_dim
        rng = random.Random(args.seed)

        # Resume from a checkpoint if requested and one exists.
        ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir \
            else Path(args.output_dir) / "checkpoint"
        start_step = 0
        if args.resume and (ckpt_dir / "trainer_state.json").exists():
            start_step, self.history = _load_jepa_checkpoint(ckpt_dir, model, opt)
            print(f"Resumed from {ckpt_dir} at step {start_step}/{total_steps}")

        def loss_fn(m, views, directions):
            V, B = views.shape[0], views.shape[1]
            flat = views.reshape(V * B, views.shape[2], views.shape[3], views.shape[4])
            z = m(flat).reshape(V, B, -1)
            return lejepa_loss(
                z, args.n_global, directions, lam=args.lam,
                n_points=args.n_points, t_max=args.t_max, sigma=args.sigma,
            )

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        def batch_stream():
            """Yield the remaining ``total_steps - start_step`` batches."""
            idx = list(range(n))
            produced = start_step
            while produced < total_steps:
                if args.shuffle:
                    rng.shuffle(idx)
                for bstart in range(0, n, args.batch_size):
                    if produced >= total_steps:
                        return
                    yield [self.train_dataset[i] for i in idx[bstart:bstart + args.batch_size]]
                    produced += 1

        if start_step == 0:
            self.history = []
        # Augment the next batch on a background thread while the GPU trains the
        # current one — augmentation (PIL multi-crop) is the dominant CPU cost.
        for step, views_np in enumerate(
            _threaded_prefetch(batch_stream(), self.data_collator, depth=2),
            start=start_step,
        ):
            views = mx.array(views_np)
            directions = sample_directions(K, args.num_slices)
            loss, grads = loss_and_grad(model, views, directions)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            l = float(loss)
            self.history.append(l)
            if step % args.log_every == 0:
                lr_now = float(schedule(step)) if callable(schedule) else float(schedule)
                print(f"step {step}/{total_steps}  loss {l:.4f}  lr {lr_now:.2e}")
            if args.save_steps and (step + 1) % args.save_steps == 0:
                _save_jepa_checkpoint(ckpt_dir, model, opt, step + 1, self.history)
        if args.save_steps:
            _save_jepa_checkpoint(ckpt_dir, model, opt, total_steps, self.history)
        return self.history


# ──────────────────────────────────────────────────────────────────────────────
# LoRA + downstream image classification (fine-tune a JEPA encoder)
# ──────────────────────────────────────────────────────────────────────────────

# LoRA target linears inside each ViT block (our naming).
_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


def _unfreeze_lora(module: nn.Module) -> None:
    """Recursively unfreeze only LoRA adapter params (lora_a / lora_b)."""
    for _, child in module.children().items():
        if "LoRA" in type(child).__name__ or "Lora" in type(child).__name__:
            if hasattr(child, "unfreeze"):
                child.unfreeze()
        elif isinstance(child, (list, tuple)):
            for c in child:
                if isinstance(c, nn.Module):
                    _unfreeze_lora(c)
        elif isinstance(child, nn.Module):
            _unfreeze_lora(child)


def apply_lora_to_encoder(encoder: ViTEncoder, r: int = 16, lora_alpha: int = 16,
                          lora_dropout: float = 0.0,
                          target_modules: Optional[List[str]] = None) -> int:
    """Swap target linears in every transformer block for LoRA layers (in place).

    Returns the number of adapted linears. Does not change freezing — the caller
    decides which params are trainable.
    """
    from mlx_lm.tuner.lora import LoRALinear

    targets = target_modules or _LORA_TARGETS
    n = 0
    for block in encoder.blocks:
        for parent, name in (
            (block.attn, "q_proj"), (block.attn, "k_proj"),
            (block.attn, "v_proj"), (block.attn, "out_proj"),
            (block, "fc1"), (block, "fc2"),
        ):
            if name not in targets:
                continue
            lin = getattr(parent, name, None)
            if isinstance(lin, nn.Linear):
                setattr(parent, name,
                        LoRALinear.from_base(lin, r=r, scale=lora_alpha / r, dropout=lora_dropout))
                n += 1
    return n


class JEPAForImageClassification(nn.Module):
    """A JEPA ViT encoder + a linear classification head.

    ``__call__((B, H, W, 3)) -> (B, num_classes)`` logits. The encoder may be
    frozen, LoRA-adapted, or fully trainable depending on how it was prepared
    (see :meth:`FastJEPAModel.for_image_classification`).
    """

    def __init__(self, encoder: ViTEncoder, num_classes: int, img_size: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.dim, num_classes)
        self.img_size = img_size
        self.num_classes = num_classes

    def __call__(self, x: mx.array) -> mx.array:
        return self.head(self.encoder(x))

    def predict(self, images: Sequence[Any], batch_size: int = 64,
                return_probs: bool = False):
        """Predict class ids (or class probabilities) for raw images.

        Applies the eval transform (resize → ImageNet-normalise), runs the
        full classifier, and returns a numpy array of predicted class ids
        ``(N,)`` — or softmax probabilities ``(N, num_classes)`` if
        ``return_probs=True``. Set the model to eval mode internally.
        """
        was = self.training
        self.eval()
        size = self.img_size
        out: List[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            chunk = [images[j] for j in range(i, min(i + batch_size, len(images)))]
            arr = np.stack([_eval_transform(im, size) for im in chunk]).astype(np.float32)
            logits = self(mx.array(arr))
            result = mx.softmax(logits, axis=-1) if return_probs else mx.argmax(logits, axis=-1)
            mx.eval(result)
            out.append(np.array(result))
        if was:
            self.train()
        return np.concatenate(out, axis=0)

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save the full classifier (encoder + LoRA adapters + head) + config.

        Writes ``model.safetensors`` (all parameters, including any fused-in
        LoRA adapter weights) and ``jepa_classifier_config.json``. Reload with
        :meth:`FastJEPAModel.load_classifier`.
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        weights = dict(tree_flatten(self.parameters()))
        mx.save_safetensors(str(p / "model.safetensors"), weights)
        cfg = dict(getattr(self.encoder, "_jepa_config", {}) or {})
        cfg.update({
            "task": "image_classification",
            "num_classes": self.num_classes,
            "img_size": self.img_size,
            "has_lora": any("lora" in k.lower() for k in weights),
        })
        (p / "jepa_classifier_config.json").write_text(json.dumps(cfg, indent=2))


class JEPAForImageRegression(nn.Module):
    """A JEPA encoder + a linear regression head (counting / scalar-vector targets).

    ``__call__((B, H, W, 3)) -> (B, out_dim)`` continuous predictions. Trained
    with MSE (see :class:`JEPARegressionTrainer`). ``out_dim=1`` is scalar
    regression — e.g. object counting, I-JEPA's headline task.
    """

    def __init__(self, encoder: ViTEncoder, out_dim: int, img_size: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.dim, out_dim)
        self.img_size = img_size
        self.out_dim = out_dim

    def __call__(self, x: mx.array) -> mx.array:
        return self.head(self.encoder(x))

    def predict(self, images: Sequence[Any], batch_size: int = 64) -> np.ndarray:
        """Predict continuous targets ``(N, out_dim)`` (or ``(N,)`` if out_dim==1)."""
        was = self.training
        self.eval()
        out: List[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            chunk = [images[j] for j in range(i, min(i + batch_size, len(images)))]
            arr = np.stack([_eval_transform(im, self.img_size) for im in chunk]).astype(np.float32)
            y = self(mx.array(arr))
            mx.eval(y)
            out.append(np.array(y))
        if was:
            self.train()
        res = np.concatenate(out, axis=0)
        return res[:, 0] if self.out_dim == 1 else res

    def save_pretrained(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        weights = dict(tree_flatten(self.parameters()))
        mx.save_safetensors(str(p / "model.safetensors"), weights)
        cfg = dict(getattr(self.encoder, "_jepa_config", {}) or {})
        cfg.update({"task": "image_regression", "out_dim": self.out_dim,
                    "img_size": self.img_size,
                    "has_lora": any("lora" in k.lower() for k in weights)})
        (p / "jepa_classifier_config.json").write_text(json.dumps(cfg, indent=2))


class JEPAForDensePrediction(nn.Module):
    """A JEPA encoder + a dense per-patch head, upsampled to image resolution.

    Consumes patch tokens (``forward_tokens``, CLS dropped), applies a per-token
    linear head, reshapes to the patch grid and bilinearly upsamples to
    ``img_size``. ``task="regression"`` → ``(B, H, W, C)`` value maps (depth);
    ``task="segmentation"`` → ``(B, H, W, C)`` class logits.
    """

    def __init__(self, encoder: ViTEncoder, out_channels: int, img_size: int,
                 task: str = "regression"):
        super().__init__()
        if task not in ("regression", "segmentation"):
            raise ValueError("task must be 'regression' or 'segmentation'")
        self.encoder = encoder
        self.head = nn.Linear(encoder.dim, out_channels)
        self.out_channels = out_channels
        self.img_size = img_size
        self.task = task

    def __call__(self, x: mx.array) -> mx.array:
        toks = self.encoder.forward_tokens(x)             # (B, N(+cls), dim)
        if self.encoder.use_cls_token:
            toks = toks[:, 1:, :]
        B, n = toks.shape[0], toks.shape[1]
        grid = _perfect_square(n)
        if grid is None:
            raise ValueError(f"non-square token count {n}")
        feat = self.head(toks).reshape(B, grid, grid, self.out_channels)
        if grid != self.img_size:
            from mlx.nn.layers.upsample import upsample_linear
            scale = (self.img_size + 0.1) / grid
            feat = upsample_linear(feat, (scale, scale), align_corners=False)
            feat = feat[:, :self.img_size, :self.img_size, :]
        return feat                                       # (B, H, W, C)

    def predict(self, images: Sequence[Any], batch_size: int = 16) -> np.ndarray:
        """Per-pixel predictions: depth maps ``(N,H,W[,C])`` or seg ids ``(N,H,W)``."""
        was = self.training
        self.eval()
        out: List[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            chunk = [images[j] for j in range(i, min(i + batch_size, len(images)))]
            arr = np.stack([_eval_transform(im, self.img_size) for im in chunk]).astype(np.float32)
            y = self(mx.array(arr))
            if self.task == "segmentation":
                y = mx.argmax(y, axis=-1)
            elif self.out_channels == 1:
                y = y[..., 0]
            mx.eval(y)
            out.append(np.array(y))
        if was:
            self.train()
        return np.concatenate(out, axis=0)

    def save_pretrained(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        weights = dict(tree_flatten(self.parameters()))
        mx.save_safetensors(str(p / "model.safetensors"), weights)
        cfg = dict(getattr(self.encoder, "_jepa_config", {}) or {})
        cfg.update({"task": "dense_prediction", "dense_task": self.task,
                    "out_channels": self.out_channels, "img_size": self.img_size,
                    "has_lora": any("lora" in k.lower() for k in weights)})
        (p / "jepa_classifier_config.json").write_text(json.dumps(cfg, indent=2))


@dataclass
class JEPAClassifierConfig:
    """Configuration for :class:`JEPAClassifierTrainer`."""

    output_dir: str = "jepa_clf_output"
    learning_rate: float = 3e-4   # LoRA-safe on a 32-layer ViT; bump for frozen/full
    weight_decay: float = 0.0
    batch_size: int = 64
    num_epochs: int = 10
    max_steps: int = -1
    warmup_ratio: float = 0.15    # warmup matters: LoRA on a deep ViT diverges without it
    img_size: int = 224
    label_smoothing: float = 0.0
    # train-time augmentation (single random-resized-crop per image)
    train_scale: Tuple[float, float] = (0.5, 1.0)
    color_jitter: float = 0.2
    hflip: bool = True
    shuffle: bool = True
    log_every: int = 10
    seed: int = 42


class JEPAClassifierTrainer:
    """Supervised fine-tuning of a :class:`JEPAForImageClassification`.

    Cross-entropy training with on-device AdamW (warmup→cosine), background
    augmentation prefetch, and an ``evaluate()`` for top-1 accuracy. Works for
    frozen-encoder linear probing, LoRA fine-tuning, or full fine-tuning — the
    encoder's freeze state is set when the classifier is built.
    """

    def __init__(self, model: JEPAForImageClassification, args: JEPAClassifierConfig,
                 train_images: Sequence[Any], train_labels: Sequence[int],
                 eval_images: Optional[Sequence[Any]] = None,
                 eval_labels: Optional[Sequence[int]] = None):
        self.model = model
        self.args = args
        self.train_images = train_images
        self.train_labels = list(train_labels)
        self.eval_images = eval_images
        self.eval_labels = list(eval_labels) if eval_labels is not None else None
        self.history: List[float] = []
        self._aug = JEPAAugment(
            img_size=args.img_size, n_global=1, n_local=0,
            global_scale=args.train_scale, hflip=args.hflip,
            color_jitter=args.color_jitter, grayscale_p=0.0, blur_p=0.0, seed=args.seed,
        )

    def _collate(self, batch):
        imgs = [self._aug(im)[0] for im, _ in batch]
        labels = [lb for _, lb in batch]
        return (np.stack(imgs).astype(np.float32), np.asarray(labels, dtype=np.int32))

    def _make_schedule(self, total_steps: int):
        lr = self.args.learning_rate
        warmup = int(self.args.warmup_ratio * total_steps)
        if warmup > 0:
            return optim.join_schedules(
                [optim.linear_schedule(0.0, lr, warmup),
                 optim.cosine_decay(lr, max(1, total_steps - warmup))],
                [warmup],
            )
        return optim.cosine_decay(lr, max(1, total_steps))

    def train(self) -> List[float]:
        args = self.args
        model = self.model
        n = len(self.train_images)
        if n == 0:
            raise ValueError("train_images is empty")
        steps_per_epoch = max(1, math.ceil(n / args.batch_size))
        total_steps = args.max_steps if args.max_steps and args.max_steps > 0 \
            else args.num_epochs * steps_per_epoch

        configure_wired_limit()
        schedule = self._make_schedule(total_steps)
        opt = optim.AdamW(learning_rate=schedule, weight_decay=args.weight_decay)
        model.train()
        rng = random.Random(args.seed)

        def loss_fn(m, x, y):
            return mx.mean(nn.losses.cross_entropy(
                m(x), y, label_smoothing=args.label_smoothing))

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        def batch_stream():
            idx = list(range(n))
            produced = 0
            while produced < total_steps:
                if args.shuffle:
                    rng.shuffle(idx)
                for bstart in range(0, n, args.batch_size):
                    if produced >= total_steps:
                        return
                    sl = idx[bstart:bstart + args.batch_size]
                    yield [(self.train_images[i], self.train_labels[i]) for i in sl]
                    produced += 1

        self.history = []
        for step, (xb, yb) in enumerate(
            _threaded_prefetch(batch_stream(), self._collate, depth=2)
        ):
            loss, grads = loss_and_grad(model, mx.array(xb), mx.array(yb))
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            self.history.append(float(loss))
            if step % args.log_every == 0:
                print(f"step {step}/{total_steps}  loss {float(loss):.4f}")
        return self.history

    def evaluate(self, images: Optional[Sequence[Any]] = None,
                 labels: Optional[Sequence[int]] = None,
                 batch_size: int = 64) -> float:
        images = images if images is not None else self.eval_images
        labels = labels if labels is not None else self.eval_labels
        if images is None or labels is None:
            raise ValueError("No eval data provided")
        self.model.eval()
        size = self.args.img_size
        correct = 0
        labels = np.asarray(labels)
        for i in range(0, len(images), batch_size):
            chunk = [images[j] for j in range(i, min(i + batch_size, len(images)))]
            arr = np.stack([_eval_transform(im, size) for im in chunk]).astype(np.float32)
            preds = np.array(mx.argmax(self.model(mx.array(arr)), axis=1))
            correct += int((preds == labels[i:i + len(chunk)]).sum())
        return correct / len(images)


def _warmup_cosine(lr: float, warmup_ratio: float, total: int):
    warmup = int(warmup_ratio * total)
    if warmup > 0:
        return optim.join_schedules(
            [optim.linear_schedule(0.0, lr, warmup),
             optim.cosine_decay(lr, max(1, total - warmup))],
            [warmup],
        )
    return optim.cosine_decay(lr, max(1, total))


class JEPARegressionTrainer:
    """Supervised regression fine-tuning of a :class:`JEPAForImageRegression`.

    MSE training with on-device AdamW (warmup→cosine); ``evaluate()`` returns
    ``{"mae", "rmse"}``. Uses a resize-only transform (no spatial augmentation)
    so targets stay valid. ``train_targets`` are scalars or ``(out_dim,)`` vectors.
    """

    def __init__(self, model: JEPAForImageRegression, args: JEPAClassifierConfig,
                 train_images: Sequence[Any], train_targets: Sequence[Any],
                 eval_images: Optional[Sequence[Any]] = None,
                 eval_targets: Optional[Sequence[Any]] = None):
        self.model = model
        self.args = args
        self.train_images = train_images
        self.train_targets = np.asarray(train_targets, dtype=np.float32)
        if self.train_targets.ndim == 1:
            self.train_targets = self.train_targets[:, None]
        self.eval_images = eval_images
        self.eval_targets = None if eval_targets is None else np.asarray(eval_targets, np.float32)
        self.history: List[float] = []

    def _collate(self, batch):
        imgs = np.stack([_eval_transform(im, self.args.img_size) for im, _ in batch]).astype(np.float32)
        tg = np.stack([t for _, t in batch]).astype(np.float32)
        return imgs, tg

    def train(self) -> List[float]:
        args, model = self.args, self.model
        n = len(self.train_images)
        if n == 0:
            raise ValueError("train_images is empty")
        steps_per_epoch = max(1, math.ceil(n / args.batch_size))
        total = args.max_steps if args.max_steps and args.max_steps > 0 else args.num_epochs * steps_per_epoch
        configure_wired_limit()
        opt = optim.AdamW(learning_rate=_warmup_cosine(args.learning_rate, args.warmup_ratio, total),
                          weight_decay=args.weight_decay)
        model.train()
        rng = random.Random(args.seed)

        def loss_fn(m, x, y):
            return mx.mean((m(x) - y) ** 2)

        lag = nn.value_and_grad(model, loss_fn)

        def stream():
            idx = list(range(n)); produced = 0
            while produced < total:
                if args.shuffle:
                    rng.shuffle(idx)
                for b in range(0, n, args.batch_size):
                    if produced >= total:
                        return
                    sl = idx[b:b + args.batch_size]
                    yield [(self.train_images[i], self.train_targets[i]) for i in sl]
                    produced += 1

        self.history = []
        for step, (xb, yb) in enumerate(_threaded_prefetch(stream(), self._collate, depth=2)):
            loss, grads = lag(model, mx.array(xb), mx.array(yb))
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            self.history.append(float(loss))
            if step % args.log_every == 0:
                print(f"step {step}/{total}  mse {float(loss):.4f}")
        return self.history

    def evaluate(self, images=None, targets=None, batch_size: int = 64) -> dict:
        images = images if images is not None else self.eval_images
        targets = targets if targets is not None else self.eval_targets
        if images is None or targets is None:
            raise ValueError("No eval data provided")
        preds = self.model.predict(images, batch_size=batch_size)
        t = np.asarray(targets, np.float32)
        p = np.asarray(preds, np.float32)
        if p.ndim == 1:
            p = p[:, None]
        if t.ndim == 1:
            t = t[:, None]
        mae = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        return {"mae": mae, "rmse": rmse}


class JEPADenseTrainer:
    """Supervised training of a :class:`JEPAForDensePrediction` (depth / segmentation).

    ``regression`` task → MSE on per-pixel value maps, ``evaluate()`` → ``{"mae"}``.
    ``segmentation`` task → per-pixel cross-entropy, ``evaluate()`` → ``{"pixel_acc"}``.
    Targets must be provided at ``img_size`` resolution (no spatial augmentation is
    applied, so image/target stay aligned).
    """

    def __init__(self, model: JEPAForDensePrediction, args: JEPAClassifierConfig,
                 train_images: Sequence[Any], train_targets: Sequence[Any],
                 eval_images: Optional[Sequence[Any]] = None,
                 eval_targets: Optional[Sequence[Any]] = None):
        self.model = model
        self.args = args
        self.train_images = train_images
        self.train_targets = train_targets
        self.eval_images = eval_images
        self.eval_targets = eval_targets
        self.history: List[float] = []

    def _collate(self, batch):
        imgs = np.stack([_eval_transform(im, self.args.img_size) for im, _ in batch]).astype(np.float32)
        if self.model.task == "segmentation":
            tg = np.stack([np.asarray(t) for _, t in batch]).astype(np.int32)
        else:
            tg = np.stack([np.asarray(t, np.float32) for _, t in batch])
            if tg.ndim == 3:                              # (B, H, W) -> (B, H, W, 1)
                tg = tg[..., None]
        return imgs, tg

    def train(self) -> List[float]:
        args, model = self.args, self.model
        n = len(self.train_images)
        if n == 0:
            raise ValueError("train_images is empty")
        steps_per_epoch = max(1, math.ceil(n / args.batch_size))
        total = args.max_steps if args.max_steps and args.max_steps > 0 else args.num_epochs * steps_per_epoch
        configure_wired_limit()
        opt = optim.AdamW(learning_rate=_warmup_cosine(args.learning_rate, args.warmup_ratio, total),
                          weight_decay=args.weight_decay)
        model.train()
        rng = random.Random(args.seed)
        seg = model.task == "segmentation"

        def loss_fn(m, x, y):
            out = m(x)                                    # (B, H, W, C)
            if seg:
                B, H, W, C = out.shape
                ce = nn.losses.cross_entropy(out.reshape(-1, C), y.reshape(-1), reduction="mean")
                return ce
            return mx.mean((out - y) ** 2)

        lag = nn.value_and_grad(model, loss_fn)

        def stream():
            idx = list(range(n)); produced = 0
            while produced < total:
                if args.shuffle:
                    rng.shuffle(idx)
                for b in range(0, n, args.batch_size):
                    if produced >= total:
                        return
                    sl = idx[b:b + args.batch_size]
                    yield [(self.train_images[i], self.train_targets[i]) for i in sl]
                    produced += 1

        self.history = []
        for step, (xb, yb) in enumerate(_threaded_prefetch(stream(), self._collate, depth=2)):
            loss, grads = lag(model, mx.array(xb), mx.array(yb))
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            self.history.append(float(loss))
            if step % args.log_every == 0:
                print(f"step {step}/{total}  loss {float(loss):.4f}")
        return self.history

    def evaluate(self, images=None, targets=None, batch_size: int = 16) -> dict:
        images = images if images is not None else self.eval_images
        targets = targets if targets is not None else self.eval_targets
        if images is None or targets is None:
            raise ValueError("No eval data provided")
        preds = self.model.predict(images, batch_size=batch_size)
        t = np.asarray(targets)
        if self.model.task == "segmentation":
            return {"pixel_acc": float((preds == t).mean())}
        p = np.asarray(preds, np.float32)
        if t.ndim == p.ndim + 1 and t.shape[-1] == 1:
            t = t[..., 0]
        return {"mae": float(np.mean(np.abs(p - t.astype(np.float32))))}


# ──────────────────────────────────────────────────────────────────────────────
# Downstream evaluation: frozen-feature linear probe
# ──────────────────────────────────────────────────────────────────────────────

def linear_probe(
    model: JEPAModelWrapper,
    train_images: Sequence[Any],
    train_labels: Sequence[int],
    test_images: Sequence[Any],
    test_labels: Sequence[int],
    num_classes: Optional[int] = None,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    encode_batch_size: int = 64,
) -> float:
    """Train a logistic-regression probe on frozen features; return test accuracy."""
    Xtr = model.encode(train_images, batch_size=encode_batch_size)
    Xte = model.encode(test_images, batch_size=encode_batch_size)
    mu = mx.mean(Xtr, axis=0)
    sd = mx.std(Xtr, axis=0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    ytr = mx.array(np.asarray(train_labels, dtype=np.int32))
    yte = np.asarray(test_labels)
    if num_classes is None:
        num_classes = int(max(int(np.max(train_labels)), int(np.max(test_labels))) + 1)

    clf = nn.Linear(Xtr.shape[1], num_classes)
    opt = optim.Adam(learning_rate=lr)

    def lf(m, x, y):
        return mx.mean(nn.losses.cross_entropy(m(x), y))

    loss_and_grad = nn.value_and_grad(clf, lf)
    N = Xtr.shape[0]
    for _ in range(epochs):
        perm = np.random.permutation(N)
        for i in range(0, N, batch_size):
            bi = mx.array(perm[i:i + batch_size])
            loss, grads = loss_and_grad(clf, Xtr[bi], ytr[bi])
            opt.update(clf, grads)
            mx.eval(clf.parameters(), opt.state)
    preds = np.array(mx.argmax(clf(Xte), axis=1))
    return float((preds == yte).mean())


def knn_probe(
    model: JEPAModelWrapper,
    train_images: Sequence[Any],
    train_labels: Sequence[int],
    test_images: Sequence[Any],
    test_labels: Sequence[int],
    k: int = 20,
    encode_batch_size: int = 64,
) -> float:
    """Weighted k-NN classifier on L2-normalised frozen features (cosine similarity).

    A standard, label-light SSL probe (DINO-style): no training, just nearest
    neighbours in feature space. Returns test accuracy.
    """
    Xtr = np.array(model.encode(train_images, batch_size=encode_batch_size))
    Xte = np.array(model.encode(test_images, batch_size=encode_batch_size))
    Xtr /= (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-8)
    Xte /= (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
    ytr = np.asarray(train_labels)
    yte = np.asarray(test_labels)
    num_classes = int(max(int(ytr.max()), int(yte.max())) + 1)
    k = min(k, Xtr.shape[0])

    sims = Xte @ Xtr.T                                  # (Nte, Ntr) cosine sims
    nn_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    preds = np.empty(Xte.shape[0], dtype=np.int64)
    for i in range(Xte.shape[0]):
        idx = nn_idx[i]
        w = sims[i, idx]                                # similarity weights
        votes = np.zeros(num_classes)
        for j, lbl in zip(idx, ytr[idx]):
            votes[lbl] += sims[i, j]
        preds[i] = int(votes.argmax())
    return float((preds == yte).mean())


class _AttentivePoolHead(nn.Module):
    """Single learned query attends over token features, then a linear classifier.

    Mirrors the attentive-pooling readout used in the I-JEPA / V-JEPA 2 papers
    (and V-JEPA 2's ``VJEPA2AttentivePooler`` classification head): far stronger
    than mean-pool + linear for these encoders, which don't pool cleanly.
    """

    def __init__(self, dim: int, num_classes: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.query = mx.random.normal((1, 1, dim)) * 0.02
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

    def __call__(self, tokens: mx.array) -> mx.array:
        B, T, D = tokens.shape
        H, hd = self.heads, self.head_dim
        q = mx.broadcast_to(self.query, (B, 1, D)).reshape(B, 1, H, hd).transpose(0, 2, 1, 3)
        k = self.k_proj(tokens).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        v = self.v_proj(tokens).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        pooled = o.transpose(0, 2, 1, 3).reshape(B, D)   # (B, D)
        return self.classifier(self.ln(pooled))


def attentive_probe(
    model: JEPAModelWrapper,
    train_images: Sequence[Any],
    train_labels: Sequence[int],
    test_images: Sequence[Any],
    test_labels: Sequence[int],
    num_classes: Optional[int] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    heads: int = 8,
    batch_size: int = 128,
    encode_batch_size: int = 64,
) -> float:
    """Train an attention-pooling head on frozen *token* features; return accuracy.

    This is the canonical evaluation for I-JEPA / V-JEPA 2 encoders, which a
    plain mean-pool linear probe under-reports. The frozen encoder is run once
    to extract token features; only the small pooling head trains.
    """
    Xtr = model.encode_tokens(train_images, batch_size=encode_batch_size)   # (N, T, D)
    Xte = model.encode_tokens(test_images, batch_size=encode_batch_size)
    ytr = mx.array(np.asarray(train_labels, dtype=np.int32))
    yte = np.asarray(test_labels)
    if num_classes is None:
        num_classes = int(max(int(np.max(train_labels)), int(np.max(test_labels))) + 1)

    head = _AttentivePoolHead(Xtr.shape[2], num_classes, heads=heads)
    mx.eval(head.parameters())
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def lf(m, x, y):
        return mx.mean(nn.losses.cross_entropy(m(x), y))

    loss_and_grad = nn.value_and_grad(head, lf)
    N = Xtr.shape[0]
    for _ in range(epochs):
        perm = np.random.permutation(N)
        for i in range(0, N, batch_size):
            bi = mx.array(perm[i:i + batch_size])
            _, grads = loss_and_grad(head, Xtr[bi], ytr[bi])
            opt.update(head, grads)
            mx.eval(head.parameters(), opt.state)
    preds = []
    for i in range(0, Xte.shape[0], batch_size):
        preds.append(np.array(mx.argmax(head(Xte[i:i + batch_size]), axis=1)))
    return float((np.concatenate(preds) == yte).mean())
