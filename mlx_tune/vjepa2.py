"""
V-JEPA 2 (Meta) video encoder for MLX-Tune — Phase 3 of the JEPA roadmap.

A native MLX port of Meta's **V-JEPA 2** video Joint-Embedding Predictive
Architecture (Bardes et al., 2025) for feature extraction and downstream
fine-tuning on Apple Silicon. Both halves of the model are ported:

* the **encoder** (representations — what you use for probing / fine-tuning);
* the **predictor** (masked latent prediction — the "world model" half): give it
  the encoder tokens of a clip plus context/target position ids and it predicts
  the latents of the target tokens. See ``VJEPA2ModelWrapper.predict_latents``
  and :func:`latent_energy` for anticipation / surprise scoring.

Meta's fine-tuned video classification checkpoints (attentive pooler + linear
head, e.g. ``facebook/vjepa2-vitl-fpc16-256-ssv2``) also load directly via
``FastVideoJEPAModel.from_pretrained`` → :class:`VJEPA2PretrainedVideoClassifier`.

Architecture (from HF ``transformers`` VJEPA2):
  * **3D tubelet patch embed** — a ``Conv3d`` over (tubelet_size, patch, patch).
  * **3D RoPE attention** — interleaved rotary embeddings split across the
    depth/height/width axes of each head (``d=h=w=2*((head_dim//3)//2)``); the
    remaining head dims are left un-rotated. Position ids are decomposed from the
    flat token index using the *config* spatial grid (``crop_size // patch_size``),
    so **inputs must be ``crop_size × crop_size``** (frame count may vary).

Verified numerically against the HF PyTorch model (see tests/test_vjepa2.py).

    from mlx_tune import FastVideoJEPAModel
    model, _ = FastVideoJEPAModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    feats = model.encode(videos)        # videos: list of (T, 256, 256, 3) uint8/float

See ``jepa.md`` at the repo root for the full roadmap and design notes.
"""

from __future__ import annotations

import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from ._perf import configure_wired_limit
from .jepa import _IMAGENET_MEAN, _IMAGENET_STD, _threaded_prefetch

_VJEPA2_PRESETS = {
    # name: (hidden, depth, heads) — for reference / validation
    "vitl": (1024, 24, 16),
    "vitg": (1408, 40, 16),
    "vith": (1280, 32, 16),
}


# ──────────────────────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────────────────────

def _rotate_queries_or_keys(x: mx.array, pos: mx.array, theta: float) -> mx.array:
    """Faithful MLX port of HF ``VJEPA2.rotate_queries_or_keys``.

    ``x`` is ``(B, H, N, sub)`` and ``pos`` is the position id per token —
    either 1-D ``(N,)`` (encoder: positions follow the token order) or 2-D
    ``(B, N)`` (predictor: positions come from the context/target masks).
    HF builds the angle table over ``sub//2`` frequencies, then **tiles** cos/sin
    along the last axis (``repeat(...,2)`` → block layout ``[θ_0..θ_{h-1}, θ_0..θ_{h-1}]``)
    while the rotated companion ``y`` uses **adjacent** pairs (``[-x_1, x_0, -x_3, x_2, ...]``).
    We replicate that exact (non-standard) combination so features match the
    pretrained weights bit-for-bit.
    """
    B, H, N, sub = x.shape
    half = sub // 2
    omega = mx.arange(half).astype(mx.float32) / (sub / 2.0)   # (half,)
    inv_freq = 1.0 / (theta ** omega)                          # (half,)
    if pos.ndim == 1:
        freq = pos.astype(mx.float32)[:, None] * inv_freq[None, :]   # (N, half)
        freq = freq[None, None]                                       # (1,1,N,half)
    else:  # (B, N) per-sample ids — broadcast over heads
        freq = pos.astype(mx.float32)[:, None, :, None] * inv_freq[None, None, None, :]
    emb_cos = mx.concatenate([mx.cos(freq), mx.cos(freq)], axis=-1)  # (...,N,sub)
    emb_sin = mx.concatenate([mx.sin(freq), mx.sin(freq)], axis=-1)
    xp = x.reshape(B, H, N, half, 2)
    y = mx.stack([-xp[..., 1], xp[..., 0]], axis=-1).reshape(B, H, N, sub)
    return x * emb_cos + y * emb_sin


class _VJEPA2RopeAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, grid_size: int, theta: float, qkv_bias: bool):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.scale = self.head_dim ** -0.5
        self.grid_size = grid_size
        self.theta = theta
        self.rot = int(2 * ((self.head_dim // 3) // 2))       # d = h = w
        self.query = nn.Linear(hidden, hidden, bias=qkv_bias)
        self.key = nn.Linear(hidden, hidden, bias=qkv_bias)
        self.value = nn.Linear(hidden, hidden, bias=qkv_bias)
        self.proj = nn.Linear(hidden, hidden)

    def _decompose_ids(self, ids: mx.array):
        """Split flat token ids into (frame, height, width) grid coordinates."""
        tpf = self.grid_size * self.grid_size
        frame = ids // tpf
        rem = ids - tpf * frame
        height = rem // self.grid_size
        width = rem - self.grid_size * height
        return frame, height, width

    def _position_ids(self, n: int):
        return self._decompose_ids(mx.arange(n))

    def _apply_rope(self, qk: mx.array, pos_ids) -> mx.array:
        d = self.rot
        frame, height, width = pos_ids
        qd = _rotate_queries_or_keys(qk[..., :d], frame, self.theta)
        qh = _rotate_queries_or_keys(qk[..., d:2 * d], height, self.theta)
        qw = _rotate_queries_or_keys(qk[..., 2 * d:3 * d], width, self.theta)
        if qk.shape[-1] > 3 * d:
            return mx.concatenate([qd, qh, qw, qk[..., 3 * d:]], axis=-1)
        return mx.concatenate([qd, qh, qw], axis=-1)

    def __call__(self, x: mx.array, position_ids: Optional[mx.array] = None) -> mx.array:
        B, N, D = x.shape
        H, hd = self.heads, self.head_dim
        q = self.query(x).reshape(B, N, H, hd).transpose(0, 2, 1, 3)
        k = self.key(x).reshape(B, N, H, hd).transpose(0, 2, 1, 3)
        v = self.value(x).reshape(B, N, H, hd).transpose(0, 2, 1, 3)
        pos = self._decompose_ids(position_ids) if position_ids is not None \
            else self._position_ids(N)
        q = self._apply_rope(q, pos)
        k = self._apply_rope(k, pos)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        o = o.transpose(0, 2, 1, 3).reshape(B, N, D)
        return self.proj(o)


class _VJEPA2MLP(nn.Module):
    def __init__(self, hidden: int, mlp_ratio: float):
        super().__init__()
        inner = int(hidden * mlp_ratio)
        self.fc1 = nn.Linear(hidden, inner)
        self.fc2 = nn.Linear(inner, hidden)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class _VJEPA2Layer(nn.Module):
    def __init__(self, hidden, heads, grid_size, theta, mlp_ratio, eps, qkv_bias):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, eps=eps)
        self.attention = _VJEPA2RopeAttention(hidden, heads, grid_size, theta, qkv_bias)
        self.norm2 = nn.LayerNorm(hidden, eps=eps)
        self.mlp = _VJEPA2MLP(hidden, mlp_ratio)

    def __call__(self, x: mx.array, position_ids: Optional[mx.array] = None) -> mx.array:
        x = x + self.attention(self.norm1(x), position_ids=position_ids)
        x = x + self.mlp(self.norm2(x))
        return x


class VideoViTEncoder(nn.Module):
    """V-JEPA 2 video ViT encoder. Input ``(B, T, H, W, 3)`` (NDHWC, MLX);
    output token sequence ``(B, N, hidden)``."""

    def __init__(self, hidden: int, depth: int, heads: int, patch_size: int,
                 tubelet_size: int, crop_size: int, mlp_ratio: float = 4.0,
                 eps: float = 1e-6, qkv_bias: bool = True, theta: float = 10000.0,
                 in_chans: int = 3):
        super().__init__()
        self.hidden = hidden
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.crop_size = crop_size
        self.grid_size = crop_size // patch_size
        self.patch_embed = nn.Conv3d(
            in_chans, hidden,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )
        self.layers = [
            _VJEPA2Layer(hidden, heads, self.grid_size, theta, mlp_ratio, eps, qkv_bias)
            for _ in range(depth)
        ]
        self.norm = nn.LayerNorm(hidden, eps=eps)

    def __call__(self, video: mx.array) -> mx.array:
        x = self.patch_embed(video)                  # (B, T', H', W', hidden)
        B = x.shape[0]
        x = x.reshape(B, -1, self.hidden)            # (B, N, hidden)  order T',H',W'
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def pooled(self, video: mx.array) -> mx.array:
        return mx.mean(self.__call__(video), axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# Predictor (masked latent prediction — the "world model" half)
# ──────────────────────────────────────────────────────────────────────────────

class _VJEPA2PredictorEmbeddings(nn.Module):
    """Context projection + learned mask tokens (HF attribute names)."""

    def __init__(self, hidden: int, pred_hidden: int, num_mask_tokens: int):
        super().__init__()
        self.predictor_embeddings = nn.Linear(hidden, pred_hidden)
        self.mask_tokens = mx.zeros((num_mask_tokens, 1, 1, pred_hidden))


class _VJEPA2Predictor(nn.Module):
    """Faithful MLX port of HF ``VJEPA2Predictor`` (masked latent prediction).

    Takes the encoder's token sequence, the flat position ids of the *context*
    tokens the predictor may see, and the position ids of the *target* tokens to
    predict; returns predicted latents for the target positions (in encoder
    hidden size). Internals mirror HF exactly: context tokens are projected to
    ``pred_hidden``, target slots use ``mask_tokens[1]`` (HF's default
    ``mask_index=1``), the combined sequence is **sorted by position id** before
    the transformer and unsorted after, and the blocks share the encoder's
    (non-standard) 3-axis RoPE driven by the explicit position ids. Attribute
    names match HF so checkpoint weights load 1:1.
    """

    def __init__(self, hidden: int, pred_hidden: int, depth: int, heads: int,
                 grid_size: int, theta: float, mlp_ratio: float, eps: float,
                 qkv_bias: bool, num_mask_tokens: int):
        super().__init__()
        self.embeddings = _VJEPA2PredictorEmbeddings(hidden, pred_hidden, num_mask_tokens)
        self.layer = [
            _VJEPA2Layer(pred_hidden, heads, grid_size, theta, mlp_ratio, eps, qkv_bias)
            for _ in range(depth)
        ]
        self.layernorm = nn.LayerNorm(pred_hidden, eps=eps)
        self.proj = nn.Linear(pred_hidden, hidden)

    def __call__(self, encoder_tokens: mx.array, context_ids: mx.array,
                 target_ids: mx.array, mask_index: int = 1) -> mx.array:
        B = encoder_tokens.shape[0]
        if context_ids.ndim == 1:
            context_ids = mx.broadcast_to(context_ids[None], (B, context_ids.shape[0]))
        if target_ids.ndim == 1:
            target_ids = mx.broadcast_to(target_ids[None], (B, target_ids.shape[0]))
        n_ctx = context_ids.shape[1]
        ctx_tok = mx.take_along_axis(encoder_tokens, context_ids[..., None], axis=1)
        ctx = self.embeddings.predictor_embeddings(ctx_tok)              # (B, Nc, P)
        m = self.embeddings.mask_tokens[mask_index % self.embeddings.mask_tokens.shape[0]]
        tgt = mx.broadcast_to(m, (B, target_ids.shape[1], m.shape[-1]))  # (B, Nt, P)
        x = mx.concatenate([ctx, tgt], axis=1)
        pos = mx.concatenate([context_ids, target_ids], axis=1)          # (B, Nc+Nt)
        order = mx.argsort(pos, axis=1)
        x = mx.take_along_axis(x, order[..., None], axis=1)
        pos_sorted = mx.take_along_axis(pos, order, axis=1)
        for blk in self.layer:
            x = blk(x, position_ids=pos_sorted)
        x = self.layernorm(x)
        inv = mx.argsort(order, axis=1)
        x = mx.take_along_axis(x, inv[..., None], axis=1)
        return self.proj(x[:, n_ctx:])


def latent_energy(pred: mx.array, target: mx.array, kind: str = "l2",
                  per_token: bool = False):
    """Surprise score between predicted and observed latents (lower = expected).

    ``kind="l2"`` → per-token mean squared distance; ``kind="cosine"`` →
    per-token ``1 − cosine``. Returns the scalar mean, or the per-token map
    when ``per_token=True`` (useful for localising *where* a clip surprised
    the model).
    """
    if kind == "l2":
        e = mx.mean((pred - target) ** 2, axis=-1)
    elif kind == "cosine":
        num = mx.sum(pred * target, axis=-1)
        den = mx.linalg.norm(pred, axis=-1) * mx.linalg.norm(target, axis=-1) + 1e-8
        e = 1.0 - num / den
    else:
        raise ValueError(f"kind must be 'l2' or 'cosine'; got {kind!r}")
    return e if per_token else mx.mean(e)


# ──────────────────────────────────────────────────────────────────────────────
# Attentive pooler (Meta's fine-tuned classification checkpoints)
# ──────────────────────────────────────────────────────────────────────────────

class _VJEPA2PoolerAttention(nn.Module):
    """Plain multi-head attention used by the attentive pooler (no RoPE).

    ``out_proj=False`` replicates HF's ``VJEPA2PoolerCrossAttention``, which has
    no output projection.
    """

    def __init__(self, hidden: int, heads: int, out_proj: bool = True):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        if out_proj:
            self.out_proj = nn.Linear(hidden, hidden)

    def __call__(self, q_in: mx.array, kv_in: mx.array) -> mx.array:
        B, Nq, D = q_in.shape
        Nk = kv_in.shape[1]
        H, hd = self.heads, self.head_dim
        q = self.q_proj(q_in).reshape(B, Nq, H, hd).transpose(0, 2, 1, 3)
        k = self.k_proj(kv_in).reshape(B, Nk, H, hd).transpose(0, 2, 1, 3)
        v = self.v_proj(kv_in).reshape(B, Nk, H, hd).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        o = o.transpose(0, 2, 1, 3).reshape(B, Nq, D)
        return self.out_proj(o) if "out_proj" in self else o


class _VJEPA2PoolerSelfAttentionLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden, eps=eps)
        self.self_attn = _VJEPA2PoolerAttention(hidden, heads, out_proj=True)
        self.layer_norm2 = nn.LayerNorm(hidden, eps=eps)
        self.mlp = _VJEPA2MLP(hidden, mlp_ratio)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.layer_norm1(x)
        x = x + self.self_attn(h, h)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class _VJEPA2PoolerCrossAttentionLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden, eps=eps)
        self.cross_attn = _VJEPA2PoolerAttention(hidden, heads, out_proj=False)
        self.layer_norm2 = nn.LayerNorm(hidden, eps=eps)
        self.mlp = _VJEPA2MLP(hidden, mlp_ratio)

    def __call__(self, queries: mx.array, tokens: mx.array) -> mx.array:
        # HF quirk: layer_norm1 is applied to the keys/values only — the queries
        # enter the attention (and the residual) un-normalised.
        kv = self.layer_norm1(tokens)
        x = queries + self.cross_attn(queries, kv)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class _VJEPA2AttentivePooler(nn.Module):
    """Faithful MLX port of HF ``VJEPA2AttentivePooler`` (HF attribute names)."""

    def __init__(self, hidden: int, heads: int, num_layers: int,
                 mlp_ratio: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.query_tokens = mx.zeros((1, 1, hidden))
        self.cross_attention_layer = _VJEPA2PoolerCrossAttentionLayer(hidden, heads, mlp_ratio, eps)
        self.self_attention_layers = [
            _VJEPA2PoolerSelfAttentionLayer(hidden, heads, mlp_ratio, eps)
            for _ in range(num_layers)
        ]

    def __call__(self, tokens: mx.array) -> mx.array:
        for lyr in self.self_attention_layers:
            tokens = lyr(tokens)
        q = mx.broadcast_to(self.query_tokens, (tokens.shape[0], 1, tokens.shape[-1]))
        return self.cross_attention_layer(q, tokens)[:, 0]


# ──────────────────────────────────────────────────────────────────────────────
# HF → MLX weight conversion
# ──────────────────────────────────────────────────────────────────────────────

def _convert_vjepa2_weights(hf: dict, depth: int, include_predictor: bool = False) -> dict:
    out = {}
    # Conv3d: HF (out,in,kT,kH,kW) → MLX (out,kT,kH,kW,in)
    out["encoder.patch_embed.weight"] = mx.transpose(
        hf["encoder.embeddings.patch_embeddings.proj.weight"], (0, 2, 3, 4, 1)
    )
    out["encoder.patch_embed.bias"] = hf["encoder.embeddings.patch_embeddings.proj.bias"]
    for i in range(depth):
        h = f"encoder.layer.{i}."
        m = f"encoder.layers.{i}."
        out[m + "norm1.weight"] = hf[h + "norm1.weight"]
        out[m + "norm1.bias"] = hf[h + "norm1.bias"]
        out[m + "norm2.weight"] = hf[h + "norm2.weight"]
        out[m + "norm2.bias"] = hf[h + "norm2.bias"]
        for proj in ("query", "key", "value", "proj"):
            out[m + f"attention.{proj}.weight"] = hf[h + f"attention.{proj}.weight"]
            out[m + f"attention.{proj}.bias"] = hf[h + f"attention.{proj}.bias"]
        out[m + "mlp.fc1.weight"] = hf[h + "mlp.fc1.weight"]
        out[m + "mlp.fc1.bias"] = hf[h + "mlp.fc1.bias"]
        out[m + "mlp.fc2.weight"] = hf[h + "mlp.fc2.weight"]
        out[m + "mlp.fc2.bias"] = hf[h + "mlp.fc2.bias"]
    out["encoder.norm.weight"] = hf["encoder.layernorm.weight"]
    out["encoder.norm.bias"] = hf["encoder.layernorm.bias"]
    if include_predictor:
        # The predictor is all LayerNorm/Linear + a mask-token parameter, and
        # our module mirrors HF's attribute names — keys pass through 1:1.
        out.update({k: v for k, v in hf.items() if k.startswith("predictor.")})
    return out


def _vjepa2_config_dict(cfg: dict) -> dict:
    out = {
        "arch": "vjepa2",
        "hidden": int(cfg["hidden_size"]),
        "depth": int(cfg["num_hidden_layers"]),
        "heads": int(cfg["num_attention_heads"]),
        "patch_size": int(cfg["patch_size"]),
        "tubelet_size": int(cfg["tubelet_size"]),
        "crop_size": int(cfg["crop_size"]),
        "mlp_ratio": float(cfg.get("mlp_ratio", 4.0)),
        "eps": float(cfg.get("layer_norm_eps", 1e-6)),
        "qkv_bias": bool(cfg.get("qkv_bias", True)),
        "theta": float(cfg.get("rope_theta", 10000.0)),
        "in_chans": int(cfg.get("in_chans", 3)),
    }
    if "pred_hidden_size" in cfg:
        out.update({
            "pred_hidden": int(cfg["pred_hidden_size"]),
            "pred_depth": int(cfg["pred_num_hidden_layers"]),
            "pred_heads": int(cfg["pred_num_attention_heads"]),
            "pred_mlp_ratio": float(cfg.get("pred_mlp_ratio", 4.0)),
            "pred_num_mask_tokens": int(cfg.get("pred_num_mask_tokens", 10)),
        })
    return out


def _encoder_from_config(config: dict) -> VideoViTEncoder:
    return VideoViTEncoder(
        hidden=config["hidden"], depth=config["depth"], heads=config["heads"],
        patch_size=config["patch_size"], tubelet_size=config["tubelet_size"],
        crop_size=config["crop_size"], mlp_ratio=config["mlp_ratio"],
        eps=config["eps"], qkv_bias=config["qkv_bias"], theta=config["theta"],
        in_chans=config["in_chans"],
    )


def _predictor_from_config(config: dict) -> Optional[_VJEPA2Predictor]:
    if "pred_hidden" not in config:
        return None
    return _VJEPA2Predictor(
        hidden=config["hidden"], pred_hidden=config["pred_hidden"],
        depth=config["pred_depth"], heads=config["pred_heads"],
        grid_size=config["crop_size"] // config["patch_size"],
        theta=config["theta"], mlp_ratio=config["pred_mlp_ratio"],
        eps=config["eps"], qkv_bias=config["qkv_bias"],
        num_mask_tokens=config["pred_num_mask_tokens"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Video preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def _prep_video(video: Any, crop_size: int, shortest_edge: Optional[int] = None,
                num_frames: Optional[int] = None) -> np.ndarray:
    """Normalise one clip to ``(T, crop, crop, 3)`` float32 (ImageNet stats).

    Accepts ``(T, H, W, 3)`` arrays (uint8 or float). By default frames are
    squash-resized to ``crop × crop`` and ``T`` is left as-is (must be a
    multiple of tubelet_size). With ``shortest_edge`` set, frames are instead
    resized so the short side equals it and then center-cropped — matching the
    HF ``VJEPA2VideoProcessor`` used by Meta's classification checkpoints.
    ``num_frames`` uniformly resamples the clip to that frame count.
    """
    from PIL import Image

    arr = np.asarray(video)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected video of shape (T, H, W, 3); got {arr.shape}")
    if num_frames is not None and arr.shape[0] != num_frames:
        idx = np.linspace(0, arr.shape[0] - 1, num_frames).round().astype(int)
        arr = arr[idx]
    frames = []
    for f in arr:
        if f.dtype != np.uint8:
            f = np.clip(f * (255.0 if float(np.nanmax(f, initial=0.0)) <= 1.0 else 1.0),
                        0, 255).astype(np.uint8)
        pil = Image.fromarray(f).convert("RGB")
        if shortest_edge is not None:
            w, h = pil.size
            s = shortest_edge / min(w, h)
            pil = pil.resize((max(crop_size, round(w * s)), max(crop_size, round(h * s))),
                             Image.BILINEAR)
            w, h = pil.size
            x, y = (w - crop_size) // 2, (h - crop_size) // 2
            pil = pil.crop((x, y, x + crop_size, y + crop_size))
        else:
            pil = pil.resize((crop_size, crop_size), Image.BILINEAR)
        a = np.asarray(pil, dtype=np.float32) / 255.0
        frames.append((a - _IMAGENET_MEAN) / _IMAGENET_STD)
    return np.stack(frames).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper + loader
# ──────────────────────────────────────────────────────────────────────────────

class VJEPA2ModelWrapper(nn.Module):
    """Holds a :class:`VideoViTEncoder` (+ optional predictor); provides
    ``encode``, ``predict_latents`` and save/load."""

    def __init__(self, encoder: VideoViTEncoder, config: dict,
                 predictor: Optional[_VJEPA2Predictor] = None):
        super().__init__()
        self.encoder = encoder
        if predictor is not None:
            self.predictor = predictor
        self.config = dict(config)
        self.crop_size = encoder.crop_size
        self.tubelet_size = encoder.tubelet_size
        self.embed_dim = encoder.hidden

    @property
    def has_predictor(self) -> bool:
        return "predictor" in self

    def __call__(self, video: mx.array) -> mx.array:
        return self.encoder(video)

    def predict_latents(
        self,
        video: Any,
        context_frames: Optional[int] = None,
        context_ids: Optional[Any] = None,
        target_ids: Optional[Any] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Masked latent prediction: predict target-token latents from context.

        Either pass ``context_frames=k`` (tokens of the first ``k`` frames are
        the context; all later tokens are the prediction targets — i.e.
        "anticipate the rest of the clip"), or explicit flat token-id arrays
        ``context_ids`` / ``target_ids`` (1-D, or ``(B, N)`` per-sample).

        Returns ``(predicted, target)`` — both ``(1, N_tgt, hidden)``: the
        predictor's latents for the target positions and the encoder's actual
        latents there. Feed them to :func:`latent_energy` for a surprise score.
        """
        if not self.has_predictor:
            raise ValueError(
                "This model has no predictor. Reload with "
                "FastVideoJEPAModel.from_pretrained(..., load_predictor=True)."
            )
        was = self.training
        self.eval()
        arr = _prep_video(video, self.crop_size)[None].astype(np.float32)
        tokens = self.encoder(mx.array(arr))                 # (1, N, hidden)
        n_tokens = tokens.shape[1]
        if context_ids is None or target_ids is None:
            if context_frames is None:
                raise ValueError("Pass context_frames or context_ids/target_ids")
            if context_frames % self.tubelet_size != 0:
                raise ValueError(
                    f"context_frames must be a multiple of tubelet_size={self.tubelet_size}")
            g = self.encoder.grid_size
            n_ctx = (context_frames // self.tubelet_size) * g * g
            if not 0 < n_ctx < n_tokens:
                raise ValueError(
                    f"context_frames={context_frames} leaves no target tokens "
                    f"(clip has {n_tokens} tokens)")
            context_ids = mx.arange(n_ctx)
            target_ids = mx.arange(n_ctx, n_tokens)
        else:
            context_ids = mx.array(np.asarray(context_ids))
            target_ids = mx.array(np.asarray(target_ids))
        pred = self.predictor(tokens, context_ids, target_ids)
        tids = target_ids if target_ids.ndim == 2 else target_ids[None]
        target = mx.take_along_axis(tokens, tids[..., None], axis=1)
        mx.eval(pred, target)
        if was:
            self.train()
        return pred, target

    def encode(self, videos: Sequence[Any], batch_size: int = 1) -> mx.array:
        """Mean-pooled clip features ``(num_videos, hidden)`` (frozen encoder)."""
        was = self.training
        self.eval()
        out: List[np.ndarray] = []
        for i in range(0, len(videos), batch_size):
            chunk = [videos[j] for j in range(i, min(i + batch_size, len(videos)))]
            arr = np.stack([_prep_video(v, self.crop_size) for v in chunk]).astype(np.float32)
            z = self.encoder.pooled(mx.array(arr))
            mx.eval(z)
            out.append(np.array(z))
        if was:
            self.train()
        return mx.array(np.concatenate(out, axis=0))

    def encode_tokens(self, videos: Sequence[Any], batch_size: int = 1) -> List[np.ndarray]:
        """Frozen *token-level* clip features (frozen encoder) for attentive probing.

        Returns a list of per-clip token arrays ``(T_tokens, hidden)`` (clips may
        differ in frame count, so token counts can differ — kept as a list).
        """
        was = self.training
        self.eval()
        out: List[np.ndarray] = []
        for v in videos:
            arr = _prep_video(v, self.crop_size)[None].astype(np.float32)
            z = self.encoder(mx.array(arr))[0]           # (T_tokens, hidden)
            mx.eval(z)
            out.append(np.array(z))
        if was:
            self.train()
        return out

    def save_pretrained(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        weights = dict(tree_flatten(self.parameters()))
        mx.save_safetensors(str(p / "model.safetensors"), weights)
        (p / "vjepa2_config.json").write_text(json.dumps(self.config, indent=2))


def _build_vjepa2(config: dict, hf_weights: dict) -> VJEPA2ModelWrapper:
    encoder = _encoder_from_config(config)
    has_pred = "pred_hidden" in config and any(
        k.startswith("predictor.") for k in hf_weights)
    predictor = _predictor_from_config(config) if has_pred else None
    if not has_pred:
        config = {k: v for k, v in config.items() if not k.startswith("pred_")}
    model = VJEPA2ModelWrapper(encoder, config, predictor=predictor)
    converted = _convert_vjepa2_weights(hf_weights, config["depth"],
                                        include_predictor=has_pred)
    model.load_weights(list(converted.items()))
    mx.eval(model.parameters())
    return model


class FastVideoJEPAModel:
    """Unsloth-style entry point for V-JEPA 2 video models.

    ``from_pretrained`` accepts a saved directory or a Hugging Face
    ``facebook/vjepa2-*`` repo. Returns ``(model, None)``.
    """

    @staticmethod
    def for_video_classification(
        model: VJEPA2ModelWrapper,
        num_classes: int,
        finetune: str = "lora",
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ) -> VJEPA2ForVideoClassification:
        """Attach a video-classification head and set the encoder trainable surface.

        ``finetune``: ``"frozen"`` (head only), ``"lora"`` (default), or ``"full"``.
        """
        clf = VJEPA2ForVideoClassification(model.encoder, num_classes)
        clf.encoder._vjepa2_config = dict(model.config)
        clf.encoder._vjepa2_config.update(
            {"finetune": finetune, "lora_r": r, "lora_alpha": lora_alpha,
             "lora_dropout": lora_dropout, "lora_targets": target_modules}
        )
        if finetune == "full":
            clf.unfreeze()
        elif finetune == "frozen":
            clf.encoder.freeze()
            clf.head.unfreeze()
        elif finetune == "lora":
            # Freeze the base FIRST, then wrap with LoRA so only the fresh
            # adapters + head are trainable (the base stays frozen).
            clf.encoder.freeze()
            n = apply_lora_to_vjepa2_encoder(
                clf.encoder, r=r, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout, target_modules=target_modules,
            )
            if n == 0:
                raise ValueError("No LoRA targets matched the V-JEPA 2 blocks")
            clf.head.unfreeze()
        else:
            raise ValueError(f"finetune must be 'frozen', 'lora', or 'full'; got {finetune!r}")
        mx.eval(clf.parameters())
        return clf

    @staticmethod
    def load_classifier(path: Union[str, Path]) -> VJEPA2ForVideoClassification:
        """Reload a video classifier saved by ``VJEPA2ForVideoClassification.save_pretrained``."""
        p = Path(path)
        cfg = json.loads((p / "vjepa2_classifier_config.json").read_text())
        encoder = _encoder_from_config(cfg)
        clf = VJEPA2ForVideoClassification(encoder, int(cfg["num_classes"]))
        if cfg.get("finetune") == "lora" or cfg.get("has_lora"):
            apply_lora_to_vjepa2_encoder(
                clf.encoder, r=int(cfg.get("lora_r", 16)),
                lora_alpha=int(cfg.get("lora_alpha", 16)),
                lora_dropout=float(cfg.get("lora_dropout", 0.0)),
                target_modules=cfg.get("lora_targets"),
            )
        weights = mx.load(str(p / "model.safetensors"))
        clf.load_weights(list(weights.items()))
        mx.eval(clf.parameters())
        return clf

    @staticmethod
    def from_pretrained(model_name: str = "facebook/vjepa2-vitl-fpc64-256",
                        load_predictor: bool = True,
                        **overrides) -> Tuple[Union[VJEPA2ModelWrapper, "VJEPA2PretrainedVideoClassifier"], None]:
        p = Path(model_name)
        if p.exists() and (p / "vjepa2_hf_classifier_config.json").exists():
            return _load_saved_vjepa2_classifier(p), None
        if p.exists() and (p / "vjepa2_config.json").exists():
            config = json.loads((p / "vjepa2_config.json").read_text())
            config.update(overrides)
            weights = mx.load(str(p / "model.safetensors"))
            has_pred = "pred_hidden" in config and any(
                k.startswith("predictor.") for k in weights)
            predictor = _predictor_from_config(config) if has_pred else None
            model = VJEPA2ModelWrapper(_encoder_from_config(config), config,
                                       predictor=predictor)
            model.load_weights(list(weights.items()))
            mx.eval(model.parameters())
            return model, None

        from huggingface_hub import hf_hub_download, snapshot_download

        cfg = json.loads(Path(hf_hub_download(model_name, "config.json")).read_text())
        if cfg.get("model_type") != "vjepa2":
            raise ValueError(
                f"'{model_name}' is not a V-JEPA 2 checkpoint "
                f"(model_type={cfg.get('model_type')!r}). Use a 'facebook/vjepa2-*' repo."
            )
        if "VJEPA2ForVideoClassification" in (cfg.get("architectures") or []):
            # Meta's fine-tuned classification checkpoint (encoder + attentive
            # pooler + linear head) — e.g. facebook/vjepa2-vitl-fpc16-256-ssv2.
            return _load_pretrained_vjepa2_classifier(model_name, cfg), None
        config = _vjepa2_config_dict(cfg)
        if not load_predictor:
            config = {k: v for k, v in config.items() if not k.startswith("pred_")}
        local = snapshot_download(model_name, allow_patterns=["*.safetensors"])
        hf: dict = {}
        for f in sorted(glob.glob(str(Path(local) / "*.safetensors"))):
            loaded = mx.load(f)
            hf.update({k: v for k, v in loaded.items()
                       if k.startswith("encoder.")
                       or (load_predictor and k.startswith("predictor."))})
        return _build_vjepa2(config, hf), None


# ──────────────────────────────────────────────────────────────────────────────
# Meta's pretrained video classification checkpoints (SSv2 / Diving48)
# ──────────────────────────────────────────────────────────────────────────────

class VJEPA2PretrainedVideoClassifier(nn.Module):
    """Meta's fine-tuned V-JEPA 2 video classifier, running natively on MLX.

    Encoder + attentive pooler + linear head, loaded from a HF
    ``VJEPA2ForVideoClassification`` checkpoint (e.g.
    ``facebook/vjepa2-vitl-fpc16-256-ssv2`` — 174 Something-Something-v2
    action classes). Built by ``FastVideoJEPAModel.from_pretrained``; no
    training needed — call :meth:`predict` on raw clips.
    """

    def __init__(self, encoder: VideoViTEncoder, pooler: _VJEPA2AttentivePooler,
                 num_classes: int, id2label: Optional[dict] = None,
                 config: Optional[dict] = None):
        super().__init__()
        self.encoder = encoder
        self.pooler = pooler
        self.classifier = nn.Linear(encoder.hidden, num_classes)
        self.num_classes = num_classes
        self.id2label = {int(k): v for k, v in (id2label or {}).items()}
        self.config = dict(config or {})
        self.crop_size = encoder.crop_size
        self.frames_per_clip = int(self.config.get("frames_per_clip", 16))
        self.shortest_edge = self.config.get("shortest_edge")

    def __call__(self, video: mx.array) -> mx.array:
        return self.classifier(self.pooler(self.encoder(video)))

    def _prep(self, video: Any) -> np.ndarray:
        return _prep_video(video, self.crop_size, shortest_edge=self.shortest_edge,
                           num_frames=self.frames_per_clip)

    def predict(self, videos: Sequence[Any], top_k: int = 1,
                batch_size: int = 1) -> List[List[dict]]:
        """Classify raw clips ``(T, H, W, 3)``.

        Frames are uniformly resampled to ``frames_per_clip`` and preprocessed
        like the HF video processor (shortest-edge resize → center crop →
        ImageNet normalise). Returns, per clip, the ``top_k`` predictions as
        ``{"id", "label", "prob"}`` dicts (label falls back to the id when the
        checkpoint has no label map).
        """
        was = self.training
        self.eval()
        results: List[List[dict]] = []
        for i in range(0, len(videos), batch_size):
            chunk = [videos[j] for j in range(i, min(i + batch_size, len(videos)))]
            arr = np.stack([self._prep(v) for v in chunk]).astype(np.float32)
            probs = mx.softmax(self(mx.array(arr)), axis=-1)
            mx.eval(probs)
            probs = np.array(probs)
            for row in probs:
                top = np.argsort(-row)[:top_k]
                results.append([
                    {"id": int(c), "label": self.id2label.get(int(c), str(int(c))),
                     "prob": float(row[c])}
                    for c in top
                ])
        if was:
            self.train()
        return results

    def save_pretrained(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(p / "model.safetensors"),
                            dict(tree_flatten(self.parameters())))
        cfg = dict(self.config)
        cfg.update({"arch": "vjepa2_classifier", "num_classes": self.num_classes,
                    "id2label": self.id2label})
        (p / "vjepa2_hf_classifier_config.json").write_text(json.dumps(cfg, indent=2))


def _build_vjepa2_classifier(config: dict, num_classes: int,
                             id2label: Optional[dict]) -> VJEPA2PretrainedVideoClassifier:
    encoder = _encoder_from_config(config)
    # The HF pooler always uses mlp_ratio 4.0 (not the encoder's), and the
    # encoder's head count.
    pooler = _VJEPA2AttentivePooler(config["hidden"], config["heads"],
                                    int(config.get("num_pooler_layers", 3)),
                                    mlp_ratio=4.0, eps=config["eps"])
    return VJEPA2PretrainedVideoClassifier(encoder, pooler, num_classes,
                                           id2label, config)


def _convert_vjepa2_classifier_weights(raw: dict, depth: int) -> dict:
    """HF ``VJEPA2ForVideoClassification`` state dict → our parameter names.

    Encoder keys live under ``vjepa2.encoder.*`` (the predictor under
    ``vjepa2.predictor.*`` is unused at inference and skipped); the pooler and
    classifier keys pass through 1:1.
    """
    enc_hf = {k[len("vjepa2."):]: v for k, v in raw.items()
              if k.startswith("vjepa2.encoder.")}
    out = _convert_vjepa2_weights(enc_hf, depth)
    out.update({k: v for k, v in raw.items()
                if k.startswith("pooler.") or k.startswith("classifier.")})
    return out


def _load_pretrained_vjepa2_classifier(repo: str, cfg: dict) -> VJEPA2PretrainedVideoClassifier:
    from huggingface_hub import hf_hub_download, snapshot_download

    config = _vjepa2_config_dict(cfg)
    config = {k: v for k, v in config.items() if not k.startswith("pred_")}
    config["arch"] = "vjepa2_classifier"
    config["frames_per_clip"] = int(cfg.get("frames_per_clip", 16))
    config["num_pooler_layers"] = int(cfg.get("num_pooler_layers", 3))
    try:
        pp = json.loads(Path(
            hf_hub_download(repo, "video_preprocessor_config.json")).read_text())
        config["shortest_edge"] = int(pp["size"]["shortest_edge"])
    except Exception:
        # Same ratio the released checkpoints use (292/256).
        config["shortest_edge"] = round(config["crop_size"] * 292 / 256)

    local = snapshot_download(repo, allow_patterns=["*.safetensors"])
    raw: dict = {}
    for f in sorted(glob.glob(str(Path(local) / "*.safetensors"))):
        raw.update(mx.load(f))
    id2label = cfg.get("id2label") or {}
    num_classes = raw["classifier.weight"].shape[0]
    clf = _build_vjepa2_classifier(config, num_classes, id2label)
    clf.load_weights(list(_convert_vjepa2_classifier_weights(raw, config["depth"]).items()))
    mx.eval(clf.parameters())
    clf.eval()
    return clf


def _load_saved_vjepa2_classifier(p: Path) -> VJEPA2PretrainedVideoClassifier:
    cfg = json.loads((p / "vjepa2_hf_classifier_config.json").read_text())
    clf = _build_vjepa2_classifier(cfg, int(cfg["num_classes"]), cfg.get("id2label"))
    clf.load_weights(str(p / "model.safetensors"))
    mx.eval(clf.parameters())
    clf.eval()
    return clf


# ──────────────────────────────────────────────────────────────────────────────
# LoRA + video classification fine-tuning
# ──────────────────────────────────────────────────────────────────────────────

_VJEPA2_LORA_TARGETS = ["query", "key", "value", "proj", "fc1", "fc2"]


def _unfreeze_lora(module: nn.Module) -> None:
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


def apply_lora_to_vjepa2_encoder(encoder: VideoViTEncoder, r: int = 16, lora_alpha: int = 16,
                                 lora_dropout: float = 0.0,
                                 target_modules: Optional[List[str]] = None) -> int:
    """Swap target linears in every V-JEPA 2 block for LoRA layers (in place)."""
    from mlx_lm.tuner.lora import LoRALinear

    targets = target_modules or _VJEPA2_LORA_TARGETS
    n = 0
    for layer in encoder.layers:
        pairs = [(layer.attention, t) for t in ("query", "key", "value", "proj")]
        pairs += [(layer.mlp, t) for t in ("fc1", "fc2")]
        for parent, name in pairs:
            if name not in targets:
                continue
            lin = getattr(parent, name, None)
            if isinstance(lin, nn.Linear):
                setattr(parent, name,
                        LoRALinear.from_base(lin, r=r, scale=lora_alpha / r, dropout=lora_dropout))
                n += 1
    return n


class VJEPA2ForVideoClassification(nn.Module):
    """V-JEPA 2 encoder + a linear head over mean-pooled clip features."""

    def __init__(self, encoder: VideoViTEncoder, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.hidden, num_classes)
        self.crop_size = encoder.crop_size
        self.num_classes = num_classes

    def __call__(self, video: mx.array) -> mx.array:
        return self.head(self.encoder.pooled(video))

    def predict(self, videos: Sequence[Any], batch_size: int = 2,
                return_probs: bool = False):
        """Predict class ids (or probabilities) for raw clips ``(T, H, W, 3)``."""
        was = self.training
        self.eval()
        out: List[np.ndarray] = []
        for i in range(0, len(videos), batch_size):
            chunk = [videos[j] for j in range(i, min(i + batch_size, len(videos)))]
            arr = np.stack([_prep_video(v, self.crop_size) for v in chunk]).astype(np.float32)
            logits = self(mx.array(arr))
            result = mx.softmax(logits, axis=-1) if return_probs else mx.argmax(logits, axis=-1)
            mx.eval(result)
            out.append(np.array(result))
        if was:
            self.train()
        return np.concatenate(out, axis=0)

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save the full video classifier (encoder + LoRA + head) + config."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        weights = dict(tree_flatten(self.parameters()))
        mx.save_safetensors(str(p / "model.safetensors"), weights)
        cfg = dict(getattr(self.encoder, "_vjepa2_config", {}) or {})
        cfg.update({
            "task": "video_classification",
            "num_classes": self.num_classes,
            "has_lora": any("lora" in k.lower() for k in weights),
        })
        (p / "vjepa2_classifier_config.json").write_text(json.dumps(cfg, indent=2))


@dataclass
class VideoClassifierConfig:
    output_dir: str = "vjepa2_clf_output"
    learning_rate: float = 3e-4   # LoRA-safe on a deep ViT; bump for frozen/full
    weight_decay: float = 0.0
    batch_size: int = 2
    num_epochs: int = 10
    max_steps: int = -1
    warmup_ratio: float = 0.15    # warmup matters: LoRA on a deep ViT diverges without it
    label_smoothing: float = 0.0
    shuffle: bool = True
    log_every: int = 10
    seed: int = 42


class VideoClassifierTrainer:
    """Supervised fine-tuning of a :class:`VJEPA2ForVideoClassification`.

    Cross-entropy with AdamW (warmup→cosine) and background preprocessing
    prefetch. The encoder's freeze state (frozen / LoRA / full) is set when the
    classifier is built via :meth:`FastVideoJEPAModel.for_video_classification`.
    """

    def __init__(self, model: VJEPA2ForVideoClassification, args: VideoClassifierConfig,
                 train_videos: Sequence[Any], train_labels: Sequence[int],
                 eval_videos: Optional[Sequence[Any]] = None,
                 eval_labels: Optional[Sequence[int]] = None):
        self.model = model
        self.args = args
        self.train_videos = train_videos
        self.train_labels = list(train_labels)
        self.eval_videos = eval_videos
        self.eval_labels = list(eval_labels) if eval_labels is not None else None
        self.crop = model.crop_size
        self.history: List[float] = []

    def _collate(self, batch):
        vids = np.stack([_prep_video(v, self.crop) for v, _ in batch]).astype(np.float32)
        labels = np.asarray([lb for _, lb in batch], dtype=np.int32)
        return vids, labels

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
        import random
        args = self.args
        model = self.model
        n = len(self.train_videos)
        if n == 0:
            raise ValueError("train_videos is empty")
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

        lg = nn.value_and_grad(model, loss_fn)

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
                    yield [(self.train_videos[i], self.train_labels[i]) for i in sl]
                    produced += 1

        self.history = []
        for step, (xb, yb) in enumerate(
            _threaded_prefetch(batch_stream(), self._collate, depth=2)
        ):
            loss, grads = lg(model, mx.array(xb), mx.array(yb))
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            self.history.append(float(loss))
            if step % args.log_every == 0:
                print(f"step {step}/{total_steps}  loss {float(loss):.4f}")
        return self.history

    def evaluate(self, videos=None, labels=None, batch_size: int = 2) -> float:
        videos = videos if videos is not None else self.eval_videos
        labels = labels if labels is not None else self.eval_labels
        if videos is None or labels is None:
            raise ValueError("No eval data provided")
        self.model.eval()
        labels = np.asarray(labels)
        correct = 0
        for i in range(0, len(videos), batch_size):
            chunk = [videos[j] for j in range(i, min(i + batch_size, len(videos)))]
            arr = np.stack([_prep_video(v, self.crop) for v in chunk]).astype(np.float32)
            preds = np.array(mx.argmax(self.model(mx.array(arr)), axis=1))
            correct += int((preds == labels[i:i + len(chunk)]).sum())
        return correct / len(videos)


# ──────────────────────────────────────────────────────────────────────────────
# Downstream: frozen-feature video linear probe
# ──────────────────────────────────────────────────────────────────────────────

def video_linear_probe(
    model: VJEPA2ModelWrapper,
    train_videos: Sequence[Any],
    train_labels: Sequence[int],
    test_videos: Sequence[Any],
    test_labels: Sequence[int],
    num_classes: Optional[int] = None,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> float:
    """Train a logistic-regression probe on frozen V-JEPA 2 clip features."""
    Xtr = model.encode(train_videos)
    Xte = model.encode(test_videos)
    mu, sd = mx.mean(Xtr, axis=0), mx.std(Xtr, axis=0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ytr = mx.array(np.asarray(train_labels, dtype=np.int32))
    yte = np.asarray(test_labels)
    if num_classes is None:
        num_classes = int(max(int(np.max(train_labels)), int(np.max(test_labels))) + 1)

    clf = nn.Linear(Xtr.shape[1], num_classes)
    opt = optim.Adam(learning_rate=lr)

    def lf(m, x, y):
        return mx.mean(nn.losses.cross_entropy(m(x), y))

    lg = nn.value_and_grad(clf, lf)
    N = Xtr.shape[0]
    for _ in range(epochs):
        perm = np.random.permutation(N)
        for i in range(0, N, batch_size):
            bi = mx.array(perm[i:i + batch_size])
            _, g = lg(clf, Xtr[bi], ytr[bi])
            opt.update(clf, g)
            mx.eval(clf.parameters(), opt.state)
    preds = np.array(mx.argmax(clf(Xte), axis=1))
    return float((preds == yte).mean())


def video_knn_probe(
    model: VJEPA2ModelWrapper,
    train_videos: Sequence[Any],
    train_labels: Sequence[int],
    test_videos: Sequence[Any],
    test_labels: Sequence[int],
    k: int = 20,
) -> float:
    """Weighted k-NN on L2-normalised frozen clip features (cosine). No training."""
    Xtr = np.array(model.encode(train_videos))
    Xte = np.array(model.encode(test_videos))
    Xtr /= (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-8)
    Xte /= (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
    ytr = np.asarray(train_labels)
    yte = np.asarray(test_labels)
    num_classes = int(max(int(ytr.max()), int(yte.max())) + 1)
    k = min(k, Xtr.shape[0])
    sims = Xte @ Xtr.T
    nn_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    preds = np.empty(Xte.shape[0], dtype=np.int64)
    for i in range(Xte.shape[0]):
        idx = nn_idx[i]
        votes = np.zeros(num_classes)
        for j in idx:
            votes[ytr[j]] += sims[i, j]
        preds[i] = int(votes.argmax())
    return float((preds == yte).mean())


def video_attentive_probe(
    model: VJEPA2ModelWrapper,
    train_videos: Sequence[Any],
    train_labels: Sequence[int],
    test_videos: Sequence[Any],
    test_labels: Sequence[int],
    num_classes: Optional[int] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    heads: int = 8,
    batch_size: int = 16,
) -> float:
    """Attention-pooling probe on frozen V-JEPA 2 *token* features (canonical eval).

    Trains a small attention-pooling head over per-clip token sequences; the
    encoder stays frozen. Stronger than the mean-pool ``video_linear_probe`` for
    V-JEPA 2, matching how the paper evaluates the encoder.
    """
    from .jepa import _AttentivePoolHead

    tr_tok = model.encode_tokens(train_videos)   # list of (T, hidden)
    te_tok = model.encode_tokens(test_videos)
    ytr = np.asarray(train_labels, dtype=np.int32)
    yte = np.asarray(test_labels)
    if num_classes is None:
        num_classes = int(max(int(ytr.max()), int(yte.max())) + 1)
    dim = tr_tok[0].shape[-1]
    head = _AttentivePoolHead(dim, num_classes, heads=heads)
    mx.eval(head.parameters())
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def lf(m, x, y):
        return mx.mean(nn.losses.cross_entropy(m(x), y))

    lg = nn.value_and_grad(head, lf)
    N = len(tr_tok)
    for _ in range(epochs):
        perm = np.random.permutation(N)
        for i in range(0, N, batch_size):
            bidx = perm[i:i + batch_size]
            # clips share token count for a fixed frame count, so we can stack
            xb = mx.array(np.stack([tr_tok[j] for j in bidx]).astype(np.float32))
            yb = mx.array(ytr[bidx])
            _, g = lg(head, xb, yb)
            opt.update(head, g)
            mx.eval(head.parameters(), opt.state)
    preds = []
    for i in range(0, len(te_tok), batch_size):
        xb = mx.array(np.stack(te_tok[i:i + batch_size]).astype(np.float32))
        preds.append(np.array(mx.argmax(head(xb), axis=1)))
    return float((np.concatenate(preds) == yte).mean())
