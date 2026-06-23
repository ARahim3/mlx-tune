"""
LeWM — LeWorldModel: a stable end-to-end Joint-Embedding Predictive *world model*.

Implements the recipe from *LeWorldModel: Stable End-to-End Joint-Embedding
Predictive Architecture from Pixels* (Maes, Le Lidec, Scieur, LeCun & Balestriero,
arXiv 2603.19312). LeWM trains a latent world model end-to-end from pixels with
**only two loss terms** and a single tunable hyperparameter:

    L = L_pred  +  lambda * SIGReg(Z)

* **L_pred** — next-embedding prediction: an action-conditioned predictor maps the
  current latent + action to the next latent, trained against the *encoder's own*
  embedding of the next observation. Crucially there is **no stop-gradient and no
  EMA target** — gradients flow through both, exactly as in LeJEPA.
* **SIGReg(Z)** — the same Sketched Isotropic Gaussian Regularization used by
  LeJEPA (reused from ``mlx_tune.jepa``). It is what stops the encoder from
  collapsing to a trivial constant (which would make next-embedding prediction
  easy but useless) — the central insight of the paper.

The trained model is a small (~10-15M param) latent world model you can **plan
with**: :func:`plan_cem` runs Cross-Entropy-Method / MPC over latent rollouts to
choose actions that drive the latent toward a goal.

This module reuses ``ViTEncoder``, ``sigreg_loss`` and ``sample_directions`` from
``mlx_tune.jepa`` and is otherwise self-contained.

Entry points: :class:`FastWorldModel`, :class:`LeWMTrainer`, :class:`LeWMConfig`,
:func:`plan_cem`, plus a toy :class:`PointMassEnv` for demos/tests.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union
import json

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_tune.jepa import ViTEncoder, sigreg_loss, sample_directions


# ──────────────────────────────────────────────────────────────────────────────
# Image helper
# ──────────────────────────────────────────────────────────────────────────────
def _images_to_array(images: Sequence[Any], size: int) -> mx.array:
    """Normalise a batch of HxWx3 uint8/float images to ``(B, size, size, 3)`` float32.

    Accepts numpy arrays or anything ``np.asarray`` can handle. Images already at
    ``size`` are used as-is; otherwise nearest-resized via PIL.
    """
    out = []
    for im in images:
        arr = np.asarray(im)
        if arr.dtype != np.float32 or arr.max() > 1.5:
            if arr.shape[:2] != (size, size):
                from PIL import Image
                arr = np.asarray(
                    Image.fromarray(arr.astype(np.uint8)).resize((size, size), Image.BILINEAR)
                )
            arr = arr.astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        out.append(arr)
    return mx.array(np.stack(out))


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class LeWMConfig:
    """Configuration for :class:`FastWorldModel` / :class:`LeWMTrainer`."""

    # encoder (small ViT — paper uses ~15M params end-to-end)
    img_size: int = 64
    patch_size: int = 8
    encoder_dim: int = 192
    encoder_depth: int = 6
    encoder_heads: int = 3
    # action conditioning
    action_dim: int = 2
    discrete_actions: bool = False
    num_actions: int = 0          # required if discrete_actions
    # predictor
    predictor_hidden: int = 512
    # objective
    sigreg_lambda: float = 0.05   # single tunable hyperparameter (paper: ~0.05)
    num_slices: int = 256         # SIGReg projection directions
    # optimisation
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    batch_size: int = 64
    epochs: int = 1
    max_steps: int = -1
    warmup_ratio: float = 0.05
    log_every: int = 20
    output_dir: str = "./lewm_output"

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ──────────────────────────────────────────────────────────────────────────────
# Action-conditioned latent predictor
# ──────────────────────────────────────────────────────────────────────────────
class _ActionEncoder(nn.Module):
    """Map an action (continuous vector or discrete id) to a ``dim`` embedding."""

    def __init__(self, dim: int, action_dim: int, discrete: bool, num_actions: int):
        super().__init__()
        self.discrete = discrete
        if discrete:
            if num_actions <= 0:
                raise ValueError("discrete_actions requires num_actions > 0")
            self.embed = nn.Embedding(num_actions, dim)
        else:
            self.proj = nn.Linear(action_dim, dim)

    def __call__(self, a: mx.array) -> mx.array:
        if self.discrete:
            return self.embed(a.astype(mx.int32).reshape(-1))
        return self.proj(a)


class _LatentPredictor(nn.Module):
    """Action-conditioned latent dynamics: ``z_{t+1} = z_t + MLP([z_t, enc(a_t)])``.

    Residual (delta) parameterisation — standard for latent dynamics, keeps the
    one-step map near identity and stabilises multi-step rollouts.
    """

    def __init__(self, dim: int, action_dim: int, hidden: int,
                 discrete: bool, num_actions: int):
        super().__init__()
        self.action_encoder = _ActionEncoder(dim, action_dim, discrete, num_actions)
        self.fc1 = nn.Linear(dim * 2, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def __call__(self, z: mx.array, a: mx.array) -> mx.array:
        ae = self.action_encoder(a)
        h = mx.concatenate([z, ae], axis=-1)
        delta = self.fc2(nn.gelu(self.fc1(h)))
        return z + delta


class WorldModel(nn.Module):
    """Encoder (ViT) + action-conditioned latent predictor.

    ``encode`` produces the latent ``z`` (mean-pooled ViT tokens, no L2-norm —
    consistent with the isotropic-Gaussian SIGReg target). ``predict_next`` and
    ``rollout`` advance the latent under actions.
    """

    def __init__(self, cfg: LeWMConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ViTEncoder(
            img_size=cfg.img_size, patch_size=cfg.patch_size, dim=cfg.encoder_dim,
            depth=cfg.encoder_depth, heads=cfg.encoder_heads,
        )
        self.predictor = _LatentPredictor(
            cfg.encoder_dim, cfg.action_dim, cfg.predictor_hidden,
            cfg.discrete_actions, cfg.num_actions,
        )

    # -- latent ops ----------------------------------------------------------
    def encode_array(self, imgs: mx.array) -> mx.array:
        """Encode a pre-normalised ``(B, H, W, 3)`` array → ``(B, dim)``."""
        return self.encoder(imgs)

    def predict_next(self, z: mx.array, a: mx.array) -> mx.array:
        return self.predictor(z, a)

    def rollout(self, z0: mx.array, actions: mx.array) -> mx.array:
        """Roll the latent forward under an action sequence.

        Args:
            z0: ``(B, dim)`` start latents.
            actions: ``(B, H, action_dim)`` (or ``(B, H)`` for discrete).

        Returns:
            ``(B, H, dim)`` predicted latents ``z_1..z_H``.
        """
        H = actions.shape[1]
        z = z0
        outs = []
        for t in range(H):
            z = self.predict_next(z, actions[:, t])
            outs.append(z)
        return mx.stack(outs, axis=1)

    # -- convenience (host-side images) --------------------------------------
    def encode(self, images: Sequence[Any], batch_size: int = 64) -> mx.array:
        """Encode host images (numpy/PIL) → ``(N, dim)``."""
        feats = []
        for i in range(0, len(images), batch_size):
            arr = _images_to_array(images[i:i + batch_size], self.cfg.img_size)
            feats.append(self.encode_array(arr))
        return mx.concatenate(feats, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────
def lewm_loss(
    model: WorldModel,
    obs: mx.array,
    actions: mx.array,
    next_obs: mx.array,
    directions: mx.array,
    sigreg_lambda: float = 0.05,
) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
    """LeWM objective: next-embedding prediction + SIGReg (no stop-gradient).

    Args:
        model: the :class:`WorldModel`.
        obs / next_obs: pre-normalised ``(B, H, W, 3)`` observation pairs.
        actions: ``(B, action_dim)`` (or ``(B,)`` discrete) actions taken at ``obs``.
        directions: SIGReg projection directions ``(dim, S)``.
        sigreg_lambda: weight on the SIGReg term.

    Returns ``(total, (pred_loss, sigreg))`` — aux for logging.
    """
    z = model.encode_array(obs)                 # (B, dim)
    z_next = model.encode_array(next_obs)       # (B, dim)  — NOT stop-grad'd
    z_pred = model.predict_next(z, actions)     # (B, dim)

    pred_loss = mx.mean((z_pred - z_next) ** 2)
    # SIGReg over both endpoints' embeddings (richer estimate of the marginal).
    zc = mx.concatenate([z, z_next], axis=0)
    sig = sigreg_loss(zc, directions)
    total = pred_loss + sigreg_lambda * sig
    return total, (pred_loss, sig)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
class FastWorldModel:
    """Unsloth-style entry point for LeWM world models.

    ``from_pretrained`` accepts either a preset size name (``"lewm-tiny"`` /
    ``"lewm-small"``) for a fresh model, or a directory saved by
    :meth:`WorldModel.save_pretrained`.
    """

    PRESETS = {
        "lewm-debug": dict(encoder_dim=64, encoder_depth=2, encoder_heads=2,
                           predictor_hidden=128, img_size=32, patch_size=8),
        "lewm-tiny": dict(encoder_dim=192, encoder_depth=6, encoder_heads=3,
                          predictor_hidden=512, img_size=64, patch_size=8),
        "lewm-small": dict(encoder_dim=384, encoder_depth=8, encoder_heads=6,
                           predictor_hidden=1024, img_size=64, patch_size=8),
    }

    @staticmethod
    def from_pretrained(name: str = "lewm-tiny", **overrides) -> WorldModel:
        p = Path(name)
        if p.exists() and (p / "lewm_config.json").exists():
            cfg_d = json.loads((p / "lewm_config.json").read_text())
            cfg_d.update(overrides)
            cfg = LeWMConfig(**cfg_d)
            model = WorldModel(cfg)
            model.load_weights(str(p / "model.safetensors"))
            mx.eval(model.parameters())
            return model
        if name in FastWorldModel.PRESETS:
            cfg_d = dict(FastWorldModel.PRESETS[name])
            cfg_d.update(overrides)
            return WorldModel(LeWMConfig(**cfg_d))
        raise ValueError(
            f"Unknown LeWM model '{name}'. Use a preset {list(FastWorldModel.PRESETS)} "
            f"or a directory saved by save_pretrained()."
        )


def _save_world_model(model: WorldModel, path: Union[str, Path]) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "lewm_config.json").write_text(json.dumps(model.cfg.to_dict(), indent=2))
    from mlx.utils import tree_flatten
    mx.save_safetensors(str(path / "model.safetensors"),
                        dict(tree_flatten(model.parameters())))


# attach save method to WorldModel (kept here to keep the class lean)
WorldModel.save_pretrained = lambda self, path: _save_world_model(self, path)


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────
def _flatten_transitions(data: Sequence[dict]) -> List[Tuple[Any, Any, Any]]:
    """Accept either explicit transitions or trajectories; return (obs, a, next).

    * transition dict: ``{"obs", "action", "next_obs"}``
    * trajectory dict: ``{"frames": [img,...], "actions": [a,...]}`` (len frames
      == len actions + 1) → consecutive transitions.
    """
    transitions: List[Tuple[Any, Any, Any]] = []
    for item in data:
        if "next_obs" in item:
            transitions.append((item["obs"], item["action"], item["next_obs"]))
        elif "frames" in item and "actions" in item:
            frames, actions = item["frames"], item["actions"]
            for t in range(len(actions)):
                transitions.append((frames[t], actions[t], frames[t + 1]))
        else:
            raise ValueError(
                "Each item needs {'obs','action','next_obs'} or {'frames','actions'}."
            )
    return transitions


class LeWMTrainer:
    """Train a :class:`WorldModel` with the two-loss LeWM objective (end-to-end)."""

    def __init__(self, model: WorldModel, args: LeWMConfig, train_data: Sequence[dict]):
        self.model = model
        self.cfg = args
        self.transitions = _flatten_transitions(train_data)
        if not self.transitions:
            raise ValueError("No transitions to train on.")
        self.output_dir = Path(args.output_dir)
        n = len(self.transitions)
        bs = max(1, args.batch_size)
        self.iters = args.max_steps if args.max_steps > 0 else max(1, (n // bs) * args.epochs)

    def _batch(self, idx: np.ndarray):
        obs = _images_to_array([self.transitions[i][0] for i in idx], self.cfg.img_size)
        nxt = _images_to_array([self.transitions[i][2] for i in idx], self.cfg.img_size)
        acts_raw = [np.asarray(self.transitions[i][1]) for i in idx]
        if self.cfg.discrete_actions:
            actions = mx.array(np.array(acts_raw).reshape(-1))
        else:
            actions = mx.array(np.stack(acts_raw).astype(np.float32))
        return obs, actions, nxt

    def train(self) -> List[float]:
        from mlx_tune._perf import configure_wired_limit
        configure_wired_limit()
        cfg = self.cfg
        model = self.model
        model.train()

        warmup = int(cfg.warmup_ratio * self.iters)
        if warmup > 0:
            sched = optim.join_schedules(
                [optim.linear_schedule(0.0, cfg.learning_rate, warmup),
                 optim.cosine_decay(cfg.learning_rate, max(1, self.iters - warmup))],
                [warmup],
            )
        else:
            sched = optim.cosine_decay(cfg.learning_rate, self.iters)
        optimizer = optim.AdamW(learning_rate=sched, weight_decay=cfg.weight_decay)

        directions = sample_directions(cfg.encoder_dim, cfg.num_slices)

        def loss_fn(m, obs, actions, nxt, directions):
            total, aux = lewm_loss(m, obs, actions, nxt, directions, cfg.sigreg_lambda)
            return total, aux

        lag = nn.value_and_grad(model, loss_fn)
        n = len(self.transitions)
        bs = max(1, cfg.batch_size)
        rng = np.random.default_rng(0)
        history: List[float] = []

        print(f"LeWMTrainer: {n} transitions, {self.iters} iters, batch {bs}")
        for step in range(self.iters):
            idx = rng.integers(0, n, size=bs)
            obs, actions, nxt = self._batch(idx)
            # fresh SIGReg directions each step (paper resamples; better coverage)
            directions = sample_directions(cfg.encoder_dim, cfg.num_slices)
            (total, (pred_l, sig)), grads = lag(model, obs, actions, nxt, directions)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, total)
            history.append(float(pred_l.item()))
            if (step + 1) % cfg.log_every == 0:
                print(f"  step {step+1}/{self.iters} | pred {float(pred_l.item()):.4f} "
                      f"| sigreg {float(sig.item()):.4f}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(self.output_dir)
        print(f"  ✓ saved world model to {self.output_dir}")
        return history


# ──────────────────────────────────────────────────────────────────────────────
# Planning (CEM / MPC over latent rollouts)
# ──────────────────────────────────────────────────────────────────────────────
def plan_cem(
    model: WorldModel,
    obs_latent: mx.array,
    goal_latent: mx.array,
    horizon: int,
    action_dim: int,
    *,
    n_samples: int = 256,
    n_iters: int = 5,
    elite_frac: float = 0.1,
    init_std: float = 0.5,
    action_low: float = -1.0,
    action_high: float = 1.0,
    return_plan: bool = False,
) -> mx.array:
    """Cross-Entropy-Method planner over latent rollouts.

    Optimises an ``horizon``-step continuous action sequence so the model's
    predicted final latent matches ``goal_latent`` (least-squares in latent
    space). Returns the first action (MPC) or the whole plan.

    Args:
        model: trained :class:`WorldModel`.
        obs_latent: ``(dim,)`` current latent (e.g. ``model.encode([obs])[0]``).
        goal_latent: ``(dim,)`` target latent.
        horizon / action_dim: planning horizon and action dimensionality.
        n_samples / n_iters / elite_frac / init_std: CEM knobs.
        action_low / action_high: action clamp range.
        return_plan: if True return the full ``(horizon, action_dim)`` mean plan.

    Returns:
        ``(action_dim,)`` first action, or ``(horizon, action_dim)`` if
        ``return_plan``.
    """
    dim = obs_latent.shape[-1]
    mean = mx.zeros((horizon, action_dim))
    std = mx.ones((horizon, action_dim)) * init_std
    z0 = mx.broadcast_to(obs_latent.reshape(1, dim), (n_samples, dim))
    goal = goal_latent.reshape(1, dim)
    n_elite = max(1, int(n_samples * elite_frac))

    for _ in range(n_iters):
        eps = mx.random.normal((n_samples, horizon, action_dim))
        samples = mx.clip(mean[None] + std[None] * eps, action_low, action_high)
        z = z0
        for t in range(horizon):
            z = model.predict_next(z, samples[:, t, :])
        cost = mx.sum((z - goal) ** 2, axis=-1)            # (n_samples,)
        order = mx.argsort(cost)
        elite = samples[order[:n_elite]]                   # (n_elite, H, A)
        mean = mx.mean(elite, axis=0)
        std = mx.std(elite, axis=0) + 1e-6
        mx.eval(mean, std)

    return mean if return_plan else mean[0]


# ──────────────────────────────────────────────────────────────────────────────
# Toy environment (for demos / tests)
# ──────────────────────────────────────────────────────────────────────────────
class PointMassEnv:
    """A 2-D point-mass with image observations.

    State is a position in ``[0, 1]^2``; a continuous 2-D action nudges it.
    ``render()`` draws a white blob at the position on a black background —
    enough learnable structure for a from-pixels world model demo.
    """

    def __init__(self, size: int = 64, speed: float = 0.12, blob: int = 6, seed: int = 0):
        self.size = size
        self.speed = speed
        self.blob = blob
        self.rng = np.random.default_rng(seed)
        self.pos = self.rng.random(2).astype(np.float32)

    def reset(self, pos: Optional[np.ndarray] = None) -> np.ndarray:
        self.pos = self.rng.random(2).astype(np.float32) if pos is None else np.array(pos, np.float32)
        return self.render()

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        self.pos = np.clip(self.pos + self.speed * a, 0.0, 1.0)
        return self.render()

    def render(self, pos: Optional[np.ndarray] = None) -> np.ndarray:
        p = self.pos if pos is None else np.asarray(pos, np.float32)
        img = np.zeros((self.size, self.size, 3), np.uint8)
        # Faint position-encoded background tint (R←x, G←y) so the *globally
        # pooled* encoder latent tracks position — a mean-pooled ViT is otherwise
        # fairly translation-invariant and the moving blob alone barely moves the
        # latent (making prediction trivially identity and planning signal-less).
        img[:, :, 0] = int(p[0] * 90)
        img[:, :, 1] = int(p[1] * 90)
        cy, cx = int(p[1] * (self.size - 1)), int(p[0] * (self.size - 1))
        r = self.blob
        y0, y1 = max(0, cy - r), min(self.size, cy + r)
        x0, x1 = max(0, cx - r), min(self.size, cx + r)
        img[y0:y1, x0:x1] = 255
        return img

    def collect(self, n_episodes: int, ep_len: int) -> List[dict]:
        """Collect random-action trajectories as ``{'frames', 'actions'}`` dicts."""
        eps = []
        for _ in range(n_episodes):
            self.reset()
            frames = [self.render()]
            actions = []
            for _ in range(ep_len):
                a = self.rng.uniform(-1, 1, size=2).astype(np.float32)
                frames.append(self.step(a))
                actions.append(a)
            eps.append({"frames": frames, "actions": actions})
        return eps
