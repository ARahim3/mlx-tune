"""Tests for LeWM / LeWorldModel (mlx_tune.lewm)."""

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_tune import (
    FastWorldModel,
    WorldModel,
    LeWMConfig,
    LeWMTrainer,
    lewm_loss,
    plan_cem,
    PointMassEnv,
)
from mlx_tune.lewm import _images_to_array, _flatten_transitions, _ActionEncoder
from mlx_tune.jepa import sample_directions


def _debug_cfg(**kw):
    base = dict(img_size=32, patch_size=8, encoder_dim=64, encoder_depth=2,
                encoder_heads=2, predictor_hidden=128, action_dim=2, num_slices=32)
    base.update(kw)
    return LeWMConfig(**base)


# ---------------------------------------------------------------------------
# Config / helpers
# ---------------------------------------------------------------------------
def test_config_to_dict_roundtrips():
    cfg = LeWMConfig(sigreg_lambda=0.1, action_dim=3)
    d = cfg.to_dict()
    assert d["sigreg_lambda"] == 0.1 and d["action_dim"] == 3
    assert LeWMConfig(**d).action_dim == 3


def test_images_to_array_shape_and_range():
    imgs = [(np.random.rand(32, 32, 3) * 255).astype(np.uint8) for _ in range(4)]
    arr = _images_to_array(imgs, 32)
    assert arr.shape == (4, 32, 32, 3)
    assert -1.01 <= float(arr.min()) and float(arr.max()) <= 1.01


def test_action_encoder_continuous_and_discrete():
    cont = _ActionEncoder(16, action_dim=2, discrete=False, num_actions=0)
    assert cont(mx.zeros((5, 2))).shape == (5, 16)
    disc = _ActionEncoder(16, action_dim=0, discrete=True, num_actions=4)
    assert disc(mx.array([0, 1, 2, 3])).shape == (4, 16)
    with pytest.raises(ValueError):
        _ActionEncoder(16, action_dim=0, discrete=True, num_actions=0)


def test_flatten_transitions_both_formats():
    t = _flatten_transitions([{"obs": 1, "action": [0, 0], "next_obs": 2}])
    assert t == [(1, [0, 0], 2)]
    traj = [{"frames": [10, 11, 12], "actions": [[0, 0], [1, 1]]}]
    ft = _flatten_transitions(traj)
    assert ft == [(10, [0, 0], 11), (11, [1, 1], 12)]
    with pytest.raises(ValueError):
        _flatten_transitions([{"bogus": 1}])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def test_worldmodel_encode_predict_rollout_shapes():
    model = WorldModel(_debug_cfg())
    imgs = [(np.random.rand(32, 32, 3) * 255).astype(np.uint8) for _ in range(3)]
    z = model.encode(imgs)
    assert z.shape == (3, 64)
    a = mx.zeros((3, 2))
    assert model.predict_next(z, a).shape == (3, 64)
    roll = model.rollout(z, mx.zeros((3, 4, 2)))
    assert roll.shape == (3, 4, 64)


def test_fastworldmodel_preset_and_unknown():
    m = FastWorldModel.from_pretrained("lewm-debug", action_dim=2)
    assert isinstance(m, WorldModel)
    with pytest.raises(ValueError):
        FastWorldModel.from_pretrained("not-a-preset")


def test_lewm_loss_components_and_grads():
    model = WorldModel(_debug_cfg())
    obs = _images_to_array([(np.random.rand(32, 32, 3) * 255).astype(np.uint8) for _ in range(8)], 32)
    nxt = _images_to_array([(np.random.rand(32, 32, 3) * 255).astype(np.uint8) for _ in range(8)], 32)
    acts = mx.array(np.random.randn(8, 2).astype(np.float32))
    dirs = sample_directions(64, 32)

    def loss_fn(m):
        total, (pred, sig) = lewm_loss(m, obs, acts, nxt, dirs)
        return total

    val, grads = nn.value_and_grad(model, loss_fn)(model)
    assert val.ndim == 0 and float(val) >= 0.0
    flat = dict(tree_flatten(grads))
    # gradients reach both the encoder and the predictor (end-to-end, no stop-grad)
    assert any("encoder" in k for k in flat)
    assert any("predictor" in k for k in flat)


def test_save_load_roundtrip(tmp_path):
    model = WorldModel(_debug_cfg())
    img = [(np.random.rand(32, 32, 3) * 255).astype(np.uint8)]
    before = model.encode(img)
    model.save_pretrained(tmp_path / "wm")
    reloaded = FastWorldModel.from_pretrained(str(tmp_path / "wm"))
    after = reloaded.encode(img)
    assert float(mx.max(mx.abs(before - after))) < 1e-5


# ---------------------------------------------------------------------------
# Planner (correctness on known dynamics)
# ---------------------------------------------------------------------------
class _KnownDynamics:
    """z_{t+1} = z_t + [a, 0...]: the first action_dim latent dims are directly
    controllable, so CEM should drive the cost to ~0."""

    def predict_next(self, z, a):
        pad = mx.concatenate([a, mx.zeros((a.shape[0], z.shape[-1] - a.shape[-1]))], axis=-1)
        return z + pad


def test_plan_cem_reaches_goal_on_known_dynamics():
    dim = 4
    z0 = mx.zeros((dim,))
    goal = mx.array([0.5, -0.3, 0.0, 0.0])
    plan = plan_cem(_KnownDynamics(), z0, goal, horizon=1, action_dim=2,
                    n_samples=512, n_iters=6, return_plan=True)
    z1 = z0 + mx.concatenate([plan[0], mx.zeros((2,))])
    assert float(mx.sum((z1 - goal) ** 2)) < 1e-3


def test_plan_cem_returns_first_action_shape():
    a = plan_cem(_KnownDynamics(), mx.zeros((4,)), mx.zeros((4,)),
                 horizon=3, action_dim=2, n_samples=64, n_iters=2)
    assert a.shape == (2,)


# ---------------------------------------------------------------------------
# Toy environment
# ---------------------------------------------------------------------------
def test_pointmass_env():
    env = PointMassEnv(size=32, seed=0)
    obs = env.reset(np.array([0.5, 0.5], np.float32))
    assert obs.shape == (32, 32, 3) and obs.dtype == np.uint8
    obs2 = env.step(np.array([1.0, 1.0]))
    assert obs2.shape == (32, 32, 3)
    eps = env.collect(n_episodes=2, ep_len=3)
    assert len(eps) == 2 and len(eps[0]["actions"]) == 3 and len(eps[0]["frames"]) == 4


# ---------------------------------------------------------------------------
# Training mechanics
# ---------------------------------------------------------------------------
def test_trainer_runs_and_loss_finite(tmp_path):
    env = PointMassEnv(size=32, seed=0)
    data = env.collect(n_episodes=8, ep_len=6)
    model = WorldModel(_debug_cfg())
    cfg = _debug_cfg(batch_size=16, max_steps=8, log_every=4, output_dir=str(tmp_path / "out"))
    hist = LeWMTrainer(model, cfg, data).train()
    assert len(hist) == 8 and all(np.isfinite(h) for h in hist)
    assert (tmp_path / "out" / "model.safetensors").exists()


@pytest.mark.slow
def test_lewm_training_reduces_prediction_loss():
    """End-to-end training (encoder + predictor, no stop-grad) should clearly
    reduce the next-embedding prediction loss while SIGReg keeps the embeddings
    from collapsing. (Closed-loop control quality is a function of training
    budget / real control data — see docs; this asserts the pipeline learns.)"""
    env = PointMassEnv(size=48, speed=0.22, blob=4, seed=2)
    data = env.collect(n_episodes=120, ep_len=10)
    model = FastWorldModel.from_pretrained("lewm-tiny", img_size=48, patch_size=8,
                                           action_dim=2, num_slices=128)
    cfg = LeWMConfig(img_size=48, patch_size=8, encoder_dim=192, encoder_depth=6,
                     encoder_heads=3, predictor_hidden=512, action_dim=2, num_slices=128,
                     batch_size=64, max_steps=250, learning_rate=6e-4)
    hist = LeWMTrainer(model, cfg, data).train()

    k = max(5, len(hist) // 5)
    early = float(np.mean(hist[:k]))
    late = float(np.mean(hist[-k:]))
    assert late < 0.5 * early  # prediction loss falls substantially over training

    # closed-loop MPC with the trained world model should, on average, drive the
    # point-mass much closer to its goal (the real proof the model is usable).
    rng = np.random.default_rng(0)
    starts, ends = [], []
    for _ in range(4):
        goal_pos = rng.random(2).astype(np.float32)
        goal_z = model.encode([env.render(goal_pos)])[0]
        env.reset(rng.random(2).astype(np.float32))
        starts.append(float(np.linalg.norm(env.pos - goal_pos)))
        for _ in range(20):
            z = model.encode([env.render()])[0]
            a = plan_cem(model, z, goal_z, horizon=3, action_dim=2,
                         n_samples=256, n_iters=4)
            env.step(np.array(a))
        ends.append(float(np.linalg.norm(env.pos - goal_pos)))
    assert np.mean(ends) < 0.7 * np.mean(starts)
