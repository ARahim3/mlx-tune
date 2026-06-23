"""
Example 62: LeWM (LeWorldModel) — train a latent world model from pixels

LeWM (arXiv 2603.19312, Maes/Le Lidec/Scieur/LeCun/Balestriero) trains a latent
world model end-to-end from pixels with only TWO loss terms and one tunable
hyperparameter:

    L = next-embedding prediction  +  lambda * SIGReg(embeddings)

There is no stop-gradient and no EMA target — SIGReg (the same regularizer as
LeJEPA) is what prevents the encoder from collapsing. The trained model is a
small latent world model you can plan with (CEM / MPC over latent rollouts).

This demo uses a toy 2-D point-mass with image observations so it runs on-device
in a minute. For real control, feed trajectories from datasets like pusht /
reacher / cube (the `{"frames", "actions"}` format below) and train longer —
convincing closed-loop planning needs the paper's training budget.

Run it:
    python examples/62_lewm_world_model.py
"""

import numpy as np
import mlx.core as mx

from mlx_tune import FastWorldModel, LeWMConfig, LeWMTrainer, plan_cem, PointMassEnv
from mlx_tune.lewm import _flatten_transitions, _images_to_array


def main():
    print("=" * 70)
    print("LeWM — latent world model from pixels (toy point-mass)")
    print("=" * 70)

    # 1. Collect random-action trajectories from the toy environment.
    env = PointMassEnv(size=48, speed=0.22, blob=4, seed=0)
    data = env.collect(n_episodes=80, ep_len=10)          # 800 transitions
    print(f"Collected {sum(len(e['actions']) for e in data)} transitions")

    # 2. Build a small world model (encoder ViT + action-conditioned predictor).
    model = FastWorldModel.from_pretrained(
        "lewm-tiny", img_size=48, patch_size=8, action_dim=2, num_slices=128
    )

    # 3. Train with the two-loss objective (next-embedding prediction + SIGReg).
    cfg = LeWMConfig(
        img_size=48, patch_size=8, encoder_dim=192, encoder_depth=6, encoder_heads=3,
        predictor_hidden=512, action_dim=2, num_slices=128,
        sigreg_lambda=0.05,                # the single tunable hyperparameter
        batch_size=64, max_steps=250, learning_rate=6e-4,
        log_every=50, output_dir="./lewm_output",
    )
    hist = LeWMTrainer(model, cfg, data).train()
    print(f"\nprediction loss: {hist[0]:.4f} -> {np.mean(hist[-20:]):.4f}")

    # 4. Latent rollout: predict future latents under a constant action.
    z0 = model.encode([env.reset()])                       # (1, dim)
    actions = mx.broadcast_to(mx.array([1.0, 0.0]), (1, 5, 2))   # constant action
    roll = model.rollout(z0, actions)
    print(f"latent rollout shape: {roll.shape}")

    # 5. Plan with the world model (CEM / MPC over latent rollouts): at each step
    #    encode the current frame, optimise an action sequence whose predicted
    #    latent matches the goal image's latent, execute the first action, repeat.
    goal_pos = np.array([0.8, 0.2], np.float32)
    goal_z = model.encode([env.render(goal_pos)])[0]
    env.reset(np.array([0.2, 0.8], np.float32))
    d0 = float(np.linalg.norm(env.pos - goal_pos))
    for _ in range(25):
        z = model.encode([env.render()])[0]
        a = plan_cem(model, z, goal_z, horizon=3, action_dim=2, n_samples=400)
        env.step(np.array(a))
    dT = float(np.linalg.norm(env.pos - goal_pos))
    print(f"closed-loop MPC distance to goal: {d0:.3f} -> {dT:.3f}")

    # 6. Save / reload.
    model.save_pretrained("./lewm_output")
    reloaded = FastWorldModel.from_pretrained("./lewm_output")
    same = float(mx.max(mx.abs(model.encode([env.render(goal_pos)]) - reloaded.encode([env.render(goal_pos)]))))
    print(f"reload max|Δ| on encode: {same:.2e}  (round-trip OK)")


if __name__ == "__main__":
    main()
