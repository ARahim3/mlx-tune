"""
Example 64: V-JEPA 2 predictor (masked latent prediction) + Meta's pretrained
video classifier — the two halves added on top of the encoder port.

Part 1 — **predictor / anticipation** (`facebook/vjepa2-vitl-fpc64-256`):
  the predictor sees the latents of the first half of a clip and predicts the
  latents of the second half — prediction in representation space, the
  "world model" half of V-JEPA 2. `latent_energy` scores how surprising the
  *actual* future was: a clip whose second half doesn't follow from its first
  half (a hard cut) should score higher energy than a coherent clip.

Part 2 — **pretrained video classification** (`facebook/vjepa2-vitl-fpc16-256-ssv2`):
  Meta's fine-tuned Something-Something-v2 checkpoint (174 action classes,
  attentive pooler + linear head) loads straight into MLX — zero training,
  call `.predict()` on raw clips.

    python examples/64_vjepa2_predictor_classifier.py

Notes:
  * Checkpoints are ~1.3 GB (encoder+predictor) and ~1.5 GB (SSv2 classifier);
    both download on first run.
  * Both ports match the HuggingFace PyTorch models numerically (cosine
    1.000000 — see tests/test_vjepa2.py predictor/ssv2 parity tests).

See `jepa.md` at the repo root for the roadmap (V-JEPA 2-AC planning is §9).
"""

import numpy as np

from mlx_tune import FastVideoJEPAModel, latent_energy

ENCODER_REPO = "facebook/vjepa2-vitl-fpc64-256"
SSV2_REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"
T = 8  # frames per clip (multiple of tubelet_size=2)


def moving_square_clip(seed=0, frames=T, reverse_second_half=False):
    """A square sliding smoothly across the frame; optionally cut to an
    unrelated motion at the halfway point (the 'surprising' clip)."""
    rng = np.random.default_rng(seed)
    clip = (rng.random((frames, 256, 256, 3)) * 40).astype(np.uint8)
    for t in range(frames):
        if reverse_second_half and t >= frames // 2:
            # hard cut: the square jumps to the opposite corner and color flips
            off = 200 - 22 * (t - frames // 2)
            clip[t, off:off + 50, 180:230] = np.array((40, 220, 40), np.uint8)
        else:
            off = 20 + 22 * t
            clip[t, 80:130, off:off + 50] = np.array((220, 40, 40), np.uint8)
    return clip


def main():
    # ── Part 1: predictor — anticipate the second half of a clip ─────────────
    print(f"Loading V-JEPA 2 encoder + predictor: {ENCODER_REPO}")
    model, _ = FastVideoJEPAModel.from_pretrained(ENCODER_REPO)
    print(f"  hidden={model.embed_dim}, predictor loaded: {model.has_predictor}")

    coherent = moving_square_clip(seed=0)
    surprising = moving_square_clip(seed=0, reverse_second_half=True)

    pred_c, target_c = model.predict_latents(coherent, context_frames=T // 2)
    pred_s, target_s = model.predict_latents(surprising, context_frames=T // 2)
    e_coherent = float(latent_energy(pred_c, target_c))
    e_surprising = float(latent_energy(pred_s, target_s))
    print(f"\n  predicted latents: {pred_c.shape} (second-half tokens)")
    print(f"  energy(coherent clip)   = {e_coherent:.4f}")
    print(f"  energy(hard-cut clip)   = {e_surprising:.4f}")
    print(f"  → the model is {'more' if e_surprising > e_coherent else 'NOT more'} "
          f"surprised by the discontinuous future (expected: more)")
    per_tok = latent_energy(pred_s, target_s, per_token=True)
    print(f"  per-token energy map: {per_tok.shape} (localise the surprise)")

    # ── Part 2: Meta's pretrained SSv2 action classifier ─────────────────────
    print(f"\nLoading pretrained SSv2 classifier: {SSV2_REPO}")
    clf, _ = FastVideoJEPAModel.from_pretrained(SSV2_REPO)
    print(f"  {clf.num_classes} classes, fpc={clf.frames_per_clip}, "
          f"pooler+head loaded from the checkpoint")

    results = clf.predict([coherent], top_k=5)
    print("\n  top-5 for the moving-square clip (synthetic — labels indicative):")
    for r in results[0]:
        print(f"    {r['prob']:.3f}  {r['label']}")

    # save → reload → identical predictions (MLX-native artifact)
    out = "vjepa2_ssv2_mlx"
    clf.save_pretrained(out)
    reloaded, _ = FastVideoJEPAModel.from_pretrained(out)
    again = reloaded.predict([coherent], top_k=1)
    assert again[0][0]["id"] == results[0][0]["id"]
    print(f"\n  saved + reloaded from '{out}' — top-1 identical ✓")


if __name__ == "__main__":
    main()
