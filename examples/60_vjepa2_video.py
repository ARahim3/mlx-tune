"""
Example 60: V-JEPA 2 (Meta) — pretrained video world model on Apple Silicon.

Loads Meta's **V-JEPA 2** video encoder (`facebook/vjepa2-vitl-fpc64-256`,
ViT-L with a 3D tubelet patch embed + 3D RoPE, self-supervised on >1M hours of
video), converts it to MLX, and shows two downstream paths:

  1. frozen-feature video linear probe (fast — encode clips once, fit a head);
  2. LoRA fine-tuning of the encoder + a video-classification head.

    python examples/60_vjepa2_video.py

Notes:
  * The checkpoint is ~1.3 GB and downloads on first run.
  * Inputs are clips shaped (T, 256, 256, 3); T must be a multiple of the
    tubelet size (2). Frames are resized to 256×256 (the crop the RoPE grid is
    tied to). Swap the synthetic clips below for real decoded video frames.
  * The MLX port matches the HuggingFace PyTorch model numerically (cosine
    1.000; see tests/test_vjepa2.py::test_vjepa2_parity_vs_hf).

See `jepa.md` at the repo root for the architecture notes and the JEPA roadmap.
"""

import numpy as np

from mlx_tune import (
    FastVideoJEPAModel,
    video_linear_probe,
    VideoClassifierTrainer,
    VideoClassifierConfig,
)

REPO = "facebook/vjepa2-vitl-fpc64-256"
T = 8  # frames per clip (multiple of tubelet_size=2)


def make_clips(n, seed):
    """Synthetic 2-class clips: a moving colored square (offline-friendly).

    Replace with real frames decoded from video files — any list of
    (T, H, W, 3) uint8 arrays works (frames are resized to 256×256 internally).
    """
    rng = np.random.default_rng(seed)
    colors = [(220, 40, 40), (40, 40, 220)]
    xs, ys = [], []
    for i in range(n):
        c = i % 2
        clip = (rng.random((T, 256, 256, 3)) * 40).astype(np.uint8)
        for t in range(T):
            off = 20 * t
            clip[t, 40 + off:120 + off, 40:120] = np.array(colors[c], dtype=np.uint8)
        xs.append(clip)
        ys.append(c)
    return xs, ys


def main():
    print(f"Loading pretrained V-JEPA 2: {REPO}")
    model, _ = FastVideoJEPAModel.from_pretrained(REPO)
    print(f"  hidden={model.embed_dim}, crop={model.crop_size}, tubelet={model.tubelet_size}")

    tr_x, tr_y = make_clips(12, seed=0)
    te_x, te_y = make_clips(8, seed=1)

    # 1) Frozen-feature video linear probe.
    print("\n[1] Frozen-feature video linear probe")
    acc = video_linear_probe(model, tr_x, tr_y, te_x, te_y, epochs=80, lr=5e-3)
    print(f"    video linear-probe accuracy: {acc:.3f}  (chance = 0.5)")

    # 2) LoRA fine-tuning of the video encoder + classification head.
    print("\n[2] LoRA fine-tuning")
    clf = FastVideoJEPAModel.for_video_classification(model, num_classes=2, finetune="lora", r=8)
    config = VideoClassifierConfig(batch_size=2, num_epochs=5,
                                   learning_rate=3e-4, warmup_ratio=0.2, log_every=2)
    trainer = VideoClassifierTrainer(clf, config, tr_x, tr_y,
                                     eval_videos=te_x, eval_labels=te_y)
    hist = trainer.train()
    print(f"    loss {hist[0]:.4f} -> {hist[-1]:.4f}")
    print(f"    fine-tuned accuracy: {trainer.evaluate():.3f}  (chance = 0.5)")

    # 3) Save → reload → run inference. Reloaded model predicts identically.
    print("\n[3] Save / load / inference")
    clf.save_pretrained("vjepa2_classifier")
    reloaded = FastVideoJEPAModel.load_classifier("vjepa2_classifier")
    preds = reloaded.predict(te_x[:4])
    print(f"    saved to ./vjepa2_classifier, reloaded and ran inference")
    print(f"    predictions (first 4): {preds.tolist()}  true: {te_y[:4]}")


if __name__ == "__main__":
    main()
