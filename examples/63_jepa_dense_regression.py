"""
Example 63: JEPA dense & regression heads (counting, depth, segmentation)

I-JEPA's own headline tasks are depth prediction and object counting — i.e.
"more than classification". This example shows the three non-classification
downstream heads on a JEPA encoder:

  * regression       — a scalar/vector target (e.g. object counting)
  * dense regression — a per-pixel value map (e.g. depth)
  * segmentation     — per-pixel class logits

Each reuses the same frozen / LoRA / full fine-tuning machinery as
classification. Uses a small from-scratch encoder + synthetic data so it runs
on-device quickly; swap in `FastJEPAModel.from_pretrained("facebook/ijepa_*")`
for the real pretrained encoder.

Run it:
    python examples/63_jepa_dense_regression.py
"""

import numpy as np

from mlx_tune import (
    FastJEPAModel, JEPAClassifierConfig,
    JEPARegressionTrainer, JEPADenseTrainer,
)


def synth(n, size=64, seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.random((size, size, 3)) * 255).astype(np.uint8) for _ in range(n)]


def main():
    print("=" * 70)
    print("JEPA dense & regression heads")
    print("=" * 70)
    imgs = synth(48)
    cfg = JEPAClassifierConfig(img_size=64, batch_size=8, num_epochs=4,
                               learning_rate=2e-3, warmup_ratio=0.1, log_every=50)

    # 1. Regression (counting / scalar value) — target = mean brightness here.
    counts = [float(np.asarray(im).mean() / 255.0) for im in imgs]
    model, _ = FastJEPAModel.from_pretrained("vit-tiny", img_size=64, patch_size=16)
    reg = FastJEPAModel.for_image_regression(model, out_dim=1, finetune="frozen")
    rt = JEPARegressionTrainer(reg, cfg, imgs, counts, eval_images=imgs, eval_targets=counts)
    rt.train()
    print("regression eval:", rt.evaluate())

    # 2. Dense regression (depth-like per-pixel map).
    depth = [(np.asarray(im).mean(-1) / 255.0).astype(np.float32) for im in imgs]
    m2, _ = FastJEPAModel.from_pretrained("vit-tiny", img_size=64, patch_size=16)
    dense = FastJEPAModel.for_dense_prediction(m2, out_channels=1, task="regression",
                                               finetune="frozen")
    dt = JEPADenseTrainer(dense, cfg, imgs, depth, eval_images=imgs, eval_targets=depth)
    dt.train()
    print("dense depth map shape:", dense.predict(imgs[:2]).shape, "| eval:", dt.evaluate())

    # 3. Segmentation (per-pixel class ids).
    seg_t = [(np.asarray(im).mean(-1) // 90).clip(0, 2).astype(np.int32) for im in imgs]
    m3, _ = FastJEPAModel.from_pretrained("vit-tiny", img_size=64, patch_size=16)
    seg = FastJEPAModel.for_dense_prediction(m3, out_channels=3, task="segmentation",
                                             finetune="frozen")
    st = JEPADenseTrainer(seg, cfg, imgs, seg_t, eval_images=imgs, eval_targets=seg_t)
    st.train()
    print("segmentation mask shape:", seg.predict(imgs[:2]).shape, "| eval:", st.evaluate())


if __name__ == "__main__":
    main()
