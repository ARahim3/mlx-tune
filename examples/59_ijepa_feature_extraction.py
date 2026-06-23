"""
Example 59: I-JEPA (Meta) — pretrained features + downstream classification.

Loads Meta's pretrained **I-JEPA** image encoder (`facebook/ijepa_vith14_1k`,
ViT-Huge/14, self-supervised on ImageNet), converts it to MLX, and shows two
downstream paths on Apple Silicon:

  1. frozen-feature linear probe (no backprop through the encoder — fast);
  2. LoRA fine-tuning of the encoder + a classification head.

    python examples/59_ijepa_feature_extraction.py

Note: `facebook/ijepa_vith14_1k` is ~2.5 GB and downloads on first run. The MLX
port is numerically identical to the HuggingFace PyTorch model (cosine 1.000;
see tests/test_jepa.py::test_ijepa_parity_vs_hf).

See `jepa.md` at the repo root for the math and the rest of the JEPA roadmap.
"""

import numpy as np

from mlx_tune import (
    FastJEPAModel,
    linear_probe,
    knn_probe,
    attentive_probe,
    JEPAClassifierTrainer,
    JEPAClassifierConfig,
)

REPO = "facebook/ijepa_vith14_1k"   # I-JEPA expects 224×224 inputs


def make_dataset(n, seed):
    """Synthetic 3-class images: a colored square on noise (offline-friendly).

    Swap this for a real dataset (e.g. `datasets.load_dataset("cifar10")`) — any
    list of HWC uint8 arrays / PIL images works.
    """
    rng = np.random.default_rng(seed)
    centers = [(230, 30, 30), (30, 230, 30), (30, 30, 230)]
    xs, ys = [], []
    for i in range(n):
        c = i % 3
        img = (rng.random((224, 224, 3)) * 60).astype(np.uint8)
        y, x = int(rng.integers(20, 150)), int(rng.integers(20, 150))
        img[y:y + 60, x:x + 60] = np.array(centers[c], dtype=np.uint8)
        xs.append(img)
        ys.append(c)
    return xs, ys


def main():
    print(f"Loading pretrained I-JEPA: {REPO}")
    model, _ = FastJEPAModel.from_pretrained(REPO)
    print(f"  encoder dim={model.embed_dim}, image size={model.img_size}")

    tr_x, tr_y = make_dataset(60, seed=0)
    te_x, te_y = make_dataset(30, seed=1)

    # 1) Frozen-feature probes — none backprop through the 630M encoder.
    #    attentive_probe is the canonical I-JEPA / V-JEPA 2 evaluation; a plain
    #    linear probe on mean-pooled features under-reads encoder quality.
    print("\n[1] Frozen-feature probes")
    print(f"    linear   : {linear_probe(model, tr_x, tr_y, te_x, te_y, epochs=100):.3f}  (chance = 0.333)")
    print(f"    kNN      : {knn_probe(model, tr_x, tr_y, te_x, te_y, k=15):.3f}")
    print(f"    attentive: {attentive_probe(model, tr_x, tr_y, te_x, te_y, epochs=60):.3f}")

    # 2) LoRA fine-tuning — adapt the encoder cheaply + train a head end-to-end.
    print("\n[2] LoRA fine-tuning")
    clf = FastJEPAModel.for_image_classification(model, num_classes=3, finetune="lora", r=8)
    config = JEPAClassifierConfig(img_size=224, batch_size=6, num_epochs=5,
                                  learning_rate=3e-4, warmup_ratio=0.2, log_every=10)
    trainer = JEPAClassifierTrainer(clf, config, tr_x, tr_y,
                                    eval_images=te_x, eval_labels=te_y)
    hist = trainer.train()
    print(f"    loss {hist[0]:.4f} -> {hist[-1]:.4f}")
    print(f"    fine-tuned accuracy: {trainer.evaluate():.3f}  (chance = 0.333)")

    # 3) Save → reload → run inference. Reloaded model predicts identically.
    print("\n[3] Save / load / inference")
    clf.save_pretrained("ijepa_classifier")
    reloaded = FastJEPAModel.load_classifier("ijepa_classifier")
    preds = reloaded.predict(te_x[:5])               # class ids
    probs = reloaded.predict(te_x[:5], return_probs=True)
    print(f"    saved to ./ijepa_classifier, reloaded and ran inference")
    print(f"    predictions (first 5): {preds.tolist()}  true: {te_y[:5]}")
    print(f"    confidence (max prob):  {probs.max(axis=1).round(3).tolist()}")


if __name__ == "__main__":
    main()
