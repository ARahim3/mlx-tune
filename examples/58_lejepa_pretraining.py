"""
Example 58: LeJEPA self-supervised pretraining (JEPA on Apple Silicon).

Trains a Vision Transformer *from scratch* with the LeJEPA objective
(Balestriero & LeCun, 2025) — a single encoder + multi-view prediction loss +
SIGReg (Sketched Isotropic Gaussian Regularization). No EMA teacher, no
predictor, no stop-gradient. Then evaluates the learned features with a frozen
linear probe.

    python examples/58_lejepa_pretraining.py

Uses a CIFAR-10 subset if `datasets` can download it; otherwise falls back to
synthetic structured images so the example always runs offline.

Scale note (important): this small config DEMONSTRATES THE API, not state-of-the-art
representations. From-scratch LeJEPA collapses at small on-device scale — with the
paper's default `lam=0.05` the encoder's embeddings collapse (per-dim std -> ~0)
within a few hundred steps, so the linear probe stays near chance. This is a
training-budget / weighting issue, NOT a bug: the SIGReg loss is verified correct
(see tests/test_jepa.py), a larger `lam` holds variance up (lam=1.0 prevents collapse
but under-weights prediction), and good features need the paper's budget
(batch >= 128, ~100 epochs). For strong JEPA features ON A MAC, load a PRETRAINED
encoder instead — e.g. `FastJEPAModel.from_pretrained("facebook/ijepa_vith14_1k")`
reaches ~0.9 linear-probe on CIFAR-10 frozen (see example 59).

See `jepa.md` at the repo root for the math and the I-JEPA / V-JEPA 2 roadmap.
"""

import numpy as np

from mlx_tune import FastJEPAModel, JEPATrainer, JEPAConfig, linear_probe

IMG_SIZE = 96
N_TRAIN = 2000
N_TEST = 1000


def load_data():
    """CIFAR-10 subset as (train_imgs, train_labels, test_imgs, test_labels)."""
    try:
        from datasets import load_dataset

        print("Loading CIFAR-10 subset via `datasets`...")
        ds = load_dataset("uoft-cs/cifar10")
        tr = ds["train"].select(range(N_TRAIN))
        te = ds["test"].select(range(N_TEST))
        tr_x = [np.array(im.convert("RGB")) for im in tr["img"]]
        te_x = [np.array(im.convert("RGB")) for im in te["img"]]
        return tr_x, list(tr["label"]), te_x, list(te["label"])
    except Exception as e:  # offline / dataset unavailable → synthetic fallback
        print(f"  ({type(e).__name__}: falling back to synthetic data)")
        centers = [(220, 40, 40), (40, 220, 40), (40, 40, 220),
                   (220, 220, 40), (220, 40, 220)]

        def make(n, seed):
            r = np.random.default_rng(seed)
            xs, ys = [], []
            for i in range(n):
                c = i % len(centers)
                img = (r.random((32, 32, 3)) * 70).astype(np.uint8)
                y, x = int(r.integers(2, 18)), int(r.integers(2, 18))
                img[y:y + 12, x:x + 12] = np.array(centers[c], dtype=np.uint8)
                xs.append(img)
                ys.append(c)
            return xs, ys

        tr_x, tr_y = make(N_TRAIN, 0)
        te_x, te_y = make(N_TEST, 1)
        return tr_x, tr_y, te_x, te_y


def main():
    tr_x, tr_y, te_x, te_y = load_data()
    print(f"train={len(tr_x)}  test={len(te_x)}")

    # 1. Build a ViT encoder (randomly initialised — LeJEPA trains from scratch).
    #    Presets: vit-debug / vit-tiny / vit-small / vit-base.
    model, _ = FastJEPAModel.from_pretrained("vit-tiny", img_size=IMG_SIZE, patch_size=16)

    # 2. Self-supervised pretraining — labels are NOT used here.
    config = JEPAConfig(
        img_size=IMG_SIZE,
        batch_size=64,
        num_epochs=3,
        learning_rate=5e-4,
        lam=0.05,            # the single LeJEPA hyperparameter (SIGReg weight)
        n_global=2,
        n_local=6,
        num_slices=512,
        log_every=20,
    )
    trainer = JEPATrainer(model, args=config, train_dataset=tr_x)
    history = trainer.train()
    print(f"\nLeJEPA loss: {history[0]:.4f} -> {history[-1]:.4f}")

    # 3. Save the pretrained encoder.
    model.save_pretrained("jepa_vit_tiny_cifar")
    print("Saved encoder to ./jepa_vit_tiny_cifar")

    # 4. Evaluate representation quality with a frozen-feature linear probe.
    acc = linear_probe(model, tr_x, tr_y, te_x, te_y, epochs=100, lr=1e-3)
    n_classes = len(set(tr_y))
    print(f"\nLinear-probe accuracy: {acc:.3f}  (chance = {1.0 / n_classes:.3f})")
    print("Note: at this small on-device scale the encoder collapses, so this probe "
          "stays near chance — see the 'Scale note' in the docstring. For strong JEPA "
          "features on a Mac, use a pretrained encoder (example 59).")


if __name__ == "__main__":
    main()
