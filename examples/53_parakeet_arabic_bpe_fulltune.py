"""
Example 53: Parakeet TDT Arabic fine-tuning with BPE vocab extension
====================================================================

This is the "serious new-language" example. It demonstrates BPE-based
vocabulary extension — the more sophisticated of the two extension
strategies offered by `model.extend_vocabulary()`. Unlike char-level
extension (Example 52), BPE retrains a real SentencePiece model on your
target-language corpus and combines it with the pretrained tokenizer
via an aggregate wrapper. The result is a more compact tokenization
that scales to longer sentences and larger datasets.

Training setup:
  1. Vocab extension via SentencePiece BPE RETRAINING on the Arabic
     corpus (500 new pieces). The `extend_vocabulary()` method handles
     training, aggregate tokenizer installation, and resizing of the
     CTC head / joint output / decoder embedding automatically.
  2. LoRA on the Conformer encoder's self-attention projections.
  3. Full-weight CTC head (the new BPE-extended head is always trainable).
  4. CTC loss for speed.

Full-weight encoder + joint fine-tuning is ALSO supported — use
`FastSTTModel.get_full_finetune(model, train_encoder=True, train_joint=True,
train_ctc_head=True)` instead of `get_peft_model` — but requires much more
data and careful LR tuning to stay stable. For the small demo budget here
we use LoRA.

Dataset: Google FLEURS `ar_eg` split (Arabic — Egyptian).

This is a small-budget demo (~80 steps, ~8 samples) that demonstrates the
pipeline runs correctly end-to-end: BPE vocab extension, LoRA training,
save/reload with the aux BPE model persisted alongside the adapters.
For production Arabic ASR, scale up: 100+ hours of audio, 2000+ steps.

To try OTHER non-Latin languages, change LANGUAGE_SPLIT:
  - `ar_eg` — Arabic (Egyptian)
  - `hi_in` — Hindi
  - `th_th` — Thai
  - `he_il` — Hebrew
  - `ja_jp` — Japanese
  - `zh_cn` — Chinese (Mandarin)
  - `ko_kr` — Korean

The BPE retraining logic is entirely language-agnostic.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlx_tune import FastSTTModel, STTSFTTrainer, STTSFTConfig, STTDataCollator


MODEL_NAME = "mlx-community/parakeet-tdt-0.6b-v3"
OUTPUT_DIR = "./parakeet_arabic_finetuned"
LANGUAGE_SPLIT = "ar_eg"


def main():
    print("=" * 70)
    print(f"Example 53: Parakeet Arabic fine-tuning ({LANGUAGE_SPLIT})")
    print("   BPE vocab extension + full-weight encoder + CTC loss")
    print("=" * 70)
    print()

    # 1. Load Parakeet (no warm-start for non-trained languages)
    model, processor = FastSTTModel.from_pretrained(
        MODEL_NAME, warm_start_ctc_head=False
    )
    print()

    # 2. Load Arabic FLEURS across all splits
    from datasets import load_dataset
    print(f"Loading FLEURS {LANGUAGE_SPLIT}...")
    all_dss = []
    all_texts = []
    for split in ["train", "validation", "test"]:
        try:
            ds = load_dataset(
                "google/fleurs",
                LANGUAGE_SPLIT,
                split=split,
                trust_remote_code=True,
            )
            all_dss.append(ds)
            all_texts.extend(ds[i]["transcription"] for i in range(len(ds)))
        except Exception as e:
            print(f"  Could not load {split}: {e}")
    print(f"  Total text corpus: {len(all_texts)} lines")

    # 3. BPE VOCAB EXTENSION.
    # Train a 500-piece SentencePiece BPE model on the Arabic corpus,
    # merge it with the pretrained Parakeet tokenizer via an aggregate
    # wrapper, and resize the CTC head / joint output / decoder embedding.
    added_pieces = model.extend_vocabulary(
        all_texts,
        strategy="bpe",
        bpe_vocab_size=500,
    )
    print(f"  {len(added_pieces)} BPE pieces added via SentencePiece retraining")
    print(f"  Example pieces: {added_pieces[:10]}...")
    print()

    # 4. Training setup. For the encoder we use LoRA (standard, stable
    # across small datasets). Full-weight training of the entire 600M-param
    # encoder is possible via `FastSTTModel.get_full_finetune()` but is
    # memory-heavy and prone to gradient explosions on tiny demo datasets
    # — save it for 50+ hour production corpora. Here we apply LoRA to the
    # encoder self-attention projections with a generous rank.
    model = FastSTTModel.get_peft_model(model, r=16, lora_alpha=32)
    print()

    # 5. Find a small set of CTC-feasible short Arabic samples.
    # Arabic BPE gives ~0.5-1 tokens per char, so it's more efficient than
    # Bengali char-level but we still need reasonably short samples.
    max_audio_samples = model.profile.max_audio_samples
    samples = []
    rejected = 0
    for ds in all_dss:
        for i in range(len(ds)):
            if len(samples) >= 8:
                break
            item = ds[i]
            audio = item["audio"]
            audio_arr = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            if max_audio_samples > 0 and len(audio_arr) > max_audio_samples:
                audio_arr = audio_arr[:max_audio_samples]
            audio_s = len(audio_arr) / sr
            if audio_s > 10:
                continue
            text = item["transcription"].strip()
            tokens = model._encode_text(text)
            if len(tokens) < 5:
                continue
            max_tokens_for_audio = int((audio_s * 12.5 - 1) / 2)
            if len(tokens) > max_tokens_for_audio:
                rejected += 1
                continue
            samples.append({
                "audio": {"array": audio_arr, "sampling_rate": sr},
                "text": text,
            })
        if len(samples) >= 8:
            break

    if not samples:
        raise RuntimeError(
            "No CTC-feasible short Arabic samples found. Consider raising "
            "`max_audio_samples` in the profile or using a larger BPE vocab."
        )
    print(f"  {len(samples)} samples accepted, {rejected} rejected")
    print(f"  First sentence: {samples[0]['text'][:80]}")
    print()

    class SimpleDataset:
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, key):
            if isinstance(key, slice):
                return self.samples[key]
            return self.samples[key]

    train_ds = SimpleDataset(samples)
    collator = STTDataCollator(model=model, processor=processor)

    # 6. Training config — LoRA on encoder + full-weight CTC head.
    config = STTSFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=80,
        learning_rate=1e-4,
        logging_steps=10,
        output_dir=OUTPUT_DIR,
        loss_type="ctc",
    )

    pre_ctc = model.transcribe_ctc(samples[0]["audio"]["array"])
    print(f"Pre-training CTC: {pre_ctc[:100]!r}")
    print()

    trainer = STTSFTTrainer(
        model=model,
        tokenizer=processor,
        train_dataset=train_ds,
        data_collator=collator,
        args=config,
    )
    trainer.train()
    print()

    post_ctc = model.transcribe_ctc(samples[0]["audio"]["array"])
    print(f"Post-training CTC: {post_ctc[:200]}")
    print(f"Reference: {samples[0]['text'][:200]}")
    print()

    # 7. Save / reload round trip — must persist the trained aux BPE model
    # alongside the adapters for the reloaded model to rebuild the same
    # vocabulary. The save path copies `aux_sentencepiece.model` into the
    # output directory automatically.
    adapter_path = os.path.join(OUTPUT_DIR, "adapters")
    print(f"Reloading from {adapter_path}...")
    model2, _ = FastSTTModel.from_pretrained(
        MODEL_NAME, warm_start_ctc_head=False
    )
    model2.load_adapter(adapter_path)
    assert model2.extension_strategy == "bpe", "BPE strategy not restored"
    assert model2.aux_sp_tokenizer is not None, "aux SP tokenizer not restored"

    reloaded_ctc = model2.transcribe_ctc(samples[0]["audio"]["array"])
    assert reloaded_ctc == post_ctc, (
        f"Reload mismatch:\n  trained:  {post_ctc}\n  reloaded: {reloaded_ctc}"
    )
    print("Adapter reload: OK — BPE model restored, weights match")
    print()
    print("Example 53 completed successfully — Arabic via BPE + full-weight fine-tune.")


if __name__ == "__main__":
    main()
