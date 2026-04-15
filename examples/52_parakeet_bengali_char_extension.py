"""
Example 52: Parakeet TDT Bengali fine-tuning with automatic vocab extension
============================================================================

Bengali is a Brahmic script language whose characters (`আ ক খ গ ঘ ...`) are
NOT covered by Parakeet v3's SentencePiece BPE vocabulary — every Bengali
character encodes to UNK. This example demonstrates how mlx-tune's
`model.extend_vocabulary()` method automatically detects the missing
characters, adds them as new tokens, and resizes the CTC head, joint
network output projection, and decoder embedding so that fine-tuning on a
Bengali dataset "just works".

The same code path handles ANY Unicode language — to fine-tune on Arabic,
Hindi, Thai, Japanese, Korean, Chinese, Hebrew, Tamil, Telugu, etc., just
change the dataset split string (e.g., `bn_in` → `ar_eg` for Arabic,
`hi_in` for Hindi, `ja_jp` for Japanese, `zh_cn` for Chinese). The vocab
extension logic is entirely language-agnostic.

Dataset: Google FLEURS `bn_in` split (Bengali, public, no auth required).

Workflow:
  1. Load Parakeet. Disable the joint-network warm-start (it biases toward
     European-script priors which hurts non-Latin convergence).
  2. Call `model.extend_vocabulary(texts, strategy="char")`. It scans the
     dataset, adds ~60 Bengali characters as new tokens, and resizes the
     CTC head / joint output / decoder embedding in lockstep.
  3. Apply LoRA and train with CTC loss.
  4. Save, reload, verify round-trip.

Demo scope: this is a small-budget demonstration that the pipeline runs
correctly end-to-end. For a production Bengali ASR model you would use
50-200 hours of audio and 2000+ training steps.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlx_tune import FastSTTModel, STTSFTTrainer, STTSFTConfig, STTDataCollator


MODEL_NAME = "mlx-community/parakeet-tdt-0.6b-v3"
OUTPUT_DIR = "./parakeet_bengali_finetuned"
LANGUAGE_SPLIT = "bn_in"  # Change to ar_eg / hi_in / ja_jp / zh_cn / etc.


def main():
    print("=" * 70)
    print(f"Example 52: Parakeet Bengali fine-tuning ({LANGUAGE_SPLIT})")
    print("        with automatic character-level vocab extension")
    print("=" * 70)
    print()

    # 1. Load Parakeet. For non-training-set languages we skip the warm-start
    # so the CTC head doesn't start with an English-biased prior over tokens.
    model, processor = FastSTTModel.from_pretrained(
        MODEL_NAME, warm_start_ctc_head=False
    )
    print()

    # 2. Load FLEURS Bengali across all splits (we need to find a handful
    # of relatively short samples for the demo — Bengali sentences in FLEURS
    # are long and char-level tokenization is token-heavy, so we scan the
    # whole corpus for samples where audio_duration * 12.5Hz >= 2*U+1).
    from datasets import load_dataset
    print(f"Loading FLEURS {LANGUAGE_SPLIT} (scanning for short samples)...")
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
            print(f"  Could not load {split} split: {e}")

    # 3. AUTO-EXTEND THE VOCABULARY BEFORE LoRA.
    # This is the critical step for non-Latin languages. We pass the full
    # set of transcripts to the wrapper; it detects every Bengali character
    # that produces SP UNK and adds them as new tokens, resizing the CTC
    # head, joint output projection, and decoder embedding.
    added_chars = model.extend_vocabulary(all_texts, strategy="char", min_count=1)
    print(f"  {len(added_chars)} new characters added: {''.join(added_chars[:30])}...")
    print()

    # 4. NOW apply LoRA (after vocab extension so the LoRA wrapping sees
    # the resized layers)
    model = FastSTTModel.get_peft_model(model, r=16, lora_alpha=32)
    print()

    # 5. Scan all splits and collect the shortest CTC-feasible samples.
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
            "No CTC-feasible short Bengali samples found in FLEURS. "
            "Char-level Bengali is token-heavy; consider the BPE strategy "
            "in Example 53, or use a dataset with shorter utterances."
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

    # 6. Pre-training CTC output (random head + resized vocab — garbage)
    pre_ctc = model.transcribe_ctc(samples[0]["audio"]["array"])
    print(f"Pre-training CTC (random head): {pre_ctc[:100]!r}")
    print()

    # 7. Train
    collator = STTDataCollator(model=model, processor=processor)
    config = STTSFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=100,
        learning_rate=1e-4,
        logging_steps=10,
        output_dir=OUTPUT_DIR,
        loss_type="ctc",
    )
    trainer = STTSFTTrainer(
        model=model,
        tokenizer=processor,
        train_dataset=train_ds,
        data_collator=collator,
        args=config,
    )
    trainer.train()
    print()

    # 8. Post-training check
    post_ctc = model.transcribe_ctc(samples[0]["audio"]["array"])
    print(f"Post-training CTC (Bengali): {post_ctc[:200]}")
    print(f"Reference:")
    print(f"  {samples[0]['text'][:200]}")
    print()

    # 9. Save / reload round trip. The save path must persist the
    # extended_chars list so reload can rebuild the same vocabulary.
    adapter_path = os.path.join(OUTPUT_DIR, "adapters")
    print(f"Reloading adapters from {adapter_path}...")
    model2, _ = FastSTTModel.from_pretrained(
        MODEL_NAME, warm_start_ctc_head=False
    )
    model2.load_adapter(adapter_path)
    # Verify the reloaded model has the same extended vocab
    assert len(model2.extended_chars) == len(added_chars), (
        f"Reload mismatch: {len(model2.extended_chars)} vs {len(added_chars)}"
    )
    reloaded_ctc = model2.transcribe_ctc(samples[0]["audio"]["array"])
    assert reloaded_ctc == post_ctc, (
        f"Reload mismatch:\n  trained:  {post_ctc}\n  reloaded: {reloaded_ctc}"
    )
    print("Adapter reload: OK — extended vocab preserved, weights match")
    print()
    print("Example 52 completed successfully — Bengali support via char vocab extension.")


if __name__ == "__main__":
    main()
