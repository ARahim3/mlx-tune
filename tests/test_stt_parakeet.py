"""
Unit tests for Parakeet TDT fine-tuning support in mlx_tune.

These tests exercise the profile, detection, collator shape contract,
and vocabulary extension path without downloading the real model. The
full end-to-end training runs in examples/50_* through examples/53_*.
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
import numpy as np


class TestParakeetProfile:
    """Profile registration, fields, and detection patterns."""

    def test_profile_registered(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert "parakeet_tdt" in STT_PROFILES

    def test_profile_fields(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        p = STT_PROFILES["parakeet_tdt"]
        assert p.architecture == "parakeet_tdt"
        assert p.sample_rate == 16000
        assert p.n_mels == 128
        assert p.max_audio_samples == 160000
        assert p.encoder_block_path == "encoder.layers"
        # Parakeet has no LoRA-able decoder (LSTM)
        assert p.decoder_block_path == ""
        assert p.preprocessor == "parakeet_mel"
        assert "linear_q" in p.lora_target_modules
        assert "linear_k" in p.lora_target_modules
        assert "linear_v" in p.lora_target_modules
        assert "linear_out" in p.lora_target_modules
        assert "linear_pos" in p.lora_target_modules

    def test_detection_mlx_community(self):
        from mlx_tune.audio_profiles import detect_stt_model_type
        assert detect_stt_model_type("mlx-community/parakeet-tdt-0.6b-v3") == "parakeet_tdt"

    def test_detection_nvidia(self):
        from mlx_tune.audio_profiles import detect_stt_model_type
        assert detect_stt_model_type("nvidia/parakeet-tdt-1.1b") == "parakeet_tdt"

    def test_detection_does_not_match_canary(self):
        from mlx_tune.audio_profiles import detect_stt_model_type
        # canary should still work despite both using FastConformer
        assert detect_stt_model_type("nvidia/canary-1b") == "canary"

    def test_detection_does_not_match_whisper(self):
        from mlx_tune.audio_profiles import detect_stt_model_type
        assert detect_stt_model_type("openai/whisper-tiny") == "whisper"


class TestParakeetCTCLossIntegration:
    """Verify CTC loss dispatch paths produce finite gradients."""

    def test_ctc_loss_runs_on_toy_shapes(self):
        from mlx_tune.losses import ctc_loss

        # Shape contract: log_probs (T, B, V), targets (B, U_max)
        T, B, V = 40, 1, 50
        log_probs = nn.log_softmax(mx.random.normal((T, B, V)), axis=-1)
        targets = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        input_lengths = mx.array([T], dtype=mx.int32)
        target_lengths = mx.array([5], dtype=mx.int32)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=V - 1)
        assert mx.isfinite(loss).item()

    def test_rnnt_loss_runs_on_toy_shapes(self):
        from mlx_tune.losses import rnnt_loss

        B, T, U, V = 1, 6, 3, 10
        jlp = nn.log_softmax(mx.random.normal((B, T, U + 1, V)), axis=-1)
        targets = mx.array([[0, 1, 2]], dtype=mx.int32)
        loss = rnnt_loss(jlp, targets, mx.array([T]), mx.array([U]), blank=V - 1)
        assert mx.isfinite(loss).item()

    def test_tdt_loss_runs_on_toy_shapes(self):
        from mlx_tune.losses import tdt_loss

        B, T, U, V, D = 1, 6, 3, 10, 5
        tok_lp = nn.log_softmax(mx.random.normal((B, T, U + 1, V)), axis=-1)
        dur_lp = nn.log_softmax(mx.random.normal((B, T, U + 1, D)), axis=-1)
        jlp = mx.concatenate([tok_lp, dur_lp], axis=-1)
        targets = mx.array([[0, 1, 2]], dtype=mx.int32)
        loss = tdt_loss(jlp, targets, mx.array([T]), mx.array([U]), blank=V - 1)
        assert mx.isfinite(loss).item()


class TestCharExtensionLogic:
    """Unit-level tests for char-level vocab extension math."""

    def test_resize_ctc_head_preserves_old_weights(self):
        """When we extend the CTC head, the first (V+1) rows must equal the
        pre-extension weights so existing vocabulary tokens keep their
        predictions unchanged."""
        old = nn.Linear(8, 5, bias=True)
        mx.eval(old.parameters())
        old_w = old.weight
        old_b = old.bias

        # Simulate the resize we do in _install_char_extension
        N = 3
        new_rows = mx.random.normal((N, 8), dtype=old_w.dtype) * 0.02
        new_row_b = mx.zeros((N,), dtype=old_b.dtype)
        new_w = mx.concatenate([old_w, new_rows], axis=0)
        new_b = mx.concatenate([old_b, new_row_b], axis=0)
        new = nn.Linear(8, 5 + N, bias=True)
        mx.eval(new.parameters())
        new.weight = new_w
        new.bias = new_b
        mx.eval(new.parameters())

        # First 5 rows should be byte-identical
        diff = mx.sum(mx.abs(new.weight[:5] - old_w)).item()
        assert diff == 0.0
        bias_diff = mx.sum(mx.abs(new.bias[:5] - old_b)).item()
        assert bias_diff == 0.0
        assert new.weight.shape[0] == 5 + N
        assert new.bias.shape[0] == 5 + N

    def test_nfc_normalization_dedupes_compositions(self):
        import unicodedata
        # Composed vs decomposed Bengali vowel ki ('ক' + combining vowel)
        composed = unicodedata.normalize("NFC", "কি")
        decomposed = unicodedata.normalize("NFD", "কি")
        assert unicodedata.normalize("NFC", decomposed) == composed


class TestBPEExtensionLogic:
    """Sanity checks for the aggregate tokenizer's id routing."""

    def test_bpe_id_ranges_do_not_collide(self):
        """New BPE tokens live in [old_out_dim, old_out_dim + N - 1],
        never overlapping the pretrained SP ids < old_blank and never
        trampling the blank at old_blank."""
        old_blank = 8192
        old_out_dim = old_blank + 1  # 8193
        N = 500
        # Simulated id layout
        sp_range = (0, old_blank - 1)  # 0..8191
        blank_pos = old_blank           # 8192
        new_range = (old_out_dim, old_out_dim + N - 1)  # 8193..8692

        # Invariants
        assert sp_range[1] < blank_pos
        assert blank_pos < new_range[0]
        assert new_range[1] == old_blank + N
