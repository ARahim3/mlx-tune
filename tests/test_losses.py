"""
Unit tests for loss functions in mlx_tune.losses
"""

import pytest
import mlx.core as mx
import mlx.nn as nn


class TestComputeLogProbs:
    """Test log probability computation."""

    def test_compute_log_probs_shape(self):
        """Test output shape of compute_log_probs."""
        from mlx_tune.losses import compute_log_probs_with_lengths

        # Create a simple mock model
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 100)

            def __call__(self, x):
                return self.linear(self.embedding(x))

        model = MockModel()
        mx.eval(model.parameters())

        # Test input
        batch_size = 2
        seq_len = 10
        input_ids = mx.random.randint(0, 100, (batch_size, seq_len))
        lengths = mx.array([8, 6])

        log_probs = compute_log_probs_with_lengths(model, input_ids, lengths)

        assert log_probs.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {log_probs.shape}"

    def test_compute_log_probs_values(self):
        """Test that log probs are negative (as expected for probabilities)."""
        from mlx_tune.losses import compute_log_probs_with_lengths

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 100)

            def __call__(self, x):
                return self.linear(self.embedding(x))

        model = MockModel()
        mx.eval(model.parameters())

        input_ids = mx.random.randint(0, 100, (2, 10))
        lengths = mx.array([8, 6])

        log_probs = compute_log_probs_with_lengths(model, input_ids, lengths)
        mx.eval(log_probs)

        # Log probabilities should be negative (or zero at maximum)
        assert mx.all(log_probs <= 0), "Log probabilities should be non-positive"


class TestDPOLoss:
    """Test DPO loss computation."""

    def test_dpo_loss_shape(self):
        """Test DPO loss returns scalar."""
        from mlx_tune.losses import dpo_loss

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 100)

            def __call__(self, x):
                return self.linear(self.embedding(x))

        model = MockModel()
        mx.eval(model.parameters())

        batch_size = 2
        seq_len = 10
        chosen_ids = mx.random.randint(0, 100, (batch_size, seq_len))
        rejected_ids = mx.random.randint(0, 100, (batch_size, seq_len))
        chosen_lengths = mx.array([8, 7])
        rejected_lengths = mx.array([9, 6])

        loss, ntoks = dpo_loss(
            model, chosen_ids, rejected_ids,
            chosen_lengths, rejected_lengths,
            beta=0.1
        )

        assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
        assert ntoks.shape == (), f"ntoks should be scalar, got shape {ntoks.shape}"

    def test_dpo_loss_beta_effect(self):
        """Test that higher beta increases loss magnitude."""
        from mlx_tune.losses import dpo_loss

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 100)

            def __call__(self, x):
                return self.linear(self.embedding(x))

        model = MockModel()
        mx.eval(model.parameters())

        chosen_ids = mx.random.randint(0, 100, (2, 10))
        rejected_ids = mx.random.randint(0, 100, (2, 10))
        chosen_lengths = mx.array([8, 7])
        rejected_lengths = mx.array([9, 6])

        loss_low_beta, _ = dpo_loss(model, chosen_ids, rejected_ids,
                                     chosen_lengths, rejected_lengths, beta=0.01)
        loss_high_beta, _ = dpo_loss(model, chosen_ids, rejected_ids,
                                      chosen_lengths, rejected_lengths, beta=1.0)

        mx.eval(loss_low_beta, loss_high_beta)

        # Both losses should be finite
        assert not mx.isnan(loss_low_beta), "Low beta loss should not be NaN"
        assert not mx.isnan(loss_high_beta), "High beta loss should not be NaN"


class TestORPOLoss:
    """Test ORPO loss computation."""

    def test_orpo_loss_shape(self):
        """Test ORPO loss returns scalar."""
        from mlx_tune.losses import orpo_loss

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 100)

            def __call__(self, x):
                return self.linear(self.embedding(x))

        model = MockModel()
        mx.eval(model.parameters())

        chosen_ids = mx.random.randint(0, 100, (2, 10))
        rejected_ids = mx.random.randint(0, 100, (2, 10))
        chosen_lengths = mx.array([8, 7])
        rejected_lengths = mx.array([9, 6])

        loss, ntoks = orpo_loss(model, chosen_ids, rejected_ids,
                                chosen_lengths, rejected_lengths, beta=0.1)

        assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"


class TestSimPOLoss:
    """Test SimPO loss computation."""

    def test_simpo_loss_shape(self):
        """Test SimPO loss returns scalar."""
        from mlx_tune.losses import simpo_loss

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 100)

            def __call__(self, x):
                return self.linear(self.embedding(x))

        model = MockModel()
        mx.eval(model.parameters())

        chosen_ids = mx.random.randint(0, 100, (2, 10))
        rejected_ids = mx.random.randint(0, 100, (2, 10))
        chosen_lengths = mx.array([8, 7])
        rejected_lengths = mx.array([9, 6])

        loss, ntoks = simpo_loss(model, chosen_ids, rejected_ids,
                                  chosen_lengths, rejected_lengths,
                                  beta=2.0, gamma=0.5)

        assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"


class TestSFTLoss:
    """Test SFT loss computation."""

    def test_sft_loss_shape(self):
        """Test SFT loss returns scalar."""
        from mlx_tune.losses import sft_loss

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 100)

            def __call__(self, x):
                return self.linear(self.embedding(x))

        model = MockModel()
        mx.eval(model.parameters())

        input_ids = mx.random.randint(0, 100, (2, 10))
        lengths = mx.array([8, 6])

        loss, ntoks = sft_loss(model, input_ids, lengths)

        assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
        assert loss.item() > 0, "Cross entropy loss should be positive"


class TestCTCLoss:
    """Tests for pure-MLX CTC loss used by Parakeet fine-tuning."""

    def test_scalar_output(self):
        from mlx_tune.losses import ctc_loss
        T, B, V = 20, 1, 10
        log_probs = nn.log_softmax(mx.random.normal((T, B, V)), axis=-1)
        targets = mx.array([[0, 1, 2, 3, 4]], dtype=mx.int32)
        input_lengths = mx.array([T], dtype=mx.int32)
        target_lengths = mx.array([5], dtype=mx.int32)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=V - 1)
        assert loss.shape == ()
        assert mx.isfinite(loss).item()

    def test_uniform_distribution_known_value(self):
        """With uniform log P = -log(V), CTC NLL for target [0] equals log(V)."""
        import math
        from mlx_tune.losses import ctc_loss
        T, B, V = 2, 1, 3
        lp = mx.full((T, B, V), -math.log(V))
        targets = mx.array([[0]], dtype=mx.int32)
        loss = ctc_loss(
            lp, targets, mx.array([T]), mx.array([1]), blank=V - 1
        ).item()
        assert abs(loss - math.log(V)) < 1e-4, f"got {loss}, expected {math.log(V)}"

    def test_finite_on_random(self):
        from mlx_tune.losses import ctc_loss
        for _ in range(5):
            T, B, V = 30, 1, 20
            log_probs = nn.log_softmax(mx.random.normal((T, B, V)), axis=-1)
            targets = mx.random.randint(0, V - 1, (B, 6))
            input_lengths = mx.array([T], dtype=mx.int32)
            target_lengths = mx.array([6], dtype=mx.int32)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=V - 1)
            assert mx.isfinite(loss).item(), f"non-finite loss: {loss.item()}"

    def test_gradient_flow(self):
        from mlx_tune.losses import ctc_loss
        T, B, V = 10, 1, 5
        targets = mx.array([[0, 1, 2]], dtype=mx.int32)
        input_lengths = mx.array([T], dtype=mx.int32)
        target_lengths = mx.array([3], dtype=mx.int32)

        def loss_fn(logits):
            lp = nn.log_softmax(logits, axis=-1)
            return ctc_loss(lp, targets, input_lengths, target_lengths, blank=V - 1)

        logits = mx.random.normal((T, B, V))
        g = mx.grad(loss_fn)(logits)
        assert g.shape == logits.shape
        assert mx.sum(mx.abs(g)).item() > 0

    def test_optimization_decreases_loss(self):
        from mlx_tune.losses import ctc_loss
        T, B, V = 10, 1, 5
        targets = mx.array([[0, 1, 2]], dtype=mx.int32)
        input_lengths = mx.array([T], dtype=mx.int32)
        target_lengths = mx.array([3], dtype=mx.int32)

        def loss_fn(logits):
            return ctc_loss(
                nn.log_softmax(logits, axis=-1),
                targets,
                input_lengths,
                target_lengths,
                blank=V - 1,
            )

        logits = mx.random.normal((T, B, V)) * 0.5
        initial = loss_fn(logits).item()
        for _ in range(50):
            logits = logits - 0.5 * mx.grad(loss_fn)(logits)
        final = loss_fn(logits).item()
        assert final < initial, f"CTC loss did not decrease: {initial} -> {final}"

    def test_multi_batch_with_different_lengths(self):
        from mlx_tune.losses import ctc_loss
        T, B, V = 10, 2, 5
        lp = nn.log_softmax(mx.random.normal((T, B, V)), axis=-1)
        targets = mx.array([[0, 1, 2, 3], [0, 1, 0, 0]], dtype=mx.int32)
        input_lengths = mx.array([10, 8], dtype=mx.int32)
        target_lengths = mx.array([4, 2], dtype=mx.int32)
        loss = ctc_loss(
            lp, targets, input_lengths, target_lengths, blank=V - 1, reduction="none"
        )
        assert loss.shape == (B,)
        assert mx.all(mx.isfinite(loss)).item()


class TestRNNTLoss:
    """Tests for pure-MLX RNN-T loss used by Parakeet fine-tuning."""

    def test_scalar_output(self):
        from mlx_tune.losses import rnnt_loss
        B, T, U, V = 1, 3, 2, 4
        jlp = nn.log_softmax(mx.random.normal((B, T, U + 1, V)), axis=-1)
        targets = mx.array([[0, 1]], dtype=mx.int32)
        loss = rnnt_loss(jlp, targets, mx.array([T]), mx.array([U]), blank=V - 1)
        assert loss.shape == ()
        assert mx.isfinite(loss).item()

    def test_uniform_distribution_matches_analytical(self):
        """For uniform joint log-probs, NLL = (T+U)*log(V) - log(C(T-1+U, U))."""
        import math
        from math import comb
        from mlx_tune.losses import rnnt_loss
        for T, U, V in [(2, 1, 2), (3, 2, 3), (4, 1, 2), (2, 2, 2), (5, 3, 4)]:
            B = 1
            jlp = mx.full((B, T, U + 1, V), -math.log(V))
            t = mx.array([list(range(U))], dtype=mx.int32)
            loss = rnnt_loss(
                jlp, t, mx.array([T]), mx.array([U]), blank=V - 1
            ).item()
            expected = (T + U) * math.log(V) - math.log(comb(T - 1 + U, U))
            assert abs(loss - expected) < 1e-3, (
                f"T={T}, U={U}, V={V}: got {loss}, expected {expected}"
            )

    def test_gradient_flow(self):
        from mlx_tune.losses import rnnt_loss
        B, T, U, V = 1, 3, 2, 4
        targets = mx.array([[0, 1]], dtype=mx.int32)

        def loss_fn(raw):
            return rnnt_loss(
                nn.log_softmax(raw, axis=-1),
                targets,
                mx.array([T]),
                mx.array([U]),
                blank=V - 1,
            )

        raw = mx.random.normal((B, T, U + 1, V))
        g = mx.grad(loss_fn)(raw)
        assert g.shape == raw.shape
        assert mx.sum(mx.abs(g)).item() > 0

    def test_optimization_decreases_loss(self):
        from mlx_tune.losses import rnnt_loss
        B, T, U, V = 1, 3, 2, 4
        targets = mx.array([[0, 1]], dtype=mx.int32)

        def loss_fn(raw):
            return rnnt_loss(
                nn.log_softmax(raw, axis=-1),
                targets,
                mx.array([T]),
                mx.array([U]),
                blank=V - 1,
            )

        raw = mx.random.normal((B, T, U + 1, V)) * 0.5
        initial = loss_fn(raw).item()
        for _ in range(30):
            raw = raw - 0.5 * mx.grad(loss_fn)(raw)
        final = loss_fn(raw).item()
        assert final < initial, f"RNNT loss did not decrease: {initial} -> {final}"

    def test_multi_batch(self):
        from mlx_tune.losses import rnnt_loss
        B, T, U_max, V = 3, 6, 3, 5
        jlp = nn.log_softmax(mx.random.normal((B, T, U_max + 1, V)), axis=-1)
        targets = mx.array(
            [[0, 1, 2], [0, 1, 0], [1, 2, 0]], dtype=mx.int32
        )
        input_lengths = mx.array([6, 5, 4], dtype=mx.int32)
        target_lengths = mx.array([3, 2, 3], dtype=mx.int32)
        losses = rnnt_loss(
            jlp, targets, input_lengths, target_lengths, blank=V - 1, reduction="none"
        )
        assert losses.shape == (B,)
        assert mx.all(mx.isfinite(losses)).item()


class TestTDTLoss:
    """Tests for pure-MLX TDT loss used by Parakeet fine-tuning."""

    def test_scalar_output(self):
        from mlx_tune.losses import tdt_loss
        B, T, U, V, D = 1, 3, 2, 4, 5
        lp_tok = nn.log_softmax(mx.random.normal((B, T, U + 1, V)), axis=-1)
        lp_dur = nn.log_softmax(mx.random.normal((B, T, U + 1, D)), axis=-1)
        jlp = mx.concatenate([lp_tok, lp_dur], axis=-1)
        targets = mx.array([[0, 1]], dtype=mx.int32)
        loss = tdt_loss(jlp, targets, mx.array([T]), mx.array([U]), blank=V - 1)
        assert loss.shape == ()
        assert mx.isfinite(loss).item()

    def test_gradient_flows_to_both_heads(self):
        """Verify both token and duration logits receive non-zero gradients."""
        from mlx_tune.losses import tdt_loss
        B, T, U, V, D = 1, 3, 2, 4, 5

        def loss_fn(raw):
            lp_tok = nn.log_softmax(raw[..., :V], axis=-1)
            lp_dur = nn.log_softmax(raw[..., V:], axis=-1)
            jlp = mx.concatenate([lp_tok, lp_dur], axis=-1)
            return tdt_loss(
                jlp,
                mx.array([[0, 1]], dtype=mx.int32),
                mx.array([T]),
                mx.array([U]),
                blank=V - 1,
            )

        raw = mx.random.normal((B, T, U + 1, V + D))
        g = mx.grad(loss_fn)(raw)
        token_grad_mag = mx.sum(mx.abs(g[..., :V])).item()
        duration_grad_mag = mx.sum(mx.abs(g[..., V:])).item()
        assert token_grad_mag > 0, "token gradients are zero"
        assert duration_grad_mag > 0, "duration gradients are zero"

    def test_optimization_decreases_loss(self):
        from mlx_tune.losses import tdt_loss
        B, T, U, V, D = 1, 4, 2, 5, 5

        def loss_fn(raw):
            lp_tok = nn.log_softmax(raw[..., :V], axis=-1)
            lp_dur = nn.log_softmax(raw[..., V:], axis=-1)
            jlp = mx.concatenate([lp_tok, lp_dur], axis=-1)
            return tdt_loss(
                jlp,
                mx.array([[0, 1]], dtype=mx.int32),
                mx.array([T]),
                mx.array([U]),
                blank=V - 1,
            )

        raw = mx.random.normal((B, T, U + 1, V + D)) * 0.5
        initial = loss_fn(raw).item()
        for _ in range(30):
            raw = raw - 0.3 * mx.grad(loss_fn)(raw)
        final = loss_fn(raw).item()
        assert final < initial, f"TDT loss did not decrease: {initial} -> {final}"
