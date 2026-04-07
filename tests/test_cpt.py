"""
Tests for CPTTrainer and CPTConfig.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestCPTConfig:
    """Test CPTConfig class."""

    def test_defaults(self):
        """Test CPTConfig has correct defaults."""
        from mlx_tune import CPTConfig

        config = CPTConfig()
        assert config.output_dir == "./cpt_outputs"
        assert config.learning_rate == 5e-5
        assert config.embedding_learning_rate == 5e-5 / 5.0  # Default 1/5
        assert config.include_embeddings is True
        assert config.dataset_text_field == "text"
        assert config.per_device_train_batch_size == 2
        assert config.gradient_accumulation_steps == 4
        assert config.lr_scheduler_type == "cosine"
        assert config.weight_decay == 0.01

    def test_custom_embedding_lr(self):
        """Test custom embedding learning rate."""
        from mlx_tune import CPTConfig

        config = CPTConfig(
            learning_rate=5e-5,
            embedding_learning_rate=5e-6,
        )
        assert config.learning_rate == 5e-5
        assert config.embedding_learning_rate == 5e-6

    def test_embedding_lr_auto_default(self):
        """Test embedding LR auto-defaults to learning_rate / 5."""
        from mlx_tune import CPTConfig

        config = CPTConfig(learning_rate=1e-4)
        assert config.embedding_learning_rate == 1e-4 / 5.0

    def test_include_embeddings_false(self):
        """Test include_embeddings can be disabled."""
        from mlx_tune import CPTConfig

        config = CPTConfig(include_embeddings=False)
        assert config.include_embeddings is False

    def test_to_dict(self):
        """Test serialization to dict."""
        from mlx_tune import CPTConfig

        config = CPTConfig(
            learning_rate=1e-4,
            embedding_learning_rate=1e-5,
            max_steps=500,
        )
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["learning_rate"] == 1e-4
        assert d["embedding_learning_rate"] == 1e-5
        assert d["max_steps"] == 500
        assert d["include_embeddings"] is True

    def test_kwargs_passthrough(self):
        """Test that extra kwargs are stored."""
        from mlx_tune import CPTConfig

        config = CPTConfig(custom_param="test_value")
        assert config.custom_param == "test_value"

    def test_max_steps_vs_epochs(self):
        """Test that max_steps takes priority."""
        from mlx_tune import CPTConfig

        config = CPTConfig(max_steps=200, num_train_epochs=5)
        assert config.max_steps == 200
        assert config.num_train_epochs == 5


class TestCPTTrainer:
    """Test CPTTrainer class."""

    def _make_mock_model(self, target_modules=None):
        """Create a mock model wrapper."""
        model = MagicMock()
        model.lora_config = {
            'r': 16,
            'target_modules': target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
            'lora_alpha': 16,
            'lora_dropout': 0.0,
        }
        model._lora_applied = True
        model.tokenizer = MagicMock()
        model.tokenizer.pad_token_id = 0
        model.model = MagicMock()
        return model

    def test_init_with_config(self):
        """Test CPTTrainer init with CPTConfig."""
        from mlx_tune import CPTConfig
        from mlx_tune.cpt_trainer import CPTTrainer

        model = self._make_mock_model()
        dataset = [{"text": "hello world"}]

        config = CPTConfig(
            learning_rate=5e-5,
            embedding_learning_rate=5e-6,
            max_steps=50,
            output_dir=tempfile.mkdtemp(),
        )

        trainer = CPTTrainer(
            model=model,
            train_dataset=dataset,
            args=config,
        )

        assert trainer.learning_rate == 5e-5
        assert trainer.embedding_learning_rate == 5e-6
        assert trainer.iters == 50
        assert trainer.include_embeddings is True

        # Cleanup
        shutil.rmtree(trainer.output_dir, ignore_errors=True)

    def test_auto_adds_embeddings(self):
        """Test embed_tokens and lm_head are auto-added to target modules."""
        from mlx_tune import CPTConfig
        from mlx_tune.cpt_trainer import CPTTrainer

        model = self._make_mock_model(target_modules=["q_proj", "v_proj"])
        dataset = [{"text": "hello"}]

        config = CPTConfig(
            include_embeddings=True,
            max_steps=10,
            output_dir=tempfile.mkdtemp(),
        )

        trainer = CPTTrainer(model=model, train_dataset=dataset, args=config)

        targets = model.lora_config['target_modules']
        assert 'embed_tokens' in targets
        assert 'lm_head' in targets

        shutil.rmtree(trainer.output_dir, ignore_errors=True)

    def test_no_auto_add_when_disabled(self):
        """Test embeddings not added when include_embeddings=False."""
        from mlx_tune import CPTConfig
        from mlx_tune.cpt_trainer import CPTTrainer

        model = self._make_mock_model(target_modules=["q_proj", "v_proj"])
        dataset = [{"text": "hello"}]

        config = CPTConfig(
            include_embeddings=False,
            max_steps=10,
            output_dir=tempfile.mkdtemp(),
        )

        trainer = CPTTrainer(model=model, train_dataset=dataset, args=config)

        targets = model.lora_config['target_modules']
        assert 'embed_tokens' not in targets
        assert 'lm_head' not in targets

        shutil.rmtree(trainer.output_dir, ignore_errors=True)

    def test_no_duplicate_embed_tokens(self):
        """Test embed_tokens isn't added twice if already present."""
        from mlx_tune import CPTConfig
        from mlx_tune.cpt_trainer import CPTTrainer

        model = self._make_mock_model(
            target_modules=["q_proj", "embed_tokens", "lm_head"]
        )
        dataset = [{"text": "hello"}]

        config = CPTConfig(
            include_embeddings=True,
            max_steps=10,
            output_dir=tempfile.mkdtemp(),
        )

        trainer = CPTTrainer(model=model, train_dataset=dataset, args=config)

        targets = model.lora_config['target_modules']
        assert targets.count('embed_tokens') == 1
        assert targets.count('lm_head') == 1

        shutil.rmtree(trainer.output_dir, ignore_errors=True)

    def test_gradient_scale_factor(self):
        """Test gradient scale factor computation."""
        from mlx_tune import CPTConfig
        from mlx_tune.cpt_trainer import CPTTrainer

        model = self._make_mock_model()
        dataset = [{"text": "hello"}]

        config = CPTConfig(
            learning_rate=5e-5,
            embedding_learning_rate=5e-6,
            max_steps=10,
            output_dir=tempfile.mkdtemp(),
        )

        trainer = CPTTrainer(model=model, train_dataset=dataset, args=config)

        assert abs(trainer._embedding_grad_scale - 0.1) < 1e-10
        assert trainer._use_decoupled_lr is True

        shutil.rmtree(trainer.output_dir, ignore_errors=True)

    def test_same_lr_no_decoupled(self):
        """Test that same LR doesn't use decoupled path."""
        from mlx_tune import CPTConfig
        from mlx_tune.cpt_trainer import CPTTrainer

        model = self._make_mock_model()
        dataset = [{"text": "hello"}]

        config = CPTConfig(
            learning_rate=5e-5,
            embedding_learning_rate=5e-5,
            max_steps=10,
            output_dir=tempfile.mkdtemp(),
        )

        trainer = CPTTrainer(model=model, train_dataset=dataset, args=config)

        assert abs(trainer._embedding_grad_scale - 1.0) < 1e-8
        assert trainer._use_decoupled_lr is False

        shutil.rmtree(trainer.output_dir, ignore_errors=True)

    def test_data_preparation(self):
        """Test CPT data preparation outputs correct format."""
        from mlx_tune import CPTConfig
        from mlx_tune.cpt_trainer import CPTTrainer
        import json

        model = self._make_mock_model()
        dataset = [
            {"text": "Document one about AI."},
            {"text": "Document two about ML."},
            {"text": "Document three about NLP."},
        ]

        tmpdir = tempfile.mkdtemp()
        config = CPTConfig(max_steps=10, output_dir=tmpdir)

        trainer = CPTTrainer(model=model, train_dataset=dataset, args=config)
        data_dir = trainer._prepare_training_data()

        # Verify train.jsonl was created with correct format
        train_file = Path(data_dir) / "train.jsonl"
        assert train_file.exists()

        with open(train_file) as f:
            lines = [json.loads(l) for l in f]

        assert len(lines) == 3
        assert all('text' in line for line in lines)
        assert lines[0]['text'] == "Document one about AI."

        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_data_preparation_custom_field(self):
        """Test CPT data prep with custom text field."""
        from mlx_tune import CPTConfig
        from mlx_tune.cpt_trainer import CPTTrainer
        import json

        model = self._make_mock_model()
        dataset = [
            {"content": "Custom field content."},
            {"content": "More content."},
        ]

        tmpdir = tempfile.mkdtemp()
        config = CPTConfig(
            max_steps=10,
            output_dir=tmpdir,
            dataset_text_field="content",
        )

        trainer = CPTTrainer(model=model, train_dataset=dataset, args=config)
        data_dir = trainer._prepare_training_data()

        train_file = Path(data_dir) / "train.jsonl"
        with open(train_file) as f:
            lines = [json.loads(l) for l in f]

        assert len(lines) == 2
        assert lines[0]['text'] == "Custom field content."

        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_iters_from_max_steps(self):
        """Test iterations computed from max_steps."""
        from mlx_tune import CPTConfig
        from mlx_tune.cpt_trainer import CPTTrainer

        model = self._make_mock_model()
        dataset = [{"text": f"doc {i}"} for i in range(100)]

        config = CPTConfig(max_steps=50, output_dir=tempfile.mkdtemp())
        trainer = CPTTrainer(model=model, train_dataset=dataset, args=config)

        assert trainer.iters == 50

        shutil.rmtree(trainer.output_dir, ignore_errors=True)

    def test_iters_from_epochs(self):
        """Test iterations computed from dataset size and epochs."""
        from mlx_tune import CPTConfig
        from mlx_tune.cpt_trainer import CPTTrainer

        model = self._make_mock_model()
        dataset = [{"text": f"doc {i}"} for i in range(100)]

        config = CPTConfig(
            num_train_epochs=2,
            per_device_train_batch_size=4,
            output_dir=tempfile.mkdtemp(),
        )
        trainer = CPTTrainer(model=model, train_dataset=dataset, args=config)

        # 100 / 4 * 2 = 50
        assert trainer.iters == 50

        shutil.rmtree(trainer.output_dir, ignore_errors=True)


class TestCPTImports:
    """Test CPT imports from mlx_tune."""

    def test_import_cpt_trainer(self):
        """Test CPTTrainer is importable from mlx_tune."""
        from mlx_tune import CPTTrainer
        assert CPTTrainer is not None

    def test_import_cpt_config(self):
        """Test CPTConfig is importable from mlx_tune."""
        from mlx_tune import CPTConfig
        assert CPTConfig is not None

    def test_in_all(self):
        """Test CPT classes are in __all__."""
        import mlx_tune
        assert "CPTTrainer" in mlx_tune.__all__
        assert "CPTConfig" in mlx_tune.__all__
