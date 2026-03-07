"""
Unit tests for FastLanguageModel
"""

import pytest
from mlx_tune import FastLanguageModel


class TestFastLanguageModel:
    """Test cases for FastLanguageModel class"""

    @pytest.fixture
    def model_name(self):
        """Fixture providing a small test model"""
        return "mlx-community/Llama-3.2-1B-Instruct-4bit"

    def test_from_pretrained_loads_model(self, model_name):
        """Test that from_pretrained successfully loads a model"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            load_in_4bit=True,
        )

        assert model is not None
        assert tokenizer is not None
        assert hasattr(model, 'model')
        assert hasattr(model, 'tokenizer')
        assert model.max_seq_length == 512
        assert model.model_name == model_name

    def test_from_pretrained_with_custom_params(self, model_name):
        """Test from_pretrained with various parameter combinations"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=1024,
            load_in_4bit=True,
            trust_remote_code=False,
        )

        assert model is not None
        assert model.max_seq_length == 1024

    def test_get_peft_model_configures_lora(self, model_name):
        """Test that get_peft_model properly configures LoRA"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            load_in_4bit=True,
        )

        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
        )

        # Check LoRA configuration
        assert hasattr(model, 'lora_config')
        assert model.lora_enabled is True
        assert model.lora_config['r'] == 8
        assert model.lora_config['lora_alpha'] == 16
        assert model.lora_config['lora_dropout'] == 0.1
        assert model.lora_config['target_modules'] == ["q_proj", "v_proj"]
        assert model.lora_config['bias'] == "none"

    def test_get_peft_model_with_default_modules(self, model_name):
        """Test that get_peft_model uses default target modules when not specified"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(model, r=16)

        assert hasattr(model, 'lora_config')
        assert model.lora_enabled is True
        # Default modules should be set
        assert len(model.lora_config['target_modules']) > 0
        expected_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        assert model.lora_config['target_modules'] == expected_modules

    def test_for_inference_enables_inference_mode(self, model_name):
        """Test that for_inference enables inference mode"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            load_in_4bit=True,
        )

        # Enable inference mode
        FastLanguageModel.for_inference(model)

        assert hasattr(model, 'inference_mode')
        assert model.inference_mode is True
        assert model.use_cache is True

    def test_for_inference_with_no_cache(self, model_name):
        """Test for_inference with caching disabled"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            load_in_4bit=True,
        )

        FastLanguageModel.for_inference(model, use_cache=False)

        assert model.inference_mode is True
        assert model.use_cache is False

    def test_tokenizer_functionality(self, model_name):
        """Test that tokenizer works correctly"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            load_in_4bit=True,
        )

        # Test encoding
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

        # Test decoding
        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, str)
        assert "Hello" in decoded or "hello" in decoded

    def test_model_wrapper_attributes(self, model_name):
        """Test that model wrapper has expected attributes"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            load_in_4bit=True,
        )

        # Check wrapper attributes
        assert hasattr(model, 'model')
        assert hasattr(model, 'tokenizer')
        assert hasattr(model, 'max_seq_length')
        assert hasattr(model, 'model_name')
        assert hasattr(model, 'lora_config')
        assert hasattr(model, 'lora_enabled')
        assert hasattr(model, 'inference_mode')
        assert hasattr(model, 'use_cache')

        # Check initial values
        assert model.lora_config is None
        assert model.lora_enabled is False
        assert model.inference_mode is False
        assert model.use_cache is True

    def test_full_workflow(self, model_name):
        """Test complete workflow: load, configure LoRA, enable inference"""
        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            load_in_4bit=True,
        )

        # Configure LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
        )

        # Enable inference
        FastLanguageModel.for_inference(model)

        # Verify all configurations
        assert model.max_seq_length == 512
        assert model.lora_enabled is True
        assert model.lora_config['r'] == 8
        assert model.inference_mode is True


class TestMLXModelWrapper:
    """Test cases for MLXModelWrapper class"""

    @pytest.fixture
    def wrapped_model(self):
        """Fixture providing a wrapped model"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
            max_seq_length=512,
            load_in_4bit=True,
        )
        return model, tokenizer

    def test_configure_lora_method(self, wrapped_model):
        """Test configure_lora method"""
        model, _ = wrapped_model

        model.configure_lora(
            r=16,
            target_modules=["q_proj"],
            lora_alpha=32,
        )

        assert model.lora_config is not None
        assert model.lora_config['r'] == 16
        assert model.lora_config['lora_alpha'] == 32

    def test_enable_inference_mode_method(self, wrapped_model):
        """Test enable_inference_mode method"""
        model, _ = wrapped_model

        model.enable_inference_mode(use_cache=True)

        assert model.inference_mode is True
        assert model.use_cache is True

    def test_wrapper_forwards_cache_argument(self):
        """Test that the wrapper exposes and forwards cache for RL decoding."""
        from mlx_tune.model import MLXModelWrapper

        class MockCacheModel:
            def __init__(self):
                self.calls = []

            def __call__(self, x, cache=None):
                self.calls.append((x, cache))
                return ("logits", {"seen": True}) if cache is not None else "logits"

        wrapped = MLXModelWrapper(
            model=MockCacheModel(),
            tokenizer=None,
            max_seq_length=128,
            model_name="mock",
        )

        assert wrapped("tokens") == "logits"
        assert wrapped("tokens", cache={"kv": 1}) == ("logits", {"seen": True})
        assert wrapped.forward_with_cache("tokens", cache={"kv": 2}) == ("logits", {"seen": True})
        assert wrapped.model.calls == [
            ("tokens", None),
            ("tokens", {"kv": 1}),
            ("tokens", {"kv": 2}),
        ]


class TestGGUFExportFix:
    """Test cases for GGUF export fix (GitHub issue #3).

    The issue was that save_pretrained_gguf was using the output directory
    as the model path, causing FileNotFoundError for config.json.
    The fix ensures we use the original model path (model_name).
    """

    @pytest.fixture
    def model_with_lora(self):
        """Fixture providing a model with LoRA configured."""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
            max_seq_length=512,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(model, r=8)
        return model, tokenizer

    def test_model_name_preserved(self, model_with_lora):
        """Test that model_name is preserved after loading."""
        model, _ = model_with_lora

        assert model.model_name == "mlx-community/Llama-3.2-1B-Instruct-4bit"

    def test_model_name_not_none(self, model_with_lora):
        """Test that model_name is not None (required for GGUF export)."""
        model, _ = model_with_lora

        assert model.model_name is not None, \
            "model_name should not be None - required for GGUF export"

    def test_model_config_preserved(self, model_with_lora):
        """Test that model config is preserved for GGUF export."""
        model, _ = model_with_lora

        # Config should be stored from mlx_load with return_config=True
        assert hasattr(model, 'config'), "Model should have config attribute"
        assert model.config is not None, "Config should not be None"

    def test_save_pretrained_gguf_requires_model_name(self):
        """Test that save_pretrained_gguf fails gracefully without model_name."""
        from mlx_tune.model import MLXModelWrapper

        # Create a wrapper without model_name
        class MockModel:
            pass

        wrapper = MLXModelWrapper(
            model=MockModel(),
            tokenizer=None,
            max_seq_length=512,
            model_name=None,  # No model name!
        )

        with pytest.raises(ValueError) as excinfo:
            wrapper.save_pretrained_gguf("output", None)

        assert "model_name" in str(excinfo.value).lower()

    def test_adapter_path_tracking(self, model_with_lora):
        """Test that adapter path can be set and retrieved."""
        model, _ = model_with_lora

        # Set adapter path
        model.set_adapter_path("/path/to/adapters")

        assert model.get_adapter_path() is not None
        assert str(model.get_adapter_path()) == "/path/to/adapters"

    def test_lora_applied_tracking(self, model_with_lora):
        """Test that _lora_applied flag is tracked correctly."""
        model, _ = model_with_lora

        # After get_peft_model, LoRA is configured but not applied yet
        assert model.lora_enabled is True
        assert model._lora_applied is False

        # Apply LoRA
        model._apply_lora()

        assert model._lora_applied is True
