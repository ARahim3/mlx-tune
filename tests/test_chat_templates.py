"""
Tests for chat_templates module - dataset format detection and conversion.

These tests ensure that various dataset formats are correctly detected and
converted to mlx-lm compatible formats.
"""

import pytest
from datasets import Dataset

from mlx_tune.chat_templates import (
    detect_dataset_format,
    standardize_sharegpt,
    convert_to_mlx_format,
    get_formatting_func,
    alpaca_to_text,
    ALPACA_TEMPLATE,
    ALPACA_TEMPLATE_NO_INPUT,
    # New chat template functions
    get_chat_template,
    list_chat_templates,
    get_template_info,
    get_template_for_model,
    CHAT_TEMPLATES,
    TEMPLATE_ALIASES,
    DEFAULT_SYSTEM_MESSAGES,
    ChatTemplateEntry,
    # Response-only training functions
    train_on_responses_only,
    _get_template_parts,
    get_response_template_ids,
)


class TestDetectDatasetFormat:
    """Test dataset format detection."""

    def test_detect_text_format(self):
        sample = {"text": "Hello world"}
        assert detect_dataset_format(sample) == "text"

    def test_detect_alpaca_format(self):
        sample = {
            "instruction": "Translate to French",
            "input": "Hello",
            "output": "Bonjour"
        }
        assert detect_dataset_format(sample) == "alpaca"

    def test_detect_alpaca_without_input(self):
        sample = {
            "instruction": "Tell me a joke",
            "output": "Why did the chicken..."
        }
        assert detect_dataset_format(sample) == "alpaca"

    def test_detect_sharegpt_format(self):
        sample = {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there!"}
            ]
        }
        assert detect_dataset_format(sample) == "sharegpt"

    def test_detect_chatml_format(self):
        sample = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"}
            ]
        }
        assert detect_dataset_format(sample) == "chatml"

    def test_detect_completions_format(self):
        sample = {
            "prompt": "What is 2+2?",
            "completion": "4"
        }
        assert detect_dataset_format(sample) == "completions"

    def test_detect_unknown_format(self):
        sample = {"foo": "bar", "baz": 123}
        assert detect_dataset_format(sample) == "unknown"


class TestAlpacaToText:
    """Test Alpaca format to text conversion."""

    def test_alpaca_with_input(self):
        sample = {
            "instruction": "Translate to French",
            "input": "Hello",
            "output": "Bonjour"
        }
        text = alpaca_to_text(sample)

        assert "Translate to French" in text
        assert "Hello" in text
        assert "Bonjour" in text
        assert "### Instruction:" in text
        assert "### Input:" in text
        assert "### Response:" in text

    def test_alpaca_without_input(self):
        sample = {
            "instruction": "Tell me a joke",
            "input": "",
            "output": "Why did the chicken cross the road?"
        }
        text = alpaca_to_text(sample)

        assert "Tell me a joke" in text
        assert "Why did the chicken" in text
        # Should NOT have ### Input: section when input is empty
        assert "### Input:" not in text

    def test_alpaca_custom_template(self):
        sample = {
            "instruction": "Do something",
            "input": "with this",
            "output": "done"
        }
        template = "Q: {instruction} {input}\nA: {output}"
        text = alpaca_to_text(sample, template=template)

        assert text == "Q: Do something with this\nA: done"


class TestStandardizeSharegpt:
    """Test ShareGPT to ChatML conversion."""

    def test_sharegpt_to_chatml(self):
        data = {
            "conversations": [
                [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi there!"}
                ]
            ]
        }
        dataset = Dataset.from_dict(data)
        converted = standardize_sharegpt(dataset)

        assert "messages" in converted[0]
        messages = converted[0]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"

    def test_sharegpt_with_system(self):
        data = {
            "conversations": [
                [
                    {"from": "system", "value": "You are helpful"},
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi!"}
                ]
            ]
        }
        dataset = Dataset.from_dict(data)
        converted = standardize_sharegpt(dataset)

        messages = converted[0]["messages"]
        assert len(messages) == 3
        assert messages[0]["role"] == "system"


class TestConvertToMlxFormat:
    """Test full dataset conversion to mlx-lm format."""

    def test_convert_alpaca_to_text(self):
        """Test that yahma/alpaca-cleaned style data is converted properly."""
        data = {
            "instruction": ["Give tips for health", "Translate hello"],
            "input": ["", "to French"],
            "output": ["Eat well, sleep well", "Bonjour"]
        }
        dataset = Dataset.from_dict(data)

        # Mock tokenizer
        class MockTokenizer:
            pass

        converted = convert_to_mlx_format(dataset, MockTokenizer(), output_format='text')

        # Should have 'text' field
        assert "text" in converted[0]
        assert "Eat well" in converted[0]["text"]
        assert "Give tips" in converted[0]["text"]

    def test_convert_sharegpt_to_chat(self):
        """Test ShareGPT to chat format conversion."""
        data = {
            "conversations": [
                [
                    {"from": "human", "value": "What is AI?"},
                    {"from": "gpt", "value": "AI is..."}
                ]
            ]
        }
        dataset = Dataset.from_dict(data)

        class MockTokenizer:
            pass

        converted = convert_to_mlx_format(dataset, MockTokenizer(), output_format='chat')

        assert "messages" in converted[0]
        assert converted[0]["messages"][0]["role"] == "user"

    def test_text_passthrough(self):
        """Test that text format passes through unchanged."""
        data = {"text": ["Hello world", "Test sample"]}
        dataset = Dataset.from_dict(data)

        class MockTokenizer:
            pass

        converted = convert_to_mlx_format(dataset, MockTokenizer(), output_format='text')

        assert converted[0]["text"] == "Hello world"


class TestGetFormattingFunc:
    """Test formatting function generation."""

    def test_formatting_func_alpaca(self):
        class MockTokenizer:
            pass

        func = get_formatting_func(MockTokenizer(), dataset_format='alpaca')

        sample = {
            "instruction": "Test instruction",
            "input": "Test input",
            "output": "Test output"
        }

        result = func(sample)
        assert isinstance(result, str)
        assert "Test instruction" in result
        assert "Test output" in result

    def test_formatting_func_text(self):
        class MockTokenizer:
            pass

        func = get_formatting_func(MockTokenizer(), dataset_format='text')

        sample = {"text": "Hello world"}
        result = func(sample)
        assert result == "Hello world"

    def test_formatting_func_auto_detect(self):
        """Test auto-detection in formatting function."""
        class MockTokenizer:
            pass

        func = get_formatting_func(MockTokenizer(), dataset_format='auto')

        # Should detect alpaca format
        alpaca_sample = {
            "instruction": "Do something",
            "input": "",
            "output": "Done"
        }
        result = func(alpaca_sample)
        assert "Do something" in result
        assert "Done" in result


class TestImports:
    """Test that all exports work correctly."""

    def test_imports_from_package(self):
        """Test importing from main package."""
        from mlx_tune import (
            detect_dataset_format,
            standardize_sharegpt,
            convert_to_mlx_format,
            get_formatting_func,
            alpaca_to_text,
        )

        # All should be callable
        assert callable(detect_dataset_format)
        assert callable(standardize_sharegpt)
        assert callable(convert_to_mlx_format)
        assert callable(get_formatting_func)
        assert callable(alpaca_to_text)

    def test_imports_chat_template_functions(self):
        """Test importing new chat template functions from main package."""
        from mlx_tune import (
            get_chat_template,
            list_chat_templates,
            get_template_info,
            get_template_for_model,
            CHAT_TEMPLATES,
            TEMPLATE_ALIASES,
            DEFAULT_SYSTEM_MESSAGES,
            ChatTemplateEntry,
        )

        # All should be callable or proper types
        assert callable(get_chat_template)
        assert callable(list_chat_templates)
        assert callable(get_template_info)
        assert callable(get_template_for_model)
        assert isinstance(CHAT_TEMPLATES, dict)
        assert isinstance(TEMPLATE_ALIASES, dict)
        assert isinstance(DEFAULT_SYSTEM_MESSAGES, dict)


# =============================================================================
# TESTS FOR GET_CHAT_TEMPLATE (Phase 1.1)
# =============================================================================

class TestChatTemplateRegistry:
    """Test the CHAT_TEMPLATES registry."""

    def test_registry_has_required_templates(self):
        """Verify all 15 required templates are present."""
        required_templates = [
            "llama-3", "llama-3.1",
            "chatml",
            "gemma-2", "gemma-3", "gemma-4",
            "qwen-2.5", "qwen-3",
            "phi-3", "phi-3.5", "phi-4",
            "mistral", "mistral-nemo",
            "deepseek-v2",
            "alpaca", "vicuna", "zephyr",
        ]
        for template in required_templates:
            assert template in CHAT_TEMPLATES, f"Missing template: {template}"

    def test_registry_entries_are_valid(self):
        """Verify all registry entries have required fields."""
        for name, entry in CHAT_TEMPLATES.items():
            assert isinstance(entry, ChatTemplateEntry), f"{name} is not ChatTemplateEntry"
            assert isinstance(entry.template, str), f"{name} template is not str"
            assert len(entry.template) > 0, f"{name} has empty template"
            assert isinstance(entry.eos_token, str), f"{name} eos_token is not str"
            assert isinstance(entry.bos_token, str), f"{name} bos_token is not str"
            assert isinstance(entry.stop_token, str), f"{name} stop_token is not str"

    def test_template_aliases_resolve(self):
        """Test that aliases resolve to valid templates."""
        for alias, target in TEMPLATE_ALIASES.items():
            assert target in CHAT_TEMPLATES, f"Alias '{alias}' points to non-existent '{target}'"


class TestListChatTemplates:
    """Test list_chat_templates function."""

    def test_returns_sorted_list(self):
        """Verify list is sorted alphabetically."""
        templates = list_chat_templates()
        assert templates == sorted(templates)

    def test_returns_all_templates(self):
        """Verify all templates are listed."""
        templates = list_chat_templates()
        assert len(templates) == len(CHAT_TEMPLATES)
        for name in CHAT_TEMPLATES.keys():
            assert name in templates


class TestGetTemplateInfo:
    """Test get_template_info function."""

    def test_get_llama3_info(self):
        """Test getting llama-3 template info."""
        info = get_template_info("llama-3")
        assert info["name"] == "llama-3"
        assert info["eos_token"] == "<|eot_id|>"
        assert info["bos_token"] == "<|begin_of_text|>"
        assert info["stop_token"] == "<|eot_id|>"
        assert "template_preview" in info

    def test_get_chatml_info(self):
        """Test getting chatml template info."""
        info = get_template_info("chatml")
        assert info["name"] == "chatml"
        assert info["eos_token"] == "<|im_end|>"

    def test_alias_resolves(self):
        """Test that alias is resolved in info."""
        info = get_template_info("llama3")  # alias
        assert info["name"] == "llama-3"

    def test_unknown_template_raises(self):
        """Test that unknown template raises ValueError."""
        with pytest.raises(ValueError, match="Unknown chat template"):
            get_template_info("nonexistent-template")


class TestGetTemplateForModel:
    """Test get_template_for_model function."""

    def test_detect_llama3(self):
        """Test detection of Llama 3 models."""
        assert get_template_for_model("meta-llama/Llama-3-8B-Instruct") == "llama-3"
        assert get_template_for_model("mlx-community/Llama-3.2-1B-Instruct-4bit") == "llama-3.1"
        assert get_template_for_model("mlx-community/Llama-3.3-70B-Instruct-4bit") == "llama-3.1"

    def test_detect_gemma(self):
        """Test detection of Gemma models."""
        assert get_template_for_model("google/gemma-2-9b-it") == "gemma-2"
        assert get_template_for_model("google/gemma-3-27b-it") == "gemma-3"
        # Gemma 4 — all 4 variants
        assert get_template_for_model("google/gemma-4-E2B-it") == "gemma-4"
        assert get_template_for_model("google/gemma-4-E4B-it") == "gemma-4"
        assert get_template_for_model("google/gemma-4-27b-it") == "gemma-4"
        assert get_template_for_model("google/gemma-4-31b-it") == "gemma-4"
        assert get_template_for_model("mlx-community/gemma-4-4b-it-4bit") == "gemma-4"

    def test_detect_qwen(self):
        """Test detection of Qwen models."""
        assert get_template_for_model("Qwen/Qwen2.5-7B-Instruct") == "qwen-2.5"
        assert get_template_for_model("Qwen/Qwen3-8B") == "qwen-3"

    def test_detect_phi(self):
        """Test detection of Phi models."""
        assert get_template_for_model("microsoft/Phi-3-mini-4k-instruct") == "phi-3"
        assert get_template_for_model("microsoft/phi-3.5-mini-instruct") == "phi-3.5"
        assert get_template_for_model("microsoft/phi-4") == "phi-4"

    def test_detect_mistral(self):
        """Test detection of Mistral models."""
        assert get_template_for_model("mistralai/Mistral-7B-Instruct-v0.3") == "mistral"
        assert get_template_for_model("mistralai/Mistral-Nemo-Instruct-2407") == "mistral-nemo"

    def test_detect_deepseek(self):
        """Test detection of DeepSeek models."""
        assert get_template_for_model("deepseek-ai/DeepSeek-V2-Chat") == "deepseek-v2"

    def test_detect_lfm2(self):
        """Test detection of Liquid Foundation Models."""
        assert get_template_for_model("LiquidAI/LFM2-1.2B") == "chatml"
        assert get_template_for_model("LiquidAI/LFM2.5-1.2B-Instruct") == "chatml"
        assert get_template_for_model("mlx-community/LFM2-2.6B-4bit") == "chatml"
        assert get_template_for_model("lmstudio-community/LFM2.5-1.2B-Instruct-MLX-8bit") == "chatml"
        assert get_template_for_model("LiquidAI/LFM2.5-1.2B-Thinking") == "chatml"

    def test_lfm2_aliases(self):
        """Test LFM2 template aliases resolve to chatml."""
        assert TEMPLATE_ALIASES["lfm2"] == "chatml"
        assert TEMPLATE_ALIASES["lfm25"] == "chatml"
        assert TEMPLATE_ALIASES["lfm2.5"] == "chatml"
        assert TEMPLATE_ALIASES["liquid"] == "chatml"

    def test_default_to_chatml(self):
        """Test that unknown models default to chatml."""
        assert get_template_for_model("unknown/some-model") == "chatml"


class TestGetChatTemplate:
    """Test get_chat_template function."""

    def test_apply_llama3_template(self):
        """Test applying llama-3 template to mock tokenizer."""
        class MockTokenizer:
            name_or_path = "test-model"
            chat_template = None

        tokenizer = MockTokenizer()
        result = get_chat_template(tokenizer, chat_template="llama-3")

        assert result is tokenizer
        assert tokenizer.chat_template is not None
        assert "<|start_header_id|>" in tokenizer.chat_template
        assert tokenizer._unsloth_chat_template_name == "llama-3"

    def test_apply_chatml_template(self):
        """Test applying chatml template."""
        class MockTokenizer:
            name_or_path = "test-model"
            chat_template = None

        tokenizer = MockTokenizer()
        result = get_chat_template(tokenizer, chat_template="chatml")

        assert "<|im_start|>" in tokenizer.chat_template
        assert "<|im_end|>" in tokenizer.chat_template

    def test_apply_gemma_template(self):
        """Test applying gemma template."""
        class MockTokenizer:
            name_or_path = "test-model"
            chat_template = None

        tokenizer = MockTokenizer()
        result = get_chat_template(tokenizer, chat_template="gemma-2")

        assert "<start_of_turn>" in tokenizer.chat_template
        assert "<end_of_turn>" in tokenizer.chat_template

    def test_alias_works(self):
        """Test that template alias works."""
        class MockTokenizer:
            name_or_path = "test-model"
            chat_template = None

        tokenizer = MockTokenizer()
        result = get_chat_template(tokenizer, chat_template="llama3")  # alias

        assert tokenizer._unsloth_chat_template_name == "llama-3"

    def test_auto_detection(self, capsys):
        """Test auto-detection of template from model name."""
        class MockTokenizer:
            name_or_path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
            chat_template = None

        tokenizer = MockTokenizer()
        result = get_chat_template(tokenizer, chat_template="auto")

        captured = capsys.readouterr()
        assert "Auto-detected" in captured.out
        assert tokenizer._unsloth_chat_template_name == "llama-3.1"

    def test_unknown_template_raises(self):
        """Test that unknown template raises ValueError."""
        class MockTokenizer:
            name_or_path = "test-model"
            chat_template = None

        tokenizer = MockTokenizer()
        with pytest.raises(ValueError, match="Unknown chat template"):
            get_chat_template(tokenizer, chat_template="nonexistent")

    def test_mapping_stored(self):
        """Test that mapping is stored on tokenizer."""
        class MockTokenizer:
            name_or_path = "test-model"
            chat_template = None

        tokenizer = MockTokenizer()
        mapping = {"role": "from", "content": "value"}
        result = get_chat_template(tokenizer, chat_template="chatml", mapping=mapping)

        assert tokenizer._unsloth_mapping == mapping

    def test_system_message_stored(self):
        """Test that custom system message is stored."""
        class MockTokenizer:
            name_or_path = "test-model"
            chat_template = None

        tokenizer = MockTokenizer()
        result = get_chat_template(
            tokenizer,
            chat_template="llama-3",
            system_message="Custom system prompt"
        )

        assert tokenizer._unsloth_system_message == "Custom system prompt"

    def test_default_system_message(self):
        """Test that default system message is applied for supported models."""
        class MockTokenizer:
            name_or_path = "test-model"
            chat_template = None

        tokenizer = MockTokenizer()
        result = get_chat_template(tokenizer, chat_template="llama-3")

        assert hasattr(tokenizer, '_unsloth_system_message')
        assert tokenizer._unsloth_system_message == "You are a helpful assistant."

    def test_stop_token_stored(self):
        """Test that stop token is stored on tokenizer."""
        class MockTokenizer:
            name_or_path = "test-model"
            chat_template = None

        tokenizer = MockTokenizer()
        result = get_chat_template(tokenizer, chat_template="llama-3")

        assert tokenizer._unsloth_stop_token == "<|eot_id|>"


class TestTemplateContents:
    """Test that templates contain expected content."""

    def test_llama3_has_special_tokens(self):
        """Test llama-3 template contains required tokens."""
        template = CHAT_TEMPLATES["llama-3"].template
        assert "<|start_header_id|>" in template
        assert "<|end_header_id|>" in template
        assert "<|eot_id|>" in template
        assert "bos_token" in template  # Uses bos_token variable

    def test_chatml_has_special_tokens(self):
        """Test chatml template contains required tokens."""
        template = CHAT_TEMPLATES["chatml"].template
        assert "<|im_start|>" in template
        assert "<|im_end|>" in template

    def test_gemma_maps_assistant_to_model(self):
        """Test gemma template uses 'model' role for assistant."""
        template = CHAT_TEMPLATES["gemma-2"].template
        assert "model" in template  # Gemma uses "model" not "assistant"

    def test_phi4_has_separator(self):
        """Test phi-4 template uses im_sep."""
        template = CHAT_TEMPLATES["phi-4"].template
        assert "<|im_sep|>" in template

    def test_mistral_has_inst_tokens(self):
        """Test mistral template has [INST] tokens."""
        template = CHAT_TEMPLATES["mistral"].template
        assert "[INST]" in template
        assert "[/INST]" in template

    def test_qwen3_has_thinking_tokens(self):
        """Test qwen-3 template supports thinking."""
        template = CHAT_TEMPLATES["qwen-3"].template
        assert "<think>" in template
        assert "</think>" in template

    def test_deepseek_has_unicode_tokens(self):
        """Test deepseek template has unicode tokens."""
        template = CHAT_TEMPLATES["deepseek-v2"].template
        assert "<｜User｜>" in template
        assert "<｜Assistant｜>" in template


# =============================================================================
# TESTS FOR TRAIN_ON_RESPONSES_ONLY (Phase 1.2)
# =============================================================================

class TestGetTemplateParts:
    """Test _get_template_parts function."""

    def test_llama3_parts(self):
        """Test llama-3 template parts."""
        parts = _get_template_parts("llama-3")
        assert parts["instruction_part"] == "<|start_header_id|>user<|end_header_id|>\n\n"
        assert parts["response_part"] == "<|start_header_id|>assistant<|end_header_id|>\n\n"

    def test_chatml_parts(self):
        """Test chatml template parts."""
        parts = _get_template_parts("chatml")
        assert parts["instruction_part"] == "<|im_start|>user\n"
        assert parts["response_part"] == "<|im_start|>assistant\n"

    def test_gemma_parts(self):
        """Test gemma template parts (uses 'model' not 'assistant')."""
        parts = _get_template_parts("gemma-2")
        assert parts["instruction_part"] == "<start_of_turn>user\n"
        assert parts["response_part"] == "<start_of_turn>model\n"

    def test_gemma4_parts(self):
        """Test gemma-4 template parts (same format as gemma-2/3)."""
        parts = _get_template_parts("gemma-4")
        assert parts["instruction_part"] == "<start_of_turn>user\n"
        assert parts["response_part"] == "<start_of_turn>model\n"

    def test_gemma4_alias(self):
        """Test gemma4 alias resolves to gemma-4."""
        from mlx_tune.chat_templates import TEMPLATE_ALIASES
        assert TEMPLATE_ALIASES.get("gemma4") == "gemma-4"

    def test_phi4_parts(self):
        """Test phi-4 template parts (uses im_sep)."""
        parts = _get_template_parts("phi-4")
        assert "<|im_sep|>" in parts["response_part"]

    def test_mistral_parts(self):
        """Test mistral template parts."""
        parts = _get_template_parts("mistral")
        assert parts["instruction_part"] == "[INST]"
        assert parts["response_part"] == "[/INST]"

    def test_alpaca_parts(self):
        """Test alpaca template parts."""
        parts = _get_template_parts("alpaca")
        assert parts["instruction_part"] == "### Instruction:\n"
        assert parts["response_part"] == "### Response:\n"

    def test_alias_resolves(self):
        """Test that aliases resolve correctly."""
        parts_alias = _get_template_parts("llama3")
        parts_full = _get_template_parts("llama-3")
        assert parts_alias == parts_full

    def test_unknown_template(self):
        """Test unknown template returns None parts."""
        parts = _get_template_parts("nonexistent")
        assert parts["instruction_part"] is None
        assert parts["response_part"] is None

    def test_all_templates_have_parts(self):
        """Test all registered templates have parts defined."""
        for name in CHAT_TEMPLATES.keys():
            parts = _get_template_parts(name)
            assert parts["instruction_part"] is not None, f"{name} missing instruction_part"
            assert parts["response_part"] is not None, f"{name} missing response_part"


class TestTrainOnResponsesOnly:
    """Test train_on_responses_only function."""

    def test_basic_usage(self, capsys):
        """Test basic usage with explicit parts."""
        class MockTrainer:
            tokenizer = None

        trainer = MockTrainer()
        result = train_on_responses_only(
            trainer,
            instruction_part="<|user|>",
            response_part="<|assistant|>",
        )

        assert result is trainer
        assert trainer._train_on_responses_only is True
        assert trainer._instruction_part == "<|user|>"
        assert trainer._response_part == "<|assistant|>"

        captured = capsys.readouterr()
        assert "train_on_responses_only enabled" in captured.out

    def test_auto_detect_from_tokenizer(self, capsys):
        """Test auto-detection of parts from tokenizer template name."""
        class MockTokenizer:
            _unsloth_chat_template_name = "llama-3"

        class MockTrainer:
            tokenizer = MockTokenizer()

        trainer = MockTrainer()
        result = train_on_responses_only(trainer)

        assert trainer._instruction_part == "<|start_header_id|>user<|end_header_id|>\n\n"
        assert trainer._response_part == "<|start_header_id|>assistant<|end_header_id|>\n\n"

    def test_auto_detect_from_model_name(self, capsys):
        """Test auto-detection from tokenizer name_or_path."""
        class MockTokenizer:
            name_or_path = "mlx-community/Llama-3.2-1B-Instruct-4bit"

        class MockTrainer:
            tokenizer = MockTokenizer()

        trainer = MockTrainer()
        result = train_on_responses_only(trainer)

        # Should detect llama-3.1 template
        assert "user<|end_header_id|>" in trainer._instruction_part
        assert "assistant<|end_header_id|>" in trainer._response_part

    def test_partial_override(self, capsys):
        """Test partial override - provide one part, auto-detect the other."""
        class MockTokenizer:
            _unsloth_chat_template_name = "chatml"

        class MockTrainer:
            tokenizer = MockTokenizer()

        trainer = MockTrainer()
        result = train_on_responses_only(
            trainer,
            instruction_part="CUSTOM_USER",  # Override instruction
            # Let response_part be auto-detected
        )

        assert trainer._instruction_part == "CUSTOM_USER"
        assert trainer._response_part == "<|im_start|>assistant\n"  # Auto-detected


class TestGetResponseTemplateIds:
    """Test get_response_template_ids function."""

    def test_with_mock_tokenizer(self):
        """Test with a mock tokenizer that has encode method."""
        class MockTokenizer:
            def encode(self, text, add_special_tokens=True):
                # Simple mock that returns character codes
                return [ord(c) for c in text[:5]]

        tokenizer = MockTokenizer()
        ids = get_response_template_ids(tokenizer, "<|assistant|>")

        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_without_encode_method(self):
        """Test with tokenizer without encode method."""
        class MockTokenizer:
            pass

        tokenizer = MockTokenizer()
        ids = get_response_template_ids(tokenizer, "<|assistant|>")

        assert ids == []


class TestTrainOnResponsesOnlyImport:
    """Test train_on_responses_only import from main package."""

    def test_import_from_package(self):
        """Test importing from main package."""
        from mlx_tune import train_on_responses_only

        assert callable(train_on_responses_only)


# =============================================================================
# TESTS FOR PHASE 2: ADVANCED DATASET FEATURES
# =============================================================================

from mlx_tune.chat_templates import (
    to_sharegpt,
    apply_column_mapping,
    infer_column_mapping,
    HFDatasetConfig,
    load_dataset_with_config,
    standardize_sharegpt_enhanced,
)


class TestToSharegpt:
    """Test to_sharegpt function for multi-turn conversation merging."""

    def test_alpaca_to_sharegpt_single_turn(self):
        """Test converting Alpaca format to ShareGPT (single turn)."""
        data = {
            "instruction": ["Translate hello", "What is AI?"],
            "input": ["to French", ""],
            "output": ["Bonjour", "Artificial Intelligence"]
        }
        dataset = Dataset.from_dict(data)

        result = to_sharegpt(dataset, output_column_name="output", conversation_extension=1)

        assert "conversations" in result[0]
        assert len(result[0]["conversations"]) == 2  # user + assistant
        assert result[0]["conversations"][0]["from"] == "human"
        assert result[0]["conversations"][1]["from"] == "gpt"
        assert "Translate hello" in result[0]["conversations"][0]["value"]
        assert "Bonjour" in result[0]["conversations"][1]["value"]

    def test_alpaca_to_sharegpt_multi_turn(self):
        """Test merging multiple rows into multi-turn conversation."""
        data = {
            "instruction": ["Q1", "Q2", "Q3", "Q4"],
            "input": ["", "", "", ""],
            "output": ["A1", "A2", "A3", "A4"]
        }
        dataset = Dataset.from_dict(data)

        result = to_sharegpt(dataset, conversation_extension=2, random_state=42)

        # With conversation_extension=2 and 4 samples, we should get 2 conversations
        assert len(result) == 2
        # Each conversation should have 4 turns (2 user + 2 assistant)
        assert len(result[0]["conversations"]) == 4

    def test_chatml_to_sharegpt(self):
        """Test converting ChatML format to ShareGPT."""
        data = {
            "messages": [[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"}
            ]]
        }
        dataset = Dataset.from_dict(data)

        result = to_sharegpt(dataset)

        assert "conversations" in result[0]
        assert result[0]["conversations"][0]["from"] == "human"
        assert result[0]["conversations"][1]["from"] == "gpt"

    def test_completions_to_sharegpt(self):
        """Test converting completions format to ShareGPT."""
        data = {
            "prompt": ["What is 2+2?"],
            "completion": ["4"]
        }
        dataset = Dataset.from_dict(data)

        result = to_sharegpt(dataset)

        assert "conversations" in result[0]
        assert result[0]["conversations"][0]["value"] == "What is 2+2?"
        assert result[0]["conversations"][1]["value"] == "4"

    def test_with_column_mapping(self):
        """Test to_sharegpt with column mapping."""
        data = {
            "question": ["What is Python?"],
            "answer": ["A programming language"]
        }
        dataset = Dataset.from_dict(data)

        result = to_sharegpt(
            dataset,
            column_mapping={"instruction": "question", "output": "answer"},
            output_column_name="output"
        )

        assert "conversations" in result[0]
        assert "What is Python?" in result[0]["conversations"][0]["value"]

    def test_empty_dataset(self):
        """Test with empty dataset."""
        dataset = Dataset.from_dict({"instruction": [], "output": []})
        result = to_sharegpt(dataset)
        assert len(result) == 0

    def test_random_state_reproducibility(self):
        """Test that random_state produces reproducible results."""
        data = {
            "instruction": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"],
            "input": ["", "", "", "", "", ""],
            "output": ["A1", "A2", "A3", "A4", "A5", "A6"]
        }
        dataset = Dataset.from_dict(data)

        result1 = to_sharegpt(dataset, conversation_extension=2, random_state=42)
        result2 = to_sharegpt(dataset, conversation_extension=2, random_state=42)

        # Results should be identical with same seed
        assert len(result1) == len(result2)


class TestApplyColumnMapping:
    """Test apply_column_mapping function."""

    def test_basic_mapping(self):
        """Test basic column renaming."""
        data = {"question": ["Q1", "Q2"], "answer": ["A1", "A2"]}
        dataset = Dataset.from_dict(data)

        result = apply_column_mapping(dataset, {
            "instruction": "question",
            "output": "answer"
        })

        assert "instruction" in result.column_names
        assert "output" in result.column_names
        assert "question" not in result.column_names
        assert "answer" not in result.column_names

    def test_partial_mapping(self):
        """Test mapping when only some columns exist."""
        data = {"question": ["Q1"], "context": ["C1"]}
        dataset = Dataset.from_dict(data)

        result = apply_column_mapping(dataset, {
            "instruction": "question",
            "output": "nonexistent"  # This column doesn't exist
        })

        assert "instruction" in result.column_names
        assert "context" in result.column_names

    def test_empty_mapping(self):
        """Test with empty mapping."""
        data = {"col1": ["v1"]}
        dataset = Dataset.from_dict(data)

        result = apply_column_mapping(dataset, {})

        assert list(result.column_names) == ["col1"]

    def test_none_mapping(self):
        """Test with None mapping."""
        data = {"col1": ["v1"]}
        dataset = Dataset.from_dict(data)

        result = apply_column_mapping(dataset, None)

        assert result is dataset


class TestInferColumnMapping:
    """Test infer_column_mapping function."""

    def test_infer_alpaca_mapping(self):
        """Test inferring mapping for Alpaca format."""
        data = {"question": ["Q"], "answer": ["A"]}
        dataset = Dataset.from_dict(data)

        mapping = infer_column_mapping(dataset, target_format="alpaca")

        assert mapping.get("instruction") == "question"
        assert mapping.get("output") == "answer"

    def test_infer_completions_mapping(self):
        """Test inferring mapping for completions format."""
        data = {"query": ["Q"], "response": ["R"]}
        dataset = Dataset.from_dict(data)

        mapping = infer_column_mapping(dataset, target_format="completions")

        assert mapping.get("prompt") == "query"
        assert mapping.get("completion") == "response"

    def test_no_mapping_needed(self):
        """Test when dataset already has correct columns."""
        data = {"instruction": ["I"], "output": ["O"]}
        dataset = Dataset.from_dict(data)

        mapping = infer_column_mapping(dataset, target_format="alpaca")

        # No mapping needed - columns already match
        assert "instruction" not in mapping
        assert "output" not in mapping


class TestHFDatasetConfig:
    """Test HFDatasetConfig class."""

    def test_config_creation(self):
        """Test creating a config."""
        config = HFDatasetConfig(
            path="yahma/alpaca-cleaned",
            train_split="train[:100]",
            valid_split="train[-10:]",
            column_mapping={"instruction": "question"},
            max_samples=50,
        )

        assert config.path == "yahma/alpaca-cleaned"
        assert config.train_split == "train[:100]"
        assert config.valid_split == "train[-10:]"
        assert config.column_mapping == {"instruction": "question"}
        assert config.max_samples == 50

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = HFDatasetConfig(
            path="test/dataset",
            conversation_extension=3,
        )

        d = config.to_dict()

        assert d["path"] == "test/dataset"
        assert d["conversation_extension"] == 3
        assert d["train_split"] == "train"

    def test_from_dict(self):
        """Test creating config from dictionary."""
        d = {
            "path": "test/dataset",
            "train_split": "train[:50]",
            "max_samples": 100,
        }

        config = HFDatasetConfig.from_dict(d)

        assert config.path == "test/dataset"
        assert config.train_split == "train[:50]"
        assert config.max_samples == 100

    def test_defaults(self):
        """Test default values."""
        config = HFDatasetConfig(path="test/dataset")

        assert config.train_split == "train"
        assert config.valid_split is None
        assert config.streaming is False
        assert config.conversation_extension == 1
        assert config.output_column == "output"


class TestStandardizeSharegptEnhanced:
    """Test standardize_sharegpt_enhanced function."""

    def test_basic_conversion(self):
        """Test basic ShareGPT to ChatML conversion."""
        data = {
            "conversations": [[
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi!"}
            ]]
        }
        dataset = Dataset.from_dict(data)

        result = standardize_sharegpt_enhanced(dataset)

        assert "messages" in result[0]
        assert result[0]["messages"][0]["role"] == "user"
        assert result[0]["messages"][1]["role"] == "assistant"

    def test_custom_role_mapping(self):
        """Test with custom role mapping."""
        data = {
            "conversations": [[
                {"from": "person", "value": "Hello"},
                {"from": "ai", "value": "Hi!"}
            ]]
        }
        dataset = Dataset.from_dict(data)

        result = standardize_sharegpt_enhanced(dataset)

        assert result[0]["messages"][0]["role"] == "user"
        assert result[0]["messages"][1]["role"] == "assistant"

    def test_different_content_field(self):
        """Test with non-standard content field."""
        data = {
            "conversations": [[
                {"from": "human", "text": "Hello"},
                {"from": "gpt", "text": "Hi!"}
            ]]
        }
        dataset = Dataset.from_dict(data)

        result = standardize_sharegpt_enhanced(dataset)

        assert result[0]["messages"][0]["content"] == "Hello"
        assert result[0]["messages"][1]["content"] == "Hi!"


class TestPhase2Imports:
    """Test Phase 2 imports from main package."""

    def test_imports_to_sharegpt(self):
        """Test importing to_sharegpt from main package."""
        from mlx_tune import to_sharegpt
        assert callable(to_sharegpt)

    def test_imports_column_mapping(self):
        """Test importing column mapping functions."""
        from mlx_tune import apply_column_mapping, infer_column_mapping
        assert callable(apply_column_mapping)
        assert callable(infer_column_mapping)

    def test_imports_hf_dataset_config(self):
        """Test importing HFDatasetConfig."""
        from mlx_tune import HFDatasetConfig, load_dataset_with_config
        assert HFDatasetConfig is not None
        assert callable(load_dataset_with_config)

    def test_imports_standardize_enhanced(self):
        """Test importing standardize_sharegpt_enhanced."""
        from mlx_tune import standardize_sharegpt_enhanced
        assert callable(standardize_sharegpt_enhanced)


class TestApplyPromptTemplate:
    """Test _apply_prompt_template helper function (via to_sharegpt)."""

    def test_basic_template(self):
        """Test basic template substitution through to_sharegpt."""
        data = {
            "name": ["Alice"],
            "age": [30],
            "output": ["Response"]
        }
        dataset = Dataset.from_dict(data)

        # Use merged_prompt with placeholders
        result = to_sharegpt(
            dataset,
            merged_prompt="User {name} is {age} years old",
            output_column_name="output"
        )

        # The user content should have the template applied
        assert "Alice" in result[0]["conversations"][0]["value"]
        assert "30" in result[0]["conversations"][0]["value"]

    def test_optional_sections(self):
        """Test optional sections with [[...]] syntax."""
        data = {
            "name": ["Bob"],
            "context": [""],  # Empty context
            "output": ["Done"]
        }
        dataset = Dataset.from_dict(data)

        # Optional section should be removed when context is empty
        result = to_sharegpt(
            dataset,
            merged_prompt="Hello {name}[[, context: {context}]]",
            output_column_name="output"
        )

        # The optional section with empty context should be omitted
        conv = result[0]["conversations"][0]["value"]
        assert "Bob" in conv
        assert "context:" not in conv


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
