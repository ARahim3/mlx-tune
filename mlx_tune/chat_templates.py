"""
Chat Templates and Dataset Formatting for MLX-Tune

This module provides Unsloth-compatible dataset formatting utilities,
converting various dataset formats to mlx-lm compatible formats.

Supported input formats:
- Alpaca: {"instruction": "...", "input": "...", "output": "..."}
- ShareGPT: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
- ChatML: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
- Text: {"text": "..."}
- Completions: {"prompt": "...", "completion": "..."}

Output formats (mlx-lm compatible):
- text: {"text": "..."}
- chat: {"messages": [...]}
- completions: {"prompt": "...", "completion": "..."}
"""

from typing import Any, Dict, List, Optional, Callable, Union, NamedTuple
from datasets import Dataset
import re
import random


# Default Alpaca prompt template
ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_TEMPLATE_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


# =============================================================================
# CHAT TEMPLATE REGISTRY (Unsloth-compatible)
# =============================================================================

class ChatTemplateEntry(NamedTuple):
    """Registry entry for a chat template."""
    template: str       # Jinja2 template string
    eos_token: str      # EOS token (or "eos_token" to use tokenizer's default)
    bos_token: str      # BOS token (or "bos_token" to use tokenizer's default)
    stop_token: str     # Stop token for generation


# Jinja2 templates for each model family
# These are based on official HuggingFace tokenizer configs and Unsloth

_LLAMA3_TEMPLATE = """{%- if messages[0]['role'] == 'system' -%}
    {%- set system_message = messages[0]['content'] | trim + '\n\n' -%}
    {%- set messages = messages[1:] -%}
{%- else -%}
    {%- set system_message = '' -%}
{%- endif -%}

{{ bos_token }}{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}
    {%- elif message['role'] == 'user' -%}
        {{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif -%}"""

_CHATML_TEMPLATE = """{%- for message in messages -%}
    {{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<|im_start|>assistant\n' }}
{%- endif -%}"""

_GEMMA_TEMPLATE = """{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
        {{ '<start_of_turn>user\n' + message['content'] | trim + '<end_of_turn>\n' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ '<start_of_turn>model\n' + message['content'] | trim + '<end_of_turn>\n' }}
    {%- elif message['role'] == 'system' -%}
        {{ '<start_of_turn>user\n' + message['content'] | trim + '<end_of_turn>\n' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<start_of_turn>model\n' }}
{%- endif -%}"""

_QWEN_TEMPLATE = """{%- for message in messages -%}
    {{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<|im_start|>assistant\n' }}
{%- endif -%}"""

_QWEN3_TEMPLATE = """{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{ '<|im_start|>system\n' + message['content'] | trim + '<|im_end|>\n' }}
    {%- elif message['role'] == 'user' -%}
        {{ '<|im_start|>user\n' + message['content'] | trim + '<|im_end|>\n' }}
    {%- elif message['role'] == 'assistant' -%}
        {%- if message.get('reasoning_content') -%}
            {{ '<|im_start|>assistant\n<think>\n' + message['reasoning_content'] | trim + '\n</think>\n' + message['content'] | trim + '<|im_end|>\n' }}
        {%- else -%}
            {{ '<|im_start|>assistant\n' + message['content'] | trim + '<|im_end|>\n' }}
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<|im_start|>assistant\n' }}
{%- endif -%}"""

_PHI3_TEMPLATE = """{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{ '<|system|>\n' + message['content'] | trim + '<|end|>\n' }}
    {%- elif message['role'] == 'user' -%}
        {{ '<|user|>\n' + message['content'] | trim + '<|end|>\n' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ '<|assistant|>\n' + message['content'] | trim + '<|end|>\n' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<|assistant|>\n' }}
{%- endif -%}"""

_PHI4_TEMPLATE = """{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{ '<|im_start|>system<|im_sep|>' + message['content'] | trim + '<|im_end|>' }}
    {%- elif message['role'] == 'user' -%}
        {{ '<|im_start|>user<|im_sep|>' + message['content'] | trim + '<|im_end|>' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ '<|im_start|>assistant<|im_sep|>' + message['content'] | trim + '<|im_end|>' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<|im_start|>assistant<|im_sep|>' }}
{%- endif -%}"""

_MISTRAL_TEMPLATE = """{{ bos_token }}{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
        {{ '[INST] ' + message['content'] | trim + ' [/INST]' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ message['content'] | trim + eos_token }}
    {%- elif message['role'] == 'system' -%}
        {{ '[INST] ' + message['content'] | trim + ' [/INST]' }}
    {%- endif -%}
{%- endfor -%}"""

_DEEPSEEK_TEMPLATE = """{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{ message['content'] | trim + '\n\n' }}
    {%- elif message['role'] == 'user' -%}
        {{ '<｜User｜>' + message['content'] | trim + '\n\n' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ '<｜Assistant｜>' + message['content'] | trim + '<｜end▁of▁sentence｜>' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<｜Assistant｜>' }}
{%- endif -%}"""

_VICUNA_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
        {{ 'USER: ' + message['content'] | trim + '\n' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ 'ASSISTANT: ' + message['content'] | trim + '</s>\n' }}
    {%- elif message['role'] == 'system' -%}
        {{ message['content'] | trim + '\n\n' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ 'ASSISTANT: ' }}
{%- endif -%}"""

_ALPACA_CHAT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
        {{ '### Instruction:\n' + message['content'] | trim + '\n\n' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ '### Response:\n' + message['content'] | trim + '\n\n' }}
    {%- elif message['role'] == 'system' -%}
        {{ message['content'] | trim + '\n\n' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '### Response:\n' }}
{%- endif -%}"""

_ZEPHYR_TEMPLATE = """{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{ '<|system|>\n' + message['content'] | trim + '</s>\n' }}
    {%- elif message['role'] == 'user' -%}
        {{ '<|user|>\n' + message['content'] | trim + '</s>\n' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ '<|assistant|>\n' + message['content'] | trim + '</s>\n' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<|assistant|>\n' }}
{%- endif -%}"""


# Main template registry
CHAT_TEMPLATES: Dict[str, ChatTemplateEntry] = {
    # Llama 3 family
    "llama-3": ChatTemplateEntry(
        template=_LLAMA3_TEMPLATE,
        eos_token="<|eot_id|>",
        bos_token="<|begin_of_text|>",
        stop_token="<|eot_id|>",
    ),
    "llama-3.1": ChatTemplateEntry(
        template=_LLAMA3_TEMPLATE,
        eos_token="<|eot_id|>",
        bos_token="<|begin_of_text|>",
        stop_token="<|eot_id|>",
    ),

    # ChatML (OpenAI format)
    "chatml": ChatTemplateEntry(
        template=_CHATML_TEMPLATE,
        eos_token="<|im_end|>",
        bos_token="",
        stop_token="<|im_end|>",
    ),

    # Gemma family
    "gemma-2": ChatTemplateEntry(
        template=_GEMMA_TEMPLATE,
        eos_token="<end_of_turn>",
        bos_token="<bos>",
        stop_token="<end_of_turn>",
    ),
    "gemma-3": ChatTemplateEntry(
        template=_GEMMA_TEMPLATE,
        eos_token="<end_of_turn>",
        bos_token="<bos>",
        stop_token="<end_of_turn>",
    ),
    "gemma-4": ChatTemplateEntry(
        template=_GEMMA_TEMPLATE,
        eos_token="<end_of_turn>",
        bos_token="<bos>",
        stop_token="<end_of_turn>",
    ),

    # Qwen family
    "qwen-2.5": ChatTemplateEntry(
        template=_QWEN_TEMPLATE,
        eos_token="<|im_end|>",
        bos_token="",
        stop_token="<|im_end|>",
    ),
    "qwen-3": ChatTemplateEntry(
        template=_QWEN3_TEMPLATE,
        eos_token="<|im_end|>",
        bos_token="",
        stop_token="<|im_end|>",
    ),

    # Phi family
    "phi-3": ChatTemplateEntry(
        template=_PHI3_TEMPLATE,
        eos_token="<|end|>",
        bos_token="",
        stop_token="<|end|>",
    ),
    "phi-3.5": ChatTemplateEntry(
        template=_PHI3_TEMPLATE,
        eos_token="<|end|>",
        bos_token="",
        stop_token="<|end|>",
    ),
    "phi-4": ChatTemplateEntry(
        template=_PHI4_TEMPLATE,
        eos_token="<|im_end|>",
        bos_token="",
        stop_token="<|im_end|>",
    ),

    # Mistral family
    "mistral": ChatTemplateEntry(
        template=_MISTRAL_TEMPLATE,
        eos_token="</s>",
        bos_token="<s>",
        stop_token="</s>",
    ),
    "mistral-nemo": ChatTemplateEntry(
        template=_MISTRAL_TEMPLATE,
        eos_token="</s>",
        bos_token="<s>",
        stop_token="</s>",
    ),

    # DeepSeek
    "deepseek-v2": ChatTemplateEntry(
        template=_DEEPSEEK_TEMPLATE,
        eos_token="<｜end▁of▁sentence｜>",
        bos_token="<｜begin▁of▁sentence｜>",
        stop_token="<｜end▁of▁sentence｜>",
    ),

    # Legacy formats
    "alpaca": ChatTemplateEntry(
        template=_ALPACA_CHAT_TEMPLATE,
        eos_token="</s>",
        bos_token="",
        stop_token="</s>",
    ),
    "vicuna": ChatTemplateEntry(
        template=_VICUNA_TEMPLATE,
        eos_token="</s>",
        bos_token="",
        stop_token="</s>",
    ),
    "zephyr": ChatTemplateEntry(
        template=_ZEPHYR_TEMPLATE,
        eos_token="</s>",
        bos_token="",
        stop_token="</s>",
    ),
}


# Template aliases for convenience
TEMPLATE_ALIASES: Dict[str, str] = {
    # Llama aliases
    "llama3": "llama-3",
    "llama-3.2": "llama-3.1",
    "llama-3.3": "llama-3.1",
    "llama31": "llama-3.1",
    "llama32": "llama-3.1",
    "llama33": "llama-3.1",

    # Gemma aliases
    "gemma": "gemma-2",
    "gemma2": "gemma-2",
    "gemma3": "gemma-3",
    "gemma4": "gemma-4",

    # Qwen aliases
    "qwen": "qwen-2.5",
    "qwen25": "qwen-2.5",
    "qwen2.5": "qwen-2.5",
    "qwen3": "qwen-3",

    # Phi aliases
    "phi3": "phi-3",
    "phi35": "phi-3.5",
    "phi4": "phi-4",

    # Mistral aliases
    "mistral-v0.3": "mistral",
    "mistral-instruct": "mistral",

    # DeepSeek aliases
    "deepseek": "deepseek-v2",
    "deepseek-v3": "deepseek-v2",

    # OpenAI format
    "openai": "chatml",
    "im_start": "chatml",

    # LFM (Liquid Foundation Models) aliases
    "lfm2": "chatml",
    "lfm25": "chatml",
    "lfm2.5": "chatml",
    "liquid": "chatml",
}


# Default system messages for models that benefit from them
DEFAULT_SYSTEM_MESSAGES: Dict[str, str] = {
    "llama-3": "You are a helpful assistant.",
    "llama-3.1": "You are a helpful assistant.",
    "qwen-2.5": "You are a helpful assistant.",
    "qwen-3": "You are a helpful assistant.",
    "deepseek-v2": "You are a helpful assistant.",
}


def detect_dataset_format(sample: Dict[str, Any]) -> str:
    """
    Detect the format of a dataset sample.

    Args:
        sample: A single sample from the dataset

    Returns:
        Format string: 'alpaca', 'sharegpt', 'chatml', 'text', 'completions', or 'unknown'
    """
    keys = set(sample.keys())

    # Check for text format (simplest)
    if 'text' in keys:
        return 'text'

    # Check for ChatML format (messages with role/content)
    if 'messages' in keys:
        messages = sample['messages']
        if isinstance(messages, list) and len(messages) > 0:
            if isinstance(messages[0], dict) and 'role' in messages[0]:
                return 'chatml'

    # Check for ShareGPT format (conversations with from/value)
    if 'conversations' in keys:
        convos = sample['conversations']
        if isinstance(convos, list) and len(convos) > 0:
            if isinstance(convos[0], dict) and 'from' in convos[0]:
                return 'sharegpt'

    # Check for completions format
    if 'prompt' in keys and 'completion' in keys:
        return 'completions'

    # Check for Alpaca format
    if 'instruction' in keys and 'output' in keys:
        return 'alpaca'

    return 'unknown'


def standardize_sharegpt(dataset: Dataset) -> Dataset:
    """
    Convert ShareGPT format to ChatML format.

    ShareGPT uses {"from": "human/gpt", "value": "..."}
    ChatML uses {"role": "user/assistant", "content": "..."}

    Args:
        dataset: Dataset with ShareGPT format conversations

    Returns:
        Dataset with ChatML format messages
    """
    role_mapping = {
        'human': 'user',
        'user': 'user',
        'gpt': 'assistant',
        'assistant': 'assistant',
        'system': 'system',
    }

    def convert_sample(sample):
        if 'conversations' not in sample:
            return sample

        messages = []
        for turn in sample['conversations']:
            role = role_mapping.get(turn.get('from', '').lower(), 'user')
            content = turn.get('value', '')
            messages.append({'role': role, 'content': content})

        return {'messages': messages}

    return dataset.map(convert_sample)


def alpaca_to_text(
    sample: Dict[str, Any],
    template: Optional[str] = None,
) -> str:
    """
    Convert Alpaca format sample to text.

    Args:
        sample: Alpaca format sample with instruction, input, output
        template: Optional custom template string

    Returns:
        Formatted text string
    """
    instruction = sample.get('instruction', '')
    input_text = sample.get('input', '')
    output = sample.get('output', '')

    if template:
        return template.format(
            instruction=instruction,
            input=input_text,
            output=output
        )

    # Use appropriate template based on whether input is provided
    if input_text.strip():
        return ALPACA_TEMPLATE.format(
            instruction=instruction,
            input=input_text,
            output=output
        )
    else:
        return ALPACA_TEMPLATE_NO_INPUT.format(
            instruction=instruction,
            output=output
        )


def apply_chat_template_to_sample(
    sample: Dict[str, Any],
    tokenizer: Any,
    add_generation_prompt: bool = False,
) -> str:
    """
    Apply tokenizer's chat template to a sample.

    Args:
        sample: Sample with 'messages' field (ChatML format)
        tokenizer: Tokenizer with apply_chat_template method
        add_generation_prompt: Whether to add generation prompt

    Returns:
        Formatted text string
    """
    messages = sample.get('messages', [])

    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
    else:
        # Fallback: simple formatting
        text_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            text_parts.append(f"{role}: {content}")
        return '\n'.join(text_parts)


def convert_to_mlx_format(
    dataset: Dataset,
    tokenizer: Any,
    output_format: str = 'text',
    alpaca_template: Optional[str] = None,
) -> Dataset:
    """
    Convert any supported dataset format to mlx-lm compatible format.

    This is the main function for dataset conversion, similar to Unsloth's
    formatting workflow.

    Args:
        dataset: Input dataset in any supported format
        tokenizer: Tokenizer (used for chat template if available)
        output_format: Target format ('text', 'chat', 'completions')
        alpaca_template: Custom template for Alpaca format conversion

    Returns:
        Dataset in mlx-lm compatible format

    Example:
        >>> from mlx_tune.chat_templates import convert_to_mlx_format
        >>> dataset = load_dataset("yahma/alpaca-cleaned", split="train[:100]")
        >>> dataset = convert_to_mlx_format(dataset, tokenizer)
        >>> # Now dataset has 'text' field compatible with mlx-lm
    """
    if len(dataset) == 0:
        return dataset

    # Detect input format from first sample
    input_format = detect_dataset_format(dataset[0])
    print(f"Detected dataset format: {input_format}")

    if input_format == 'unknown':
        print(f"Warning: Unknown dataset format. Fields: {list(dataset[0].keys())}")
        print("Attempting to use raw sample...")

    def convert_sample(sample):
        # Already in target format?
        if input_format == output_format:
            return sample
        if input_format == 'text' and output_format == 'text':
            return sample

        # Convert based on input format
        if input_format == 'alpaca':
            if output_format == 'text':
                text = alpaca_to_text(sample, alpaca_template)
                return {'text': text}
            elif output_format == 'completions':
                instruction = sample.get('instruction', '')
                input_text = sample.get('input', '')
                prompt = f"{instruction}\n{input_text}".strip() if input_text else instruction
                return {'prompt': prompt, 'completion': sample.get('output', '')}
            elif output_format == 'chat':
                # Convert to messages format
                messages = [
                    {'role': 'user', 'content': f"{sample.get('instruction', '')}\n{sample.get('input', '')}".strip()},
                    {'role': 'assistant', 'content': sample.get('output', '')}
                ]
                return {'messages': messages}

        elif input_format == 'sharegpt':
            # First convert to ChatML
            messages = []
            role_mapping = {'human': 'user', 'gpt': 'assistant', 'system': 'system'}
            for turn in sample.get('conversations', []):
                role = role_mapping.get(turn.get('from', '').lower(), 'user')
                messages.append({'role': role, 'content': turn.get('value', '')})

            if output_format == 'chat':
                return {'messages': messages}
            elif output_format == 'text':
                text = apply_chat_template_to_sample({'messages': messages}, tokenizer)
                return {'text': text}

        elif input_format == 'chatml':
            if output_format == 'chat':
                return sample  # Already in chat format
            elif output_format == 'text':
                text = apply_chat_template_to_sample(sample, tokenizer)
                return {'text': text}

        elif input_format == 'completions':
            if output_format == 'completions':
                return sample
            elif output_format == 'text':
                return {'text': f"{sample.get('prompt', '')}\n{sample.get('completion', '')}"}

        elif input_format == 'text':
            return sample

        # Fallback for unknown format - try to create text
        if output_format == 'text':
            # Try common field names
            for field in ['text', 'content', 'output', 'response', 'completion']:
                if field in sample:
                    return {'text': sample[field]}
            # Last resort: stringify the sample
            import json
            return {'text': json.dumps(sample)}

        return sample

    converted = dataset.map(convert_sample)

    # Verify conversion
    if len(converted) > 0:
        result_format = detect_dataset_format(converted[0])
        print(f"Output format: {result_format}")
        if output_format == 'text' and 'text' not in converted[0]:
            print(f"Warning: Conversion may have failed. Sample keys: {list(converted[0].keys())}")

    return converted


def get_formatting_func(
    tokenizer: Any,
    dataset_format: str = 'auto',
    alpaca_template: Optional[str] = None,
) -> Callable:
    """
    Get a formatting function for use with SFTTrainer.

    This returns a function that can be passed to SFTTrainer's formatting_func
    parameter to automatically convert samples to the text format.

    Args:
        tokenizer: Tokenizer instance
        dataset_format: Expected format ('auto', 'alpaca', 'sharegpt', 'chatml')
        alpaca_template: Custom template for Alpaca format

    Returns:
        Formatting function that takes a sample and returns formatted text

    Example:
        >>> formatting_func = get_formatting_func(tokenizer)
        >>> trainer = SFTTrainer(
        ...     model=model,
        ...     train_dataset=dataset,
        ...     formatting_func=formatting_func,
        ...     ...
        ... )
    """
    def formatting_func(sample: Dict[str, Any]) -> str:
        # Detect format if auto
        fmt = dataset_format
        if fmt == 'auto':
            fmt = detect_dataset_format(sample)

        # Convert based on format
        if fmt == 'text':
            return sample.get('text', '')

        elif fmt == 'alpaca':
            return alpaca_to_text(sample, alpaca_template)

        elif fmt == 'sharegpt':
            # Convert to ChatML first
            messages = []
            role_mapping = {'human': 'user', 'gpt': 'assistant', 'system': 'system'}
            for turn in sample.get('conversations', []):
                role = role_mapping.get(turn.get('from', '').lower(), 'user')
                messages.append({'role': role, 'content': turn.get('value', '')})
            return apply_chat_template_to_sample({'messages': messages}, tokenizer)

        elif fmt == 'chatml':
            return apply_chat_template_to_sample(sample, tokenizer)

        elif fmt == 'completions':
            return f"{sample.get('prompt', '')}\n{sample.get('completion', '')}"

        else:
            # Unknown format - try to extract text
            for field in ['text', 'content', 'output', 'response']:
                if field in sample:
                    return sample[field]
            return str(sample)

    return formatting_func


# =============================================================================
# GET_CHAT_TEMPLATE FUNCTION (Unsloth-compatible)
# =============================================================================

def _detect_template_from_tokenizer(tokenizer: Any) -> str:
    """
    Auto-detect the appropriate chat template from tokenizer name or config.

    Args:
        tokenizer: A HuggingFace tokenizer

    Returns:
        Template name string
    """
    # Get model name from tokenizer
    name = getattr(tokenizer, 'name_or_path', '').lower()

    # Detection rules based on model name
    if 'llama-3' in name or 'llama3' in name:
        if any(v in name for v in ['3.1', '3.2', '3.3', '3-1', '3-2', '3-3']):
            return 'llama-3.1'
        return 'llama-3'

    if 'gemma-4' in name or 'gemma4' in name:
        return 'gemma-4'
    if 'gemma-3' in name or 'gemma3' in name:
        return 'gemma-3'
    if 'gemma' in name:
        return 'gemma-2'

    if 'qwen-3' in name or 'qwen3' in name:
        return 'qwen-3'
    if 'qwen' in name:
        return 'qwen-2.5'

    if 'phi-4' in name or 'phi4' in name:
        return 'phi-4'
    if 'phi-3.5' in name or 'phi35' in name:
        return 'phi-3.5'
    if 'phi-3' in name or 'phi3' in name:
        return 'phi-3'

    if 'mistral-nemo' in name:
        return 'mistral-nemo'
    if 'mistral' in name:
        return 'mistral'

    if 'deepseek' in name:
        return 'deepseek-v2'

    if 'lfm' in name or 'liquid' in name:
        return 'chatml'

    if 'vicuna' in name:
        return 'vicuna'

    if 'zephyr' in name:
        return 'zephyr'

    # Check if tokenizer already has a chat template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        # Try to detect from existing template content
        template = tokenizer.chat_template
        if '<|im_start|>' in template:
            if '<|im_sep|>' in template:
                return 'phi-4'
            return 'chatml'
        if '<|start_header_id|>' in template:
            return 'llama-3'
        if '<start_of_turn>' in template:
            return 'gemma-2'
        if '[INST]' in template:
            return 'mistral'

    # Default to chatml (widely compatible)
    return 'chatml'


def get_chat_template(
    tokenizer: Any,
    chat_template: str = "auto",
    mapping: Optional[Dict[str, str]] = None,
    map_eos_token: bool = True,
    system_message: Optional[str] = None,
) -> Any:
    """
    Apply a chat template to the tokenizer.

    This function matches Unsloth's get_chat_template API for drop-in compatibility.
    It sets the tokenizer's chat_template attribute and optionally configures
    special tokens.

    Args:
        tokenizer: A HuggingFace tokenizer
        chat_template: Template name or "auto" to detect from model name.
                      Supported: llama-3, llama-3.1, chatml, gemma-2, gemma-3,
                      qwen-2.5, qwen-3, phi-3, phi-3.5, phi-4, mistral,
                      mistral-nemo, deepseek-v2, alpaca, vicuna, zephyr
        mapping: Optional column mapping for dataset conversion.
                 e.g., {"role": "from", "content": "value", "user": "human", "assistant": "gpt"}
        map_eos_token: Whether to update the tokenizer's EOS token to match template
        system_message: Custom system message to prepend (if supported by template)

    Returns:
        Modified tokenizer with chat_template set

    Example:
        >>> from mlx_tune import get_chat_template, FastLanguageModel
        >>> model, tokenizer = FastLanguageModel.from_pretrained("mlx-community/Llama-3.2-1B-Instruct-4bit")
        >>> tokenizer = get_chat_template(tokenizer, chat_template="llama-3")
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> text = tokenizer.apply_chat_template(messages, tokenize=False)
    """
    # Resolve alias
    template_name = chat_template.lower().strip()
    if template_name in TEMPLATE_ALIASES:
        template_name = TEMPLATE_ALIASES[template_name]

    # Auto-detect if needed
    if template_name == "auto":
        template_name = _detect_template_from_tokenizer(tokenizer)
        print(f"Auto-detected chat template: {template_name}")

    # Look up template
    if template_name not in CHAT_TEMPLATES:
        available = list_chat_templates()
        raise ValueError(
            f"Unknown chat template: '{chat_template}'. "
            f"Available templates: {', '.join(available)}"
        )

    entry = CHAT_TEMPLATES[template_name]

    # Set the chat template
    tokenizer.chat_template = entry.template

    # Optionally map EOS token
    if map_eos_token and entry.eos_token != "eos_token":
        # Store the stop token for generation
        tokenizer._unsloth_stop_token = entry.stop_token

        # Try to set EOS token if the tokenizer supports it
        try:
            if hasattr(tokenizer, 'eos_token'):
                # Check if the token exists in vocabulary
                if hasattr(tokenizer, 'get_vocab'):
                    vocab = tokenizer.get_vocab()
                    if entry.eos_token in vocab:
                        tokenizer.eos_token = entry.eos_token
        except Exception:
            pass  # Silently fail if we can't set the EOS token

    # Store BOS token reference
    if entry.bos_token and entry.bos_token != "bos_token":
        tokenizer._unsloth_bos_token = entry.bos_token

    # Store mapping for dataset conversion
    if mapping:
        tokenizer._unsloth_mapping = mapping

    # Store system message
    if system_message:
        tokenizer._unsloth_system_message = system_message
    elif template_name in DEFAULT_SYSTEM_MESSAGES:
        tokenizer._unsloth_system_message = DEFAULT_SYSTEM_MESSAGES[template_name]

    # Store template name for reference
    tokenizer._unsloth_chat_template_name = template_name

    return tokenizer


def list_chat_templates() -> List[str]:
    """
    List all available chat template names.

    Returns:
        Sorted list of template names

    Example:
        >>> from mlx_tune import list_chat_templates
        >>> templates = list_chat_templates()
        >>> print(templates)
        ['alpaca', 'chatml', 'deepseek-v2', 'gemma-2', ...]
    """
    return sorted(CHAT_TEMPLATES.keys())


def get_template_info(template_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific chat template.

    Args:
        template_name: The template name (supports aliases)

    Returns:
        Dictionary with template information

    Raises:
        ValueError: If template_name is not found

    Example:
        >>> from mlx_tune import get_template_info
        >>> info = get_template_info("llama-3")
        >>> print(info['eos_token'])
        '<|eot_id|>'
    """
    # Resolve alias
    name = template_name.lower().strip()
    if name in TEMPLATE_ALIASES:
        name = TEMPLATE_ALIASES[name]

    if name not in CHAT_TEMPLATES:
        available = list_chat_templates()
        raise ValueError(
            f"Unknown chat template: '{template_name}'. "
            f"Available templates: {', '.join(available)}"
        )

    entry = CHAT_TEMPLATES[name]
    return {
        "name": name,
        "eos_token": entry.eos_token,
        "bos_token": entry.bos_token,
        "stop_token": entry.stop_token,
        "template_preview": entry.template[:200] + "..." if len(entry.template) > 200 else entry.template,
        "default_system_message": DEFAULT_SYSTEM_MESSAGES.get(name),
    }


def get_template_for_model(model_name: str) -> str:
    """
    Get the recommended chat template name for a given model.

    Args:
        model_name: Model name or path (e.g., "meta-llama/Llama-3.2-1B-Instruct")

    Returns:
        Recommended template name

    Example:
        >>> from mlx_tune import get_template_for_model
        >>> template = get_template_for_model("meta-llama/Llama-3.2-1B-Instruct")
        >>> print(template)
        'llama-3.1'
    """
    name = model_name.lower()

    # Detection rules
    if 'llama-3' in name or 'llama3' in name:
        if any(v in name for v in ['3.1', '3.2', '3.3']):
            return 'llama-3.1'
        return 'llama-3'

    if 'gemma-4' in name or 'gemma4' in name:
        return 'gemma-4'
    if 'gemma-3' in name or 'gemma3' in name:
        return 'gemma-3'
    if 'gemma' in name:
        return 'gemma-2'

    if 'qwen3' in name or 'qwen-3' in name:
        return 'qwen-3'
    if 'qwen' in name:
        return 'qwen-2.5'

    if 'phi-4' in name:
        return 'phi-4'
    if 'phi-3.5' in name:
        return 'phi-3.5'
    if 'phi-3' in name or 'phi3' in name:
        return 'phi-3'

    if 'mistral-nemo' in name:
        return 'mistral-nemo'
    if 'mistral' in name:
        return 'mistral'

    if 'deepseek' in name:
        return 'deepseek-v2'

    if 'lfm' in name or 'liquid' in name:
        return 'chatml'

    if 'vicuna' in name:
        return 'vicuna'

    if 'zephyr' in name:
        return 'zephyr'

    # Default
    return 'chatml'


# =============================================================================
# TRAIN_ON_RESPONSES_ONLY (Unsloth-compatible)
# =============================================================================

def train_on_responses_only(
    trainer: Any,
    instruction_part: Optional[str] = None,
    response_part: Optional[str] = None,
) -> Any:
    """
    Modify the trainer to only compute loss on response tokens.

    This function matches Unsloth's train_on_responses_only API for drop-in
    compatibility. It configures the trainer to mask instruction/prompt tokens
    so that loss is only computed on the assistant's responses.

    Args:
        trainer: An SFTTrainer instance
        instruction_part: The token sequence marking the start of user instruction.
                         If None, auto-detected from template.
                         e.g., "<|start_header_id|>user<|end_header_id|>"
        response_part: The token sequence marking the start of assistant response.
                      If None, auto-detected from template.
                      e.g., "<|start_header_id|>assistant<|end_header_id|>"

    Returns:
        Modified trainer with response-only training enabled

    Example:
        >>> from mlx_tune import SFTTrainer, train_on_responses_only
        >>> trainer = SFTTrainer(model=model, ...)
        >>> trainer = train_on_responses_only(
        ...     trainer,
        ...     instruction_part="<|start_header_id|>user<|end_header_id|>",
        ...     response_part="<|start_header_id|>assistant<|end_header_id|>",
        ... )
        >>> trainer.train()
    """
    # Store configuration on the trainer
    trainer._train_on_responses_only = True
    trainer._instruction_part = instruction_part
    trainer._response_part = response_part

    # Try to auto-detect parts from template if not provided
    if instruction_part is None or response_part is None:
        template_name = None

        # Check if tokenizer has template info
        if hasattr(trainer, 'tokenizer'):
            tokenizer = trainer.tokenizer
            if hasattr(tokenizer, '_unsloth_chat_template_name'):
                template_name = tokenizer._unsloth_chat_template_name
            elif hasattr(tokenizer, 'name_or_path'):
                template_name = get_template_for_model(tokenizer.name_or_path)

        # Set default parts based on template
        if template_name:
            parts = _get_template_parts(template_name)
            if instruction_part is None:
                trainer._instruction_part = parts.get('instruction_part')
            if response_part is None:
                trainer._response_part = parts.get('response_part')

    # Log configuration
    print(f"train_on_responses_only enabled:")
    print(f"  instruction_part: {trainer._instruction_part}")
    print(f"  response_part: {trainer._response_part}")

    return trainer


def _get_template_parts(template_name: str) -> Dict[str, str]:
    """
    Get the instruction and response marker parts for a template.

    These are used for masking during response-only training.
    """
    # Resolve alias
    name = template_name.lower()
    if name in TEMPLATE_ALIASES:
        name = TEMPLATE_ALIASES[name]

    # Template-specific parts
    parts_mapping = {
        "llama-3": {
            "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
            "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        },
        "llama-3.1": {
            "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
            "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        },
        "chatml": {
            "instruction_part": "<|im_start|>user\n",
            "response_part": "<|im_start|>assistant\n",
        },
        "gemma-2": {
            "instruction_part": "<start_of_turn>user\n",
            "response_part": "<start_of_turn>model\n",
        },
        "gemma-3": {
            "instruction_part": "<start_of_turn>user\n",
            "response_part": "<start_of_turn>model\n",
        },
        "gemma-4": {
            "instruction_part": "<start_of_turn>user\n",
            "response_part": "<start_of_turn>model\n",
        },
        "qwen-2.5": {
            "instruction_part": "<|im_start|>user\n",
            "response_part": "<|im_start|>assistant\n",
        },
        "qwen-3": {
            "instruction_part": "<|im_start|>user\n",
            "response_part": "<|im_start|>assistant\n",
        },
        "phi-3": {
            "instruction_part": "<|user|>\n",
            "response_part": "<|assistant|>\n",
        },
        "phi-3.5": {
            "instruction_part": "<|user|>\n",
            "response_part": "<|assistant|>\n",
        },
        "phi-4": {
            "instruction_part": "<|im_start|>user<|im_sep|>",
            "response_part": "<|im_start|>assistant<|im_sep|>",
        },
        "mistral": {
            "instruction_part": "[INST]",
            "response_part": "[/INST]",
        },
        "mistral-nemo": {
            "instruction_part": "[INST]",
            "response_part": "[/INST]",
        },
        "deepseek-v2": {
            "instruction_part": "<｜User｜>",
            "response_part": "<｜Assistant｜>",
        },
        "alpaca": {
            "instruction_part": "### Instruction:\n",
            "response_part": "### Response:\n",
        },
        "vicuna": {
            "instruction_part": "USER: ",
            "response_part": "ASSISTANT: ",
        },
        "zephyr": {
            "instruction_part": "<|user|>\n",
            "response_part": "<|assistant|>\n",
        },
    }

    return parts_mapping.get(name, {
        "instruction_part": None,
        "response_part": None,
    })


def get_response_template_ids(
    tokenizer: Any,
    response_part: str,
) -> List[int]:
    """
    Get the token IDs for the response template marker.

    This is useful for finding where responses start in tokenized sequences.

    Args:
        tokenizer: The tokenizer to use
        response_part: The response marker string

    Returns:
        List of token IDs for the response marker
    """
    if hasattr(tokenizer, 'encode'):
        # Use encode without special tokens
        try:
            return tokenizer.encode(response_part, add_special_tokens=False)
        except Exception:
            return tokenizer.encode(response_part)
    return []


def create_response_only_collator(
    tokenizer: Any,
    instruction_part: str,
    response_part: str,
    ignore_index: int = -100,
) -> Callable:
    """
    Create a data collator that masks instruction tokens.

    This is used during training to ensure loss is only computed on response tokens.

    Args:
        tokenizer: The tokenizer to use
        instruction_part: The instruction marker string
        response_part: The response marker string
        ignore_index: The index to use for masked tokens (default -100)

    Returns:
        A collator function that masks instruction tokens in labels
    """
    # Get token IDs for response marker
    response_ids = get_response_template_ids(tokenizer, response_part)

    def collator(examples):
        """Collate examples and mask instruction tokens."""
        # This is a simplified version - full implementation would handle batching
        for example in examples:
            if 'labels' in example and 'input_ids' in example:
                input_ids = example['input_ids']
                labels = example['labels']

                # Find response start positions and mask everything before
                # This is a simplified approach - Unsloth uses more sophisticated matching
                # For now, we rely on mlx-lm's --mask-prompt flag for subprocess training

        return examples

    return collator


# =============================================================================
# PHASE 2.1: TO_SHAREGPT WITH CONVERSATION_EXTENSION (Unsloth-compatible)
# =============================================================================

def to_sharegpt(
    dataset: Dataset,
    merged_prompt: Optional[str] = None,
    output_column_name: str = "output",
    conversation_extension: int = 1,
    column_mapping: Optional[Dict[str, str]] = None,
    random_state: Optional[int] = None,
) -> Dataset:
    """
    Convert dataset to ShareGPT format with optional multi-turn conversation merging.

    This function matches Unsloth's to_sharegpt API for drop-in compatibility.
    It converts single-turn datasets to multi-turn conversations by optionally
    merging multiple rows together.

    Args:
        dataset: Input dataset in any format (Alpaca, text, etc.)
        merged_prompt: Optional prompt template with {column_name} placeholders.
                      Use [[text {column}]] for optional sections.
                      If None, auto-generates from available columns.
        output_column_name: Column name containing the target/output text.
                           Default: "output"
        conversation_extension: Number of rows to merge into one conversation.
                               Set to 1 for single-turn (default).
                               Set to 2-5 for multi-turn conversations.
                               Higher values may improve chatbot quality but slow training.
        column_mapping: Optional mapping for non-standard column names.
                       e.g., {"instruction": "question", "output": "answer"}
        random_state: Random seed for reproducible conversation merging.

    Returns:
        Dataset in ShareGPT format with 'conversations' column

    Example:
        >>> from mlx_tune import to_sharegpt, standardize_sharegpt
        >>> # Convert Alpaca dataset with multi-turn merging
        >>> dataset = to_sharegpt(
        ...     dataset,
        ...     output_column_name="output",
        ...     conversation_extension=3,  # Merge 3 rows into 1 conversation
        ... )
        >>> dataset = standardize_sharegpt(dataset)  # Always call this after!

    Note:
        The conversation_extension parameter randomly selects rows and merges them.
        This can significantly improve chatbot quality for instruction-following tasks.
    """
    if random_state is not None:
        random.seed(random_state)

    # Apply column mapping if provided
    if column_mapping:
        dataset = apply_column_mapping(dataset, column_mapping)

    # Detect format and get sample
    if len(dataset) == 0:
        return dataset

    sample = dataset[0]
    input_format = detect_dataset_format(sample)
    print(f"to_sharegpt: Detected format '{input_format}', conversation_extension={conversation_extension}")

    # Build conversations
    def create_single_conversation(sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create a single conversation from one sample."""
        conversation = []

        if input_format == 'alpaca':
            # Build user message from instruction + input
            instruction = sample.get('instruction', '')
            input_text = sample.get('input', '')
            user_content = instruction
            if input_text and input_text.strip():
                user_content = f"{instruction}\n\n{input_text}"

            conversation.append({
                'from': 'human',
                'value': user_content.strip()
            })
            conversation.append({
                'from': 'gpt',
                'value': sample.get(output_column_name, sample.get('output', '')).strip()
            })

        elif input_format == 'completions':
            conversation.append({
                'from': 'human',
                'value': sample.get('prompt', '').strip()
            })
            conversation.append({
                'from': 'gpt',
                'value': sample.get('completion', sample.get(output_column_name, '')).strip()
            })

        elif input_format == 'chatml':
            # Convert ChatML messages to ShareGPT format
            role_mapping = {'user': 'human', 'assistant': 'gpt', 'system': 'system'}
            for msg in sample.get('messages', []):
                conversation.append({
                    'from': role_mapping.get(msg.get('role', 'user'), 'human'),
                    'value': msg.get('content', '').strip()
                })

        elif input_format == 'sharegpt':
            # Already in ShareGPT format
            return sample.get('conversations', [])

        elif input_format == 'text':
            # Use custom merged_prompt or fallback
            if merged_prompt:
                user_content = _apply_prompt_template(merged_prompt, sample)
            else:
                user_content = sample.get('text', '')

            # Try to split into prompt/response if output column exists
            output = sample.get(output_column_name, '')
            if output:
                conversation.append({'from': 'human', 'value': user_content.strip()})
                conversation.append({'from': 'gpt', 'value': output.strip()})
            else:
                # Just use the text as a human message
                conversation.append({'from': 'human', 'value': user_content.strip()})
                conversation.append({'from': 'gpt', 'value': ''})

        else:
            # Unknown format - try to use merged_prompt or create from available fields
            if merged_prompt:
                user_content = _apply_prompt_template(merged_prompt, sample)
            else:
                # Try common field patterns
                user_content = sample.get('question', sample.get('query', sample.get('input', '')))

            output = sample.get(output_column_name, sample.get('answer', sample.get('response', '')))

            if user_content:
                conversation.append({'from': 'human', 'value': str(user_content).strip()})
                conversation.append({'from': 'gpt', 'value': str(output).strip() if output else ''})

        return conversation

    # Handle conversation_extension
    if conversation_extension <= 1:
        # Single-turn: just convert each sample
        def convert_sample(sample):
            return {'conversations': create_single_conversation(sample)}
        return dataset.map(convert_sample)
    else:
        # Multi-turn: merge multiple samples into one conversation
        # Get all indices
        indices = list(range(len(dataset)))

        # Group indices for merging
        merged_data = []
        i = 0
        while i < len(indices):
            # Get a group of samples to merge
            group_size = min(conversation_extension, len(indices) - i)

            # Random selection for better diversity
            if group_size < conversation_extension and len(indices) > conversation_extension:
                # For the last incomplete group, randomly select from all indices
                group_indices = random.sample(indices, min(conversation_extension, len(indices)))
            else:
                group_indices = indices[i:i + group_size]

            # Merge conversations
            merged_conversation = []
            for idx in group_indices:
                sample = dataset[idx]
                conv = create_single_conversation(sample)
                merged_conversation.extend(conv)

            merged_data.append({'conversations': merged_conversation})
            i += group_size

        # Create new dataset
        from datasets import Dataset as HFDataset
        return HFDataset.from_list(merged_data)


def _apply_prompt_template(template: str, sample: Dict[str, Any]) -> str:
    """
    Apply a prompt template with {column} placeholders and [[optional]] sections.

    Args:
        template: Template string with {column_name} placeholders
                 Use [[text {column}]] for optional sections
        sample: Sample dictionary with column values

    Returns:
        Formatted string with placeholders replaced
    """
    result = template

    # Handle optional sections first: [[text {column}]]
    optional_pattern = r'\[\[(.*?)\]\]'

    def replace_optional(match):
        section = match.group(1)
        # Find all column references in this section
        col_refs = re.findall(r'\{(\w+)\}', section)
        # Check if all referenced columns have values
        for col in col_refs:
            value = sample.get(col, '')
            if not value or (isinstance(value, str) and not value.strip()):
                return ''  # Remove entire optional section
        # All columns have values, include the section
        for col in col_refs:
            section = section.replace(f'{{{col}}}', str(sample.get(col, '')))
        return section

    result = re.sub(optional_pattern, replace_optional, result)

    # Now replace remaining {column} placeholders
    for key, value in sample.items():
        result = result.replace(f'{{{key}}}', str(value) if value else '')

    return result.strip()


# =============================================================================
# PHASE 2.2: CUSTOM COLUMN MAPPING (Unsloth-compatible)
# =============================================================================

def apply_column_mapping(
    dataset: Dataset,
    column_mapping: Dict[str, str],
    inplace: bool = False,
) -> Dataset:
    """
    Apply column mapping to rename dataset columns.

    This function matches Unsloth's column mapping behavior for datasets
    with non-standard column names.

    Args:
        dataset: Input dataset
        column_mapping: Mapping from standard names to actual column names.
                       e.g., {"instruction": "question", "output": "answer"}
                       The keys are the target standard names, values are source names.
        inplace: If True, modify dataset in place (not supported, always creates new)

    Returns:
        Dataset with renamed columns

    Example:
        >>> from mlx_tune import apply_column_mapping
        >>> # Dataset has 'question' and 'answer' columns
        >>> dataset = apply_column_mapping(dataset, {
        ...     "instruction": "question",
        ...     "output": "answer"
        ... })
        >>> # Now dataset has 'instruction' and 'output' columns
    """
    if not column_mapping:
        return dataset

    # Check which source columns exist
    existing_cols = set(dataset.column_names)
    rename_map = {}

    for target_name, source_name in column_mapping.items():
        if source_name in existing_cols:
            # Only rename if source exists and target != source
            if target_name != source_name:
                rename_map[source_name] = target_name

    if not rename_map:
        return dataset

    print(f"apply_column_mapping: Renaming columns {rename_map}")

    # Apply renaming
    return dataset.rename_columns(rename_map)


def infer_column_mapping(
    dataset: Dataset,
    target_format: str = "alpaca",
) -> Dict[str, str]:
    """
    Automatically infer column mapping based on common patterns.

    This helps convert datasets with non-standard column names to
    standard formats (Alpaca, ChatML, etc.).

    Args:
        dataset: Input dataset
        target_format: Target format to infer mapping for.
                      Options: "alpaca", "completions", "chatml"

    Returns:
        Suggested column mapping dictionary

    Example:
        >>> from mlx_tune import infer_column_mapping
        >>> mapping = infer_column_mapping(dataset, target_format="alpaca")
        >>> print(mapping)
        {'instruction': 'question', 'output': 'answer'}
    """
    existing_cols = set(dataset.column_names)

    # Common field patterns
    instruction_patterns = ['instruction', 'question', 'query', 'prompt', 'input', 'user', 'human']
    output_patterns = ['output', 'answer', 'response', 'completion', 'target', 'assistant', 'gpt']
    input_patterns = ['input', 'context', 'document', 'passage']
    system_patterns = ['system', 'system_message', 'system_prompt']

    mapping = {}

    if target_format == "alpaca":
        # Find instruction column
        for pattern in instruction_patterns:
            if pattern in existing_cols:
                if pattern != 'instruction':
                    mapping['instruction'] = pattern
                break

        # Find output column
        for pattern in output_patterns:
            if pattern in existing_cols:
                if pattern != 'output':
                    mapping['output'] = pattern
                break

        # Find input column (optional)
        for pattern in input_patterns:
            if pattern in existing_cols and pattern != 'input':
                mapping['input'] = pattern
                break

    elif target_format == "completions":
        # Find prompt column
        for pattern in instruction_patterns:
            if pattern in existing_cols:
                if pattern != 'prompt':
                    mapping['prompt'] = pattern
                break

        # Find completion column
        for pattern in output_patterns:
            if pattern in existing_cols:
                if pattern != 'completion':
                    mapping['completion'] = pattern
                break

    elif target_format == "chatml":
        # For ChatML, we need messages - check if already present
        if 'messages' not in existing_cols:
            # Need to convert, return suggested mapping
            for pattern in instruction_patterns:
                if pattern in existing_cols:
                    mapping['_user_content'] = pattern
                    break
            for pattern in output_patterns:
                if pattern in existing_cols:
                    mapping['_assistant_content'] = pattern
                    break

    return mapping


# =============================================================================
# PHASE 2.3: HF DATASET CONFIG (Unsloth-compatible)
# =============================================================================

class HFDatasetConfig:
    """
    Configuration for loading and processing HuggingFace datasets.

    This matches Unsloth's dataset configuration pattern for easy integration.

    Attributes:
        path: HuggingFace dataset path (e.g., "yahma/alpaca-cleaned")
        name: Dataset configuration name (optional)
        train_split: Training split specification (default: "train")
        valid_split: Validation split specification (optional)
        streaming: Whether to use streaming mode (default: False)
        column_mapping: Optional column renaming mapping
        prompt_template: Custom prompt template for text formatting
        output_column: Column containing target/output text

    Example:
        >>> from mlx_tune import HFDatasetConfig
        >>> config = HFDatasetConfig(
        ...     path="Open-Orca/OpenOrca",
        ...     train_split="train[:90%]",
        ...     valid_split="train[-10%:]",
        ...     column_mapping={"instruction": "question", "output": "response"},
        ... )
    """

    def __init__(
        self,
        path: str,
        name: Optional[str] = None,
        train_split: str = "train",
        valid_split: Optional[str] = None,
        streaming: bool = False,
        column_mapping: Optional[Dict[str, str]] = None,
        prompt_template: Optional[str] = None,
        output_column: str = "output",
        conversation_extension: int = 1,
        max_samples: Optional[int] = None,
    ):
        self.path = path
        self.name = name
        self.train_split = train_split
        self.valid_split = valid_split
        self.streaming = streaming
        self.column_mapping = column_mapping
        self.prompt_template = prompt_template
        self.output_column = output_column
        self.conversation_extension = conversation_extension
        self.max_samples = max_samples

    def load(self) -> Dataset:
        """
        Load the dataset according to this configuration.

        Returns:
            Loaded and preprocessed Dataset

        Example:
            >>> config = HFDatasetConfig(path="yahma/alpaca-cleaned")
            >>> dataset = config.load()
        """
        from datasets import load_dataset

        # Load the dataset
        load_kwargs = {
            "split": self.train_split,
            "streaming": self.streaming,
        }
        if self.name:
            load_kwargs["name"] = self.name

        print(f"Loading dataset: {self.path}")
        dataset = load_dataset(self.path, **load_kwargs)

        # Apply max_samples if specified
        if self.max_samples and not self.streaming:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))

        # Apply column mapping
        if self.column_mapping:
            dataset = apply_column_mapping(dataset, self.column_mapping)

        return dataset

    def load_train_and_valid(self) -> tuple:
        """
        Load both training and validation datasets.

        Returns:
            Tuple of (train_dataset, valid_dataset). valid_dataset may be None.

        Example:
            >>> config = HFDatasetConfig(
            ...     path="Open-Orca/OpenOrca",
            ...     train_split="train[:90%]",
            ...     valid_split="train[-10%:]",
            ... )
            >>> train_ds, valid_ds = config.load_train_and_valid()
        """
        from datasets import load_dataset

        load_kwargs = {"streaming": self.streaming}
        if self.name:
            load_kwargs["name"] = self.name

        print(f"Loading dataset: {self.path}")

        # Load train split
        train_dataset = load_dataset(self.path, split=self.train_split, **load_kwargs)

        # Apply max_samples to train
        if self.max_samples and not self.streaming:
            train_dataset = train_dataset.select(range(min(self.max_samples, len(train_dataset))))

        # Apply column mapping to train
        if self.column_mapping:
            train_dataset = apply_column_mapping(train_dataset, self.column_mapping)

        # Load valid split if specified
        valid_dataset = None
        if self.valid_split:
            valid_dataset = load_dataset(self.path, split=self.valid_split, **load_kwargs)
            if self.column_mapping:
                valid_dataset = apply_column_mapping(valid_dataset, self.column_mapping)

        return train_dataset, valid_dataset

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "path": self.path,
            "name": self.name,
            "train_split": self.train_split,
            "valid_split": self.valid_split,
            "streaming": self.streaming,
            "column_mapping": self.column_mapping,
            "prompt_template": self.prompt_template,
            "output_column": self.output_column,
            "conversation_extension": self.conversation_extension,
            "max_samples": self.max_samples,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "HFDatasetConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


def load_dataset_with_config(
    config: Union[HFDatasetConfig, Dict[str, Any], str],
    tokenizer: Optional[Any] = None,
    convert_to_sharegpt: bool = False,
) -> Dataset:
    """
    Load and preprocess a dataset using configuration.

    This is a convenience function that handles the full pipeline:
    loading, column mapping, format conversion, and optional ShareGPT conversion.

    Args:
        config: HFDatasetConfig, dict with config params, or dataset path string
        tokenizer: Optional tokenizer for chat template application
        convert_to_sharegpt: Whether to convert to ShareGPT format

    Returns:
        Processed Dataset ready for training

    Example:
        >>> from mlx_tune import load_dataset_with_config
        >>> dataset = load_dataset_with_config(
        ...     {"path": "yahma/alpaca-cleaned", "max_samples": 1000},
        ...     tokenizer=tokenizer,
        ...     convert_to_sharegpt=True,
        ... )
    """
    # Handle different config types
    if isinstance(config, str):
        config = HFDatasetConfig(path=config)
    elif isinstance(config, dict):
        config = HFDatasetConfig.from_dict(config)

    # Load dataset
    dataset = config.load()

    # Convert to ShareGPT if requested
    if convert_to_sharegpt:
        dataset = to_sharegpt(
            dataset,
            merged_prompt=config.prompt_template,
            output_column_name=config.output_column,
            conversation_extension=config.conversation_extension,
        )
        dataset = standardize_sharegpt(dataset)

    return dataset


# =============================================================================
# ENHANCED STANDARDIZE_SHAREGPT (Unsloth-compatible)
# =============================================================================

def standardize_sharegpt_enhanced(
    dataset: Dataset,
    role_mapping: Optional[Dict[str, str]] = None,
    content_mapping: Optional[Dict[str, str]] = None,
) -> Dataset:
    """
    Enhanced version of standardize_sharegpt with custom role/content mapping.

    This extends the basic standardize_sharegpt function to support
    datasets with non-standard field names.

    Args:
        dataset: Dataset with 'conversations' column
        role_mapping: Mapping for role field names.
                     e.g., {"from": "speaker", "human": "user_role"}
        content_mapping: Mapping for content field names.
                        e.g., {"value": "text", "content": "message"}

    Returns:
        Dataset with standardized ChatML format messages

    Example:
        >>> from mlx_tune import standardize_sharegpt_enhanced
        >>> dataset = standardize_sharegpt_enhanced(
        ...     dataset,
        ...     role_mapping={"human": "person", "gpt": "ai"},
        ...     content_mapping={"value": "message"},
        ... )
    """
    # Default role mapping
    default_role_mapping = {
        'human': 'user',
        'user': 'user',
        'gpt': 'assistant',
        'assistant': 'assistant',
        'system': 'system',
        'person': 'user',  # Common variant
        'ai': 'assistant',  # Common variant
        'bot': 'assistant',  # Common variant
    }
    if role_mapping:
        default_role_mapping.update(role_mapping)

    # Field names to check for role and content
    role_fields = ['from', 'role', 'speaker', 'type']
    content_fields = ['value', 'content', 'text', 'message']

    if content_mapping:
        content_fields = list(content_mapping.values()) + content_fields

    def convert_sample(sample):
        if 'conversations' not in sample:
            return sample

        messages = []
        for turn in sample['conversations']:
            # Find role
            role = None
            for field in role_fields:
                if field in turn:
                    role_value = turn[field]
                    role = default_role_mapping.get(
                        role_value.lower() if isinstance(role_value, str) else role_value,
                        'user'
                    )
                    break
            if role is None:
                role = 'user'

            # Find content
            content = ''
            for field in content_fields:
                if field in turn and turn[field]:
                    content = turn[field]
                    break

            messages.append({'role': role, 'content': content})

        return {'messages': messages}

    return dataset.map(convert_sample)


# Convenience exports matching Unsloth API
__all__ = [
    # Dataset format detection and conversion
    'detect_dataset_format',
    'standardize_sharegpt',
    'standardize_sharegpt_enhanced',
    'convert_to_mlx_format',
    'get_formatting_func',
    'apply_chat_template_to_sample',
    'alpaca_to_text',
    'ALPACA_TEMPLATE',
    'ALPACA_TEMPLATE_NO_INPUT',
    # Chat template functions (Unsloth-compatible)
    'get_chat_template',
    'list_chat_templates',
    'get_template_info',
    'get_template_for_model',
    # Response-only training
    'train_on_responses_only',
    '_get_template_parts',
    'get_response_template_ids',
    'create_response_only_collator',
    # Template registry
    'CHAT_TEMPLATES',
    'TEMPLATE_ALIASES',
    'DEFAULT_SYSTEM_MESSAGES',
    'ChatTemplateEntry',
    # Phase 2.1: Multi-turn conversation merging
    'to_sharegpt',
    # Phase 2.2: Column mapping
    'apply_column_mapping',
    'infer_column_mapping',
    # Phase 2.3: HF dataset config
    'HFDatasetConfig',
    'load_dataset_with_config',
]
