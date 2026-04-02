"""
OCR Fine-Tuning Support for MLX-Tune

Provides OCR-specific API built on top of the VLM infrastructure:
- FastOCRModel — loads any OCR or VLM model with OCR-optimized defaults
- OCRSFTTrainer — SFT training with post-training CER/WER evaluation
- OCRGRPOTrainer — GRPO training with character-level reward functions
- Evaluation metrics — CER, WER, exact match
- Dataset helpers — convert common OCR dataset formats
- Reward functions — for GRPO training with OCR objectives

Two training tracks:
1. Fine-tune a dedicated OCR model (DeepSeek-OCR, GLM-OCR, etc.) on domain-specific data
2. Fine-tune a general VLM (Qwen3.5, Pixtral, etc.) for OCR tasks

Usage (matches Unsloth API pattern):
    from mlx_tune import FastOCRModel, OCRSFTTrainer, OCRSFTConfig

    model, processor = FastOCRModel.from_pretrained(
        "mlx-community/DeepSeek-OCR-8bit",
    )
    model = FastOCRModel.get_peft_model(model, r=16, lora_alpha=16)

    # OCR-specific methods
    text = model.transcribe(image)
    metrics = model.evaluate(test_images, ground_truths)
"""

from typing import Optional, Any, List, Dict, Union


# ============================================================================
# OCR Evaluation Metrics (pure Python, no extra dependencies)
# ============================================================================

def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertion, deletion, substitution
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,       # insertion
                prev_row[j + 1] + 1,   # deletion
                prev_row[j] + cost,    # substitution
            ))
        prev_row = curr_row
    return prev_row[-1]


def compute_cer(prediction: str, reference: str) -> float:
    """
    Compute Character Error Rate (CER) between prediction and reference.

    CER = edit_distance(prediction, reference) / len(reference)

    Args:
        prediction: Predicted text from OCR model
        reference: Ground truth text

    Returns:
        CER value (0.0 = perfect, >1.0 possible with many insertions)
    """
    if len(reference) == 0:
        return 0.0 if len(prediction) == 0 else float(len(prediction))
    return _levenshtein_distance(prediction, reference) / len(reference)


def _word_levenshtein_distance(words1: List[str], words2: List[str]) -> int:
    """Compute Levenshtein distance between two word sequences."""
    if len(words1) < len(words2):
        return _word_levenshtein_distance(words2, words1)
    if len(words2) == 0:
        return len(words1)

    prev_row = list(range(len(words2) + 1))
    for i, w1 in enumerate(words1):
        curr_row = [i + 1]
        for j, w2 in enumerate(words2):
            cost = 0 if w1 == w2 else 1
            curr_row.append(min(
                curr_row[j] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + cost,
            ))
        prev_row = curr_row
    return prev_row[-1]


def compute_wer(prediction: str, reference: str) -> float:
    """
    Compute Word Error Rate (WER) between prediction and reference.

    WER = edit_distance(pred_words, ref_words) / len(ref_words)

    Args:
        prediction: Predicted text from OCR model
        reference: Ground truth text

    Returns:
        WER value (0.0 = perfect, >1.0 possible with many insertions)
    """
    pred_words = prediction.split()
    ref_words = reference.split()
    if len(ref_words) == 0:
        return 0.0 if len(pred_words) == 0 else float(len(pred_words))

    return _word_levenshtein_distance(pred_words, ref_words) / len(ref_words)


def compute_exact_match(prediction: str, reference: str, normalize: bool = True) -> float:
    """
    Compute exact match score between prediction and reference.

    Args:
        prediction: Predicted text from OCR model
        reference: Ground truth text
        normalize: If True, strip whitespace and lowercase before comparing

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    if normalize:
        prediction = prediction.strip().lower()
        reference = reference.strip().lower()
    return 1.0 if prediction == reference else 0.0


def compute_ocr_metrics(
    predictions: List[str],
    references: List[str],
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Compute all OCR metrics over a batch of predictions.

    Args:
        predictions: List of predicted texts
        references: List of ground truth texts
        normalize: If True, normalize for exact match comparison

    Returns:
        Dict with keys: "cer", "wer", "exact_match" (averaged over batch)
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )
    if len(predictions) == 0:
        return {"cer": 0.0, "wer": 0.0, "exact_match": 0.0}

    cers = [compute_cer(p, r) for p, r in zip(predictions, references)]
    wers = [compute_wer(p, r) for p, r in zip(predictions, references)]
    ems = [compute_exact_match(p, r, normalize) for p, r in zip(predictions, references)]

    return {
        "cer": sum(cers) / len(cers),
        "wer": sum(wers) / len(wers),
        "exact_match": sum(ems) / len(ems),
    }


# ============================================================================
# GRPO Reward Functions for OCR
# ============================================================================

def cer_reward(response: str, ground_truth: str) -> float:
    """
    Reward function based on Character Error Rate.

    Returns 1.0 - CER, clamped to [0.0, 1.0].
    Perfect transcription = 1.0, complete mismatch = 0.0.
    """
    return max(0.0, 1.0 - compute_cer(response, ground_truth))


def exact_match_reward(response: str, ground_truth: str) -> float:
    """
    Binary reward: 1.0 if exact match (after normalization), 0.0 otherwise.
    """
    return compute_exact_match(response, ground_truth, normalize=True)


def combined_ocr_reward(
    response: str,
    ground_truth: str,
    cer_weight: float = 0.7,
    em_weight: float = 0.3,
) -> float:
    """
    Combined OCR reward: weighted sum of CER-based and exact match rewards.

    Default: 70% character accuracy + 30% exact match bonus.
    """
    return cer_weight * cer_reward(response, ground_truth) + em_weight * exact_match_reward(response, ground_truth)


# ============================================================================
# Dataset Helpers
# ============================================================================

def convert_ocr_pairs_to_messages(
    image: Any,
    text: str,
    instruction: Optional[str] = None,
) -> Dict:
    """
    Convert an (image, text) pair to VLM conversation format.

    Args:
        image: PIL Image, file path, or URL
        text: Ground truth transcription
        instruction: Custom instruction prompt (default: "Transcribe the text in this image.")

    Returns:
        Dict with "messages" key in VLM conversation format
    """
    if instruction is None:
        instruction = "Transcribe the text in this image."

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": image},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            },
        ]
    }


def load_ocr_dataset(
    dataset_name_or_path: str,
    image_column: str = "image",
    text_column: str = "text",
    instruction: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Load an OCR dataset from HuggingFace and convert to VLM message format.

    Args:
        dataset_name_or_path: HuggingFace dataset name or local path
        image_column: Column name containing images
        text_column: Column name containing ground truth text
        instruction: Custom instruction prompt
        split: Dataset split to load (supports slicing like "train[:50]")
        max_samples: Maximum number of samples to load (alternative to split slicing)

    Returns:
        List of dicts in VLM conversation format
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name_or_path, split=split)

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    # Auto-detect column names if defaults don't exist
    cols = dataset.column_names
    if image_column not in cols:
        # Try common alternatives
        for alt in ["image", "img", "picture", "photo", "input_image"]:
            if alt in cols:
                image_column = alt
                break
        else:
            raise ValueError(
                f"Image column '{image_column}' not found. Available: {cols}"
            )

    if text_column not in cols:
        for alt in ["text", "label", "transcription", "ground_truth", "gt", "caption", "ocr_text"]:
            if alt in cols:
                text_column = alt
                break
        else:
            raise ValueError(
                f"Text column '{text_column}' not found. Available: {cols}"
            )

    converted = []
    for sample in dataset:
        converted.append(convert_ocr_pairs_to_messages(
            image=sample[image_column],
            text=str(sample[text_column]),
            instruction=instruction,
        ))

    return converted


# ============================================================================
# OCR Models Registry
# ============================================================================

OCR_MODELS = {
    # Dedicated OCR models (pre-trained specifically for document understanding)
    "mlx-community/DeepSeek-OCR-8bit": {
        "type": "dedicated",
        "params": "0.9B",
        "description": "DeepSeek's document OCR model with 32x vision token compression",
        "quantization": "8-bit",
    },
    "mlx-community/DeepSeek-OCR-4bit": {
        "type": "dedicated",
        "params": "0.9B",
        "description": "DeepSeek's document OCR model (4-bit quantized)",
        "quantization": "4-bit",
    },
    "mlx-community/DeepSeek-OCR-2-8bit": {
        "type": "dedicated",
        "params": "1B",
        "description": "DeepSeek OCR v2 with improved accuracy",
        "quantization": "8-bit",
    },
    "mlx-community/GLM-OCR-bf16": {
        "type": "dedicated",
        "params": "0.9B",
        "description": "GLM-OCR: lightweight model beating larger models on OmniDocBench",
        "quantization": "bf16",
    },
    "mlx-community/GLM-OCR-4bit": {
        "type": "dedicated",
        "params": "0.9B",
        "description": "GLM-OCR (4-bit quantized)",
        "quantization": "4-bit",
    },
    "mlx-community/dots.ocr-4bit": {
        "type": "dedicated",
        "params": "~1B",
        "description": "DOTS OCR specialized model",
        "quantization": "4-bit",
    },
    # Derived OCR models (VLMs fine-tuned for OCR)
    "mlx-community/olmOCR-2-7B-1025-5bit": {
        "type": "derived",
        "params": "7B",
        "description": "olmOCR-2: Qwen2.5-VL fine-tuned on 270K PDF pages",
        "quantization": "5-bit",
    },
    "mlx-community/LightOnOCR-1B-1025-bf16": {
        "type": "derived",
        "params": "1B",
        "description": "LightOnOCR: end-to-end multilingual OCR model",
        "quantization": "bf16",
    },
    # General VLMs excellent for OCR fine-tuning
    "mlx-community/Qwen3.5-0.8B-bf16": {
        "type": "general_vlm",
        "params": "0.8B",
        "description": "Qwen3.5 VLM — lightweight, great for OCR fine-tuning",
        "quantization": "bf16",
    },
    "mlx-community/Qwen2.5-VL-7B-Instruct-4bit": {
        "type": "general_vlm",
        "params": "7B",
        "description": "Qwen2.5-VL — strong vision understanding for OCR tasks",
        "quantization": "4-bit",
    },
    "mlx-community/pixtral-12b-4bit": {
        "type": "general_vlm",
        "params": "12B",
        "description": "Pixtral — Mistral's vision model with 400M vision encoder",
        "quantization": "4-bit",
    },
}


# ============================================================================
# OCR Model Wrapper
# ============================================================================

class OCRModelWrapper:
    """
    Wrapper around VLMModelWrapper with OCR-specific methods.

    Adds transcribe(), batch_transcribe(), and evaluate() while delegating
    all standard VLM operations (generate, save, load, merge) to the
    underlying VLMModelWrapper.

    Example:
        >>> model, processor = FastOCRModel.from_pretrained(
        ...     "mlx-community/DeepSeek-OCR-8bit",
        ... )
        >>> text = model.transcribe(image)
        >>> metrics = model.evaluate(test_images, ground_truths)
    """

    # Default prompts per model type
    _DEFAULT_PROMPTS = {
        "deepseek": "OCR the text in this image.",
        "glm": "OCR this image.",
        "olmocr": "Extract all text from this document image.",
        "default": "Transcribe the text in this image.",
    }

    def __init__(self, vlm_wrapper: Any, model_name: Optional[str] = None):
        self._vlm = vlm_wrapper
        self._model_name = model_name or ""

    def _get_default_prompt(self) -> str:
        """Get model-appropriate default OCR prompt."""
        name_lower = self._model_name.lower()
        for key, prompt in self._DEFAULT_PROMPTS.items():
            if key in name_lower:
                return prompt
        return self._DEFAULT_PROMPTS["default"]

    def transcribe(
        self,
        image: Any,
        prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """
        Transcribe text from a single image.

        Args:
            image: PIL Image, file path, or URL
            prompt: Custom OCR prompt (uses model-appropriate default if None)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy, recommended for OCR)

        Returns:
            Transcribed text
        """
        if prompt is None:
            prompt = self._get_default_prompt()
        return self._vlm.generate(
            prompt=prompt,
            image=image,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def batch_transcribe(
        self,
        images: List[Any],
        prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        verbose: bool = True,
        **kwargs,
    ) -> List[str]:
        """
        Transcribe text from multiple images sequentially.

        Args:
            images: List of PIL Images, file paths, or URLs
            prompt: Custom OCR prompt (uses model-appropriate default if None)
            max_tokens: Maximum tokens to generate per image
            temperature: Sampling temperature
            verbose: Print progress

        Returns:
            List of transcribed texts
        """
        results = []
        for i, img in enumerate(images):
            if verbose:
                print(f"  Transcribing {i+1}/{len(images)}...", end="\r")
            results.append(self.transcribe(
                image=img, prompt=prompt, max_tokens=max_tokens,
                temperature=temperature, **kwargs,
            ))
        if verbose:
            print(f"  Transcribed {len(images)} images.          ")
        return results

    def evaluate(
        self,
        images: List[Any],
        references: List[str],
        prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        verbose: bool = True,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Evaluate OCR performance on a set of images with ground truth.

        Args:
            images: List of test images
            references: List of ground truth texts
            prompt: Custom OCR prompt
            max_tokens: Maximum tokens per transcription
            temperature: Sampling temperature
            verbose: Print progress and results

        Returns:
            Dict with "cer", "wer", "exact_match" metrics
        """
        if len(images) != len(references):
            raise ValueError(
                f"Length mismatch: {len(images)} images vs {len(references)} references"
            )

        predictions = self.batch_transcribe(
            images, prompt=prompt, max_tokens=max_tokens,
            temperature=temperature, verbose=verbose, **kwargs,
        )

        metrics = compute_ocr_metrics(predictions, references)

        if verbose:
            print(f"  CER:         {metrics['cer']:.4f}")
            print(f"  WER:         {metrics['wer']:.4f}")
            print(f"  Exact Match: {metrics['exact_match']:.4f}")

        return metrics

    # Delegate everything else to the underlying VLMModelWrapper
    def generate(self, *args, **kwargs):
        return self._vlm.generate(*args, **kwargs)

    def stream_generate(self, *args, **kwargs):
        return self._vlm.stream_generate(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        return self._vlm.save_pretrained(*args, **kwargs)

    def load_adapter(self, *args, **kwargs):
        return self._vlm.load_adapter(*args, **kwargs)

    def save_pretrained_merged(self, *args, **kwargs):
        return self._vlm.save_pretrained_merged(*args, **kwargs)

    def save_pretrained_gguf(self, *args, **kwargs):
        return self._vlm.save_pretrained_gguf(*args, **kwargs)

    def train(self):
        return self._vlm.model.train()

    def eval(self):
        return self._vlm.model.eval()

    @property
    def model(self):
        return self._vlm.model

    @property
    def processor(self):
        return self._vlm.processor

    @property
    def config(self):
        return self._vlm.config

    @property
    def lora_config(self):
        return self._vlm.lora_config

    @property
    def _lora_applied(self):
        return self._vlm._lora_applied

    def __getattr__(self, name):
        # Avoid infinite recursion for our own attributes
        if name in ("_vlm", "_model_name"):
            raise AttributeError(name)
        return getattr(self._vlm, name)


# ============================================================================
# FastOCRModel API
# ============================================================================

class FastOCRModel:
    """
    Unsloth-compatible API for OCR model fine-tuning on Apple Silicon.

    Wraps FastVisionModel with OCR-optimized defaults:
    - Vision layers frozen by default (dedicated OCR models have pre-optimized encoders)
    - Language layers fine-tuned for domain adaptation
    - Returns OCRModelWrapper with transcribe() and evaluate() methods

    Two training tracks:
    1. Dedicated OCR models: DeepSeek-OCR, GLM-OCR, DOTS-OCR, etc.
    2. General VLMs for OCR: Qwen3.5, Qwen2.5-VL, Pixtral, etc.

    Example:
        >>> model, processor = FastOCRModel.from_pretrained(
        ...     "mlx-community/DeepSeek-OCR-8bit",
        ... )
        >>> model = FastOCRModel.get_peft_model(model, r=16, lora_alpha=16)
        >>> text = model.transcribe(image)
    """

    @staticmethod
    def from_pretrained(
        model_name: str,
        load_in_4bit: bool = False,
        use_gradient_checkpointing: Union[bool, str] = False,
        **kwargs,
    ):
        """
        Load an OCR or VLM model for OCR fine-tuning.

        Args:
            model_name: HuggingFace model name (e.g., "mlx-community/DeepSeek-OCR-8bit")
            load_in_4bit: Use 4-bit quantization
            use_gradient_checkpointing: Enable gradient checkpointing

        Returns:
            Tuple of (OCRModelWrapper, processor)
        """
        from mlx_tune.vlm import FastVisionModel

        vlm_wrapper, processor = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing=use_gradient_checkpointing,
            **kwargs,
        )

        ocr_wrapper = OCRModelWrapper(vlm_wrapper, model_name=model_name)
        return ocr_wrapper, processor

    @staticmethod
    def get_peft_model(
        model: "OCRModelWrapper",
        finetune_vision_layers: bool = False,
        finetune_language_layers: bool = True,
        finetune_attention_modules: bool = True,
        finetune_mlp_modules: bool = True,
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0,
        bias: str = "none",
        random_state: int = 3407,
        use_rslora: bool = False,
        loftq_config: Any = None,
        **kwargs,
    ) -> "OCRModelWrapper":
        """
        Add LoRA adapters to the OCR model.

        Key difference from FastVisionModel: defaults finetune_vision_layers=False.
        Dedicated OCR models have pre-optimized vision encoders — fine-tuning only
        the language layers is more parameter-efficient for domain adaptation.

        Set finetune_vision_layers=True when fine-tuning a general VLM for OCR
        from scratch (e.g., Qwen3.5 on LaTeX OCR).

        Args:
            model: OCRModelWrapper from from_pretrained()
            finetune_vision_layers: Fine-tune vision encoder (default: False)
            finetune_language_layers: Fine-tune language model (default: True)
            r: LoRA rank
            lora_alpha: LoRA alpha (recommended: alpha == r)

        Returns:
            Same OCRModelWrapper with LoRA applied
        """
        from mlx_tune.vlm import FastVisionModel

        vlm_wrapper = model._vlm
        FastVisionModel.get_peft_model(
            vlm_wrapper,
            finetune_vision_layers=finetune_vision_layers,
            finetune_language_layers=finetune_language_layers,
            finetune_attention_modules=finetune_attention_modules,
            finetune_mlp_modules=finetune_mlp_modules,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            random_state=random_state,
            use_rslora=use_rslora,
            loftq_config=loftq_config,
            **kwargs,
        )
        return model

    @staticmethod
    def for_training(model: "OCRModelWrapper"):
        """Switch model to training mode."""
        from mlx_tune.vlm import FastVisionModel
        FastVisionModel.for_training(model._vlm)

    @staticmethod
    def for_inference(model: "OCRModelWrapper"):
        """Switch model to inference mode."""
        from mlx_tune.vlm import FastVisionModel
        FastVisionModel.for_inference(model._vlm)


# ============================================================================
# OCR Training Configs
# ============================================================================

class OCRSFTConfig:
    """
    Training configuration for OCR SFT fine-tuning.

    Extends VLMSFTConfig with OCR-tuned defaults:
    - Lower learning rate (5e-5) for stable domain adaptation
    - Longer max_length (4096) for complex documents
    - train_on_completions=True (only learn from transcription, not prompt)

    All parameters from VLMSFTConfig are supported.
    """

    def __init__(
        self,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 5,
        max_steps: Optional[int] = None,
        num_train_epochs: int = 1,
        learning_rate: float = 5e-5,
        logging_steps: int = 1,
        optim: str = "adam",
        weight_decay: float = 0.01,
        lr_scheduler_type: str = "linear",
        seed: int = 3407,
        output_dir: str = "ocr_outputs",
        report_to: str = "none",
        remove_unused_columns: bool = False,
        dataset_text_field: str = "",
        dataset_kwargs: Optional[Dict] = None,
        max_length: int = 4096,
        train_on_completions: bool = True,
        # OCR-specific
        eval_dataset: Optional[Any] = None,
        eval_steps: Optional[int] = None,
        **kwargs,
    ):
        from mlx_tune.vlm import VLMSFTConfig
        self._vlm_config = VLMSFTConfig(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            optim=optim,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            seed=seed,
            output_dir=output_dir,
            report_to=report_to,
            remove_unused_columns=remove_unused_columns,
            dataset_text_field=dataset_text_field,
            dataset_kwargs=dataset_kwargs,
            max_length=max_length,
            train_on_completions=train_on_completions,
            **kwargs,
        )
        self.eval_dataset = eval_dataset
        self.eval_steps = eval_steps

    def __getattr__(self, name):
        if name in ("_vlm_config", "eval_dataset", "eval_steps"):
            raise AttributeError(name)
        return getattr(self._vlm_config, name)


class OCRGRPOConfig:
    """
    Configuration for OCR GRPO training.

    Extends VLMGRPOConfig with OCR-tuned defaults.
    Uses combined_ocr_reward (CER + exact match) as default reward function.
    """

    def __init__(
        self,
        beta: float = 0.04,
        num_generations: int = 2,
        temperature: float = 0.7,
        max_completion_length: int = 512,
        output_dir: str = "./ocr_grpo_outputs",
        learning_rate: float = 1e-6,
        max_steps: int = -1,
        num_train_epochs: int = 1,
        logging_steps: int = 1,
        save_steps: int = 100,
        reward_fn: Optional[Any] = None,
    ):
        from mlx_tune.vlm import VLMGRPOConfig
        self._vlm_config = VLMGRPOConfig(
            beta=beta,
            num_generations=num_generations,
            temperature=temperature,
            max_completion_length=max_completion_length,
            output_dir=output_dir,
            learning_rate=learning_rate,
            max_steps=max_steps,
            num_train_epochs=num_train_epochs,
            logging_steps=logging_steps,
            save_steps=save_steps,
            reward_fn=reward_fn,
        )

    def __getattr__(self, name):
        if name == "_vlm_config":
            raise AttributeError(name)
        return getattr(self._vlm_config, name)


# ============================================================================
# OCR Trainers
# ============================================================================

class OCRSFTTrainer:
    """
    OCR Supervised Fine-Tuning Trainer.

    Wraps VLMSFTTrainer with optional post-training evaluation using
    CER/WER/exact match metrics.

    Example:
        >>> trainer = OCRSFTTrainer(
        ...     model=model,
        ...     processor=processor,
        ...     train_dataset=train_data,
        ...     args=OCRSFTConfig(max_steps=30, eval_dataset=eval_data),
        ... )
        >>> stats = trainer.train()
    """

    def __init__(
        self,
        model: Any,
        processor: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        data_collator: Optional[Any] = None,
        train_dataset: Optional[Any] = None,
        args: Optional[OCRSFTConfig] = None,
        **kwargs,
    ):
        from mlx_tune.vlm import VLMSFTTrainer, UnslothVisionDataCollator

        if args is None:
            args = OCRSFTConfig()

        self.ocr_config = args
        self.eval_dataset = args.eval_dataset
        self.eval_steps = args.eval_steps

        # Get the underlying VLM wrapper for the collator
        proc = processor or tokenizer
        vlm_wrapper = model._vlm if isinstance(model, OCRModelWrapper) else model

        if data_collator is None:
            data_collator = UnslothVisionDataCollator(vlm_wrapper, proc)

        self._trainer = VLMSFTTrainer(
            model=vlm_wrapper,
            tokenizer=proc,
            data_collator=data_collator,
            train_dataset=train_dataset,
            args=args._vlm_config,
            **kwargs,
        )

        # Keep reference to OCR wrapper for evaluation
        self._ocr_model = model if isinstance(model, OCRModelWrapper) else None

    def train(self) -> Any:
        """
        Run OCR SFT training, optionally evaluating after training.

        Returns:
            Training statistics from VLMSFTTrainer
        """
        stats = self._trainer.train()

        # Post-training evaluation if eval_dataset provided
        if self.eval_dataset is not None and self._ocr_model is not None:
            print("\n--- Post-Training OCR Evaluation ---")
            images = []
            references = []
            for sample in self.eval_dataset:
                msgs = sample.get("messages", [])
                for msg in msgs:
                    if msg.get("role") == "user":
                        for content in msg.get("content", []):
                            if content.get("type") == "image":
                                images.append(content.get("image"))
                    if msg.get("role") == "assistant":
                        for content in msg.get("content", []):
                            if content.get("type") == "text":
                                references.append(content.get("text", ""))

            if images and references and len(images) == len(references):
                FastOCRModel.for_inference(self._ocr_model)
                self._ocr_model.evaluate(images, references)
                FastOCRModel.for_training(self._ocr_model)

        return stats

    @property
    def metrics(self):
        return getattr(self._trainer, "metrics", {})


class OCRGRPOTrainer:
    """
    OCR GRPO (Group Relative Policy Optimization) Trainer.

    Wraps VLMGRPOTrainer with CER-based reward function as default.
    Uses character accuracy and exact match rewards to optimize OCR output.

    Example:
        >>> trainer = OCRGRPOTrainer(
        ...     model=model,
        ...     train_dataset=grpo_dataset,
        ...     processor=processor,
        ...     args=OCRGRPOConfig(num_generations=2, max_steps=10),
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: Any,
        train_dataset: Any,
        processor: Optional[Any] = None,
        reward_fn: Optional[Any] = None,
        args: Optional[OCRGRPOConfig] = None,
        **kwargs,
    ):
        from mlx_tune.vlm import VLMGRPOTrainer

        if args is None:
            args = OCRGRPOConfig()

        # Default to combined OCR reward if no reward function provided
        if reward_fn is None and args.reward_fn is None:
            reward_fn = combined_ocr_reward

        vlm_wrapper = model._vlm if isinstance(model, OCRModelWrapper) else model

        self._trainer = VLMGRPOTrainer(
            model=vlm_wrapper,
            train_dataset=train_dataset,
            processor=processor,
            reward_fn=reward_fn,
            args=args._vlm_config,
            **kwargs,
        )

    def train(self) -> Any:
        """Run OCR GRPO training."""
        return self._trainer.train()
