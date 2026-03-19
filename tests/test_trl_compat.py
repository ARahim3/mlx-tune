import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest


class TinyModel(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def __call__(self, x):
        return self.output(self.embedding(x))


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True):
        token_ids = [((ord(char) % 10) + 3) for char in text]
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        return token_ids

    def decode(self, ids, skip_special_tokens: bool = True):
        if skip_special_tokens:
            ids = [token for token in ids if token not in (self.pad_token_id, self.eos_token_id, self.bos_token_id)]
        return "".join(chr(65 + (token % 26)) for token in ids)


@pytest.fixture(autouse=True)
def isolated_trl_modules():
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "trl" or name.startswith("trl.")
    }
    for name in list(saved):
        sys.modules.pop(name, None)
    yield
    for name in list(sys.modules):
        if name == "trl" or name.startswith("trl."):
            sys.modules.pop(name, None)
    sys.modules.update(saved)


def test_patch_fast_rl_creates_fallback_trl_module_when_missing():
    from mlx_tune import GRPOConfig as MLXGRPOConfig
    from mlx_tune import GRPOTrainer as MLXGRPOTrainer
    from mlx_tune import PatchFastRL
    from mlx_tune import SFTConfig as MLXSFTConfig
    from mlx_tune import SFTTrainer as MLXSFTTrainer

    PatchFastRL()

    trl = importlib.import_module("trl")
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

    assert trl.__MLX_TUNE_PATCHED__ is True
    assert trl.trainer.__MLX_TUNE_PATCHED__ is True
    assert issubclass(GRPOConfig, MLXGRPOConfig)
    assert issubclass(GRPOTrainer, MLXGRPOTrainer)
    assert issubclass(SFTConfig, MLXSFTConfig)
    assert issubclass(SFTTrainer, MLXSFTTrainer)
    assert trl.trainer.GRPOTrainer is GRPOTrainer
    assert trl.trainer.SFTConfig is SFTConfig


def test_patch_fast_rl_mutates_existing_module_in_place_and_is_idempotent():
    from mlx_tune import PatchFastRL

    trl_module = ModuleType("trl")
    trl_module.__package__ = "trl"
    trl_module.__path__ = []
    trl_module.keep_me = "present"
    trainer_module = ModuleType("trl.trainer")
    trainer_module.keep_me_too = "present"
    trl_module.trainer = trainer_module
    sys.modules["trl"] = trl_module
    sys.modules["trl.trainer"] = trainer_module

    PatchFastRL()
    first_grpo_trainer = trl_module.GRPOTrainer

    PatchFastRL()

    assert sys.modules["trl"] is trl_module
    assert sys.modules["trl.trainer"] is trainer_module
    assert trl_module.trainer is trainer_module
    assert trl_module.keep_me == "present"
    assert trainer_module.keep_me_too == "present"
    assert trl_module.GRPOTrainer is first_grpo_trainer
    assert trainer_module.GRPOTrainer is first_grpo_trainer


def test_grpo_config_normalizes_reward_aliases():
    from mlx_tune import PatchFastRL

    PatchFastRL()

    from trl import GRPOConfig

    reward_fn = lambda response, context: float(len(response) + len(context))
    reward_funcs = [reward_fn, reward_fn]

    config = GRPOConfig(
        reward_funcs=reward_funcs,
        reward_func=reward_fn,
        generations_per_prompt=3,
        baseline_mode="rloo",
    )

    assert config.reward_sources[:2] == reward_funcs
    assert config.reward_sources[2]["source"] is reward_fn
    assert config.reward_fn is reward_fn
    assert config.num_generations == 3
    assert config.advantage_estimator == "rloo"


def test_grpo_trainer_accepts_processing_class_and_foreign_args(tmp_path):
    from mlx_tune import PatchFastRL

    PatchFastRL()

    from trl import GRPOTrainer

    reward_fn = lambda response, context: float(len(response) + len(context))
    tokenizer = TinyTokenizer()
    model = TinyModel()
    mx.eval(model.parameters())
    output_dir = tmp_path / "grpo"

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=[{"prompt": "hi"}],
        eval_dataset=[{"prompt": "bye"}],
        args=SimpleNamespace(
            output_dir=str(output_dir),
            learning_rate=1e-5,
            per_device_train_batch_size=1,
            num_train_epochs=1,
            max_steps=1,
            logging_steps=1,
            save_steps=1,
            max_seq_length=8,
            max_completion_length=2,
            generations_per_prompt=2,
            reward_funcs=[reward_fn],
        ),
    )

    assert trainer.tokenizer is tokenizer
    assert trainer.eval_dataset == [{"prompt": "bye"}]
    assert trainer.config.num_generations == 2
    assert trainer.reward_sources == [reward_fn]
    assert trainer.output_dir == output_dir


def test_dpo_trainer_runs_via_patched_trl_import(tmp_path):
    from mlx_tune import PatchFastRL

    PatchFastRL()

    from trl import DPOTrainer

    tokenizer = TinyTokenizer()
    model = TinyModel()
    mx.eval(model.parameters())

    trainer = DPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=[
            {"prompt": "a", "chosen": "b", "rejected": "c"},
            {"prompt": "d", "chosen": "e", "rejected": "f"},
        ],
        args=SimpleNamespace(
            output_dir=str(tmp_path / "dpo"),
            learning_rate=1e-4,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            max_steps=1,
            warmup_steps=0,
            logging_steps=1,
            save_steps=1,
            max_seq_length=12,
            max_prompt_length=6,
        ),
    )

    result = trainer.train()

    assert result["status"] == "success"
    assert result["global_step"] == 1
    assert Path(result["adapter_path"]).exists()
