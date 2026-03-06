import mlx.core as mx
import mlx.nn as nn
import pytest


class SmallBackbone(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]

    def __call__(self, x):
        h = self.embedding(x)
        for layer in self.layers:
            h = mx.maximum(layer(h), 0)
        return h


class SmallLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.model = SmallBackbone(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers)
        self.output = nn.Linear(hidden_size, vocab_size)

    def __call__(self, x):
        return self.output(self.model(x))


class MockTokenizer:
    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True):
        ids = [((ord(char) % (self.vocab_size - 3)) + 3) for char in text[:32]]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids, skip_special_tokens: bool = True):
        if skip_special_tokens:
            ids = [token for token in ids if token not in (self.pad_token_id, self.eos_token_id, self.bos_token_id)]
        return "".join(chr(65 + (token % 26)) for token in ids)


class MockModelWrapper:
    def __init__(self, model: SmallLanguageModel):
        self.model = model
        self._lora_applied = False

    def __call__(self, x):
        return self.model(x)

    def _apply_lora(self):
        self._lora_applied = True
        return True


def make_model(seed: int) -> MockModelWrapper:
    mx.random.seed(seed)
    model = SmallLanguageModel()
    mx.eval(model.parameters())
    return MockModelWrapper(model)


def test_prepare_rl_dataset_auto_detect_and_chat_adaptation():
    from mlx_tune import prepare_rl_dataset

    prompt_dataset = prepare_rl_dataset([{"prompt": "Solve 2 + 2", "answer": "4"}])
    assert prompt_dataset.mode == "prompt"
    assert prompt_dataset.samples[0]["reward_context"] == "4"

    chat_dataset = prepare_rl_dataset(
        [
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ],
                "score": 1.0,
            }
        ],
        mode="reward_scalar",
        tokenizer=MockTokenizer(),
    )
    assert chat_dataset.mode == "reward_scalar"
    assert chat_dataset.adapter_name == "chat_reward_scalar"
    assert chat_dataset.samples[0]["response"] == "Hello"


def test_prepare_rl_dataset_raises_on_ambiguous_pairwise_schema():
    from mlx_tune import prepare_rl_dataset

    with pytest.raises(ValueError, match="Ambiguous RL dataset schema"):
        prepare_rl_dataset([{"prompt": "Q", "chosen": "A", "rejected": "B"}])


def test_resume_from_checkpoint_returns_bundle_with_manifest_fields(tmp_path):
    from mlx_tune import RewardConfig, RewardTrainer, build_reward_model, resume_from_checkpoint

    tokenizer = MockTokenizer()
    reward_model = build_reward_model(make_model(100))
    trainer = RewardTrainer(
        model=reward_model,
        train_dataset=[
            {"prompt": "Q:", "response": "good", "score": 1.0},
            {"prompt": "Q:", "response": "bad", "score": 0.0},
        ],
        tokenizer=tokenizer,
        args=RewardConfig(
            learning_rate=1e-2,
            max_steps=1,
            logging_steps=1,
            save_steps=1,
            output_dir=str(tmp_path / "reward"),
        ),
    )

    trainer.train()
    bundle = resume_from_checkpoint(tmp_path / "reward")

    assert bundle.algorithm == "reward"
    assert "reward_model" in bundle.restored_roles
    assert bundle.trainer_state["global_step"] == 1
    assert bundle.metrics_history
    assert bundle.source_format == "manifest"
