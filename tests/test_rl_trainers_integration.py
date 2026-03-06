"""
Integration tests for RL trainers.
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten


class SmallLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        self.output = nn.Linear(hidden_size, vocab_size)

    def __call__(self, x):
        h = self.embedding(x)
        for layer in self.layers:
            h = mx.maximum(layer(h), 0)
        return self.output(h)


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
        self.lora_config = None
        self._adapter_path = None

    def __call__(self, x):
        return self.model(x)

    def _apply_lora(self):
        self._lora_applied = True
        return True

    def set_adapter_path(self, path: str):
        self._adapter_path = path


def make_model(seed: int) -> MockModelWrapper:
    mx.random.seed(seed)
    model = SmallLanguageModel()
    mx.eval(model.parameters())
    return MockModelWrapper(model)


def parameter_snapshot(model_wrapper: MockModelWrapper):
    return {name: mx.array(value) for name, value in tree_flatten(model_wrapper.model.parameters())}


def parameters_changed(before, after_wrapper: MockModelWrapper) -> bool:
    after = {name: value for name, value in tree_flatten(after_wrapper.model.parameters())}
    for name, before_value in before.items():
        delta = mx.max(mx.abs(after[name] - before_value)).item()
        if delta > 1e-6:
            return True
    return False


def rollout_sequence_tensors(rollout_batch, index: int):
    sequence = rollout_batch.prompt_ids[index] + rollout_batch.completion_ids[index]
    return (
        mx.array([sequence]),
        mx.array([len(rollout_batch.prompt_ids[index])]),
        mx.array([rollout_batch.completion_lengths[index]]),
    )


@pytest.fixture
def tokenizer():
    return MockTokenizer()


@pytest.fixture
def preference_dataset():
    return [
        {
            "prompt": "What is machine learning?",
            "chosen": "Machine learning is learning from data.",
            "rejected": "Computers do stuff.",
        },
        {
            "prompt": "Explain Python.",
            "chosen": "Python is a high-level programming language.",
            "rejected": "Python is only a snake.",
        },
        {
            "prompt": "What is deep learning?",
            "chosen": "Deep learning uses many neural-network layers.",
            "rejected": "It is just regular learning.",
        },
    ]


@pytest.fixture
def kto_dataset():
    return [
        {"text": "Machine learning uses data.", "label": 1},
        {"text": "Computers maybe stuff.", "label": 0},
        {"text": "Python is a programming language.", "label": 1},
        {"text": "Snake only.", "label": 0},
    ]


@pytest.fixture
def grpo_dataset():
    return [
        {"prompt": "What is 2 + 2?", "answer": "4"},
        {"prompt": "What is 5 * 3?", "answer": "15"},
        {"prompt": "What is 10 - 7?", "answer": "3"},
    ]


@pytest.mark.integration
class TestDPOTrainerIntegration:
    def test_dpo_frozen_reference_stays_unchanged_across_policy_updates(
        self,
        tmp_path,
        tokenizer,
        preference_dataset,
    ):
        from mlx_tune import DPOConfig, DPOTrainer, compute_reference_logprobs

        model = make_model(0)
        trainer = DPOTrainer(
            model=model,
            train_dataset=preference_dataset,
            tokenizer=tokenizer,
            args=DPOConfig(
                learning_rate=5e-2,
                max_steps=3,
                logging_steps=1,
                save_steps=1,
                output_dir=str(tmp_path),
            ),
        )

        result = trainer.train()
        assert result["global_step"] == 3
        assert trainer.reference_policy is not None

        pad_id = tokenizer.pad_token_id
        sample = trainer.train_samples[0]
        chosen = mx.array([sample["chosen_ids"]])
        rejected = mx.array([sample["rejected_ids"]])
        chosen_lengths = mx.array([sample["chosen_length"]])
        rejected_lengths = mx.array([sample["rejected_length"]])

        ref_chosen, ref_rejected = compute_reference_logprobs(
            trainer.reference_policy.model.model,
            chosen,
            rejected,
            chosen_lengths,
            rejected_lengths,
        )

        assert mx.allclose(ref_chosen, mx.array([sample["reference_chosen_logprobs"]]))
        assert mx.allclose(ref_rejected, mx.array([sample["reference_rejected_logprobs"]]))
        assert model._lora_applied

    def test_dpo_loss_changes_when_reference_cache_changes(
        self,
        tokenizer,
        preference_dataset,
    ):
        from mlx_tune import dpo_loss, precompute_preference_reference_logprobs

        policy = make_model(1)
        reference_a = make_model(2)
        reference_b = make_model(3)

        prompt = preference_dataset[0]["prompt"]
        chosen = tokenizer.encode(prompt + preference_dataset[0]["chosen"])
        rejected = tokenizer.encode(prompt + preference_dataset[0]["rejected"])

        chosen_ids = mx.array([chosen])
        rejected_ids = mx.array([rejected])
        chosen_lengths = mx.array([len(chosen)])
        rejected_lengths = mx.array([len(rejected)])

        ref_a = precompute_preference_reference_logprobs(
            reference_a.model,
            chosen_ids,
            rejected_ids,
            chosen_lengths,
            rejected_lengths,
        )
        ref_b = precompute_preference_reference_logprobs(
            reference_b.model,
            chosen_ids,
            rejected_ids,
            chosen_lengths,
            rejected_lengths,
        )

        loss_a, _ = dpo_loss(
            policy.model,
            chosen_ids,
            rejected_ids,
            chosen_lengths,
            rejected_lengths,
            beta=0.1,
            reference_chosen_logprobs=ref_a[0],
            reference_rejected_logprobs=ref_a[1],
        )
        loss_b, _ = dpo_loss(
            policy.model,
            chosen_ids,
            rejected_ids,
            chosen_lengths,
            rejected_lengths,
            beta=0.1,
            reference_chosen_logprobs=ref_b[0],
            reference_rejected_logprobs=ref_b[1],
        )

        assert abs(loss_a.item() - loss_b.item()) > 1e-6

    def test_dpo_resume_restores_state_and_cache(
        self,
        tmp_path,
        tokenizer,
        preference_dataset,
    ):
        from mlx_tune import DPOConfig, DPOTrainer

        output_dir = tmp_path / "dpo_resume"
        trainer = DPOTrainer(
            model=make_model(4),
            train_dataset=preference_dataset,
            tokenizer=tokenizer,
            args=DPOConfig(
                learning_rate=1e-2,
                max_steps=2,
                logging_steps=1,
                save_steps=1,
                output_dir=str(output_dir),
            ),
        )
        trainer.train()

        resumed = DPOTrainer(
            model=make_model(5),
            train_dataset=preference_dataset,
            tokenizer=tokenizer,
            args=DPOConfig(
                learning_rate=1e-2,
                max_steps=4,
                logging_steps=1,
                save_steps=1,
                output_dir=str(output_dir),
            ),
        )
        result = resumed.train(resume_from_checkpoint=str(output_dir))

        assert result["global_step"] == 4
        assert resumed.cache_metadata == trainer.cache_metadata
        assert resumed.optimizer is not None
        assert resumed.optimizer.state["step"].item() == 4
        assert mx.allclose(
            mx.array([sample["reference_chosen_logprobs"] for sample in resumed.train_samples]),
            mx.array([sample["reference_chosen_logprobs"] for sample in trainer.train_samples]),
        )


@pytest.mark.integration
class TestGRPOTrainerIntegration:
    def test_grpo_training_changes_parameters_and_increases_rewarded_logprob(
        self,
        tmp_path,
        tokenizer,
        grpo_dataset,
    ):
        from mlx_tune import GRPOConfig, GRPOTrainer, compute_completion_log_probs

        mx.random.seed(7)
        model = make_model(6)
        before = parameter_snapshot(model)

        trainer = GRPOTrainer(
            model=model,
            train_dataset=grpo_dataset,
            tokenizer=tokenizer,
            reward_fn=lambda response, context: float(len(response)),
            args=GRPOConfig(
                learning_rate=5e-2,
                beta=0.01,
                num_generations=3,
                max_completion_length=4,
                max_steps=1,
                logging_steps=1,
                save_steps=1,
                output_dir=str(tmp_path),
            ),
        )

        result = trainer.train()
        rollout = trainer._last_rollout_batch

        assert result["global_step"] == 1
        assert rollout is not None
        assert parameters_changed(before, model)

        best_index = int(mx.argmax(rollout.rewards).item())
        input_ids, prompt_lengths, completion_lengths = rollout_sequence_tensors(rollout, best_index)
        updated_logprob = compute_completion_log_probs(
            model.model,
            input_ids,
            prompt_lengths,
            completion_lengths,
        )[0].item()
        rollout_logprob = rollout.rollout_logprobs[best_index].item()

        assert updated_logprob > rollout_logprob

    def test_grpo_prefers_answer_context_over_prompt(
        self,
        tmp_path,
        tokenizer,
    ):
        from mlx_tune import GRPOConfig, GRPOTrainer

        seen_contexts = []

        def reward_fn(response: str, context: str) -> float:
            seen_contexts.append(context)
            return float(len(response))

        trainer = GRPOTrainer(
            model=make_model(7),
            train_dataset=[{"prompt": "Solve 2 + 2", "answer": "4"}],
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            args=GRPOConfig(
                learning_rate=1e-2,
                num_generations=2,
                max_completion_length=3,
                max_steps=1,
                logging_steps=1,
                save_steps=1,
                output_dir=str(tmp_path),
            ),
        )
        trainer.train()

        assert seen_contexts
        assert all(context == "4" for context in seen_contexts)

    def test_grpo_resume_restores_rng_and_optimizer_state(
        self,
        tmp_path,
        tokenizer,
        grpo_dataset,
    ):
        from mlx_tune import GRPOConfig, GRPOTrainer

        output_dir = tmp_path / "grpo_resume"
        config = GRPOConfig(
            learning_rate=1e-2,
            beta=0.01,
            num_generations=2,
            max_completion_length=4,
            max_steps=1,
            logging_steps=1,
            save_steps=1,
            output_dir=str(output_dir),
        )

        mx.random.seed(11)
        trainer = GRPOTrainer(
            model=make_model(8),
            train_dataset=grpo_dataset,
            tokenizer=tokenizer,
            reward_fn=lambda response, context: float(len(response)),
            args=config,
        )
        trainer.train()

        def load_and_rollout(seed: int):
            restored = GRPOTrainer(
                model=make_model(seed),
                train_dataset=grpo_dataset,
                tokenizer=tokenizer,
                reward_fn=lambda response, context: float(len(response)),
                args=GRPOConfig(
                    learning_rate=1e-2,
                    beta=0.01,
                    num_generations=2,
                    max_completion_length=4,
                    max_steps=2,
                    logging_steps=1,
                    save_steps=1,
                    output_dir=str(output_dir),
                ),
            )
            restored._apply_lora_if_needed()
            restored._prepare_prompt_samples()
            optimizer = restored._optimizer_for_training()
            restored.optimizer = optimizer
            restored.load_state(optimizer, Path(output_dir))
            rollout = restored._collect_rollout_batch(restored._next_samples(restored.prompt_samples))
            return restored, rollout

        restored_a, rollout_a = load_and_rollout(9)
        restored_b, rollout_b = load_and_rollout(9)

        assert restored_a.optimizer.state["step"].item() == 1
        assert rollout_a.completion_ids == rollout_b.completion_ids
        assert mx.allclose(rollout_a.rollout_logprobs, rollout_b.rollout_logprobs)

        resumed = GRPOTrainer(
            model=make_model(12),
            train_dataset=grpo_dataset,
            tokenizer=tokenizer,
            reward_fn=lambda response, context: float(len(response)),
            args=GRPOConfig(
                learning_rate=1e-2,
                beta=0.01,
                num_generations=2,
                max_completion_length=4,
                max_steps=2,
                logging_steps=1,
                save_steps=1,
                output_dir=str(output_dir),
            ),
        )
        result = resumed.train(resume_from_checkpoint=str(output_dir))
        assert result["global_step"] == 2


@pytest.mark.integration
class TestKTOTrainerIntegration:
    def test_kto_uses_cached_reference_logprobs_instead_of_live_policy_outputs(
        self,
        tmp_path,
        tokenizer,
        kto_dataset,
    ):
        from mlx_tune import KTOTrainer, compute_log_probs_with_lengths, kto_loss

        trainer = KTOTrainer(
            model=make_model(13),
            train_dataset=kto_dataset,
            tokenizer=tokenizer,
            learning_rate=2e-2,
            max_steps=2,
            logging_steps=1,
            save_steps=1,
            output_dir=str(tmp_path),
        )
        trainer.train()

        sample = trainer.train_samples[0]
        input_ids, lengths, labels, cached_reference = trainer._build_batch([sample])
        cached_loss, _ = kto_loss(
            trainer.model.model,
            input_ids,
            lengths,
            labels,
            beta=trainer.beta,
            reference_logprobs=cached_reference,
        )
        live_policy_reference = compute_log_probs_with_lengths(
            trainer.model.model,
            input_ids,
            lengths,
        )
        live_loss, _ = kto_loss(
            trainer.model.model,
            input_ids,
            lengths,
            labels,
            beta=trainer.beta,
            reference_logprobs=live_policy_reference,
        )
        assert abs(cached_loss.item() - live_loss.item()) > 1e-6


@pytest.mark.integration
class TestOtherRLTrainers:
    def test_orpo_and_simpo_still_train(
        self,
        tmp_path,
        tokenizer,
        preference_dataset,
    ):
        from mlx_tune import ORPOConfig, ORPOTrainer, SimPOTrainer

        orpo = ORPOTrainer(
            model=make_model(14),
            train_dataset=preference_dataset,
            tokenizer=tokenizer,
            args=ORPOConfig(
                learning_rate=1e-2,
                max_steps=1,
                output_dir=str(tmp_path / "orpo"),
            ),
        )
        simpo = SimPOTrainer(
            model=make_model(15),
            train_dataset=preference_dataset,
            tokenizer=tokenizer,
            learning_rate=1e-2,
            max_steps=1,
            output_dir=str(tmp_path / "simpo"),
        )

        assert orpo.train()["status"] == "success"
        assert simpo.train()["status"] == "success"
