"""
Integration tests for RL trainers.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten


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
        self.lora_config = None
        self._adapter_path = None

    def __call__(self, x):
        return self.model(x)

    def _apply_lora(self):
        self._lora_applied = True
        return True

    def set_adapter_path(self, path: str):
        self._adapter_path = path


def write_legacy_rl_checkpoint(trainer, checkpoint_dir: Path):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = checkpoint_dir / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_weights = dict(tree_flatten(trainer.model.model.trainable_parameters()))
    mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), adapter_weights)

    state_arrays = {
        f"optimizer.{key}": value
        for key, value in tree_flatten(trainer.optimizer.state)
    }
    state_arrays.update({f"rng.{idx}": state for idx, state in enumerate(mx.random.state)})
    if hasattr(trainer, "_extra_state_arrays"):
        state_arrays.update(trainer._extra_state_arrays())
    mx.save_safetensors(str(checkpoint_dir / "trainer_state.safetensors"), state_arrays)

    metadata = {
        "algorithm": trainer.algorithm,
        "config": trainer.config.to_dict() if hasattr(trainer.config, "to_dict") else dict(trainer.config),
        "global_step": trainer.global_step,
        "dataset_cursor": trainer.dataset_cursor,
        "cache_metadata": trainer.cache_metadata,
    }
    (checkpoint_dir / "trainer_state.json").write_text(json.dumps(metadata, indent=2))

    if trainer.reference_policy is not None:
        reference_weights = dict(tree_flatten(trainer.reference_policy.model.model.parameters()))
        mx.save_safetensors(str(checkpoint_dir / "reference_model.safetensors"), reference_weights)
        (checkpoint_dir / "reference_metadata.json").write_text(
            json.dumps(
                {
                    "source": trainer.reference_policy.source,
                    "metadata": trainer.reference_policy.metadata,
                },
                indent=2,
            )
        )


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
        mx.array([int(rollout_batch.completion_lengths[index].item())]),
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
class TestRewardAndPPOIntegration:
    def test_reward_trainer_and_scoring_helper_save_manifest_checkpoint(
        self,
        tmp_path,
        tokenizer,
    ):
        from mlx_tune import RewardConfig, RewardTrainer, build_reward_model, score_reward_model

        reward_model = build_reward_model(make_model(50))
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

        result = trainer.train()
        scores = score_reward_model(
            reward_model,
            [{"prompt": "Q:", "response": "good"}],
            batch_size=1,
            tokenizer=tokenizer,
        )

        assert result["status"] == "success"
        assert len(scores) == 1
        assert (tmp_path / "reward" / "manifest.json").exists()
        assert (tmp_path / "reward" / "reward_model" / "head.safetensors").exists()

    def test_ppo_trainer_persists_policy_and_value_optimizers(
        self,
        tmp_path,
        tokenizer,
        grpo_dataset,
    ):
        from mlx_tune import PPOConfig, PPOTrainer, build_value_model

        trainer = PPOTrainer(
            model=make_model(51),
            train_dataset=grpo_dataset,
            tokenizer=tokenizer,
            reward_fn=lambda response, context: float(len(response)),
            value_model=build_value_model(make_model(52)),
            args=PPOConfig(
                learning_rate=1e-2,
                value_learning_rate=2e-2,
                max_steps=1,
                logging_steps=1,
                save_steps=1,
                num_generations=2,
                max_completion_length=4,
                output_dir=str(tmp_path / "ppo"),
            ),
        )

        result = trainer.train()

        assert result["status"] == "success"
        assert trainer._last_rollout_batch is not None
        assert trainer._last_rollout_batch.value_predictions is not None
        assert trainer._last_rollout_batch.returns is not None
        assert (tmp_path / "ppo" / "optimizers" / "policy" / "state.safetensors").exists()
        assert (tmp_path / "ppo" / "optimizers" / "value" / "state.safetensors").exists()

    def test_ppo_trainer_uses_requested_advantage_estimator(
        self,
        tmp_path,
        tokenizer,
        grpo_dataset,
    ):
        from mlx_tune import PPOConfig, PPOTrainer, build_value_model
        from mlx_tune._rl_runtime import compute_returns_and_advantages

        trainer = PPOTrainer(
            model=make_model(60),
            ref_model=make_model(61),
            train_dataset=grpo_dataset,
            tokenizer=tokenizer,
            reward_fn=lambda response, context: float(len(response)),
            value_model=build_value_model(make_model(62)),
            args=PPOConfig(
                advantage_estimator="rloo",
                normalize_advantages=False,
                temperature=0.0,
                max_steps=1,
                logging_steps=1,
                save_steps=1,
                num_generations=2,
                max_completion_length=4,
                output_dir=str(tmp_path / "ppo_rloo"),
            ),
        )

        trainer._ensure_reference_policy()
        trainer._prepare_prompt_samples()
        rollout_batch = trainer._collect_rollout_batch(trainer._next_prompt_batch())

        _, expected_advantages = compute_returns_and_advantages(
            rewards=rollout_batch.rewards,
            prompt_group_indices=rollout_batch.prompt_group_indices,
            mode="rloo",
            gamma=trainer.gamma,
            gae_lambda=trainer.gae_lambda,
            normalize=False,
        )

        assert mx.allclose(rollout_batch.advantages, expected_advantages)

    def test_on_policy_trainers_activate_kl_controls(
        self,
        tmp_path,
        tokenizer,
        grpo_dataset,
    ):
        from mlx_tune import (
            GRPOConfig,
            GRPOTrainer,
            OnlineDPOConfig,
            OnlineDPOTrainer,
            PPOConfig,
            PPOTrainer,
            build_value_model,
        )

        grpo = GRPOTrainer(
            model=make_model(70),
            ref_model=make_model(71),
            train_dataset=grpo_dataset,
            tokenizer=tokenizer,
            reward_fn=lambda response, context: float(len(response)),
            args=GRPOConfig(
                beta=0.3,
                kl_target=1e-4,
                max_steps=1,
                logging_steps=1,
                save_steps=1,
                num_generations=2,
                max_completion_length=3,
                output_dir=str(tmp_path / "grpo_kl"),
            ),
        )
        grpo._ensure_reference_policy()
        grpo._prepare_prompt_samples()
        grpo_rollout = grpo._collect_rollout_batch(grpo._next_prompt_batch())
        assert grpo._effective_kl_beta(grpo_rollout) != grpo.beta

        ppo = PPOTrainer(
            model=make_model(72),
            ref_model=make_model(73),
            train_dataset=grpo_dataset,
            tokenizer=tokenizer,
            reward_fn=lambda response, context: float(len(response)),
            value_model=build_value_model(make_model(74)),
            args=PPOConfig(
                beta=0.4,
                kl_penalty_mode="none",
                max_steps=1,
                logging_steps=1,
                save_steps=1,
                num_generations=2,
                max_completion_length=3,
                output_dir=str(tmp_path / "ppo_kl_none"),
            ),
        )
        ppo._ensure_reference_policy()
        ppo._prepare_prompt_samples()
        ppo_rollout = ppo._collect_rollout_batch(ppo._next_prompt_batch())
        assert ppo._effective_kl_beta(ppo_rollout) == 0.0

        online_dpo = OnlineDPOTrainer(
            model=make_model(75),
            ref_model=make_model(76),
            train_dataset=grpo_dataset,
            tokenizer=tokenizer,
            reward_fn=lambda response, context: float(len(response)),
            args=OnlineDPOConfig(
                beta=0.2,
                kl_target=1e-4,
                max_steps=1,
                logging_steps=1,
                save_steps=1,
                num_generations=2,
                max_completion_length=3,
                output_dir=str(tmp_path / "online_dpo_kl"),
            ),
        )
        online_dpo._ensure_reference_policy()
        online_dpo._prepare_prompt_samples()
        online_rollout = online_dpo._collect_rollout_batch(online_dpo._next_prompt_batch())
        assert online_dpo._effective_kl_beta(online_rollout) != online_dpo.beta


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

    def test_grpo_phase1_accepts_documented_loss_type_aliases(
        self,
        tmp_path,
        tokenizer,
        grpo_dataset,
    ):
        from mlx_tune import GRPOConfig, GRPOTrainer

        for loss_type in ["grpo", "dr_grpo", "dapo", "bnpo"]:
            trainer = GRPOTrainer(
                model=make_model(20),
                train_dataset=grpo_dataset,
                tokenizer=tokenizer,
                reward_fn=lambda response, context: float(len(response)),
                args=GRPOConfig(
                    loss_type=loss_type,
                    learning_rate=1e-2,
                    beta=0.01,
                    num_generations=2,
                    max_completion_length=3,
                    max_steps=1,
                    logging_steps=1,
                    save_steps=1,
                    output_dir=str(tmp_path / loss_type),
                ),
            )
            result = trainer.train()
            assert result["status"] == "success"
            assert trainer.phase1_loss_type == "phase1_shared_rollout_recompute"

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

    def test_grpo_prefers_learned_reward_model_over_reward_fn(
        self,
        tmp_path,
        tokenizer,
        grpo_dataset,
    ):
        from mlx_tune import GRPOConfig, GRPOTrainer, build_reward_model

        reward_model = build_reward_model(make_model(30))
        reward_model.head.update(
            {
                "weight": mx.zeros_like(reward_model.head.weight),
                "bias": mx.array([1.0], dtype=mx.float32),
            },
            strict=False,
        )
        mx.eval(reward_model.head.parameters())

        trainer = GRPOTrainer(
            model=make_model(31),
            train_dataset=grpo_dataset,
            tokenizer=tokenizer,
            reward_model=reward_model,
            reward_fn=lambda response, context: (_ for _ in ()).throw(RuntimeError("reward_fn should not run")),
            args=GRPOConfig(
                learning_rate=1e-2,
                beta=0.01,
                num_generations=2,
                max_completion_length=4,
                max_steps=1,
                logging_steps=1,
                save_steps=1,
                output_dir=str(tmp_path),
            ),
        )

        result = trainer.train()
        assert result["status"] == "success"
        assert trainer._last_rollout_batch is not None
        assert mx.allclose(
            trainer._last_rollout_batch.rewards,
            mx.ones_like(trainer._last_rollout_batch.rewards),
        )

    def test_grpo_offline_reward_source_uses_dataset_rewards_without_reward_evaluator(
        self,
        tmp_path,
        tokenizer,
    ):
        from mlx_tune import GRPOConfig, GRPOTrainer

        trainer = GRPOTrainer(
            model=make_model(63),
            train_dataset=[
                {"prompt": "Solve 2 + 2", "completion": "4", "reward": 1.0},
                {"prompt": "Solve 2 + 2", "completion": "5", "reward": -1.0},
            ],
            tokenizer=tokenizer,
            reward_fn=lambda *_: (_ for _ in ()).throw(RuntimeError("reward_fn should not run")),
            args=GRPOConfig(
                learning_rate=1e-2,
                beta=0.01,
                num_generations=2,
                reward_source="offline",
                max_steps=1,
                logging_steps=1,
                save_steps=1,
                output_dir=str(tmp_path / "grpo_offline"),
            ),
        )

        result = trainer.train()

        assert result["status"] == "success"
        assert trainer._last_rollout_batch is not None
        assert trainer._last_rollout_batch.rewards.tolist() == [1.0, -1.0]

    def test_grpo_single_reward_source_component_preserves_weight_semantics(
        self,
        tmp_path,
        tokenizer,
    ):
        from mlx_tune import GRPOConfig, GRPOTrainer

        trainer = GRPOTrainer(
            model=make_model(64),
            train_dataset=[{"prompt": "Solve 2 + 2", "answer": "4"}],
            tokenizer=tokenizer,
            args=GRPOConfig(
                learning_rate=1e-2,
                beta=0.01,
                num_generations=2,
                max_completion_length=3,
                max_steps=1,
                logging_steps=1,
                save_steps=1,
                reward_sources=[
                    {"name": "zeroed", "source": "length", "weight": 0.0},
                ],
                output_dir=str(tmp_path / "grpo_weighted_single_reward"),
            ),
        )

        result = trainer.train()

        assert result["status"] == "success"
        assert trainer._last_rollout_batch is not None
        assert mx.allclose(
            trainer._last_rollout_batch.rewards,
            mx.zeros_like(trainer._last_rollout_batch.rewards),
        )

    def test_grpo_prompt_rollouts_respect_rollout_batch_size(
        self,
        tmp_path,
        tokenizer,
    ):
        from mlx_tune import GRPOConfig, GRPOTrainer

        trainer = GRPOTrainer(
            model=make_model(65),
            train_dataset=[
                {"prompt": "What is 1 + 1?", "answer": "2"},
                {"prompt": "What is 2 + 2?", "answer": "4"},
                {"prompt": "What is 3 + 3?", "answer": "6"},
            ],
            tokenizer=tokenizer,
            reward_fn=lambda response, context: float(len(response)),
            args=GRPOConfig(
                learning_rate=1e-2,
                beta=0.01,
                per_device_train_batch_size=1,
                rollout_batch_size=3,
                num_generations=2,
                max_completion_length=3,
                max_steps=1,
                logging_steps=1,
                save_steps=1,
                output_dir=str(tmp_path / "grpo_rollout_batch_size"),
            ),
        )

        result = trainer.train()

        assert result["status"] == "success"
        assert trainer._last_rollout_batch is not None
        assert len(trainer._last_rollout_batch.prompt_ids) == 6
        assert sorted(set(trainer._last_rollout_batch.prompt_group_indices.tolist())) == [0, 1, 2]

    def test_grpo_manifest_checkpoint_persists_roles_and_metrics(
        self,
        tmp_path,
        tokenizer,
        grpo_dataset,
    ):
        from mlx_tune import GRPOConfig, GRPOTrainer, build_reward_model, build_value_model

        output_dir = tmp_path / "grpo_manifest"
        reward_model = build_reward_model(make_model(33))
        value_model = build_value_model(make_model(34))
        score_sequence = mx.array([[2, 4, 7]], dtype=mx.int32)
        score_sequence_lengths = mx.array([3], dtype=mx.int32)
        score_prompt_lengths = mx.array([1], dtype=mx.int32)
        score_completion_lengths = mx.array([2], dtype=mx.int32)
        saved_reward_score = reward_model.score(
            score_sequence,
            sequence_lengths=score_sequence_lengths,
            prompt_lengths=score_prompt_lengths,
            completion_lengths=score_completion_lengths,
        )
        saved_value_score = value_model.predict(
            score_sequence,
            sequence_lengths=score_sequence_lengths,
            prompt_lengths=score_prompt_lengths,
            completion_lengths=score_completion_lengths,
        )

        trainer = GRPOTrainer(
            model=make_model(32),
            train_dataset=grpo_dataset,
            tokenizer=tokenizer,
            reward_model=reward_model,
            value_model=value_model,
            args=GRPOConfig(
                learning_rate=1e-2,
                beta=0.01,
                num_generations=2,
                max_completion_length=4,
                max_steps=1,
                logging_steps=1,
                save_steps=1,
                output_dir=str(output_dir),
            ),
        )
        trainer.train()

        assert (output_dir / "manifest.json").exists()
        assert (output_dir / "policy" / "role.json").exists()
        assert (output_dir / "reference" / "weights.safetensors").exists()
        assert (output_dir / "reward_model" / "weights.safetensors").exists()
        assert (output_dir / "reward_model" / "head.safetensors").exists()
        assert (output_dir / "value_model" / "weights.safetensors").exists()
        assert (output_dir / "value_model" / "head.safetensors").exists()
        assert (output_dir / "optimizer" / "state.safetensors").exists()
        assert (output_dir / "scheduler" / "state.json").exists()
        assert (output_dir / "trainer" / "state.json").exists()
        assert (output_dir / "trainer" / "rng.safetensors").exists()
        assert (output_dir / "metrics" / "history.jsonl").exists()
        assert trainer.metrics_history

        resumed = GRPOTrainer(
            model=make_model(35),
            train_dataset=grpo_dataset,
            tokenizer=tokenizer,
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
        assert resumed.reward_model is not None
        assert resumed.value_model is not None
        assert mx.allclose(
            resumed.reward_model.score(
                score_sequence,
                sequence_lengths=score_sequence_lengths,
                prompt_lengths=score_prompt_lengths,
                completion_lengths=score_completion_lengths,
            ),
            saved_reward_score,
        )
        assert mx.allclose(
            resumed.value_model.predict(
                score_sequence,
                sequence_lengths=score_sequence_lengths,
                prompt_lengths=score_prompt_lengths,
                completion_lengths=score_completion_lengths,
            ),
            saved_value_score,
        )
        assert len(resumed.metrics_history) >= len(trainer.metrics_history)
        assert resumed.loaded_checkpoint_manifest is not None


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
        batch = trainer._build_batch([sample])
        cached_loss, _ = kto_loss(
            trainer.model.model,
            batch.input_ids,
            batch.sequence_lengths,
            batch.labels,
            beta=trainer.beta,
            reference_logprobs=batch.reference_logprobs,
        )
        live_policy_reference = compute_log_probs_with_lengths(
            trainer.model.model,
            batch.input_ids,
            batch.sequence_lengths,
        )
        live_loss, _ = kto_loss(
            trainer.model.model,
            batch.input_ids,
            batch.sequence_lengths,
            batch.labels,
            beta=trainer.beta,
            reference_logprobs=live_policy_reference,
        )
        assert abs(cached_loss.item() - live_loss.item()) > 1e-6

    def test_legacy_rl_checkpoint_loads_and_resaves_manifest_layout(
        self,
        tmp_path,
        tokenizer,
        kto_dataset,
    ):
        from mlx_tune import KTOTrainer

        source_trainer = KTOTrainer(
            model=make_model(40),
            train_dataset=kto_dataset,
            tokenizer=tokenizer,
            learning_rate=2e-2,
            max_steps=1,
            logging_steps=1,
            save_steps=1,
            output_dir=str(tmp_path / "source"),
        )
        source_trainer.train()

        legacy_dir = tmp_path / "legacy_kto"
        write_legacy_rl_checkpoint(source_trainer, legacy_dir)

        resumed = KTOTrainer(
            model=make_model(41),
            train_dataset=kto_dataset,
            tokenizer=tokenizer,
            learning_rate=2e-2,
            max_steps=2,
            logging_steps=1,
            save_steps=1,
            output_dir=str(legacy_dir),
        )
        result = resumed.train(resume_from_checkpoint=str(legacy_dir))

        assert result["global_step"] == 2
        assert resumed.cache_metadata == source_trainer.cache_metadata
        assert (legacy_dir / "manifest.json").exists()
        assert (legacy_dir / "reference" / "weights.safetensors").exists()
        assert (legacy_dir / "runtime" / "cache.safetensors").exists()


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

    def test_online_dpo_uses_reward_model_preference_ordering(
        self,
        tmp_path,
        tokenizer,
    ):
        from mlx_tune import OnlineDPOConfig, OnlineDPOTrainer, build_reward_model

        reward_model = build_reward_model(make_model(60))
        reward_model.head.update(
            {
                "weight": mx.zeros_like(reward_model.head.weight),
                "bias": mx.array([1.0], dtype=mx.float32),
            },
            strict=False,
        )
        mx.eval(reward_model.head.parameters())

        trainer = OnlineDPOTrainer(
            model=make_model(61),
            train_dataset=[{"prompt": "Solve 1 + 1", "reward_context": "2"}],
            tokenizer=tokenizer,
            reward_model=reward_model,
            reward_fn=lambda response, context: (_ for _ in ()).throw(RuntimeError("reward_fn should not run")),
            args=OnlineDPOConfig(
                learning_rate=1e-2,
                max_steps=1,
                logging_steps=1,
                save_steps=1,
                num_generations=2,
                max_completion_length=3,
                output_dir=str(tmp_path / "online_dpo"),
            ),
        )

        result = trainer.train()

        assert result["status"] == "success"
        assert trainer._last_rollout_batch is not None
        assert mx.allclose(
            trainer._last_rollout_batch.rewards,
            mx.ones_like(trainer._last_rollout_batch.rewards),
        )

    def test_online_dpo_prefers_answer_context_over_prompt(
        self,
        tmp_path,
        tokenizer,
    ):
        from mlx_tune import OnlineDPOConfig, OnlineDPOTrainer

        seen_contexts = []

        def reward_fn(response: str, context: str) -> float:
            seen_contexts.append(context)
            return float(len(response))

        trainer = OnlineDPOTrainer(
            model=make_model(62),
            train_dataset=[{"prompt": "Solve 2 + 2", "answer": "4"}],
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            args=OnlineDPOConfig(
                learning_rate=1e-2,
                max_steps=1,
                logging_steps=1,
                save_steps=1,
                num_generations=2,
                max_completion_length=3,
                output_dir=str(tmp_path / "online_dpo_answer_context"),
            ),
        )

        result = trainer.train()

        assert result["status"] == "success"
        assert seen_contexts
        assert all(context == "4" for context in seen_contexts)

    def test_online_dpo_prompt_rollouts_respect_rollout_batch_size(
        self,
        tmp_path,
        tokenizer,
    ):
        from mlx_tune import OnlineDPOConfig, OnlineDPOTrainer

        trainer = OnlineDPOTrainer(
            model=make_model(66),
            train_dataset=[
                {"prompt": "Solve 1 + 1", "answer": "2"},
                {"prompt": "Solve 2 + 2", "answer": "4"},
                {"prompt": "Solve 3 + 3", "answer": "6"},
            ],
            tokenizer=tokenizer,
            reward_fn=lambda response, context: float(len(response)),
            args=OnlineDPOConfig(
                learning_rate=1e-2,
                per_device_train_batch_size=1,
                rollout_batch_size=3,
                max_steps=1,
                logging_steps=1,
                save_steps=1,
                num_generations=2,
                max_completion_length=3,
                output_dir=str(tmp_path / "online_dpo_rollout_batch_size"),
            ),
        )

        result = trainer.train()

        assert result["status"] == "success"
        assert trainer._last_rollout_batch is not None
        assert len(trainer._last_rollout_batch.prompt_ids) == 6
        assert sorted(set(trainer._last_rollout_batch.prompt_group_indices.tolist())) == [0, 1, 2]


@pytest.mark.integration
def test_grpo_evaluate_records_namespaced_metrics_and_preference_accuracy(tmp_path, tokenizer):
    from mlx_tune import GRPOConfig, GRPOTrainer

    output_dir = tmp_path / "grpo_eval"
    trainer = GRPOTrainer(
        model=make_model(80),
        train_dataset=[{"prompt": "Solve 2 + 2", "answer": "4"}],
        eval_dataset=[{"prompt": "Solve 3 + 3", "answer": "6"}],
        eval_preference_dataset=[
            {"prompt": "Q", "chosen": "AAAA", "rejected": "B"},
            {"prompt": "Q", "chosen": "CCCC", "rejected": "D"},
        ],
        tokenizer=tokenizer,
        reward_fn=lambda response, context: float(len(response)),
        args=GRPOConfig(
            seed=123,
            learning_rate=1e-2,
            beta=0.01,
            num_generations=2,
            max_completion_length=3,
            max_steps=1,
            eval_steps=1,
            logging_steps=1,
            save_steps=1,
            output_dir=str(output_dir),
        ),
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    history_rows = [json.loads(line) for line in (output_dir / "metrics" / "history.jsonl").read_text().splitlines()]

    assert "eval/reward_mean" in eval_metrics
    assert "eval/preference_win_rate" in eval_metrics
    assert any("train/policy_loss" in row for row in history_rows)
    assert any("eval/reward_mean" in row for row in history_rows)
    assert any("eval/preference_win_rate" in row for row in history_rows)


@pytest.mark.integration
@pytest.mark.parametrize("trainer_kind", ["grpo", "ppo", "online_dpo"])
def test_on_policy_trainers_with_same_seed_are_reproducible(tmp_path, tokenizer, grpo_dataset, trainer_kind):
    from mlx_tune import (
        GRPOConfig,
        GRPOTrainer,
        OnlineDPOConfig,
        OnlineDPOTrainer,
        PPOConfig,
        PPOTrainer,
        build_value_model,
    )

    def build_trainer(output_dir):
        if trainer_kind == "grpo":
            return GRPOTrainer(
                model=make_model(81),
                train_dataset=grpo_dataset,
                tokenizer=tokenizer,
                reward_fn=lambda response, context: float(len(response)),
                args=GRPOConfig(
                    seed=777,
                    learning_rate=1e-2,
                    beta=0.01,
                    num_generations=2,
                    max_completion_length=3,
                    max_steps=1,
                    logging_steps=1,
                    save_steps=1,
                    output_dir=str(output_dir),
                ),
            )
        if trainer_kind == "ppo":
            return PPOTrainer(
                model=make_model(81),
                train_dataset=grpo_dataset,
                tokenizer=tokenizer,
                reward_fn=lambda response, context: float(len(response)),
                value_model=build_value_model(make_model(82)),
                args=PPOConfig(
                    seed=777,
                    learning_rate=1e-2,
                    value_learning_rate=1e-2,
                    num_generations=2,
                    max_completion_length=3,
                    max_steps=1,
                    logging_steps=1,
                    save_steps=1,
                    output_dir=str(output_dir),
                ),
            )
        return OnlineDPOTrainer(
            model=make_model(81),
            train_dataset=grpo_dataset,
            tokenizer=tokenizer,
            reward_fn=lambda response, context: float(len(response)),
            args=OnlineDPOConfig(
                seed=777,
                learning_rate=1e-2,
                num_generations=2,
                max_completion_length=3,
                max_steps=1,
                logging_steps=1,
                save_steps=1,
                output_dir=str(output_dir),
            ),
        )

    trainer_a = build_trainer(tmp_path / f"{trainer_kind}_a")
    trainer_b = build_trainer(tmp_path / f"{trainer_kind}_b")
    trainer_a.train()
    trainer_b.train()

    assert trainer_a.metrics_history == trainer_b.metrics_history
    assert trainer_a._last_rollout_batch is not None
    assert trainer_b._last_rollout_batch is not None
    assert trainer_a._last_rollout_batch.completion_ids == trainer_b._last_rollout_batch.completion_ids


@pytest.mark.integration
def test_grpo_checkpoint_records_step_boundary_and_independent_cursors(tmp_path, tokenizer):
    from mlx_tune import GRPOConfig, GRPOTrainer, resume_from_checkpoint

    output_dir = tmp_path / "grpo_cursor_checkpoint"
    trainer = GRPOTrainer(
        model=make_model(83),
        train_dataset=[
            {"prompt": "Q1", "completion": "A1", "reward": 1.0},
            {"prompt": "Q2", "completion": "A2", "reward": 0.5},
            {"prompt": "Q3", "completion": "A3", "reward": 0.2},
            {"prompt": "Q4", "completion": "A4", "reward": -0.1},
        ],
        tokenizer=tokenizer,
        args=GRPOConfig(
            seed=99,
            learning_rate=1e-2,
            reward_source="offline",
            num_generations=2,
            max_steps=1,
            logging_steps=1,
            save_steps=1,
            output_dir=str(output_dir),
        ),
    )

    trainer.train()
    bundle = resume_from_checkpoint(output_dir)
    state = bundle.trainer_state

    assert bundle.manifest["format_version"] == 4
    assert state["seed"] == 99
    assert state["trainer_state"]["step_boundary"]["completed_optimizer_step"] == 1
    assert state["trainer_state"]["step_boundary"]["checkpoint_authoritative"] is True
    assert state["trainer_state"]["cursors"]["prompt_dataset"] == 0
    assert state["trainer_state"]["cursors"]["offline_rollout_dataset"] == 2


@pytest.mark.integration
def test_grpo_reuses_precomputed_rollout_reference_cache_after_resume(tmp_path, tokenizer, monkeypatch):
    import mlx_tune.rl_trainers as rl_trainers_module
    from mlx_tune import GRPOConfig, GRPOTrainer

    output_dir = tmp_path / "grpo_cache_resume"
    train_dataset = [
        {"prompt": "Q1", "completion": "A1", "reward": 1.0},
        {"prompt": "Q2", "completion": "A2", "reward": 0.5},
        {"prompt": "Q3", "completion": "A3", "reward": -0.2},
    ]
    trainer = GRPOTrainer(
        model=make_model(84),
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        args=GRPOConfig(
            seed=11,
            learning_rate=1e-2,
            reward_source="offline",
            num_generations=2,
            precompute_reference_scores=True,
            max_steps=1,
            logging_steps=1,
            save_steps=1,
            output_dir=str(output_dir),
        ),
    )
    trainer.train()

    resumed = GRPOTrainer(
        model=make_model(85),
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        args=GRPOConfig(
            seed=11,
            learning_rate=1e-2,
            reward_source="offline",
            num_generations=2,
            precompute_reference_scores=True,
            max_steps=2,
            logging_steps=1,
            save_steps=1,
            output_dir=str(output_dir),
        ),
    )
    resumed._apply_lora_if_needed()
    resumed._prepare_prompt_samples()
    optimizer = resumed._optimizer_for_training()
    resumed.optimizer = optimizer
    resumed.load_state(optimizer=optimizer, checkpoint_dir=output_dir)
    resumed._ensure_reference_policy()

    original = rl_trainers_module.score_policy_in_chunks
    reference_model = resumed.reference_policy.model.model
    calls = {"reference": 0}

    def wrapped(model, *args, **kwargs):
        if model is reference_model:
            calls["reference"] += 1
        return original(model, *args, **kwargs)

    monkeypatch.setattr(rl_trainers_module, "score_policy_in_chunks", wrapped)
    batch = resumed._collect_fixed_rollout_batch(
        resumed.rollout_samples[:2],
        cache_key="grpo.train_rollout_reference_logprobs",
    )

    assert calls["reference"] == 0
    assert batch.reference_logprobs is not None


@pytest.mark.integration
def test_grpo_invalidates_precomputed_rollout_reference_cache_when_dataset_changes(
    tmp_path,
    tokenizer,
    monkeypatch,
):
    import mlx_tune.rl_trainers as rl_trainers_module
    from mlx_tune import GRPOConfig, GRPOTrainer

    output_dir = tmp_path / "grpo_cache_dataset_drift"
    original_dataset = [
        {"prompt": "Q1", "completion": "A1", "reward": 1.0},
        {"prompt": "Q2", "completion": "A2", "reward": 0.5},
    ]
    changed_dataset = [
        {"prompt": "!!!!!!!!", "completion": "zzzzzzzz", "reward": 1.0},
        {"prompt": "????????", "completion": "yyyyyyyy", "reward": 0.5},
    ]
    trainer = GRPOTrainer(
        model=make_model(88),
        train_dataset=original_dataset,
        tokenizer=tokenizer,
        args=GRPOConfig(
            seed=19,
            learning_rate=1e-2,
            reward_source="offline",
            num_generations=2,
            precompute_reference_scores=True,
            max_steps=1,
            logging_steps=1,
            save_steps=1,
            output_dir=str(output_dir),
        ),
    )
    trainer.train()
    original_scores = mx.array(trainer.runtime_cache_arrays["grpo.train_rollout_reference_logprobs"])

    resumed = GRPOTrainer(
        model=make_model(89),
        train_dataset=changed_dataset,
        tokenizer=tokenizer,
        args=GRPOConfig(
            seed=19,
            learning_rate=1e-2,
            reward_source="offline",
            num_generations=2,
            precompute_reference_scores=True,
            max_steps=2,
            logging_steps=1,
            save_steps=1,
            output_dir=str(output_dir),
        ),
    )
    resumed._apply_lora_if_needed()
    resumed._prepare_prompt_samples()
    optimizer = resumed._optimizer_for_training()
    resumed.optimizer = optimizer
    resumed.load_state(optimizer=optimizer, checkpoint_dir=output_dir)
    resumed._ensure_reference_policy()

    original = rl_trainers_module.score_policy_in_chunks
    reference_model = resumed.reference_policy.model.model
    calls = {"reference": 0}

    def wrapped(model, *args, **kwargs):
        if model is reference_model:
            calls["reference"] += 1
        return original(model, *args, **kwargs)

    monkeypatch.setattr(rl_trainers_module, "score_policy_in_chunks", wrapped)
    batch = resumed._collect_fixed_rollout_batch(
        resumed.rollout_samples,
        cache_key="grpo.train_rollout_reference_logprobs",
    )

    assert calls["reference"] > 0
    assert not mx.allclose(batch.reference_logprobs, original_scores)


@pytest.mark.integration
def test_grpo_resume_rejects_objective_and_update_shape_drift(tmp_path, tokenizer, grpo_dataset):
    from mlx_tune import GRPOConfig, GRPOTrainer

    output_dir = tmp_path / "grpo_resume_config_drift"
    trainer = GRPOTrainer(
        model=make_model(90),
        train_dataset=grpo_dataset,
        tokenizer=tokenizer,
        reward_fn=lambda response, context: float(len(response)),
        args=GRPOConfig(
            seed=31,
            learning_rate=1e-2,
            beta=0.01,
            clip_epsilon=0.2,
            rollout_batch_size=1,
            minibatch_reuse_steps=1,
            num_generations=2,
            max_completion_length=4,
            max_steps=1,
            logging_steps=1,
            save_steps=1,
            output_dir=str(output_dir),
        ),
    )
    trainer.train()

    resumed = GRPOTrainer(
        model=make_model(91),
        train_dataset=grpo_dataset,
        tokenizer=tokenizer,
        reward_fn=lambda response, context: float(len(response)),
        args=GRPOConfig(
            seed=31,
            learning_rate=1e-2,
            beta=0.99,
            clip_epsilon=0.33,
            rollout_batch_size=5,
            minibatch_reuse_steps=7,
            num_generations=2,
            max_completion_length=4,
            max_steps=2,
            logging_steps=1,
            save_steps=1,
            output_dir=str(output_dir),
        ),
    )
    resumed._apply_lora_if_needed()
    resumed._prepare_prompt_samples()
    optimizer = resumed._optimizer_for_training()
    resumed.optimizer = optimizer

    with pytest.raises(ValueError, match="fingerprint"):
        resumed.load_state(optimizer=optimizer, checkpoint_dir=output_dir)


@pytest.mark.integration
def test_online_dpo_eval_preference_reference_cache_reused_after_resume(tmp_path, tokenizer, monkeypatch):
    import mlx_tune.rl_trainers as rl_trainers_module
    from mlx_tune import OnlineDPOConfig, OnlineDPOTrainer

    output_dir = tmp_path / "online_dpo_eval_cache"
    eval_preference_dataset = [
        {"prompt": "Q", "chosen": "AA", "rejected": "B"},
        {"prompt": "Q", "chosen": "CC", "rejected": "D"},
    ]
    trainer = OnlineDPOTrainer(
        model=make_model(86),
        train_dataset=[{"prompt": "Solve 2 + 2", "answer": "4"}],
        eval_preference_dataset=eval_preference_dataset,
        tokenizer=tokenizer,
        reward_fn=lambda response, context: float(len(response)),
        args=OnlineDPOConfig(
            seed=22,
            learning_rate=1e-2,
            num_generations=2,
            max_completion_length=3,
            max_steps=1,
            eval_steps=1,
            precompute_reference_scores=True,
            logging_steps=1,
            save_steps=1,
            output_dir=str(output_dir),
        ),
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_state(optimizer=trainer.optimizer)

    resumed = OnlineDPOTrainer(
        model=make_model(87),
        train_dataset=[{"prompt": "Solve 2 + 2", "answer": "4"}],
        eval_preference_dataset=eval_preference_dataset,
        tokenizer=tokenizer,
        reward_fn=lambda response, context: float(len(response)),
        args=OnlineDPOConfig(
            seed=22,
            learning_rate=1e-2,
            num_generations=2,
            max_completion_length=3,
            max_steps=2,
            eval_steps=1,
            precompute_reference_scores=True,
            logging_steps=1,
            save_steps=1,
            output_dir=str(output_dir),
        ),
    )
    resumed._apply_lora_if_needed()
    resumed._prepare_prompt_samples()
    optimizer = resumed._optimizer_for_training()
    resumed.optimizer = optimizer
    resumed.load_state(optimizer=optimizer, checkpoint_dir=output_dir)

    original = rl_trainers_module.score_policy_in_chunks
    reference_model = resumed.reference_policy.model.model
    calls = {"reference": 0}

    def wrapped(model, *args, **kwargs):
        if model is reference_model:
            calls["reference"] += 1
        return original(model, *args, **kwargs)

    monkeypatch.setattr(rl_trainers_module, "score_policy_in_chunks", wrapped)
    eval_metrics = resumed.evaluate()

    assert calls["reference"] == 0
    assert "eval/preference_win_rate" in eval_metrics
