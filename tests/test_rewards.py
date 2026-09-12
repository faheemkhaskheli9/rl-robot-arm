import numpy as np
import pytest

from rl_robot_arm.rewards import RewardConfig, compute_reward, load_reward_config

ZERO_ACTION = np.zeros(2)


def test_dense_reward_is_negative_distance_when_far_from_target():
    config = RewardConfig(mode="dense", control_cost=0.0, success_bonus=1.0)
    reward = compute_reward(distance=2.0, action=ZERO_ACTION, terminated=False, config=config)
    assert reward == -2.0


def test_dense_reward_applies_control_cost_penalty():
    config = RewardConfig(mode="dense", control_cost=0.5, success_bonus=1.0)
    action = np.array([1.0, 1.0])  # ||action||^2 == 2
    reward = compute_reward(distance=0.5, action=action, terminated=False, config=config)
    assert reward == pytest.approx(-0.5 - 0.5 * 2.0)


def test_dense_reward_adds_success_bonus_on_termination():
    config = RewardConfig(mode="dense", control_cost=0.0, success_bonus=5.0)
    reward = compute_reward(distance=0.01, action=ZERO_ACTION, terminated=True, config=config)
    assert reward == pytest.approx(-0.01 + 5.0)


def test_sparse_reward_is_zero_signal_except_on_success():
    config = RewardConfig(mode="sparse", success_reward=1.0, failure_reward=0.0)
    assert compute_reward(distance=3.0, action=ZERO_ACTION, terminated=False, config=config) == 0.0
    assert compute_reward(distance=0.01, action=ZERO_ACTION, terminated=True, config=config) == 1.0


def test_sparse_reward_ignores_distance_and_action_magnitude():
    config = RewardConfig(mode="sparse", success_reward=1.0, failure_reward=-0.1)
    big_action = np.array([1.0, 1.0])
    near = compute_reward(distance=0.001, action=big_action, terminated=False, config=config)
    far = compute_reward(distance=100.0, action=big_action, terminated=False, config=config)
    assert near == far == -0.1


def test_unknown_mode_raises_value_error():
    with pytest.raises(ValueError):
        RewardConfig(mode="not-a-mode")


def test_load_reward_config_dense_preset(tmp_path):
    path = tmp_path / "dense.yaml"
    path.write_text("mode: dense\ncontrol_cost: 0.02\nsuccess_bonus: 2.0\n")
    config = load_reward_config(path)
    assert config.mode == "dense"
    assert config.control_cost == 0.02
    assert config.success_bonus == 2.0


def test_load_reward_config_sparse_preset(tmp_path):
    path = tmp_path / "sparse.yaml"
    path.write_text("mode: sparse\nsuccess_reward: 1.0\nfailure_reward: -1.0\n")
    config = load_reward_config(path)
    assert config.mode == "sparse"
    assert config.failure_reward == -1.0


def test_load_reward_config_missing_file_raises_not_silently_defaults(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_reward_config(tmp_path / "does-not-exist.yaml")


def test_load_reward_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("mode: dense\nnot_a_real_field: 123\n")
    with pytest.raises(ValueError):
        load_reward_config(path)


def test_shipped_dense_and_sparse_presets_load_cleanly():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    dense = load_reward_config(repo_root / "configs" / "reward_dense.yaml")
    sparse = load_reward_config(repo_root / "configs" / "reward_sparse.yaml")
    assert dense.mode == "dense"
    assert sparse.mode == "sparse"
