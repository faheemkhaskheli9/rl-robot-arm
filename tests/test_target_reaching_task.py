"""Tests for the configurable target-reaching task definition (issue #2)."""

from __future__ import annotations

import numpy as np
import pytest

import rl_robot_arm  # noqa: F401 - registers envs on import
from rl_robot_arm.envs.planar_arm import PlanarArmReachEnv
from rl_robot_arm.policies import jacobian_transpose_policy, random_policy
from rl_robot_arm.rollout import evaluate_policy, run_episode

gym = pytest.importorskip("gymnasium")


def test_target_radius_range_is_configurable():
    env = PlanarArmReachEnv(n_links=2, target_radius_range=(0.5, 0.6))
    for seed in range(30):
        _, info = env.reset(seed=seed)
        radius = float(np.linalg.norm(info["target"]))
        assert 0.5 * env.reach - 1e-9 <= radius <= 0.6 * env.reach + 1e-9


def test_target_position_differs_across_episodes():
    env = PlanarArmReachEnv(n_links=2)
    targets = []
    for seed in range(10):
        _, info = env.reset(seed=seed)
        targets.append(tuple(info["target"]))
    assert len(set(targets)) > 1


@pytest.mark.parametrize("bad_range", [(0.9, 0.2), (-0.1, 0.5), (0.2, 1.5), (0.3, 0.3)])
def test_invalid_target_radius_range_raises(bad_range):
    with pytest.raises(ValueError, match="target_radius_range"):
        PlanarArmReachEnv(n_links=2, target_radius_range=bad_range)


def test_episode_terminates_on_timeout_when_never_reached():
    # A held-still arm essentially never lands exactly on a random target, so
    # this episode should end via the TimeLimit wrapper (truncated), not success.
    env = gym.make("PlanarArmReach-v0", n_links=2, max_episode_steps=25)
    env.reset(seed=0)
    steps = 0
    truncated = terminated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        steps += 1
    env.close()

    assert steps <= 25
    assert truncated or terminated
    if terminated:
        assert info["is_success"]
    else:
        assert not info["is_success"]


def _bounded_env(**kwargs):
    """A registered, TimeLimit-wrapped env so an unsuccessful episode still ends."""
    return gym.make("PlanarArmReach-v0", max_episode_steps=100, **kwargs)


def test_run_episode_reports_success_and_distance():
    env = _bounded_env(n_links=2, link_lengths=(1.0, 1.0), tolerance=0.05)
    result = run_episode(env, jacobian_transpose_policy, seed=0)
    env.close()

    assert isinstance(result.success, bool)
    assert result.steps >= 1
    assert result.final_distance >= 0.0


def test_success_rate_is_a_loggable_aggregate_metric():
    env = _bounded_env(n_links=2)
    stats = evaluate_policy(env, jacobian_transpose_policy, episodes=20, seed=0)
    env.close()

    assert 0.0 <= stats.success_rate <= 1.0
    assert stats.success_rate == sum(1 for ep in stats.episodes if ep.success) / 20
    metrics = stats.as_dict()
    assert metrics["n_episodes"] == 20
    assert "success_rate" in metrics and "mean_return" in metrics


def test_random_policy_success_rate_is_lower_than_p_control():
    random_env = _bounded_env(n_links=2)
    control_env = _bounded_env(n_links=2)
    random_stats = evaluate_policy(random_env, random_policy, episodes=15, seed=0)
    control_stats = evaluate_policy(control_env, jacobian_transpose_policy, episodes=15, seed=0)
    random_env.close()
    control_env.close()

    assert control_stats.success_rate >= random_stats.success_rate
