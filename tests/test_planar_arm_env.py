"""Tests for the Phase 1 simulated planar-arm environment (issue #1)."""

from __future__ import annotations

import numpy as np
import pytest

import rl_robot_arm  # noqa: F401 - registers envs on import
from rl_robot_arm.envs.planar_arm import PlanarArmReachEnv
from rl_robot_arm.policies import jacobian_transpose_policy

gym = pytest.importorskip("gymnasium")


def test_follows_gymnasium_api_shapes():
    env = PlanarArmReachEnv(n_links=2)
    obs, info = env.reset(seed=0)

    assert env.observation_space.contains(obs)
    assert obs.shape == (3 * 2 + 6,)
    assert set(info) >= {"distance", "is_success", "ee_pos", "target"}

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert env.observation_space.contains(obs2)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_reset_is_deterministic_given_seed():
    a = PlanarArmReachEnv()
    b = PlanarArmReachEnv()
    obs_a, info_a = a.reset(seed=123)
    obs_b, info_b = b.reset(seed=123)
    np.testing.assert_allclose(obs_a, obs_b)
    np.testing.assert_allclose(info_a["target"], info_b["target"])


def test_registered_env_runs_through_timelimit():
    env = gym.make("PlanarArmReach-v0", max_episode_steps=10)
    env.reset(seed=0)
    truncated = False
    steps = 0
    while not truncated:
        _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        steps += 1
        if terminated:
            break
    assert steps <= 10
    assert truncated or terminated
    env.close()


def test_targets_are_within_reach():
    env = PlanarArmReachEnv(n_links=2)
    for seed in range(50):
        _, info = env.reset(seed=seed)
        assert np.linalg.norm(info["target"]) <= env.reach + 1e-9


def test_forward_kinematics_matches_hand_calc():
    env = PlanarArmReachEnv(n_links=2, link_lengths=(1.0, 1.0))
    env.reset(seed=0, options={"target": (0.0, 0.0)})
    env._q = np.array([0.0, np.pi / 2])
    pts = env.forward_kinematics()
    np.testing.assert_allclose(pts[0], [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(pts[1], [1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(pts[2], [1.0, 1.0], atol=1e-9)


def test_reaching_target_terminates_with_success_bonus():
    env = PlanarArmReachEnv(n_links=2, link_lengths=(1.0, 1.0), tolerance=0.05)
    env.reset(seed=1, options={"target": (2.0, 0.0)})  # straight-out pose
    env._q = np.array([0.0, 0.0])
    _, reward, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert terminated
    assert info["is_success"]
    assert reward > 0  # success bonus dominates the tiny distance penalty


def test_p_control_policy_reduces_distance_over_an_episode():
    env = PlanarArmReachEnv(n_links=2)
    successes = 0
    for seed in range(20):
        obs, info = env.reset(seed=seed)
        start = info["distance"]
        for _ in range(100):
            obs, _, terminated, _, info = env.step(jacobian_transpose_policy(env, obs))
            if terminated:
                break
        # every episode makes progress toward the target...
        assert info["distance"] < start
        successes += int(info["is_success"])
    # ...and a clear majority of targets are actually reached by a crude
    # local controller (the rest stall at singularities -- Phase 2/3 territory).
    assert successes >= 12


def test_invalid_construction_raises():
    with pytest.raises(ValueError):
        PlanarArmReachEnv(n_links=0)
    with pytest.raises(ValueError):
        PlanarArmReachEnv(n_links=2, link_lengths=(1.0,))
    with pytest.raises(ValueError):
        PlanarArmReachEnv(render_mode="human")


def test_rgb_array_render_returns_image():
    env = PlanarArmReachEnv(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame is not None
    assert frame.shape == (128, 128, 3)
    assert frame.dtype == np.uint8
