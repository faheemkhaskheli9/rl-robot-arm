from __future__ import annotations

import numpy as np
import pytest

from rl_robot_arm.ddpg.replay_buffer import ReplayBuffer


def make_transition(rng, obs_dim=4, action_dim=2):
    return (
        rng.normal(size=obs_dim).astype(np.float32),
        rng.uniform(-1, 1, size=action_dim).astype(np.float32),
        float(rng.normal()),
        rng.normal(size=obs_dim).astype(np.float32),
        bool(rng.integers(0, 2)),
    )


def test_len_grows_with_pushes():
    buffer = ReplayBuffer(capacity=10, obs_dim=4, action_dim=2)
    rng = np.random.default_rng(0)
    assert len(buffer) == 0
    for i in range(5):
        buffer.push(*make_transition(rng))
        assert len(buffer) == i + 1


def test_capacity_is_respected_ring_buffer():
    buffer = ReplayBuffer(capacity=5, obs_dim=4, action_dim=2)
    rng = np.random.default_rng(0)
    for _ in range(20):
        buffer.push(*make_transition(rng))
    assert len(buffer) == 5


def test_sample_returns_requested_batch_size():
    buffer = ReplayBuffer(capacity=50, obs_dim=4, action_dim=2, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(30):
        buffer.push(*make_transition(rng))

    batch = buffer.sample(16)
    assert batch.obs.shape == (16, 4)
    assert batch.action.shape == (16, 2)
    assert batch.reward.shape == (16, 1)
    assert batch.next_obs.shape == (16, 4)
    assert batch.done.shape == (16, 1)


def test_sample_more_than_available_raises():
    buffer = ReplayBuffer(capacity=50, obs_dim=4, action_dim=2)
    rng = np.random.default_rng(0)
    buffer.push(*make_transition(rng))
    with pytest.raises(ValueError):
        buffer.sample(5)


def test_invalid_capacity_raises():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0, obs_dim=4, action_dim=2)


def test_ring_buffer_overwrites_oldest_first():
    buffer = ReplayBuffer(capacity=3, obs_dim=1, action_dim=1)
    for i in range(5):
        buffer.push(np.array([float(i)]), np.array([0.0]), 0.0, np.array([0.0]), False)
    # After pushing 0..4 into capacity=3, slots hold {2, 3, 4} (write index
    # wrapped) -- 0 and 1 were overwritten. Sampling is with replacement
    # (each call draws at most `len(buffer)` values), so repeat several
    # draws and union them rather than assuming one draw covers all 3.
    seen = set()
    for _ in range(20):
        seen.update(buffer.sample(3).obs.flatten().tolist())
    assert seen == {2.0, 3.0, 4.0}
