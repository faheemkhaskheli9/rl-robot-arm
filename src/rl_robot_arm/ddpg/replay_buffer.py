"""Fixed-capacity off-policy replay buffer (issue #5).

A plain ring buffer over numpy arrays -- no external dependency, deterministic
given a seeded RNG for sampling.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = ["ReplayBuffer", "Batch"]


class Batch(NamedTuple):
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_obs: np.ndarray
    done: np.ndarray


class ReplayBuffer:
    """Stores transitions and samples uniform random batches.

    Capacity is fixed at construction; once full, the oldest transition is
    overwritten (a ring buffer), which is the standard off-policy behaviour
    -- training only ever sees the most recent `capacity` transitions.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)

        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._action = np.zeros((capacity, action_dim), dtype=np.float32)
        self._reward = np.zeros((capacity, 1), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._done = np.zeros((capacity, 1), dtype=np.float32)

        self._write_idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self._write_idx
        self._obs[i] = obs
        self._action[i] = action
        self._reward[i] = reward
        self._next_obs[i] = next_obs
        self._done[i] = float(done)

        self._write_idx = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer holding {self._size}"
            )
        idx = self._rng.integers(0, self._size, size=batch_size)
        return Batch(
            obs=self._obs[idx],
            action=self._action[idx],
            reward=self._reward[idx],
            next_obs=self._next_obs[idx],
            done=self._done[idx],
        )
