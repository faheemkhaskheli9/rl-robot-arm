"""Configurable reward functions for the target-reaching task.

Pulled out of :class:`~rl_robot_arm.envs.planar_arm.PlanarArmReachEnv` so the
reward shape is a variable the Phase 4 reward-design experiments can swap
without touching the environment's kinematics, and so it can be unit tested
against known arm poses independent of the ``gymnasium`` API.

Two modes:

* ``"dense"`` -- shaped, every step: ``-distance - control_cost * ||action||^2``,
  plus ``+success_bonus`` on the step the target is reached. Gives the agent a
  gradient to follow even far from the target.
* ``"sparse"`` -- ``+success_reward`` on the step the target is reached,
  ``failure_reward`` (default 0.0) every other step. No shaping signal at all,
  which is the harder-to-learn-from baseline reward-design experiments compare
  the dense mode against.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["RewardConfig", "compute_reward", "load_reward_config"]

_VALID_MODES = ("dense", "sparse")


@dataclass(frozen=True)
class RewardConfig:
    """Parameters for one reward mode. Construct via :func:`load_reward_config`
    or directly for tests."""

    mode: str = "dense"
    control_cost: float = 0.01
    success_bonus: float = 1.0
    success_reward: float = 1.0
    failure_reward: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {self.mode!r}")


def compute_reward(
    *,
    distance: float,
    action: np.ndarray,
    terminated: bool,
    config: RewardConfig,
) -> float:
    """Return the scalar reward for one step under ``config``'s mode.

    ``distance`` is the end-effector-to-target distance *after* the step,
    ``action`` the clipped joint-velocity command that produced it, and
    ``terminated`` whether the target was reached this step (distance within
    tolerance) -- all values the caller already has from its own step logic.
    """
    if config.mode == "sparse":
        return config.success_reward if terminated else config.failure_reward

    # dense
    reward = -float(distance) - config.control_cost * float(np.sum(np.asarray(action) ** 2))
    if terminated:
        reward += config.success_bonus
    return reward


def load_reward_config(path: str | os.PathLike[str]) -> RewardConfig:
    """Load a :class:`RewardConfig` from a YAML file under ``configs/``.

    A path the caller explicitly passed that doesn't exist is a hard error
    (never silently falls back to defaults) -- see ``configs/reward_dense.yaml``
    / ``configs/reward_sparse.yaml`` for the shipped presets.
    """
    import yaml  # lazy: only needed when loading from disk, not for compute_reward

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Reward config not found: {path}")

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(raw).__name__}")

    try:
        return RewardConfig(**raw)
    except TypeError as exc:
        raise ValueError(f"{path} has an invalid reward config: {exc}") from exc
