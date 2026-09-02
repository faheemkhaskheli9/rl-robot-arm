"""Reinforcement-learning robotic arm.

Phase 1 provides a headless, CPU-only simulated planar arm wrapped in the
Gymnasium API so later phases (DDPG, HER) can train against a standard
``reset()`` / ``step()`` interface.
"""

from __future__ import annotations

from gymnasium.envs.registration import register

from .envs.planar_arm import PlanarArmReachEnv

__all__ = ["PlanarArmReachEnv", "register_envs"]

_REGISTERED = False


def register_envs() -> None:
    """Register this project's environments with Gymnasium (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    register(
        id="PlanarArmReach-v0",
        entry_point="rl_robot_arm.envs.planar_arm:PlanarArmReachEnv",
        max_episode_steps=100,
    )
    _REGISTERED = True


register_envs()
