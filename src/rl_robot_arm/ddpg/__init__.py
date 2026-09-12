"""DDPG agent (Phase 2): actor-critic networks, target soft updates, replay
buffer, and the training loop that ties them to the Phase 1 environment.
"""
from __future__ import annotations

from .agent import DDPGAgent, DDPGConfig, load_ddpg_config
from .replay_buffer import ReplayBuffer
from .train import TrainingLog, train_ddpg

__all__ = [
    "DDPGAgent",
    "DDPGConfig",
    "load_ddpg_config",
    "ReplayBuffer",
    "TrainingLog",
    "train_ddpg",
]
