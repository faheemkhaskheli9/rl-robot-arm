"""Training loop tying the replay buffer + DDPG agent to a Gymnasium env
(issue #5): episode reward/success logging, checkpointing the
best-performing agent seen so far.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .agent import DDPGAgent
from .replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)

__all__ = ["EpisodeLog", "TrainingLog", "train_ddpg"]


@dataclass(frozen=True)
class EpisodeLog:
    episode: int
    total_return: float
    steps: int
    success: bool


@dataclass
class TrainingLog:
    """Per-episode reward/success history plus which episode produced the
    best-performing checkpoint (if any was saved)."""

    episodes: list[EpisodeLog] = field(default_factory=list)
    best_return: float = float("-inf")
    best_episode: int | None = None
    checkpoint_path: Path | None = None

    def success_rate(self, last_n: int | None = None) -> float:
        eps = self.episodes[-last_n:] if last_n else self.episodes
        if not eps:
            return 0.0
        return sum(1 for e in eps if e.success) / len(eps)

    def mean_return(self, last_n: int | None = None) -> float:
        eps = self.episodes[-last_n:] if last_n else self.episodes
        if not eps:
            return 0.0
        return float(np.mean([e.total_return for e in eps]))


def _write_checkpoint_atomic(path: Path, agent: DDPGAgent) -> None:
    """Save `agent`'s weights atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(agent.state_dict(), tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def train_ddpg(
    env,
    agent: DDPGAgent,
    buffer: ReplayBuffer,
    *,
    episodes: int,
    seed: int = 0,
    updates_per_step: int = 1,
    checkpoint_dir: str | os.PathLike[str] | None = None,
) -> TrainingLog:
    """Train `agent` against `env` for `episodes` episodes.

    Each step: act with exploration noise, store the transition, and once
    the buffer holds at least one batch and `config.warmup_steps` random-ish
    steps have passed, take `updates_per_step` gradient steps. Logs every
    episode's return/success; when `checkpoint_dir` is given, atomically
    overwrites `<checkpoint_dir>/best.pt` whenever an episode's return beats
    every prior one (checkpointing the best-performing agent, not just the
    latest).
    """
    log = TrainingLog()
    checkpoint_path = Path(checkpoint_dir) / "best.pt" if checkpoint_dir else None
    total_steps = 0

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        total_return = 0.0
        steps = 0
        done = False
        while not done:
            if total_steps < agent.config.warmup_steps:
                # Warmup: pure random exploration fills the buffer with
                # diverse transitions before the actor's own (initially
                # poor) policy starts driving data collection.
                action = env.action_space.sample()
            else:
                action = agent.act(obs, explore=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.push(obs, action, reward, next_obs, terminated)
            obs = next_obs
            total_return += float(reward)
            steps += 1
            total_steps += 1

            if len(buffer) >= agent.config.batch_size and total_steps >= agent.config.warmup_steps:
                for _ in range(updates_per_step):
                    agent.update(buffer.sample(agent.config.batch_size))

        success = bool(info.get("is_success", False))
        log.episodes.append(EpisodeLog(episode=ep, total_return=total_return, steps=steps, success=success))

        if total_return > log.best_return:
            log.best_return = total_return
            log.best_episode = ep
            if checkpoint_path is not None:
                _write_checkpoint_atomic(checkpoint_path, agent)
                log.checkpoint_path = checkpoint_path

        logger.info(
            "episode %d: return=%.3f steps=%d success=%s (buffer=%d)",
            ep, total_return, steps, success, len(buffer),
        )

    return log
