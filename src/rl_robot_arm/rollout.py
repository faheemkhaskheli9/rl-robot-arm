"""Reusable episode rollout + aggregate metrics for the reaching task.

Pulled out of the CLI so ``success_rate`` (and the per-episode return/steps
it's built from) is a value any caller -- CLI, a test, a future training
loop's eval callback -- can get back and log, not just something printed to
stdout.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class EpisodeResult:
    """Outcome of one rolled-out episode."""

    total_return: float
    steps: int
    success: bool
    final_distance: float


@dataclass(frozen=True)
class RolloutStats:
    """Aggregate metrics over a batch of episodes."""

    episodes: list[EpisodeResult]

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for ep in self.episodes if ep.success) / len(self.episodes)

    @property
    def mean_return(self) -> float:
        if not self.episodes:
            return 0.0
        return statistics.fmean(ep.total_return for ep in self.episodes)

    @property
    def mean_steps(self) -> float:
        if not self.episodes:
            return 0.0
        return statistics.fmean(ep.steps for ep in self.episodes)

    def as_dict(self) -> dict[str, float]:
        return {
            "success_rate": self.success_rate,
            "mean_return": self.mean_return,
            "mean_steps": self.mean_steps,
            "n_episodes": len(self.episodes),
        }


def run_episode(env, policy: Callable[[object, np.ndarray], np.ndarray], seed: int) -> EpisodeResult:
    """Roll out ``policy`` in ``env`` for one episode and return its outcome.

    ``env`` must itself bound episode length (e.g. via Gymnasium's
    ``TimeLimit``, as ``gym.make("PlanarArmReach-v0", ...)`` does) so an
    episode that never reaches the target still ends with ``truncated=True``
    -- the raw, unwrapped :class:`PlanarArmReachEnv` never truncates on its
    own and a policy that never succeeds would loop here forever.
    """
    obs, info = env.reset(seed=seed)
    total = 0.0
    steps = 0
    done = False
    while not done:
        action = policy(env.unwrapped if hasattr(env, "unwrapped") else env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        steps += 1
        done = terminated or truncated

    return EpisodeResult(
        total_return=total,
        steps=steps,
        success=bool(info["is_success"]),
        final_distance=float(info["distance"]),
    )


def evaluate_policy(
    env,
    policy: Callable[[object, np.ndarray], np.ndarray],
    *,
    episodes: int,
    seed: int = 0,
) -> RolloutStats:
    """Roll out ``policy`` for ``episodes`` episodes and return aggregate stats."""
    results = [run_episode(env, policy, seed=seed + i) for i in range(episodes)]
    return RolloutStats(episodes=results)
