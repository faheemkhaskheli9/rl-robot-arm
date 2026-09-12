"""DDPG agent: actor/critic networks with target-network soft updates
(issue #4).

Hyperparameters are configurable via `configs/ddpg_default.yaml` (loaded
through `load_ddpg_config`, following the same pattern as
`rewards.load_reward_config`: an explicit path that doesn't exist is a hard
error, never a silent fallback to defaults).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

__all__ = ["DDPGConfig", "load_ddpg_config", "Actor", "Critic", "DDPGAgent"]


@dataclass(frozen=True)
class DDPGConfig:
    """Hyperparameters for one DDPG agent. Construct via
    :func:`load_ddpg_config` or directly for tests."""

    hidden_sizes: tuple[int, ...] = (64, 64)
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005  # soft-update rate for target networks
    buffer_capacity: int = 50_000
    batch_size: int = 64
    warmup_steps: int = 200  # random-action steps before training starts
    exploration_noise_std: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.hidden_sizes or any(h <= 0 for h in self.hidden_sizes):
            raise ValueError("hidden_sizes must be a non-empty tuple of positive ints")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError("tau must be in (0, 1]")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1]")
        if self.batch_size <= 0 or self.buffer_capacity <= 0:
            raise ValueError("batch_size and buffer_capacity must be positive")


def load_ddpg_config(path: str | os.PathLike[str]) -> DDPGConfig:
    """Load a :class:`DDPGConfig` from a YAML file under `configs/`.

    A path the caller explicitly passed that doesn't exist is a hard error
    (never silently falls back to defaults) -- see
    `configs/ddpg_default.yaml` for the shipped preset.
    """
    import yaml  # lazy: only needed when loading from disk

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DDPG config not found: {path}")

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(raw).__name__}")
    if "hidden_sizes" in raw and raw["hidden_sizes"] is not None:
        raw["hidden_sizes"] = tuple(raw["hidden_sizes"])

    try:
        return DDPGConfig(**raw)
    except TypeError as exc:
        raise ValueError(f"{path} has an invalid DDPG config: {exc}") from exc


def _mlp(sizes: list[int], activation: type[nn.Module], out_activation: type[nn.Module] | None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = i == len(sizes) - 2
        if not is_last:
            layers.append(activation())
        elif out_activation is not None:
            layers.append(out_activation())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Maps an observation to an action in [-1, 1]^action_dim (tanh output,
    matching `PlanarArmReachEnv`'s `Box(-1, 1, ...)` action space)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        self.net = _mlp([obs_dim, *hidden_sizes, action_dim], nn.ReLU, nn.Tanh)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    """Maps (observation, action) to a scalar Q-value estimate."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        self.net = _mlp([obs_dim + action_dim, *hidden_sizes, 1], nn.ReLU, None)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """target <- tau * source + (1 - tau) * target, in place."""
    with torch.no_grad():
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.mul_(1.0 - tau).add_(s_param, alpha=tau)


class DDPGAgent:
    """Actor-critic DDPG agent with target networks and soft (Polyak)
    updates.

    All state (networks, RNG for exploration noise) is created from
    `config.seed`, so a fixed seed makes a training run reproducible.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: DDPGConfig | None = None):
        self.config = config or DDPGConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        torch.manual_seed(self.config.seed)
        self._noise_rng = np.random.default_rng(self.config.seed)

        h = self.config.hidden_sizes
        self.actor = Actor(obs_dim, action_dim, h)
        self.critic = Critic(obs_dim, action_dim, h)
        self.target_actor = Actor(obs_dim, action_dim, h)
        self.target_critic = Critic(obs_dim, action_dim, h)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_actor.parameters():
            p.requires_grad_(False)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)

    def act(self, obs: np.ndarray, *, explore: bool = True) -> np.ndarray:
        """Return an action for `obs`. With `explore=True`, adds Gaussian
        exploration noise (std = `config.exploration_noise_std`) and clips
        back into [-1, 1]."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = self.actor(obs_t).squeeze(0).numpy()
        if explore:
            action = action + self._noise_rng.normal(
                0.0, self.config.exploration_noise_std, size=self.action_dim
            ).astype(np.float32)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def update(self, batch) -> dict[str, float]:
        """One gradient step on a sampled `ddpg.replay_buffer.Batch`.

        Returns the critic/actor losses for logging.
        """
        obs = torch.as_tensor(batch.obs, dtype=torch.float32)
        action = torch.as_tensor(batch.action, dtype=torch.float32)
        reward = torch.as_tensor(batch.reward, dtype=torch.float32)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32)
        done = torch.as_tensor(batch.done, dtype=torch.float32)

        # --- critic: TD target from target networks (no grad through them) ---
        with torch.no_grad():
            next_action = self.target_actor(next_obs)
            target_q = self.target_critic(next_obs, next_action)
            y = reward + self.config.gamma * (1.0 - done) * target_q
        q = self.critic(obs, action)
        critic_loss = nn.functional.mse_loss(q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- actor: maximize Q(obs, actor(obs)) i.e. minimize -Q ---
        actor_loss = -self.critic(obs, self.actor(obs)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        _soft_update(self.target_actor, self.actor, self.config.tau)
        _soft_update(self.target_critic, self.critic, self.config.tau)

        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item())}

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_actor.load_state_dict(state["target_actor"])
        self.target_critic.load_state_dict(state["target_critic"])
