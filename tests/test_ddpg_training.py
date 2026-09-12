"""Integration test for issue #4/#5's core claim: the DDPG agent trains and
improves success rate over episodes on the dense-reward environment.

Uses a deliberately easy configuration (a single-link arm with targets
sampled near its reachable circle, a generous tolerance) so a real DDPG
update loop demonstrably improves within a test-suite-sized episode budget
(no mocking of the learning algorithm itself -- this is CPU-only and takes
several seconds, not a network/hardware boundary that needs mocking).
Every seed (env, buffer sampling, torch weights, exploration noise) is
fixed, so the run is deterministic.
"""
from __future__ import annotations

import gymnasium as gym

from rl_robot_arm import register_envs
from rl_robot_arm.ddpg.agent import DDPGAgent, DDPGConfig
from rl_robot_arm.ddpg.replay_buffer import ReplayBuffer
from rl_robot_arm.ddpg.train import train_ddpg

register_envs()


def _train(tmp_path, episodes=180):
    env = gym.make(
        "PlanarArmReach-v0",
        n_links=1,
        max_episode_steps=60,
        tolerance=0.15,
        target_radius_range=(0.85, 0.98),
    )
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    config = DDPGConfig(
        hidden_sizes=(32, 32),
        batch_size=32,
        buffer_capacity=8000,
        warmup_steps=300,
        actor_lr=2e-3,
        critic_lr=2e-3,
        seed=0,
    )
    agent = DDPGAgent(obs_dim, action_dim, config)
    buffer = ReplayBuffer(capacity=config.buffer_capacity, obs_dim=obs_dim, action_dim=action_dim, seed=0)
    log = train_ddpg(env, agent, buffer, episodes=episodes, seed=0, checkpoint_dir=tmp_path)
    env.close()
    return log


def test_success_rate_improves_over_training(tmp_path):
    log = _train(tmp_path)
    window = 40
    assert len(log.episodes) >= 2 * window

    early = sum(1 for e in log.episodes[:window] if e.success) / window
    late = sum(1 for e in log.episodes[-window:] if e.success) / window
    assert late > early


def test_mean_return_improves_over_training(tmp_path):
    log = _train(tmp_path)
    window = 40
    early_mean = sum(e.total_return for e in log.episodes[:window]) / window
    late_mean = sum(e.total_return for e in log.episodes[-window:]) / window
    assert late_mean > early_mean


def test_training_log_tracks_best_episode(tmp_path):
    log = _train(tmp_path, episodes=60)
    assert log.best_episode is not None
    assert log.best_return == max(e.total_return for e in log.episodes)


def test_checkpoint_is_saved_for_best_agent(tmp_path):
    log = _train(tmp_path, episodes=60)
    assert log.checkpoint_path is not None
    assert log.checkpoint_path.exists()
    assert not log.checkpoint_path.with_suffix(".pt.tmp").exists()


def test_checkpoint_can_be_loaded_back_into_a_fresh_agent(tmp_path):
    import torch

    log = _train(tmp_path, episodes=60)
    config = DDPGConfig(hidden_sizes=(32, 32), seed=123)
    fresh_agent = DDPGAgent(obs_dim=9, action_dim=1, config=config)
    state = torch.load(log.checkpoint_path, weights_only=True)
    fresh_agent.load_state_dict(state)  # must not raise
