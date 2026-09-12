from __future__ import annotations

import numpy as np
import pytest

from rl_robot_arm.ddpg.agent import DDPGAgent, DDPGConfig, load_ddpg_config
from rl_robot_arm.ddpg.replay_buffer import ReplayBuffer


def test_default_config_is_valid():
    config = DDPGConfig()
    agent = DDPGAgent(obs_dim=12, action_dim=2, config=config)
    assert agent.obs_dim == 12
    assert agent.action_dim == 2


def test_invalid_tau_raises():
    with pytest.raises(ValueError):
        DDPGConfig(tau=0.0)


def test_invalid_hidden_sizes_raises():
    with pytest.raises(ValueError):
        DDPGConfig(hidden_sizes=())


def test_load_ddpg_config_reads_shipped_default():
    config = load_ddpg_config("configs/ddpg_default.yaml")
    assert config.hidden_sizes == (64, 64)
    assert config.batch_size == 64


def test_load_ddpg_config_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        load_ddpg_config("configs/does-not-exist.yaml")


def test_act_returns_action_in_bounds():
    agent = DDPGAgent(obs_dim=6, action_dim=2, config=DDPGConfig(seed=1))
    obs = np.zeros(6, dtype=np.float32)
    action = agent.act(obs, explore=True)
    assert action.shape == (2,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_act_without_exploration_is_deterministic():
    agent = DDPGAgent(obs_dim=6, action_dim=2, config=DDPGConfig(seed=1))
    obs = np.ones(6, dtype=np.float32) * 0.3
    a1 = agent.act(obs, explore=False)
    a2 = agent.act(obs, explore=False)
    np.testing.assert_array_equal(a1, a2)


def test_target_networks_start_equal_to_online_networks():
    agent = DDPGAgent(obs_dim=6, action_dim=2, config=DDPGConfig(seed=0))
    for p, tp in zip(agent.actor.parameters(), agent.target_actor.parameters()):
        assert (p == tp).all()
    for p, tp in zip(agent.critic.parameters(), agent.target_critic.parameters()):
        assert (p == tp).all()


def test_update_moves_target_networks_via_soft_update():
    agent = DDPGAgent(obs_dim=4, action_dim=2, config=DDPGConfig(seed=0, tau=0.5, batch_size=8))
    buffer = ReplayBuffer(capacity=100, obs_dim=4, action_dim=2, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        buffer.push(
            rng.normal(size=4).astype(np.float32),
            rng.uniform(-1, 1, size=2).astype(np.float32),
            float(rng.normal()),
            rng.normal(size=4).astype(np.float32),
            False,
        )

    before = [p.clone() for p in agent.target_actor.parameters()]
    losses = agent.update(buffer.sample(8))
    after = list(agent.target_actor.parameters())

    assert "critic_loss" in losses and "actor_loss" in losses
    assert any(not (b == a).all() for b, a in zip(before, after))


def test_target_networks_do_not_require_grad():
    agent = DDPGAgent(obs_dim=4, action_dim=2, config=DDPGConfig(seed=0))
    assert all(not p.requires_grad for p in agent.target_actor.parameters())
    assert all(not p.requires_grad for p in agent.target_critic.parameters())


def test_state_dict_roundtrip():
    agent = DDPGAgent(obs_dim=4, action_dim=2, config=DDPGConfig(seed=0))
    other = DDPGAgent(obs_dim=4, action_dim=2, config=DDPGConfig(seed=99))
    other.load_state_dict(agent.state_dict())
    for p, q in zip(agent.actor.parameters(), other.actor.parameters()):
        assert (p == q).all()
