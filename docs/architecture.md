# Architecture Notes: Reinforcement-Learning Robotic Arm

## Pipeline

```text
Simulated Arm Env -> DDPG + HER Agent -> Reward Signal -> Training Loop -> Evaluation on Target-Reaching
```

## Components

- Simulated robotic arm environment
- DDPG algorithm implementation
- HER (Hindsight Experience Replay)
- Target-reaching task
- Reward design experimentation
- Training curve visualization

## Design Notes

- Keep provider/model choices swappable behind interfaces (see `multi-llm-router`
  and similar projects in this portfolio for the general pattern).
- Prefer configuration-driven pipelines (YAML/JSON in `configs/`) over hardcoded
  parameters so experiments are reproducible.

## Phase 1 environment: `PlanarArmReach-v0`

Implemented in `src/rl_robot_arm/envs/planar_arm.py`. A kinematic (no external
physics engine) `n_links` planar arm anchored at the origin, integrated as
`q <- q + a * max_joint_speed * dt`. Fully deterministic given a seed and
runs headless / CPU-only, so CI exercises it directly.

### Action space

`Box(low=-1.0, high=1.0, shape=(n_links,), dtype=float32)` — per-joint angular
velocity command, scaled internally by `max_joint_speed` (default 2.0 rad/s).
Values outside `[-1, 1]` are clipped.

### Observation space

`Box(low=-inf, high=+inf, shape=(3 * n_links + 6,), dtype=float32)`, laid out as:

| Slice | Size | Meaning |
|-------|------|---------|
| `cos(q)` | `n_links` | cosine of each joint angle |
| `sin(q)` | `n_links` | sine of each joint angle |
| `q_dot` | `n_links` | last commanded joint velocity (rad/s) |
| `ee_xy` | 2 | end-effector position |
| `target_xy` | 2 | target position |
| `target_xy - ee_xy` | 2 | vector from end-effector to target |

For the default 2-link arm this is a 12-vector.

### Reward (Phase 1 default — refined in issue #3)

`reward = -distance(ee, target) - control_cost * ||a||^2`, plus a
`+success_bonus` (default 1.0) on the step the target is reached.

### Episode termination

- `terminated`: end-effector within `tolerance` (default 0.05) of the target.
- `truncated`: emitted by the Gymnasium `TimeLimit` wrapper at
  `max_episode_steps=100` (set in `register`).

### `info` dict

`distance`, `is_success`, `ee_pos`, `target`, `joint_angles`.

## Phase 2 agent: DDPG (`src/rl_robot_arm/ddpg/`)

- `ddpg.agent` — `Actor` (MLP, tanh output matching the `[-1, 1]` action
  space), `Critic` (MLP over `[obs; action] -> Q`), and `DDPGAgent`, which
  owns both plus a frozen (`requires_grad=False`) target copy of each,
  updated via Polyak/soft update (`target <- tau*source + (1-tau)*target`)
  after every gradient step rather than a hard periodic copy.
- `ddpg.replay_buffer` — a fixed-capacity ring buffer of transitions;
  sampling is uniform-random with replacement.
- `ddpg.train.train_ddpg` — the training loop: acts with Gaussian
  exploration noise (random actions during `warmup_steps` so the buffer
  isn't filled entirely by the untrained actor's narrow policy), stores
  transitions, and once the buffer holds a full batch takes one gradient
  step per env step. Logs every episode's return/success/step-count
  (`TrainingLog`) and atomically checkpoints (`configs/ddpg_default.yaml`'s
  companion checkpoint dir) whenever an episode's return beats every prior
  one — the *best*-seen agent, not just the latest.
- Hyperparameters live in `configs/ddpg_default.yaml`, loaded via
  `ddpg.agent.load_ddpg_config` (same explicit-path-must-exist contract as
  `rewards.load_reward_config`).
