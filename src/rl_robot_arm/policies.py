"""Simple hand-written policies for smoke-testing the environment.

No learning here -- these just show the Phase 1 env is controllable before
the DDPG agent (Phase 2) exists.
"""

from __future__ import annotations

import numpy as np

from .envs.planar_arm import PlanarArmReachEnv


def random_policy(env: PlanarArmReachEnv, _obs: np.ndarray) -> np.ndarray:
    return env.action_space.sample()


def jacobian_transpose_policy(
    env: PlanarArmReachEnv, _obs: np.ndarray, gain: float = 4.0
) -> np.ndarray:
    """Move the end-effector toward the target via the transpose Jacobian.

    ``dq ~ J^T (target - ee)``, then scale into the [-1, 1] action range.
    This is only a controllability smoke check -- being a local method it
    can stall at a kinematic singularity, which is exactly the kind of
    reward-shaping / exploration problem Phases 2-3 exist to solve.
    """
    q = env._q  # noqa: SLF001 - intentional white-box helper
    angles = np.cumsum(q)
    # Column i of the planar Jacobian: sum of downstream link contributions.
    jac = np.zeros((2, env.n_links))
    for i in range(env.n_links):
        downstream = env.link_lengths[i:]
        downstream_angles = angles[i:]
        jac[0, i] = -np.sum(downstream * np.sin(downstream_angles))
        jac[1, i] = np.sum(downstream * np.cos(downstream_angles))

    error = env._target - env.end_effector  # noqa: SLF001
    dq = gain * jac.T @ error
    scaled = dq / (env.max_joint_speed * env.dt)
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


POLICIES = {
    "random": random_policy,
    "p-control": jacobian_transpose_policy,
}
