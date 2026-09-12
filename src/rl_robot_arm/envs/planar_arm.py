"""A headless planar robotic-arm reaching environment (Gymnasium API).

An ``n_links`` planar arm is anchored at the origin. The agent commands
joint angular velocities; the arm integrates them kinematically (no
external physics engine, so it runs CPU-only and deterministically in CI).
The task is to bring the end-effector onto a randomly placed reachable
target.

Spaces
------
action : Box(-1, 1, shape=(n_links,)), float32
    Per-joint velocity command, scaled by ``max_joint_speed`` (rad/s).
observation : Box(shape=(3 * n_links + 6,)), float32
    ``[cos(q), sin(q), q_dot, ee_xy, target_xy, (target - ee)_xy]``.

Reward (Phase 1 default -- refined in issue #3)
    Dense: ``-distance(ee, target) - control_cost * ||action||^2``.
    ``+success_bonus`` on the step the target is reached.

Episode end
    terminated : end-effector within ``tolerance`` of the target.
    truncated  : handled by the Gymnasium ``TimeLimit`` wrapper
                 (``max_episode_steps=100`` from ``register``).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

__all__ = ["PlanarArmReachEnv"]


class PlanarArmReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        n_links: int = 2,
        link_lengths: tuple[float, ...] | None = None,
        max_joint_speed: float = 2.0,
        dt: float = 0.1,
        tolerance: float = 0.05,
        control_cost: float = 0.01,
        success_bonus: float = 1.0,
        target_radius_range: tuple[float, float] = (0.2, 0.9),
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if n_links < 1:
            raise ValueError("n_links must be >= 1")
        if link_lengths is None:
            link_lengths = tuple(1.0 / n_links for _ in range(n_links))
        if len(link_lengths) != n_links:
            raise ValueError(f"link_lengths must have {n_links} entries, got {len(link_lengths)}")
        if any(length <= 0 for length in link_lengths):
            raise ValueError("link lengths must be positive")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        low, high = target_radius_range
        if not (0.0 <= low < high <= 1.0):
            raise ValueError(
                "target_radius_range must satisfy 0 <= low < high <= 1, "
                f"got {target_radius_range}"
            )

        self.n_links = n_links
        self.link_lengths = np.asarray(link_lengths, dtype=np.float64)
        self.reach = float(self.link_lengths.sum())
        self.max_joint_speed = float(max_joint_speed)
        self.dt = float(dt)
        self.tolerance = float(tolerance)
        self.control_cost = float(control_cost)
        self.success_bonus = float(success_bonus)
        self.target_radius_range = (float(low), float(high))
        self.render_mode = render_mode

        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_links,), dtype=np.float32)
        obs_dim = 3 * n_links + 6
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._q = np.zeros(n_links, dtype=np.float64)
        self._q_dot = np.zeros(n_links, dtype=np.float64)
        self._target = np.zeros(2, dtype=np.float64)

    # ------------------------------------------------------------------ kinematics
    def forward_kinematics(self, q: np.ndarray | None = None) -> np.ndarray:
        """Return every joint position including base (0,0) and end-effector."""
        q = self._q if q is None else np.asarray(q, dtype=np.float64)
        angles = np.cumsum(q)
        segments = self.link_lengths[:, None] * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        )
        points = np.vstack([np.zeros((1, 2)), np.cumsum(segments, axis=0)])
        return points

    @property
    def end_effector(self) -> np.ndarray:
        return self.forward_kinematics()[-1]

    def _sample_reachable_target(self) -> np.ndarray:
        low, high = self.target_radius_range
        radius = self.np_random.uniform(low * self.reach, high * self.reach)
        angle = self.np_random.uniform(-np.pi, np.pi)
        return np.array([radius * np.cos(angle), radius * np.sin(angle)])

    def _distance(self) -> float:
        return float(np.linalg.norm(self.end_effector - self._target))

    def _get_obs(self) -> np.ndarray:
        ee = self.end_effector
        return np.concatenate(
            [
                np.cos(self._q),
                np.sin(self._q),
                self._q_dot,
                ee,
                self._target,
                self._target - ee,
            ]
        ).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        distance = self._distance()
        return {
            "distance": distance,
            "is_success": distance <= self.tolerance,
            "ee_pos": self.end_effector.copy(),
            "target": self._target.copy(),
            "joint_angles": self._q.copy(),
        }

    # ------------------------------------------------------------------ gym API
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._q = self.np_random.uniform(-np.pi, np.pi, size=self.n_links)
        self._q_dot = np.zeros(self.n_links, dtype=np.float64)

        if options and "target" in options:
            self._target = np.asarray(options["target"], dtype=np.float64)
        else:
            self._target = self._sample_reachable_target()

        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(self.n_links)
        action = np.clip(action, -1.0, 1.0)

        self._q_dot = action * self.max_joint_speed
        self._q = self._wrap(self._q + self._q_dot * self.dt)

        distance = self._distance()
        terminated = distance <= self.tolerance
        reward = -distance - self.control_cost * float(np.sum(action**2))
        if terminated:
            reward += self.success_bonus

        return self._get_obs(), reward, terminated, False, self._get_info()

    @staticmethod
    def _wrap(q: np.ndarray) -> np.ndarray:
        return (q + np.pi) % (2 * np.pi) - np.pi

    # ------------------------------------------------------------------ render
    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        size = 128
        img = np.full((size, size, 3), 255, dtype=np.uint8)
        scale = (size / 2 - 4) / max(self.reach, 1e-6)
        center = np.array([size / 2, size / 2])

        def to_px(p: np.ndarray) -> np.ndarray:
            px = center + np.array([p[0], -p[1]]) * scale
            return np.clip(px, 0, size - 1).astype(int)

        pts = self.forward_kinematics()
        for a, b in zip(pts[:-1], pts[1:]):
            self._draw_line(img, to_px(a), to_px(b), (30, 30, 30))
        tx, ty = to_px(self._target)
        img[max(ty - 2, 0) : ty + 3, max(tx - 2, 0) : tx + 3] = (220, 40, 40)
        return img

    @staticmethod
    def _draw_line(img: np.ndarray, a: np.ndarray, b: np.ndarray, color) -> None:
        steps = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]), 1))
        for t in np.linspace(0, 1, steps + 1):
            x, y = (a + t * (b - a)).astype(int)
            img[y, x] = color
