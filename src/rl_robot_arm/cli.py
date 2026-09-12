"""Command-line entrypoint: roll out a policy in the planar-arm env.

    python -m rl_robot_arm --episodes 3 --policy p-control
    python -m rl_robot_arm --episodes 5 --policy random --links 3 --seed 0
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import gymnasium as gym

from . import register_envs
from .policies import POLICIES
from .rollout import evaluate_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rl_robot_arm",
        description="Roll out a hand-written policy in the PlanarArmReach-v0 environment.",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--policy", choices=sorted(POLICIES), default="p-control")
    parser.add_argument("--links", type=int, default=2, help="Number of arm joints.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-steps", type=int, default=100, help="Step cap per episode."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    register_envs()

    env = gym.make(
        "PlanarArmReach-v0",
        n_links=args.links,
        max_episode_steps=args.max_steps,
    )
    policy = POLICIES[args.policy]

    stats = evaluate_policy(env, policy, episodes=args.episodes, seed=args.seed)
    for i, ep in enumerate(stats.episodes):
        print(
            f"episode {i + 1:>2}: return={ep.total_return:8.3f}  steps={ep.steps:>3}  "
            f"final_dist={ep.final_distance:.4f}  success={ep.success}"
        )

    env.close()
    print(
        f"\n{args.policy}: mean_return={stats.mean_return:.3f}  "
        f"success_rate={stats.success_rate:.2f}  "
        f"({sum(1 for ep in stats.episodes if ep.success)}/{len(stats.episodes)})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
