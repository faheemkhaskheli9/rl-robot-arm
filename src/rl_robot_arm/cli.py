"""Command-line entrypoint: roll out a policy in the planar-arm env.

    python -m rl_robot_arm --episodes 3 --policy p-control
    python -m rl_robot_arm --episodes 5 --policy random --links 3 --seed 0
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Sequence

import gymnasium as gym

from . import register_envs
from .policies import POLICIES


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

    returns: list[float] = []
    successes = 0
    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        total = 0.0
        done = False
        steps = 0
        while not done:
            action = policy(env.unwrapped, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            done = terminated or truncated
        successes += int(info["is_success"])
        returns.append(total)
        print(
            f"episode {episode + 1:>2}: return={total:8.3f}  steps={steps:>3}  "
            f"final_dist={info['distance']:.4f}  success={info['is_success']}"
        )

    env.close()
    print(
        f"\n{args.policy}: mean_return={statistics.fmean(returns):.3f}  "
        f"success_rate={successes / args.episodes:.2f}  ({successes}/{args.episodes})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
