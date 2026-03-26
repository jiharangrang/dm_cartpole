from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import SAC

from utils.make_env import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC policy.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sac_cartpole_swingup.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a saved SB3 model zip file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override for evaluation.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy actions instead of deterministic actions.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    config_path = args.config
    if not config_path.is_absolute():
        config_path = base_dir / config_path

    model_path = args.model_path
    if not model_path.is_absolute():
        model_path = base_dir / model_path

    config = load_config(config_path)
    seed = args.seed if args.seed is not None else config["seed"]
    success_threshold = config["evaluation"]["success_reward_threshold"]

    env = make_env(config, seed=seed)
    model = SAC.load(str(model_path), env=env)

    episode_returns = []
    success_count = 0

    for episode_idx in range(args.episodes):
        observation, _ = env.reset(seed=seed + episode_idx)
        done = False
        episode_return = 0.0

        while not done:
            action, _ = model.predict(observation, deterministic=not args.stochastic)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated

        episode_returns.append(episode_return)
        if episode_return >= success_threshold:
            success_count += 1

    env.close()

    returns = np.asarray(episode_returns, dtype=np.float32)
    print(f"Model: {model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean episode reward: {returns.mean():.2f}")
    print(f"Std episode reward: {returns.std():.2f}")
    print(f"Min episode reward: {returns.min():.2f}")
    print(f"Max episode reward: {returns.max():.2f}")
    print(f"Success threshold: {success_threshold:.2f}")
    print(f"Success rate: {success_count / args.episodes:.2%}")
    print(f"Deterministic evaluation: {not args.stochastic}")


if __name__ == "__main__":
    main()
