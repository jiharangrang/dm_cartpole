from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import yaml
from stable_baselines3 import SAC

from utils.make_env import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a trained SAC policy and save a GIF.")
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
        default=1,
        help="Number of episodes to record.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override for playback.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output GIF path. Defaults to videos/playback.gif",
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
    render_fps = config["env"].get("render_fps", 20)

    default_output = base_dir / "videos" / "playback.gif"
    output_path = args.output if args.output is not None else default_output
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(config, seed=seed, render_mode="rgb_array")
    model = SAC.load(str(model_path), env=env)

    frames = []
    for episode_idx in range(args.episodes):
        observation, _ = env.reset(seed=seed + episode_idx)
        frames.append(env.render())
        done = False

        while not done:
            action, _ = model.predict(observation, deterministic=not args.stochastic)
            observation, _, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            done = terminated or truncated

    env.close()
    imageio.mimsave(output_path, frames, fps=render_fps)

    print(f"Saved playback to: {output_path}")
    print(f"Frames: {len(frames)}")
    print(f"Deterministic playback: {not args.stochastic}")


if __name__ == "__main__":
    main()
