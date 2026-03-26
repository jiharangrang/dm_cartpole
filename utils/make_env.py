from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3.common.monitor import Monitor

from envs.dm_cartpole_swingup_env import DMCartpoleSwingupEnv


def make_env(
    config: dict[str, Any],
    *,
    seed: int,
    render_mode: str | None = None,
    monitor_path: str | None = None,
):
    env_config = config["env"]
    env = DMCartpoleSwingupEnv(
        flatten_observation=env_config.get("flatten_observation", True),
        max_episode_steps=env_config.get("max_episode_steps", 500),
        seed=seed,
        render_mode=render_mode,
        frame_width=env_config.get("frame_width", 640),
        frame_height=env_config.get("frame_height", 480),
    )
    env.metadata["render_fps"] = env_config.get("render_fps", env.metadata["render_fps"])

    if monitor_path is None:
        return env

    monitor_file = Path(monitor_path)
    monitor_file.parent.mkdir(parents=True, exist_ok=True)
    return Monitor(env, filename=str(monitor_file))


def smoke_test_env(env, rollout_steps: int = 50) -> None:
    env.reset()
    for _ in range(rollout_steps):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
