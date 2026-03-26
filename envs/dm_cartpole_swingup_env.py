from __future__ import annotations

from collections import OrderedDict
from typing import Any

import gymnasium as gym
import numpy as np
from dm_control import suite
from dm_env import specs


def _flatten_value(value: np.ndarray | float | int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return array.reshape(-1)


class DMCartpoleSwingupEnv(gym.Env[np.ndarray, np.ndarray]):
    """Minimal Gymnasium wrapper around dm_control cartpole swingup."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        *,
        flatten_observation: bool = True,
        max_episode_steps: int = 500,
        seed: int | None = None,
        render_mode: str | None = None,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> None:
        super().__init__()
        self.domain_name = "cartpole"
        self.task_name = "swingup"
        self.flatten_observation = flatten_observation
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.frame_width = frame_width
        self.frame_height = frame_height

        self._seed = seed
        self._episode_steps = 0
        self._env = None
        self._time_step = None
        self._action_low = None
        self._action_high = None
        self._action_dtype = np.float32
        self._observation_keys: list[str] = []

        self._build_env(seed=seed)

    def _build_env(self, seed: int | None) -> None:
        task_kwargs: dict[str, Any] = {}
        if seed is not None:
            task_kwargs["random"] = seed

        self._env = suite.load(
            domain_name=self.domain_name,
            task_name=self.task_name,
            task_kwargs=task_kwargs,
        )
        self._seed = seed
        self._episode_steps = 0

        action_spec = self._env.action_spec()
        self._action_low = np.asarray(action_spec.minimum, dtype=np.float32)
        self._action_high = np.asarray(action_spec.maximum, dtype=np.float32)
        self._action_dtype = action_spec.dtype

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=action_spec.shape,
            dtype=np.float32,
        )

        self._observation_keys = list(self._env.observation_spec().keys())
        self.observation_space = self._build_observation_space(self._env.observation_spec())

    def _build_observation_space(self, observation_spec: OrderedDict[str, specs.Array]) -> gym.Space:
        if self.flatten_observation:
            lows: list[np.ndarray] = []
            highs: list[np.ndarray] = []
            for key in self._observation_keys:
                value_spec = observation_spec[key]
                if isinstance(value_spec, specs.BoundedArray):
                    low = np.asarray(value_spec.minimum, dtype=np.float32)
                    high = np.asarray(value_spec.maximum, dtype=np.float32)
                else:
                    low = np.full(value_spec.shape, -np.inf, dtype=np.float32)
                    high = np.full(value_spec.shape, np.inf, dtype=np.float32)
                lows.append(low.reshape(-1))
                highs.append(high.reshape(-1))
            return gym.spaces.Box(
                low=np.concatenate(lows).astype(np.float32),
                high=np.concatenate(highs).astype(np.float32),
                dtype=np.float32,
            )

        dict_space: dict[str, gym.Space] = {}
        for key in self._observation_keys:
            value_spec = observation_spec[key]
            if isinstance(value_spec, specs.BoundedArray):
                low = np.asarray(value_spec.minimum, dtype=np.float32)
                high = np.asarray(value_spec.maximum, dtype=np.float32)
            else:
                low = np.full(value_spec.shape, -np.inf, dtype=np.float32)
                high = np.full(value_spec.shape, np.inf, dtype=np.float32)
            dict_space[key] = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return gym.spaces.Dict(dict_space)

    def _convert_observation(self, observation: OrderedDict[str, Any]) -> np.ndarray | dict[str, np.ndarray]:
        if self.flatten_observation:
            parts = [_flatten_value(observation[key]) for key in self._observation_keys]
            return np.concatenate(parts).astype(np.float32)
        return {
            key: np.asarray(observation[key], dtype=np.float32)
            for key in self._observation_keys
        }

    def _rescale_action(self, action: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        scale = (self._action_high - self._action_low) / 2.0
        midpoint = (self._action_high + self._action_low) / 2.0
        return (midpoint + clipped * scale).astype(self._action_dtype)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        del options

        if seed is not None and seed != self._seed:
            self._build_env(seed=seed)

        self._episode_steps = 0
        self._time_step = self._env.reset()
        observation = self._convert_observation(self._time_step.observation)
        info = {
            "discount": float(self._time_step.discount or 1.0),
            "raw_reward": float(self._time_step.reward or 0.0),
        }
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        scaled_action = self._rescale_action(action)
        self._time_step = self._env.step(scaled_action)
        self._episode_steps += 1

        observation = self._convert_observation(self._time_step.observation)
        reward = float(self._time_step.reward or 0.0)
        terminated = bool(self._time_step.last())
        truncated = self._episode_steps >= self.max_episode_steps and not terminated
        info = {
            "discount": float(self._time_step.discount or 1.0),
            "raw_action": scaled_action,
            "episode_steps": self._episode_steps,
        }
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        if self.render_mode != "rgb_array":
            raise NotImplementedError("Only render_mode='rgb_array' is supported in this sandbox.")
        return self._env.physics.render(
            height=self.frame_height,
            width=self.frame_width,
            camera_id=0,
        )

    def close(self) -> None:
        self._env = None
        self._time_step = None
