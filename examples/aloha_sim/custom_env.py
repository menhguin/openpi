import os
import pathlib

import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gymnasium import spaces
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from custom_task import CustomTransferTask
from gym_aloha.constants import ACTIONS, JOINTS, DT
from gym_aloha.utils import sample_box_pose

class CustomGymEnv(gym.Env):
    """Gymnasium environment wrapper for our custom task."""
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=640,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
        seed=None,
    ):
        super().__init__()
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        # Create the task and environment
        self.task = CustomTransferTask(random=np.random.RandomState(seed))
        self._env = self.task._env

        # Define spaces
        if self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "top": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8,
                            )
                        }
                    ),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(len(JOINTS),),
                        dtype=np.float64,
                    ),
                }
            )
        else:
            raise ValueError(f"Unsupported observation type: {self.obs_type}")

        self.action_space = spaces.Box(low=-1, high=1, shape=(len(ACTIONS),), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Seed the task's RNG
        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)

        # Sample a new initial position
        self.task.set_initial_pose(sample_box_pose(seed))

        # Reset the environment
        timestep = self._env.reset()
        observation = self._format_obs(timestep.observation)

        info = {"is_success": False}
        return observation, info

    def step(self, action):
        assert action.ndim == 1
        timestep = self._env.step(action)

        observation = self._format_obs(timestep.observation)
        reward = timestep.reward or 0.0
        terminated = reward == 4  # Success is when reward reaches maximum
        truncated = False
        info = {"is_success": terminated}

        return observation, reward, terminated, truncated, info

    def _format_obs(self, raw_obs):
        if self.obs_type == "pixels_agent_pos":
            return {
                "pixels": {"top": raw_obs["images"]["top"].copy()},
                "agent_pos": raw_obs["qpos"],
            }
        else:
            raise ValueError(f"Unsupported observation type: {self.obs_type}")

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        assert self.render_mode == "rgb_array"
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        return self._env.physics.render(height=height, width=width, camera_id="top")

    def close(self):
        pass


class CustomAlohaEnv(_environment.Environment):
    """OpenPI client environment wrapper for our custom environment."""

    def __init__(self, seed: int = 0) -> None:
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)

        self._gym = CustomGymEnv(seed=seed)
        self._last_obs = None
        self._done = True
        self._episode_reward = 0.0

    @override
    def reset(self) -> None:
        gym_obs, _ = self._gym.reset(seed=int(self._rng.integers(2**32 - 1)))
        self._last_obs = self._convert_observation(gym_obs)
        self._done = False
        self._episode_reward = 0.0

    @override
    def is_episode_complete(self) -> bool:
        return self._done

    @override
    def get_observation(self) -> dict:
        if self._last_obs is None:
            raise RuntimeError("Observation is not set. Call reset() first.")
        return self._last_obs

    @override
    def apply_action(self, action: dict) -> None:
        gym_obs, reward, terminated, truncated, info = self._gym.step(action["actions"])
        self._last_obs = self._convert_observation(gym_obs)
        self._done = terminated or truncated
        self._episode_reward = max(self._episode_reward, reward)

    def _convert_observation(self, gym_obs: dict) -> dict:
        img = gym_obs["pixels"]["top"]
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        # Convert axis order from [H, W, C] --> [C, H, W]
        img = np.transpose(img, (2, 0, 1))

        return {
            "state": gym_obs["agent_pos"],
            "images": {"cam_high": img},
        } 