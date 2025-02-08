import os
import pathlib

import gym_aloha
from openpi_client.runtime import environment as _environment
from typing_extensions import override
import numpy as np

from custom_task import CustomTransferTask

class CustomAlohaEnv(_environment.Environment):
    """Custom Aloha environment with a water tank object."""
    
    def __init__(self, seed: int = 0):
        # Get the path to our custom assets
        assets_dir = pathlib.Path(__file__).parent / "assets"
        
        # Create our custom task
        self._task = CustomTransferTask(
            object_file=str(assets_dir / "watertank_removal.stl"),
            object_scale=0.5,  # Scale the object size
            object_pos=np.array([0.5, 0.0, 0.1]),  # Position on the table
            object_euler=np.array([0, 0, 0]),  # Rotation in Euler angles
        )
        
        self._last_obs = None
        self._done = True
        self._episode_reward = 0.0
    
    @override
    def reset(self) -> None:
        timestep = self._task._env.reset()
        self._last_obs = self._convert_observation(timestep.observation)
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
        timestep = self._task._env.step(action["actions"])
        self._last_obs = self._convert_observation(timestep.observation)
        self._done = timestep.last()
        self._episode_reward = max(self._episode_reward, timestep.reward or 0.0)
    
    def _convert_observation(self, raw_obs: dict) -> dict:
        img = raw_obs["images"]["top"]
        # Convert axis order from [H, W, C] --> [C, H, W]
        img = np.transpose(img, (2, 0, 1))
        
        return {
            "state": raw_obs["qpos"],
            "images": {"cam_high": img},
        } 