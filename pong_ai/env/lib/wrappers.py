from collections import deque

import cv2
import gymnasium as gym
from ale_py import ALEInterface
from gymnasium.spaces import Box
import numpy as np
from numpy import ndarray

from constants import *

class FireEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(FireEnv, self).__init__(env)
        if not (env.unwrapped.get_action_meanings()[1] == 'FIRE' and
                len(env.unwrapped.get_action_meanings()) >= 3):
            raise ValueError(
                "The environment does not support the FIRE action or "
                "has insufficient action meanings."
            )

    def step(self, action: int) -> tuple[ndarray, float, bool, bool, dict]:
        result = self.env.step(action)
        observation, reward, terminated, truncated, info = result
        return observation, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> tuple[ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        result = self.env.step(1)
        obs, _, terminated, truncated, info = result
        done = terminated or truncated
        if done:
            obs, info = self.env.reset(**kwargs)
        result = self.env.step(2)
        obs, _, terminated, truncated, info = result
        done = terminated or truncated
        if done:
            obs, info = self.env.reset(**kwargs)
        return obs, info
    

class MaxSkipEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
        super(MaxSkipEnv, self).__init__(env)
        self.obs_buffer = deque(maxlen=2)
        self.skip = skip
    
    def step(self, action: int) -> tuple[ndarray, float, bool, bool, dict]:
        total_reward = 0.0
        for _ in range(self.skip):
            result = self.env.step(action)
            obs, reward, terminated, truncated, info = result
            is_done = terminated or truncated
            self.obs_buffer.append(obs)
            total_reward += reward
            if is_done:
                break
        max_frame = np.max(np.stack(self.obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> tuple[ndarray, dict]:
        self.obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self.obs_buffer.append(obs)
        return obs, info


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, observation: ndarray) -> ndarray:
        return ProcessFrame84.process(observation)

    @staticmethod
    def process(frame: ndarray) -> ndarray:
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else: 
            raise ValueError(f"Unknown resolution: {frame.size}")
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + \
              img[:, :, 2] * 0.114
        resized_screen = cv2.resize(
            img, (84, 110), interpolation=cv2.INTER_AREA
            )
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)
    

class ImageToTorch(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super(ImageToTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32
        )

    def observation(self, observation: ndarray) -> ndarray:
        return np.moveaxis(observation, 2, 0)


    
class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, observation: ndarray) -> ndarray:
        return np.array(observation).astype(np.float32) / 255.0
    

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, n_steps: int, dtype: type = np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0),
            dtype=dtype
        )

    def reset(self, **kwargs) -> tuple[ndarray, dict]:
        self.buffer = np.zeros_like(
            self.observation_space.low, dtype=self.dtype
        )
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs.squeeze()), info 
    
    def observation(self, observation: ndarray) -> ndarray:
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def make_env(env_name: str) -> gym.Env:
    env = gym.make(env_name, render_mode=None)
    env = MaxSkipEnv(env)
    env = FireEnv(env)
    env = ProcessFrame84(env)
    env = ImageToTorch(env)
    env = BufferWrapper(env, n_steps=4)
    env = ScaledFloatFrame(env)
    return env
