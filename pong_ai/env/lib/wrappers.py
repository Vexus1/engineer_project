from collections import deque

import cv2
import gymnasium as gym
from ale_py import ALEInterface
import numpy as np
from numpy import ndarray

from constants import *

class FireResetEnv(gym.Wrapper):
    """A wrapper for environments requiring a FIRE action to start the game."""

    def __init__(self, env: gym.Env):
        super(FireResetEnv, self).__init__(env)
        if not (env.unwrapped.get_action_meanings()[1] == 'FIRE' and
                len(env.unwrapped.get_action_meanings()) >= 3):
            raise ValueError(
                "The environment does not support the FIRE action or "
                "has insufficient action meanings."
            )

    def step(self, action: int) -> tuple[ndarray, float, bool, bool, dict]:
        """Executes a step in the environment."""
        return self.env.step(action)
    
    def reset(self, **kwargs) -> tuple[ndarray, dict]:
        """Resets the environment and performs the FIRE action."""
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        done = terminated or truncated
        if done:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        done = terminated or truncated
        if done:
            self.env.reset(**kwargs)
        return obs, info
    

class MaxAndSkipEnv(gym.Wrapper): 
    """
    Skips n frames to improve efficiency and aggregates rewards over skipped steps.
    Outputs the maximum pixel value across two recent frames to reduce flicker.
    """

    def __init__(self, env: gym.Env = None, skip: int = 4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action: int) -> tuple[ndarray, float, bool, bool, dict]:
        """
        Executes an action for multiple frames and aggregates rewards.
        Retains the maximum pixel value of two consecutive frames.
        """
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> ndarray:
        """Resets the environment and clears the frame buffer."""
        self._obs_buffer.clear()
        obs, _ = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    """
    Preprocesses frames by:
        - Converting them to grayscale.
        - Resizing to 84x84 resolution.
    """
    def __init__(self, env: gym.Env = None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, observation: ndarray) -> ndarray:
        """Processes a single observation frame into grayscale and resizes it."""
        return ProcessFrame84.process(observation)

    @staticmethod
    def process(frame: ndarray) -> ndarray:
        """ Converts a frame to grayscale, resizes it to 84x84, and normalizes values."""
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            raise ValueError(f"Unknown resolution: {frame.size}")
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + \
              img[:, :, 2] * 0.114
        resized_screen = cv2.resize(
            img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)
    

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Converts image format to match PyTorch requirements.
    Changes the image's channel order to (channels, height, width).
    """
    
    def __init__(self, env: gym.Env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def observation(self, observation: ndarray) -> ndarray:
        """Converts an image from HWC (height, width, channels) to CHW format."""
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Scales pixel values in frames from [0, 255] to [0, 1].
    Helps normalize inputs for the neural network.
    """

    def observation(self, observation: ndarray) -> ndarray:
        """Scales pixel values to the range [0, 1]."""
        return np.array(observation).astype(np.float32) / 255.0
    

class BufferWrapper(gym.ObservationWrapper):
    """
    Maintains a buffer of n most recent observations.
    Combines these observations into a single input for the neural network.
    """

    def __init__(self, env: gym.Env, n_steps: int, dtype: type = np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self, **kwargs) -> tuple[ndarray, dict]:
        """Resets the environment and initializes the observation buffer."""
        self.buffer = np.zeros_like(
            self.observation_space.low, dtype=self.dtype)
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, observation: ndarray) -> ndarray:
        """Updates the buffer with a new observation."""
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def make_env(env_name, render_mode=None):
    """
    Creates a processed environment with a sequence of wrappers.
    Includes skipping frames, preprocessing images, and normalizing data.
    """
    env = gym.make(env_name, render_mode=render_mode)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)
