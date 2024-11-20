#!/usr/bin/env python3
from collections import defaultdict

import gym
import tictactoe_gym
from tensorboardX import SummaryWriter

ENV_NAME = "TTT-v0"
GAMMA = 0.9
EPSILON = 0.1

class Agent:
    def __init__(self):
        self.env: gym.Env = gym.make(ENV_NAME)
        self.state: tuple[int] = self.env.reset()
        self.q_table = defaultdict(float)

    def play_n_random_steps(self, count: int) -> None:
        for _ in range(count):
            action: int = self.env.action_space.sample()
            new_state: tuple[int]
            reward: float
            is_done: bool
            new_state, reward, is_done, _ = self.env.step(action)
            
    def select_action(self, state: tuple[int]) -> int:
        pass

    def play_episode(self, env: gym.Env) -> float:
        pass

    def update_q_table(self, state: tuple[int], action: int,
                       reward: float, next_stage: tuple[int]) -> None:
        pass


