#!/usr/bin/env python3
from __future__ import annotations
from collections import defaultdict

import gym
import tictactoe_gym
import numpy as np

from constants import *

class Agent:
    def __init__(self):
        self.env: gym.Env = gym.make(ENV_NAME)
        self.state: tuple[int] = self.env.reset()
        self.q_table = defaultdict(float)
        self.epsilon = EPSILON

    def play_n_random_steps(self, count: int) -> None:
        for _ in range(count):
            action: int = self.env.action_space.sample()
            new_state: tuple[int]
            reward: float
            is_done: bool
            new_state, reward, is_done, _ = self.env.step(action)
            self.update_q_table(self.state, action, reward, new_state)
            if is_done:
                self.state = self.env.reset()
            else:
                self.state = new_state
    
    def update_q_table(self, state: tuple[int], action: int,
                       reward: float, next_state: tuple[int]) -> None:
        max_q_next = max([self.q_table[(next_state, _action)]
                          for _action in range(self.env.action_space.n)],
                          default=0)
        old_val = self.q_table[(state, action)]
        q_update = old_val + ALPHA * (reward + GAMMA * max_q_next - old_val)
        self.q_table[(state, action)] = q_update

    @classmethod
    def update_epsiolon(cls, agent: Agent, epsilon_decay: float,
                        epsilon_min: float) -> None:
        agent.epsilon = max(agent.epsilon * epsilon_decay, epsilon_min)

    def select_action(self, state: tuple[int]) -> int:
        if np.random.rand() < EPSILON:
            return self.env.action_space.sample()
        q_values = [self.q_table[(state, action)]
                    for action in range(self.env.action_space.n)]
        return int(np.argmax(q_values))

    def play_episode(self, env: gym.Env) -> float:
        total_reward = 0.0
        state: tuple[int] = env.reset()
        while True:
            action: int = self.select_action(state)
            new_state: tuple[int]
            reward: float
            is_done: bool
            new_state, reward, is_done, _ = env.step(action)
            self.update_q_table(state, action, reward, new_state)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward
