#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from typing import Any

import gym
import tictactoe_gym
import numpy as np
from numpy import ndarray
from icecream import ic

from constants import *

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

@dataclass(frozen=True)
class State:
    board: tuple[int, ...]
    metadata: frozenset[tuple[str, Any]]

    @classmethod
    def from_raw(cls, raw_state: tuple[ndarray, dict]) -> State:
        board = tuple(map(int, raw_state[0].flatten()))
        metadata = cls._make_hashable(raw_state[1])
        return cls(board=board, metadata=metadata)

    @staticmethod
    def _make_hashable(obj: Any) -> Any:
        """
        Recursively converts lists in a dictionary or other structures to tuples
        to ensure hashability.
        """
        if isinstance(obj, list):
            return tuple(State._make_hashable(i) for i in obj)
        elif isinstance(obj, np.ndarray):  
            return tuple(obj.flatten())
        elif isinstance(obj, dict):
            return frozenset((k, State._make_hashable( v)) for k, v in obj.items())
        return obj


class Agent:
    def __init__(self):
        self.env: gym.Env = gym.make(ENV_NAME)
        self.state = State.from_raw(self.env.reset())
        self.q_table = defaultdict(float)
        self.epsilon = EPSILON

    def play_n_random_steps(self, count: int) -> None:
        for _ in range(count):
            action: int = self.env.action_space.sample()
            new_state: tuple[ndarray, dict]
            reward: float
            is_done: bool
            new_state, reward, terminated, truncated, _ = self.env.step(action)
            is_done = terminated or truncated  
            new_state = State.from_raw(new_state)
            self.update_q_table(self.state, action, reward, new_state)
            if is_done:
                self.state = State.from_raw(self.env.reset())
            else:
                self.state = new_state
    
    def update_q_table(self, state: State, action: int,
                       reward: float, next_state: State) -> None:
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

    def select_action(self, state: ndarray) -> int:
        if np.random.rand() < EPSILON:
            return self.env.action_space.sample()
        state = tuple(state.flatten())
        q_values =   [self.q_table[(state, action)]
                    for action in range(self.env.action_space.n)]
        return int(np.argmax(q_values))

    def play_episode(self, env: gym.Env) -> float:
        total_reward = 0.0
        state = State.from_raw(env.reset())
        while True:
            action: int = self.select_action(state)
            new_state: ndarray
            reward: float
            is_done: bool
            new_state, reward, terminated, truncated, _ = self.env.step(action)
            is_done = terminated or truncated  
            new_state = State.from_raw(new_state)
            self.update_q_table(state, action, reward, new_state)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward
