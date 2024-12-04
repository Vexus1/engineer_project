import os
from dataclasses import dataclass
from collections import deque

import argparse
import time
import numpy as np
from numpy import ndarray
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from env.lib import wrappers
from env.lib import dqn_model
from constants import *

@dataclass(frozen=True)
class Experience:
    """
    Represents a single experience step in the environment.

    Args
    ----------
    state : ndarray
        Current state of the environment.
    action : int
        Action taken by the agent.
    reward : float
        Reward received after the action.
    done : bool
        Boolean indicating whether the episode ended.
    new_state : ndarray
        State resulting from the action.
    """
    
    state: ndarray
    action: int
    reward: float
    done: bool
    new_state: ndarray

    def __iter__(self):
        return iter((self.state, self.action, 
                     self.reward, self.done, self.new_state))


class ExperienceBuffer:
    """
    Stores and manages a fixed-size buffer of experience tuples for replay.
    Provides random sampling of experiences for training the Q-network.

    Args
    ----------
    capacity : int
        Maximum number of experiences to store.
    """

    def __init__(self, capacity: int):
        self.buffer: deque[Experience] = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience) -> None: 
        """Adds an experience to the buffer."""   
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> tuple[ndarray, ndarray,
                                               ndarray, ndarray, ndarray]:
        """Randomly samples a batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)
    

class Agent:
    """
    Manages the interaction between the agent and the environment.
    Responsible for:

        - Selecting actions based on epsilon-greedy policy.
        - Collecting experiences.
        - Resetting the environment after an episode.

    Args
    ----------
    - env : wrappers.make_env
        The game environment.
    - exp_buffer : ExperienceBuffer
        ExperienceBuffer for storing gameplay experiences.
    """

    def __init__(self, env: wrappers.make_env, exp_buffer: ExperienceBuffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self) -> None:
        """Resets the agent's state and reward for a new episode."""
        self.state, _ = env.reset()  
        self.state = np.asarray(self.state, dtype=np.float32)  
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net: nn.Module,
                epsilon: float = 0.0, device: str = "cpu") -> float | None:
        """
        Executes a single step in the environment, appending the
        experience to the buffer and returning 
        the total reward if the episode ends.
        """
        done_reward = None
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.asarray([self.state]) 
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated
        self.total_reward += reward
        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch: tuple[ndarray, ndarray, ndarray, ndarray, ndarray],
              net: nn.Module, tgt_net: nn.Module,
              device: str = "cpu") -> torch.Tensor:
    """Calculates the mean squared error loss for the Q-network."""
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(np.array(
        states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(
        next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()
    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=ENV_NAME,
                        help="Name of the environment, default=" + ENV_NAME)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    env = wrappers.make_env(args.env)
    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape,
                            env.action_space.n).to(device)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    writer = SummaryWriter(comment="-" + args.env)
    print(net)
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START -
                      frame_idx / EPSILON_DECAY_LAST_FRAME)
        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, reward %.3f, "
                  "eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), m_reward, epsilon,
                speed
            ))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), args.env +
                           "-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (
                        best_m_reward, m_reward))
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break
        if len(buffer) < REPLAY_START_SIZE:
            continue
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()