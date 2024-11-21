import os

import gym
from tensorboardX import SummaryWriter

from env.qlearn_agent import Agent
from constants import *

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    os.chdir(script_dir) 
    log_dir = os.path.join(script_dir, "logs")  
    test_env: gym.Env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(logdir=log_dir, comment="tic-tac-toe-q-learning")
    iter_num = 0
    best_reward = 0.0
    while True:
        iter_num += 1
        agent.play_n_random_steps(10)
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_num)
        if reward > best_reward:
            print(f"Best reward updated {best_reward:.3f} -> {reward:.3f}")
            best_reward = reward
        agent.update_epsiolon(agent, EPSILON_DECAY, EPSILON_MIN)
        if reward > 0.95:
            print(f"solved in {iter_num} iterations")
            break
    writer.close()
