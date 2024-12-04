#!/usr/bin/env python3
import gymnasium as gym
import time
import argparse
import numpy as np
import collections
import torch
import os
from ale_py import ALEInterface

from env.lib import wrappers
from env.lib import dqn_model
from constants import *

FPS = 25  

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
                        help="Model file to load (e.g., PongNoFrameskip-v4-best_19.dat)")
    parser.add_argument("-e", "--env", default=ENV_NAME,
                        help=f"Environment name to use, default={ENV_NAME}")
    parser.add_argument("-r", "--record", help="Directory to save video recording")
    parser.add_argument("--no-vis", dest='vis', action='store_false',
                        help="Disable visualization")
    args = parser.parse_args()
    env = wrappers.make_env(args.env, render_mode='human')  
    net = dqn_model.DQN((4, 84, 84), env.action_space.n) 
    state_dict = torch.load(args.model, map_location=lambda stg, _: stg)
    net.load_state_dict(state_dict)
    state, _ = env.reset()
    total_reward = 0.0
    action_counter = collections.Counter()
    while True:
        start_ts = time.time()
        if args.vis:
            frame = env.render()
        state_v = torch.tensor(np.asarray([state]), dtype=torch.float32)
        q_vals = net(state_v).detach().numpy()[0]
        action = np.argmax(q_vals)
        action_counter[action] += 1
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        if done:
            break
        if args.vis:
            delta = 1 / FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", action_counter)
    if args.record:
        env.close()
