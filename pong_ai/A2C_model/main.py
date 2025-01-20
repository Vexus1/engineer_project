import os
import argparse
from collections import deque 

import gymnasium as gym
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim

from env.A2C_model import A2CModel
from env import wrappers
from env.agent import PolicyAgent
from env.experience_source import ExperienceSourceFirstLast
from env.trackers import RewardTracker, TensorBoardMeanTracker
from constants import *

def unpack_batch(batch, net, device='cpu'):
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.asarray(exp.state))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.asarray(exp.last_state))
    states_v = torch.FloatTensor(
        np.asarray(states)).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(np.asarray(last_states)).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        last_vals_np *= GAMMA ** REWARD_STEPS
        rewards_np[not_done_idx] += last_vals_np
    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_t, ref_vals_v

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=ENV_NAME, help="Environment name")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the training run (folder name)")
    parser.add_argument("--load_model", type=str, default=None, help="Path to a saved model to load")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(script_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)  
    log_dir = os.path.join(runs_dir, args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    models_dir = os.path.join(log_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    make_env = lambda: wrappers.make_env(args.env)
    envs = [make_env() for _ in range(NUM_ENVS)]
    net = A2CModel(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print(net)
    
    # if args.load_model:
    #     model_path = os.path.join(script_dir, args.load_model)
    #     net.load_state_dict(torch.load(model_path, map_location=device))
    #     print(f"Model załadowany z {model_path}")
    agent = PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_source = ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
    batch = []
    best_m_reward = None
    total_rewards = deque(maxlen=100)  
    with RewardTracker(writer, stop_reward=18) as tracker:
        with TensorBoardMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    for new_reward in new_rewards:
                        total_rewards.append(new_reward)
                        m_reward = np.mean(total_rewards)
                        if best_m_reward is None or m_reward > best_m_reward:
                            best_m_reward = m_reward
                            save_path = os.path.join(models_dir, f"{args.env}-best_{m_reward:.0f}.dat")
                            torch.save(net.state_dict(), save_path)
                            print(f"Nowy najlepszy wynik: {best_m_reward:.3f} - Model zapisany jako {save_path}")
                    if tracker.reward(new_rewards[0], step_idx):
                        break
                if len(batch) < BATCH_SIZE:
                    continue
                states_v, actions_t, vals_ref_v = unpack_batch(batch, net, device=device)
                batch.clear()
                optimizer.zero_grad()
                logits_v, value_v = net(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.detach()
                log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()
                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()
                loss_policy_v.backward(retain_graph=True)
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in net.parameters()
                                        if p.grad is not None])
                loss_v = entropy_loss_v + loss_value_v
                loss_v.backward()
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                loss_v += loss_policy_v

                tb_tracker.track("advantage",       adv_v, step_idx)
                tb_tracker.track("values",          value_v, step_idx)
                tb_tracker.track("batch_rewards",   vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy",    entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy",     loss_policy_v, step_idx)
                tb_tracker.track("loss_value",      loss_value_v, step_idx)
                tb_tracker.track("loss_total",      loss_v, step_idx)
                tb_tracker.track("grad_l2",         np.sqrt(np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max",        np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var",        np.var(grads), step_idx)
