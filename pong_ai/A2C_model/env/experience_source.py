from collections import deque
from dataclasses import dataclass

import numpy as np
import gymnasium as gym

@dataclass(frozen=True)
class Experience:
    state: np.ndarray
    action: int
    reward: float
    done: bool

    def __iter__(self):
        return iter((self.state, self.action, self.reward, self.done))
    

class ExperienceSource:
    def __init__(self, env: gym.Env, agent, steps_count=2,
                 steps_delta=1, vectorized=False):
        if isinstance(env, (list, tuple)): # To rework
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []
        self.vectorized = vectorized

    def __iter__(self):
        states, agent_states, histories = [], [], []
        env_lens, cur_rewards, cur_steps = [], [], []
        for env in self.pool:
            obs, _ = env.reset() 
            if self.vectorized:
                obs_len = len(obs)
                states.extend(obs)
            else:
                obs_len = 1
                states.append(obs)
            env_lens.append(obs_len)
            for _ in range(obs_len):
                histories.append(deque(maxlen=self.steps_count))
                cur_rewards.append(0.0)
                cur_steps.append(0)
                agent_states.append(None) # warning

        iter_idx = 0
        while True:
            actions = [None] * len(states)
            states_input = []
            states_indices = []
            for idx, state in enumerate(states):
                if state is None:
                    actions[idx] = self.pool[0].action_space.sample()
                else:
                    states_input.append(state)
                    states_indices.append(idx)
            if states_input:
                states_actions, new_agent_states = self.agent(states_input,
                                                              agent_states)
                for idx, action in enumerate(states_actions):
                    g_idx = states_indices[idx]
                    actions[g_idx] = action
                    agent_states[g_idx] = new_agent_states[idx]
            grouped_actions = group_list(actions, env_lens)

            global_ofs = 0
            for _, (env, action_n) in enumerate(zip(self.pool, grouped_actions)):
                if self.vectorized:
                    next_state_n, r_n, truncated, terminated, _ = env.step(action_n)
                    is_done_n = truncated or terminated
                else:
                    next_state, r, truncated, terminated, _ = env.step(action_n[0])
                    is_done = truncated or terminated
                    next_state_n, r_n, is_done_n = [next_state], [r], [is_done]
                for ofs, (action, next_state, r, is_done) in enumerate(
                        zip(action_n, next_state_n, r_n, is_done_n)):
                    idx = global_ofs + ofs
                    state = states[idx]
                    history: deque = histories[idx]
                    cur_rewards[idx] += r
                    cur_steps[idx] += 1
                    if state is not None:
                        history.append(Experience(state=state, action=action,
                                                  reward=r, done=is_done))
                    if len(history) == self.steps_count and \
                        iter_idx % self.steps_delta == 0:
                        yield tuple(history)
                    states[idx] = next_state
                    if is_done:
                        if 0 < len(history) < self.steps_count:
                            yield tuple(history)
                        while len(history) > 1:
                            history.popleft()
                            yield tuple(history)
                        self.total_rewards.append(cur_rewards[idx])
                        self.total_steps.append(cur_steps[idx])
                        cur_rewards[idx] = 0.0
                        cur_steps[idx] = 0
                        if not self.vectorized:
                            states[idx], _ = env.reset()
                        else:
                            states[idx] = None
                        agent_states[idx] = None # warning
                        history.clear()
                global_ofs += len(action_n)
            iter_idx += 1

    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r
    
    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res
    

def group_list(items, lens):
    result = []
    cur_ofs = 0
    for g_len in lens:
        result.append(items[cur_ofs:cur_ofs+g_len])
        cur_ofs += g_len
    return result


@dataclass(frozen=True)
class ExperienceFirstLast:
    state: np.ndarray
    action: int
    reward: float
    last_state: np.ndarray | None

    def __iter__(self):
        return iter((self.state, self.action, self.reward, self.last_state))
    

class ExperienceSourceFirstLast(ExperienceSource):
    def __init__(self, env: gym.Env, agent, gamma, steps_count=1,
                 steps_delta=1, vectorized=False):
        super(ExperienceSourceFirstLast, self).__init__(env, agent,
                                                        steps_count+1,
                                                        steps_delta,
                                                        vectorized=vectorized)
        self.gamma = gamma
        self.steps = steps_count

    def __iter__(self):
        for exp in super(ExperienceSourceFirstLast, self).__iter__():
            if exp[-1].done and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward
            yield ExperienceFirstLast(state=exp[0].state, action=exp[0].action,
                                      reward=total_reward, last_state=last_state)
