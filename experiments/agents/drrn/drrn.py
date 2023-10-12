'''
This code has been taken from https://github.com/microsoft/tdqn and modified to match our needs
'''
import numpy as np
import logging

logger = logging.getLogger(__name__)
from tqdm import tqdm
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import DRRN
from .utils.memory import PrioritizedReplayMemory, Transition, State
import sentencepiece as spm

import pickle

# from experiments.agents.base_agent import BaseAgent
from agents.base_agent import BaseAgent

# Accelerate
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.state.device


class DRRNAgent(BaseAgent):
    def __init__(self, envs, reshape_reward, spm_path, saving_path, gamma=0.9, batch_size=64, memory_size=5000000,
                 priority_fraction=0, clip=5, embedding_dim=128, hidden_dim=128, lr=0.0001, max_steps=64, save_frequency=10, nbr_obs=2, test=False):
        super().__init__(envs)
        self.filter_candidates_fn = lambda candidates: [[valid_action[7:-1] if valid_action.startswith('search[') else valid_action[6:-1] \
                                                            for valid_action in valid_actions] for valid_actions in candidates]
        self.infos, self.subgoals= [], []
        self.rewards_envs, self.dones_envs = [], []
        self.n_envs = len(self.env)
        self.obs_queue = [deque([], maxlen=nbr_obs) for _ in range(self.n_envs)]
        self.acts_queue = [deque([], maxlen=nbr_obs - 1) for _ in range(self.n_envs)]
        self.reshape_reward = reshape_reward
        self.gamma = gamma
        self.batch_size = batch_size
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_path)
        ## self.memory = ReplayMemory(memory_size)     ## PJ: Changing to more memory efficient memory, since the pickle files are enormous
        self.memory = PrioritizedReplayMemory(capacity=memory_size,
                                              priority_fraction=priority_fraction)  ## PJ: Changing to more memory efficient memory, since the pickle files are enormous
        self.clip = clip
        self.network = DRRN(len(self.sp), embedding_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.max_steps = max_steps

        # Stateful env
        logging.info("resetting environment")
        for i, env in enumerate(self.env):
            if test:
                ob, info = env.reset(i)
            else:
                ob, info = env.reset()
            self.infos.append(info)
            self.obs_queue[i].append(ob.lower())
            self.subgoals.append(info['valid'])
        logging.info("reset environment")
        prompts = [self.generate_prompt_webshop_v2(goal=self.infos[j]['goal'], subgoals=self.subgoals[j],
                                           deque_obs=self.obs_queue[j], deque_actions=self.acts_queue[j])
                      for j in range(self.n_envs)]
        self.states = self.build_state(prompts)
        self.encoded_actions = self.encode_actions(self.filter_candidates_fn(self.subgoals))
        self.logs = {
            "return_per_episode": [],
            "reshaped_return_per_episode": [],
            "reshaped_return_bonus_per_episode": [],
            "num_frames_per_episode": [],
            "num_frames": self.max_steps,
            "episodes_done": 0,
            "entropy": 0,
            "policy_loss": 0,
            "value_loss": 0,
            "grad_norm": 0,
            "loss": 0
        }
        self.returns = [0 for _ in range(self.n_envs)]
        self.frames_per_episode = [0 for _ in range(self.n_envs)]

        self.save_frequency = save_frequency
        self.saving_path = saving_path
        self.__inner_counter = 0

    def observe(self, state, act, rew, next_state, next_acts, done):
        self.memory.push(False, state, act, rew, next_state, next_acts, done)

    def build_state(self, obs):
        return [State(self.sp.EncodeAsIds(o)) for o in obs]

    def encode_actions(self, acts):
        return [self.sp.EncodeAsIds(a) for a in acts]

    def act(self, states, poss_acts, sample=True):
        """ Returns a string action from poss_acts. """
        act_values = self.network.forward(states, poss_acts)
        if sample:
            act_probs = [F.softmax(vals, dim=0) for vals in act_values]
            act_idxs = [torch.multinomial(probs, num_samples=1).item()
                        for probs in act_probs]
        else:
            act_idxs = [vals.argmax(dim=0).item() for vals in act_values]

        act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(act_idxs)]
        return act_ids, act_idxs, act_values

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute Q(s', a') for all a'
        next_qvals = self.network(batch.next_state, batch.next_acts)
        # Take the max over next q-values
        next_qvals = torch.tensor([vals.max() for vals in next_qvals], device=device)
        # Zero all the next_qvals that are done
        next_qvals = next_qvals * (1 - torch.tensor(batch.done, dtype=torch.float, device=device))
        targets = torch.tensor(batch.reward, dtype=torch.float, device=device) + self.gamma * next_qvals

        # Next compute Q(s, a)
        # Nest each action in a list - so that it becomes the only admissible cmd
        nested_acts = tuple([[a] for a in batch.act])
        qvals = self.network(batch.state, nested_acts)
        # Combine the qvals: Maybe just do a greedy max for generality
        qvals = torch.cat(qvals)

        # Compute Huber loss
        loss = F.smooth_l1_loss(qvals, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        # loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.clip)
        self.optimizer.step()
        return loss

    def update_parameters(self):
        episodes_done = 0
        for i in tqdm(range(self.max_steps // self.n_envs), ascii=" " * 9 + ">", ncols=100):
            action_ids, action_idxs, _ = self.act(self.states, self.encoded_actions, sample=True)
            actions = [_subgoals[idx] for _subgoals, idx in zip(self.subgoals, action_idxs)]
            
            self.infos, self.subgoals = [], []
            self.rewards_envs, self.dones_envs = [], []
            for j, env in enumerate(self.env):
                obs, reward, done, info = env.step(actions[j])
                self.rewards_envs.append(reward)
                self.dones_envs.append(done)
                self.infos.append(info)
                self.subgoals.append(info['valid'])
                self.returns[j] += reward
                self.frames_per_episode[j] += 1
                self.acts_queue[j].append(actions[j])
                self.obs_queue[j].append(obs.lower())
                if done:
                    self.obs_queue[j].clear()
                    self.acts_queue[j].clear()
                    obs, info = env.reset()
                    self.infos[-1] = info
                    self.subgoals[-1] = info['valid']
                    self.obs_queue[j].append(obs.lower())
                    episodes_done += 1
                    self.logs["num_frames_per_episode"].append(self.frames_per_episode[j])
                    self.frames_per_episode[j] = 0
                    self.logs["return_per_episode"].append(self.returns[j])
                    self.returns[j] = 0
                    
                    

            next_prompts = [self.generate_prompt_webshop_v2(goal=self.infos[j]['goal'], subgoals=self.subgoals[j],
                                           deque_obs=self.obs_queue[j], deque_actions=self.acts_queue[j])
                            for j in range(self.n_envs)]
            next_states = self.build_state(next_prompts)
            next_poss_actions = self.encode_actions(self.filter_candidates_fn(self.subgoals))
            for state, act, rew, next_state, next_poss_acts, done in \
                    zip(self.states, action_ids, self.rewards_envs, next_states, next_poss_actions, self.dones_envs):
                self.observe(state, act, rew, next_state, next_poss_acts, done)
            self.states = next_states
            self.encoded_actions = next_poss_actions
            # self.logs["num_frames"] += self.n_envs

        loss = self.update()
        self.__inner_counter += 1
        if self.__inner_counter % self.save_frequency == 0:
            self.save()

        if loss is not None:
            self.logs["loss"] = loss.detach().cpu().item()

        logs = {}
        for k, v in self.logs.items():
            if isinstance(v, list):
                logs[k] = v[:-episodes_done]
            else:
                logs[k] = v
        logs["episodes_done"] = episodes_done
        return logs

    def generate_trajectories(self, n_tests):

        episodes_done = 0
        remove_indexes = []
        reset_index = self.n_envs
        pbar = tqdm(range(n_tests), ascii=" " * 9 + ">", ncols=100)
        while episodes_done < n_tests:

            _, action_idxs, _ = self.act(self.states, self.encoded_actions, sample=True)
            actions = [_subgoals[idx] for _subgoals, idx in zip(self.subgoals, action_idxs)]

            self.infos, self.subgoals = [], []
            for j, env in enumerate(self.env):
                obs, reward, done, info = env.step(actions[j])
                self.infos.append(info)
                self.subgoals.append(info['valid'])
                self.returns[j] += reward
                # self.reshaped_returns[j] += reshaped_rewards[j]
                self.frames_per_episode[j] += 1
                self.acts_queue[j].append(actions[j])
                self.obs_queue[j].append(obs.lower())
                if done:
                    self.obs_queue[j].clear()
                    self.acts_queue[j].clear()
                    obs, info = env.reset(reset_index)
                    reset_index += 1
                    if len(self.env[0].goal_idxs) == reset_index:
                        reset_index = 0
                    episodes_done += 1
                    if episodes_done > n_tests:
                        remove_indexes.append(j)
                    self.infos[-1] = info
                    self.subgoals[-1] = info['valid']
                    self.obs_queue[j].append(obs.lower())
                    self.logs["num_frames_per_episode"].append(self.frames_per_episode[j])
                    self.frames_per_episode[j] = 0
                    self.logs["return_per_episode"].append(self.returns[j])
                    self.returns[j] = 0
                    pbar.update(1)
                    
            # removing environments
            if len(remove_indexes) > 0:
                for index in sorted(remove_indexes, reverse=True):
                    del self.env[index]
                    del self.infos[index]
                    del self.subgoals[index]
                    del self.returns[index]
                    del self.frames_per_episode[index]
                    del self.obs_queue[index]
                    del self.acts_queue[index]
                    
                    
            next_prompts = [self.generate_prompt_webshop_v2(goal=self.infos[j]['goal'], subgoals=self.subgoals[j],
                                           deque_obs=self.obs_queue[j], deque_actions=self.acts_queue[j])
                            for j in range(self.n_envs)]
            self.states = self.build_state(next_prompts)
            
            self.encoded_actions = self.encode_actions(self.filter_candidates_fn(self.subgoals))
            
            # self.logs["num_frames"] += self.n_envs
        pbar.close()

        logs = {}
        for k, v in self.logs.items():
            if isinstance(v, list):
                logs[k] = v[:]
            else:
                logs[k] = v
        logs["episodes_done"] = episodes_done
        return logs

    def load(self):
        try:
            with open(self.saving_path + "/memory.pkl", 'rb') as _file:
                saved_memory = pickle.load(_file)
            self.memory = saved_memory
            self.optimizer.load_state_dict(torch.load(self.saving_path + "/optimizer.checkpoint"))
        except Exception as err:
            print(f"Encountered the following exception when trying to load the memory, an empty memory will be used instead: {err}")

        self.network.load_state_dict(torch.load(self.saving_path + "/model.checkpoint"))


    def save(self):
        torch.save(self.network.state_dict(), self.saving_path + "/model.checkpoint")
        torch.save(self.optimizer.state_dict(), self.saving_path + "/optimizer.checkpoint")
        with open(self.saving_path + "/memory.pkl", 'wb') as _file:
            pickle.dump(self.memory, _file)