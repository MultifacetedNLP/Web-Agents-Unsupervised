from .base_ppo_agent import BasePPOAgent

import babyai.utils
from babyai.rl.utils import DictList

import os
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
from collections import deque
import logging
from transformers import BartForConditionalGeneration, BartTokenizer
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

class LLMPPOAgentWebshop(BasePPOAgent):
    def __init__(self, envs, lm_server, llm_scoring_module_key, nbr_llms=None, num_frames_per_proc=None, discount=0.99,
                 lr=7e-4, beta1=0.9, beta2=0.999, gae_lambda=0.95, entropy_coef=0.01, value_loss_coef=0.5,
                 max_grad_norm=0.5, adam_eps=1e-5, clip_eps=0.2, epochs=4, prioritization_best_trajectories=2, batch_size=64, reshape_reward=None,
                 name_experiment=None, saving_path_model=None, saving_path_logs=None, number_envs=None, subgoals=None,
                 nbr_obs=3, id_expe=None, template_test=1, aux_info=None, debug=False, test=False):
        super().__init__(envs, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef, value_loss_coef,
                         max_grad_norm, reshape_reward, aux_info, device=torch.device("cpu"))

        self.lm_server = lm_server
        self.llm_scoring_module_key = llm_scoring_module_key
        # Useful filter to avoid computing score of each candidate when using additional heads directly
        if llm_scoring_module_key == "score":
            self.filter_candidates_fn = lambda candidates: [[valid_action[7:-1] if valid_action.startswith('search[') else valid_action[6:-1] \
                                                            for valid_action in valid_actions] for valid_actions in candidates]
        elif llm_scoring_module_key == "policy_head":
            self.filter_candidates_fn = lambda candidates: None
        else:
            raise NotImplementedError()

        self.nbr_obs = nbr_obs
        self.obs_queue = [deque([], maxlen=self.nbr_obs) for _ in range(self.num_procs)]
        self.acts_queue = [deque([], maxlen=self.nbr_obs - 1) for _ in range(self.num_procs)]
        # self.subgoals = subgoals
        shape = (self.num_frames_per_proc, self.num_procs)
        self.rewards_envs, self.dones_envs = [], []
        self.obs, self.infos, self.subgoals = [], [], []
        self.subgoalss = [None] * (shape[0])
        logging.info("resetting environment")
        for i, env in enumerate(self.env):
            if test:
                print(i)
                ob, info = env.reset(i)
            else:
                ob, info = env.reset()
            self.infos.append(info)
            self.obs_queue[i].append(ob)
            self.subgoals.append(info['valid'])
        logging.info("reset environment")

        self.prompts = [None] * (shape[0])
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)

        self.nbr_llms = nbr_llms

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.prioritization_best_trajectories = prioritization_best_trajectories
        self.batch_size = batch_size
        self.debug = debug

        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_eps = adam_eps

        self.name_experiment = name_experiment
        self.saving_path_model = saving_path_model
        self.saving_path_logs = saving_path_logs
        self.number_envs = number_envs

        self.id_expe = id_expe
        self.template_test = template_test
        self.number_updates = 0

        if self.saving_path_logs and id_expe:
            self.experiment_path = os.path.join(self.saving_path_logs, id_expe)

    def collect_experiences(self, debug=False):
        """Collects rollouts and computes advantages.
        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.
        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        # self.lm_server.generate(contexts=[""], do_sample=True, top_k=2)

        for i in tqdm(range(self.num_frames_per_proc), ascii=" " * 9 + ">", ncols=100):
            # Do one agent-environment interaction

            prompt = [self.generate_prompt_webshop_v2(goal=self.infos[j]['goal'], subgoals=self.subgoals[j],
                                           deque_obs=self.obs_queue[j], deque_actions=self.acts_queue[j])
                      for j in range(self.num_procs)]
            
            # [[self.lm_server.generate(contexts=[promp], do_sample=True, top_k=2)[0][0]['text']] if valid_actions[0].startswith('search[') else valid_actions for valid_actions, promp in zip(self.subgoals, prompt)]

            output = self.lm_server.custom_module_fns(module_function_keys=[self.llm_scoring_module_key, 'value'],
                                                      contexts=prompt,
                                                      candidates=self.filter_candidates_fn(self.subgoals))
            
            # scores = torch.stack([_o[self.llm_scoring_module_key] for _o in output]).squeeze()
            # dist = Categorical(logits=scores)
            scores = [_o[self.llm_scoring_module_key] for _o in output]
            dists = [Categorical(logits=score) for score in scores]
            action = torch.stack([dist.sample() for dist in dists])
            a = action.cpu().numpy()
            
            # action = dist.sample()
            # a = action.cpu().numpy()
            
            values = torch.stack([_o["value"][0] for _o in output])
            
            actions_str = []
            for j in range(self.num_procs):
                actions_str.append(self.subgoals[j][int(a[j])])
                self.acts_queue[j].append(actions_str[j])

            
            
            # take a step based on the calculated action
            self.subgoalss[i] = self.subgoals
            self.infos, self.subgoals = [], []
            self.rewards_envs, self.dones_envs = [], []
            for j, env in enumerate(self.env):
                obs, reward, done, info = env.step(actions_str[j])
                self.rewards_envs.append(reward)
                self.dones_envs.append(done)
                self.infos.append(info)
                self.subgoals.append(info['valid'])
                self.obs_queue[j].append(obs)
                if done:
                    # reinitialise memory of past observations and actions
                    self.obs_queue[j].clear()
                    self.acts_queue[j].clear()
                    obs, info = env.reset()
                    self.infos[-1] = info
                    self.subgoals[-1] = info['valid']
                    self.obs_queue[j].append(obs)


            if self.aux_info:
                env_info = self.aux_info_collector.process(env_info)
                # env_info = self.process_aux_info(env_info)

            # Update experiences values

            # self.obss[i] = self.obs
            # self.obs = obs

            self.prompts[i] = prompt

            self.dones[i] = torch.tensor(self.dones_envs, device=self.device, dtype=torch.float)
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(self.dones_envs, device=self.device, dtype=torch.float)

            self.actions[i] = action
            self.values[i] = values.squeeze()

            if self.reshape_reward is not None:
                rewards_shaped = torch.tensor([
                    self.reshape_reward(subgoal_proba=None, reward=reward_, policy_value=None, llm_0=None)
                    for reward_ in self.rewards_envs
                ], device=self.device)
                self.rewards[i] = rewards_shaped[:, 0]
                self.rewards_bonus[i] = rewards_shaped[:, 1]
            else:
                self.rewards[i] = torch.tensor(self.rewards_envs, device=self.device)

            # log_prob = dist.log_prob(action)
            log_prob = torch.stack([dist.log_prob(action[j]) for j, dist in enumerate(dists)])
            

            if len(log_prob.shape) > 1:
                log_prob = log_prob.sum(dim=-1)
            self.log_probs[i] = log_prob

            # Update log values

            self.log_episode_return += torch.tensor(self.rewards_envs, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_reshaped_return_bonus += self.rewards_bonus[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(self.dones_envs):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item() * 10)
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item() * 10)
                    self.log_reshaped_return_bonus.append(self.log_episode_reshaped_return_bonus[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_reshaped_return_bonus *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences
        prompt = [self.generate_prompt_webshop_v2(goal=self.infos[i]['goal'], subgoals=self.subgoals[i],
                                       deque_obs=self.obs_queue[i], deque_actions=self.acts_queue[i])
                  for i in range(self.num_procs)]
        output = self.lm_server.custom_module_fns(module_function_keys=['value'], contexts=prompt)
        next_value = torch.stack([_o["value"] for _o in output]).squeeze()

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i + 1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i + 1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i + 1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk

        exps = DictList()
        exps.prompt = np.array([self.prompts[i][j]
                                for j in range(self.num_procs)
                                for i in range(self.num_frames_per_proc)])
        exps.subgoal = np.array([self.subgoalss[i][j]
                                 for j in range(self.num_procs)
                                 for i in range(self.num_frames_per_proc)])
        # In commments below T is self.num_frames_per_proc, P is self.num_procs,
        # D is the dimensionality

        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)

        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)
        self.dones[-1, :] = 1
        exps.dones = self.dones.transpose(0, 1).reshape(-1)

        if self.aux_info:
            exps = self.aux_info_collector.end_collection(exps)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "reshaped_return_bonus_per_episode": self.log_reshaped_return_bonus[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "episodes_done": self.log_done_counter,
        }
        

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_reshaped_return_bonus = self.log_reshaped_return_bonus[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, log
    
    def best_trajectories_indexs(self, rewards, dones, best_reward=10):
        last_reward = 0
        rewards_length = len(rewards)

        best_indexes = []
        for i in range(rewards_length - 1, -1, -1):
            if dones[i] == 1:
                last_reward = rewards[i]
                    
            if last_reward == best_reward:
                best_indexes.append(i)

        return best_indexes

    def update_parameters(self):
        # Collect experiences
        exps, logs = self.collect_experiences(debug=self.debug)
        # print(exps.action)
        # action_counts = exps.action.unique(return_counts=True)
        # pi_l_action_counts = exps.pi_l_action.unique(return_counts=True)
        '''
        exps is a DictList with the following keys ['prompt', 'action', 'value', 'reward',
         'advantage', 'returnn', 'log_prob'] and ['collected_info', 'extra_predictions'] if we use aux_info
        exps.prompt is a (n_procs * n_frames_per_proc) of prompt
        if we use aux_info: exps.collected_info and exps.extra_predictions are DictLists with keys
        being the added information. They are either (n_procs * n_frames_per_proc) 1D tensors or
        (n_procs * n_frames_per_proc) x k 2D tensors where k is the number of classes for multiclass classification
        '''
        
        if self.prioritization_best_trajectories>0:
            best_indexes = self.best_trajectories_indexs(exps.reward, exps.dones) * \
                            self.prioritization_best_trajectories
        else:
            best_indexes = []
        
        lm_server_update_first_call = True
        for _ in tqdm(range(self.epochs), ascii=" " * 9 + "<", ncols=100):
            # Initialize log values

            log_entropies = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            log_losses = []

            # Create minibatch of size self.batch_size*self.nbr_llms
            # each llm receive a batch of size batch_size
            for inds in self._get_batches_starting_indexes(best_indexes):
                # inds is a numpy array of indices that correspond to the beginning of a sub-batch
                # there are as many inds as there are batches

                exps_batch = exps[inds]

                # return the list of dict_return calculate by each llm
                list_dict_return = self.lm_server.update(exps_batch.prompt,
                                                         self.filter_candidates_fn(exps_batch.subgoal),
                                                         exps=dict(exps_batch),
                                                         lr=self.lr,
                                                         beta1=self.beta1,
                                                         beta2=self.beta2,
                                                         adam_eps=self.adam_eps,
                                                         clip_eps=self.clip_eps,
                                                         entropy_coef=self.entropy_coef,
                                                         value_loss_coef=self.value_loss_coef,
                                                         max_grad_norm=self.max_grad_norm,
                                                         nbr_llms=self.nbr_llms,
                                                         id_expe=self.id_expe,
                                                         lm_server_update_first_call=lm_server_update_first_call,
                                                         saving_path_model=self.saving_path_model,
                                                         experiment_path=self.experiment_path,
                                                         number_updates=self.number_updates,
                                                         scoring_module_key=self.llm_scoring_module_key,
                                                         template_test=self.template_test)

                lm_server_update_first_call = False

                log_losses.append(np.mean([d["loss"] for d in list_dict_return]))
                log_entropies.append(np.mean([d["entropy"] for d in list_dict_return]))
                log_policy_losses.append(np.mean([d["policy_loss"] for d in list_dict_return]))
                log_value_losses.append(np.mean([d["value_loss"] for d in list_dict_return]))
                log_grad_norms.append(np.mean([d["grad_norm"] for d in list_dict_return]))

        # Log some values

        logs["entropy"] = np.mean(log_entropies)
        logs["policy_loss"] = np.mean(log_policy_losses)
        logs["value_loss"] = np.mean(log_value_losses)
        logs["grad_norm"] = np.mean(log_grad_norms)
        logs["loss"] = np.mean(log_losses)

        return logs

    def _get_batches_starting_indexes(self, best_indexes=[]):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.
        Returns
        -------
        batches_starting_indexes : list of lists filter_candidates_fnof int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = np.arange(0, self.num_frames)
        if best_indexes:
            indexes.extend(best_indexes)
        indexes = np.random.permutation(indexes)

        num_indexes = self.batch_size
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes

    
    def process_goal(self, state):
        state = state.lower().replace('"', '').replace("'", "")
        state = state.replace('amazon shopping game\ninstruction:', '').replace('instruction:', '')
        state = state.replace('\n[button] search [button_]', '').strip()
        if ', and price lower than' in state:
            state = state.split(', and price lower than')[0]
        return state
    
    
    def bart_predict(self, input, model, skip_special_tokens=True, **kwargs):
        input_ids = bart_tokenizer(input)['input_ids']
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        output = model.generate(input_ids, max_length=512, **kwargs)
        return bart_tokenizer.batch_decode(output.tolist(), skip_special_tokens=skip_special_tokens)

    

    def generate_trajectories(self, n_tests, sample=False, deactivte_RL_for_search=False, bart_path="", generate_query=False):
        """Generates trajectories and calculates relevant metrics.
        Runs several environments concurrently.
        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        
        if bart_path:
            bart_model = BartForConditionalGeneration.from_pretrained(bart_path)
            print('bart model loaded', bart_path)
        
        pbar = tqdm(range(n_tests), ascii=" " * 9 + ">", ncols=100)
        reset_index = self.num_procs
        while self.log_done_counter < n_tests:
            
            if bart_path or generate_query:
                actions_str = []
                for j, subgoal in enumerate(self.subgoals):
                    if subgoal[0].startswith('search['):
                        if bart_path:
                            goal = self.process_goal(self.infos[j]['goal'])
                            query = self.bart_predict(goal, bart_model, num_return_sequences=5, num_beams=5)
                            query = query[0]
                            actions_str.append(f'search[{query}]')
                            self.acts_queue[j].append(actions_str[j])
                        else:
                            
                            prompt = self.generate_prompt_webshop_v2(goal=self.infos[j]['goal'], subgoals=self.subgoals[j],
                                                            deque_obs=self.obs_queue[j],
                                                            deque_actions=self.acts_queue[j])
                            
                            
                            query = self.lm_server.generate(contexts=[prompt], num_beams=5)
                            query = query[0][0]["text"].replace("[", "").replace("]", "")
                            print(query)
                            actions_str.append(f'search[{query}]')
                            self.acts_queue[j].append(actions_str[j])
                    else:
                        prompt = self.generate_prompt_webshop_v2(goal=self.infos[j]['goal'], subgoals=self.subgoals[j],
                                                            deque_obs=self.obs_queue[j],
                                                            deque_actions=self.acts_queue[j])
                        
                        output = self.lm_server.custom_module_fns(
                            module_function_keys=[self.llm_scoring_module_key], # 'value'
                            contexts=[prompt],
                            candidates=self.filter_candidates_fn([self.subgoals[j]]))
                        
                        scores = output[0][self.llm_scoring_module_key]
                        
                        if sample:
                            dist = Categorical(logits=scores)
                            action = dist.sample()
                            a = action.cpu().numpy()
                        else:
                            a = torch.argmax(scores).item()
                            
                        actions_str.append(self.subgoals[j][int(a)])
                        self.acts_queue[j].append(actions_str[j])
            else:
                # Do one agent-environment interaction
                prompt = [self.generate_prompt_webshop_v2(goal=self.infos[j]['goal'], subgoals=self.subgoals[j],
                                                            deque_obs=self.obs_queue[j],
                                                            deque_actions=self.acts_queue[j])
                        for j in range(self.num_procs)]

                output = self.lm_server.custom_module_fns(
                    module_function_keys=[self.llm_scoring_module_key], # 'value'
                    contexts=prompt,
                    candidates=self.filter_candidates_fn(self.subgoals))
                scores = [_o[self.llm_scoring_module_key] for _o in output]
                # vals = torch.stack([_o["value"][0] for _o in output]).cpu().numpy()
                
                if sample:
                    # a = [torch.multinomial(F.softmax(score, dim=0), 1)[0].item() for score in scores]
                    dists = [Categorical(logits=score) for score in scores]
                    action = torch.stack([dist.sample() for dist in dists])
                    a = action.cpu().numpy()
                else:
                    a = [torch.argmax(score).item() for score in scores]
                
                if deactivte_RL_for_search:
                    a = [-1 if subgoal[0].startswith('search[') else action for (action, subgoal) in zip(a, self.subgoals)]
                

                actions_str = []
                for j in range(self.num_procs):
                    actions_str.append(self.subgoals[j][int(a[j])])
                    # self.actions.append(self.subgoals[j][int(a[j])])
                    self.acts_queue[j].append(actions_str[j])


            # take a step based on the calculated action
            self.infos, self.subgoals = [], []
            self.rewards_envs, self.dones_envs = [], []
            for j, env in enumerate(self.env):
                # self.vals.append(vals[j][0])
                # self.prompts.append(prompt[j])
                obs, reward, done, info = env.step(actions_str[j])
                self.rewards_envs.append(reward)
                self.dones_envs.append(done)
                self.infos.append(info)
                self.subgoals.append(info['valid'])
                self.obs_queue[j].append(obs)
                if done:
                    # reinitialise memory of past observations and actions
                    self.obs_queue[j].clear()
                    self.acts_queue[j].clear()
                    print("reset_index: ", reset_index)
                    print("reward: ", reward)
                    obs, info = env.reset(reset_index)
                    reset_index += 1
                    self.infos[-1] = info
                    self.subgoals[-1] = info['valid']
                    self.obs_queue[j].append(obs)

            # self.obs = obs

            self.mask = 1 - torch.tensor(self.dones_envs, device=self.device, dtype=torch.float)


            self.log_episode_return += torch.tensor(self.rewards_envs, device=self.device, dtype=torch.float)
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(self.dones_envs):
                if done_ and self.log_done_counter < n_tests:
                    self.log_done_counter += 1
                    pbar.update(1)
                        
                    self.log_return.append(self.log_episode_return[i].item() * 10)
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_num_frames *= self.mask

        pbar.close()

        log = {
            "return_per_episode": self.log_return[-self.log_done_counter:],
            "num_frames_per_episode": self.log_num_frames[-self.log_done_counter:]
        }

        return log