# load WebEnV
import os
import torch
from web_agent_site import WebEnv
import argparse
from dotmap import DotMap
from agents.ppo.llm_ppo_agent_webshop import LLMPPOAgentWebshop
import hydra
from lamorel import Caller, lamorel_init
from lamorel import BaseUpdater
from train_language_agent import ValueHeadModuleFn, LogScoringModuleFn, ActionHeadsModuleFn
import json
import babyai.utils as utils
import csv

from colorama import Fore

from accelerate import Accelerator
import logging
import numpy as np

lamorel_init()
logger = logging.getLogger(__name__)
accelerator = Accelerator()


class LoadSpecificWeightsUpdater(BaseUpdater):
    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        if not hasattr(self, "is_loaded"):
            try:
                self._llm_module.load_state_dict(torch.load(kwargs["saving_path_model"] +
                                                            "/" + kwargs["id_expe"] + "/last/model.checkpoint", map_location='cuda:0'))
                self.is_loaded = True
                print("Last")
            except:
                self._llm_module.load_state_dict(torch.load(kwargs["saving_path_model"] +
                                                            "/" + kwargs["id_expe"] + "/backup/model.checkpoint"), map_location='cuda:0')
                self.is_loaded = True
                print("Backup")


def run_agent(algo, saving_path_logs, id_expe, n_tests):
        format_str = ("Reward: {: .2f} +- {: .2f}  (Min: {: .2f} Max: {: .2f}) |\
        Success Rate: {: .2f} |")


        test_path = os.path.join(os.path.join(saving_path_logs, id_expe), 'test')
        csv_path = os.path.join(test_path, 'log.csv')
        first_created = not os.path.exists(csv_path)
        
        csv_writer = csv.writer(open(csv_path, 'a', 1))
        if first_created:
            csv_writer.writerow(["return_per_episode", "success_per_episode"])

        
        logs = algo.generate_trajectories(n_tests)

        return_per_episode = utils.synthesize(logs["return_per_episode"])
        success_rates = [1 if r == 100.0 else 0 for r in logs["return_per_episode"]]
        success_per_episode = utils.synthesize(success_rates)
        # num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        data = [*return_per_episode.values(), success_per_episode['mean'] * 100]

        logger.info(Fore.YELLOW + format_str.format(*data) + Fore.RESET)
        print('------------------------------------------')
        print('avg test score', return_per_episode['mean'])
        print('avg test success rate %', success_per_episode['mean'] * 100)
        
        for each_return_per_episode, success_rate in zip(logs["return_per_episode"], success_rates):
            csv_writer.writerow([each_return_per_episode, success_rate])



@hydra.main(config_path='config', config_name='config')
def main(config_args):
    
    
    custom_lamorel_module_functions = {
        'value': ValueHeadModuleFn(config_args.lamorel_args.llm_args.model_type)
    }
    if config_args.rl_script_args.use_action_heads:
        custom_lamorel_module_functions['policy_head'] = ActionHeadsModuleFn(
            config_args.lamorel_args.llm_args.model_type,
            len(config_args.rl_script_args.action_space)
        )
        lamorel_scoring_module_key = "policy_head"
    else:
        custom_lamorel_module_functions['score'] = LogScoringModuleFn(
            config_args.lamorel_args.llm_args.model_type
        )
        lamorel_scoring_module_key = "score"

    lm_server = Caller(config_args.lamorel_args, custom_updater=LoadSpecificWeightsUpdater(),
                       custom_module_functions=custom_lamorel_module_functions)
        
        
    log_path = os.path.join(config_args.rl_script_args.saving_path_logs, config_args.rl_script_args.id_expe)
    # create the folder for the tests results and return_per_episode
    test_path = os.path.join(log_path, 'test')
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    
    
    lm_server.update([None for _ in range(config_args.lamorel_args.distributed_setup_args.n_llm_processes)],
                    [[None] for _ in range(config_args.lamorel_args.distributed_setup_args.n_llm_processes)],
                    id_expe=config_args.rl_script_args.id_expe, saving_path_model=config_args.rl_script_args.saving_path_model)
    
    
    envs = []
    number_envs = config_args.rl_script_args.number_envs
    
    test_env = WebEnv(config_args.rl_script_args.environment_args, split='test')
    server = test_env.env.server
    for i in range(number_envs):
        env = WebEnv(config_args.rl_script_args.environment_args, split='test', server=server, id=f'test{i}_')
        envs.append(env)
    print('envs loaded')
    
    
    
    algo = LLMPPOAgentWebshop(envs = envs, lm_server = lm_server, llm_scoring_module_key=lamorel_scoring_module_key,
                              num_frames_per_proc=config_args.rl_script_args.number_episodes, test=True)
    
    
    run_agent(algo, config_args.rl_script_args.saving_path_logs, config_args.rl_script_args.id_expe,
              config_args.rl_script_args.number_episodes)
    
    if config_args.lamorel_args.distributed_setup_args.n_llm_processes > 0:
        lm_server.close()
    
    
if __name__ == "__main__":
    main()