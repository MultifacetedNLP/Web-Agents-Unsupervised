# load WebEnV
import os
import torch
from web_agent_site import WebEnv
from agents.ppo.llm_ppo_agent_webshop import LLMPPOAgentWebshop
import hydra
from lamorel import Caller, lamorel_init
from lamorel import BaseUpdater
from train_language_agent import ValueHeadModuleFn, LogScoringModuleFn
import csv

from colorama import Fore

from accelerate import Accelerator
import logging
import numpy

lamorel_init()
logger = logging.getLogger(__name__)
accelerator = Accelerator()


class LoadSpecificWeightsUpdater(BaseUpdater):
    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        if not hasattr(self, "is_loaded"):
            try:
                self._llm_module.load_state_dict(torch.load(kwargs["saving_path_model"] +
                                                            "/" + kwargs["run_name"] + "/last/model.checkpoint", map_location='cuda:0'))
                self.is_loaded = True
                print("Last")
            except:
                try:
                    self._llm_module.load_state_dict(torch.load(kwargs["saving_path_model"] +
                                                                "/" + kwargs["run_name"] + "/backup/model.checkpoint"), map_location='cuda:0')
                    self.is_loaded = True
                    print("Backup")
                except:
                    print("No RL model loaded")


def run_agent(args, algo, saving_path_logs, run_name, n_tests):
    format_str = ("Reward: {: .2f} +- {: .2f}  (Min: {: .2f} Max: {: .2f}) |\
    Success Rate: {: .2f} |")


    test_path = os.path.join(os.path.join(saving_path_logs, run_name), 'test')
    
    csv_path = os.path.join(test_path, 'log.csv')
    first_created = not os.path.exists(csv_path) 
    csv_writer = csv.writer(open(csv_path, 'a', 1))
    if first_created:
        csv_writer.writerow(["return_per_episode", "success_per_episode"])

        
    logs = algo.generate_trajectories(n_tests, sample=args.rl_script_args.sample, deactivte_RL_for_search=args.rl_script_args.deactivte_RL_for_search,
                                      bart_path=args.rl_script_args.bart_path, generate_query=args.rl_script_args.generate_query)

    success_rates = [1 if r == 100.0 else 0 for r in logs["return_per_episode"]]

    average_score = f"avg test score: {numpy.mean(logs['return_per_episode'])}"
    average_test_success_rate = f"avg test success rate %: {numpy.mean(success_rates) * 100}"
    print('------------------------------------------')
    print(average_score)
    print(average_test_success_rate)
    
    
    txt_path = os.path.join(test_path, 'final.txt')
    with open(txt_path, 'w') as file:
        # Write content to the file
        file.write(average_score + "\n")
        file.write(average_test_success_rate)
        
    for each_return_per_episode, success_rate in zip(logs["return_per_episode"], success_rates):
        csv_writer.writerow([each_return_per_episode, success_rate])



@hydra.main(config_path='config', config_name='config')
def main(config_args):
    
    
    custom_lamorel_module_functions = {
        'value': ValueHeadModuleFn(config_args.lamorel_args.llm_args.model_type)
    }

    custom_lamorel_module_functions['score'] = LogScoringModuleFn(
        config_args.lamorel_args.llm_args.model_type
    )
    lamorel_scoring_module_key = "score"

    lm_server = Caller(config_args.lamorel_args, custom_updater=LoadSpecificWeightsUpdater(),
                       custom_module_functions=custom_lamorel_module_functions)
        
        
    log_path = os.path.join(config_args.rl_script_args.saving_path_logs, config_args.rl_script_args.run_name)
    # create the folder for the tests results and return_per_episode
    test_path = os.path.join(log_path, 'test')
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    
    
    lm_server.update([None for _ in range(config_args.lamorel_args.distributed_setup_args.n_llm_processes)],
                [[None] for _ in range(config_args.lamorel_args.distributed_setup_args.n_llm_processes)],
                run_name=config_args.rl_script_args.run_name, saving_path_model=config_args.rl_script_args.saving_path_model)
    
    
    envs = []
    number_envs = config_args.rl_script_args.number_envs
    
    test_env = WebEnv(config_args.rl_script_args.environment_args, split='test')
    server = test_env.env.server
    for i in range(number_envs):
        env = WebEnv(config_args.rl_script_args.environment_args, split='test', server=server, id=f'test{i}_')
        envs.append(env)
    print('envs loaded')
    
    
    
    algo = LLMPPOAgentWebshop(envs = envs, lm_server = lm_server, llm_scoring_module_key=lamorel_scoring_module_key,
                              num_frames_per_proc=config_args.rl_script_args.number_episodes,
                              nbr_obs=config_args.rl_script_args.nbr_obs, test=True)
    
    
    run_agent(config_args, algo, config_args.rl_script_args.saving_path_logs, config_args.rl_script_args.run_name,
              config_args.rl_script_args.number_episodes)
    
    if config_args.lamorel_args.distributed_setup_args.n_llm_processes > 0:
        lm_server.close()
    
    
if __name__ == "__main__":
    main()