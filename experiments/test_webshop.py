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
from agents.drrn.drrn import DRRNAgent

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
                                                            "/" + kwargs["id_expe"] + "/last/model.checkpoint", map_location='cuda:0'))
                self.is_loaded = True
                print("Last")
            except:
                try:
                    self._llm_module.load_state_dict(torch.load(kwargs["saving_path_model"] +
                                                                "/" + kwargs["id_expe"] + "/backup/model.checkpoint"), map_location='cuda:0')
                    self.is_loaded = True
                    print("Backup")
                except:
                    print("No RL model loaded")

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def run_agent_drrn(args, envs):
    
    algo = DRRNAgent(envs, None, args.rl_script_args.spm_path,
                    max_steps=len(envs) * 4,
                    saving_path=args.rl_script_args.saving_path_model + "/" + args.rl_script_args.id_expe,
                    nbr_obs=args.rl_script_args.nbr_obs, test=True)
    algo.load()
    
    logs = algo.generate_trajectories(args.rl_script_args.number_episodes)

    success_rates = [1 if r == 100.0 else 0 for r in logs["return_per_episode"]]
    fail_rate = [1 if r == 0.0 else 0 for r in logs["return_per_episode"]]

    average_score = f"avg test score: {numpy.mean(logs['return_per_episode'])}"
    std_score = f"std of test score: {numpy.std(logs['return_per_episode'])}"
    average_success_rate = f"avg test success rate %: {numpy.mean(success_rates) * 100}"
    average_fail_rate = f"avg test fail rate %: {numpy.mean(fail_rate) * 100}"
    print('------------------------------------------')
    print(average_score)
    print(std_score)
    print(average_success_rate)
    print(average_fail_rate)
    
    test_path = os.path.join(os.path.join(args.rl_script_args.saving_path_logs, args.rl_script_args.id_expe), 'test')
    
    name = f"number_episodes_{args.rl_script_args.number_episodes}"
    
    csv_path = os.path.join(test_path, f'log_{name}.csv')
    csv_path = uniquify(csv_path)
    csv_writer = csv.writer(open(csv_path, 'a', 1))
    csv_writer.writerow(["return_per_episode", "success_per_episode"])
    
    for each_return_per_episode, success_rate in zip(logs["return_per_episode"], success_rates):
        csv_writer.writerow([each_return_per_episode, success_rate])
    
    
    txt_path = os.path.join(test_path, f'final_{name}.txt')
    txt_path = uniquify(txt_path)
    with open(txt_path, 'w') as file:
        # Write content to the file
        file.write(average_score + "\n")
        file.write(std_score + "\n")
        file.write(average_success_rate + "\n")
        file.write(average_fail_rate)


def run_agent(args, envs, lm_server, lamorel_scoring_module_key):
    
    algo = LLMPPOAgentWebshop(envs = envs, lm_server = lm_server, llm_scoring_module_key=lamorel_scoring_module_key,
                              num_frames_per_proc=args.rl_script_args.number_episodes,
                              nbr_obs=args.rl_script_args.nbr_obs, test=True)
    
    
    logs = algo.generate_trajectories(args.rl_script_args.number_episodes, epsilon=args.rl_script_args.epsilon,
                                      top_k=args.rl_script_args.top_k, top_p=args.rl_script_args.top_p,
                                      generate_query=args.rl_script_args.generate_query)

    success_rates = [1 if r == 100.0 else 0 for r in logs["return_per_episode"]]
    fail_rate = [1 if r == 0.0 else 0 for r in logs["return_per_episode"]]

    average_score = f"avg test score: {numpy.mean(logs['return_per_episode'])}"
    std_score = f"std of test score: {numpy.std(logs['return_per_episode'])}"
    average_success_rate = f"avg test success rate %: {numpy.mean(success_rates) * 100}"
    average_fail_rate = f"avg test fail rate %: {numpy.mean(fail_rate) * 100}"
    print('------------------------------------------')
    print(average_score)
    print(std_score)
    print(average_success_rate)
    print(average_fail_rate)
    
    
    test_path = os.path.join(os.path.join(args.rl_script_args.saving_path_logs, args.rl_script_args.id_expe), 'test')
    
    name = f"epsilon_{args.rl_script_args.epsilon}_top_k_" + \
    f"{args.rl_script_args.top_k}_top_p_{args.rl_script_args.top_p}_generate_query_{args.rl_script_args.generate_query}_number_episodes_" + \
    f"{args.rl_script_args.number_episodes}"
    
    csv_path = os.path.join(test_path, f'log_{name}.csv')
    csv_path = uniquify(csv_path)
    csv_writer = csv.writer(open(csv_path, 'a', 1))
    csv_writer.writerow(["return_per_episode", "success_per_episode"])
    
    for each_return_per_episode, success_rate in zip(logs["return_per_episode"], success_rates):
        csv_writer.writerow([each_return_per_episode, success_rate])
    
    
    txt_path = os.path.join(test_path, f'final_{name}.txt')
    txt_path = uniquify(txt_path)
    with open(txt_path, 'w') as file:
        # Write content to the file
        file.write(average_score + "\n")
        file.write(std_score + "\n")
        file.write(average_success_rate + "\n")
        file.write(average_fail_rate)



@hydra.main(config_path='config', config_name='config')
def main(config_args):
    
    if config_args.lamorel_args.distributed_setup_args.n_llm_processes > 0:
        custom_lamorel_module_functions = {
            'value': ValueHeadModuleFn(config_args.lamorel_args.llm_args.model_type)
        }

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
        
    envs = []
    number_envs = config_args.rl_script_args.number_envs
        
    test_env = WebEnv(config_args.rl_script_args.environment_args, split='test')
    server = test_env.env.server
    for i in range(number_envs):
        env = WebEnv(config_args.rl_script_args.environment_args, split='test', server=server, id=f'test{i}_')
        envs.append(env)
        
    if config_args.rl_script_args.environment_args.category:
        config_args.rl_script_args.number_episodes = 4 * len(envs[0].goal_idxs)
    print('envs loaded')
    
    # Flan-T5 model
    if config_args.lamorel_args.distributed_setup_args.n_llm_processes > 0:
        
        lm_server.update([None for _ in range(config_args.lamorel_args.distributed_setup_args.n_llm_processes)],
                    [[None] for _ in range(config_args.lamorel_args.distributed_setup_args.n_llm_processes)],
                    id_expe=config_args.rl_script_args.id_expe, saving_path_model=config_args.rl_script_args.saving_path_model)
        
        
        run_agent(config_args, envs, lm_server,  lamorel_scoring_module_key)
        
        torch.cuda.empty_cache()
        config_args.rl_script_args.epsilon = 2  # sample
        
        run_agent(config_args, envs, lm_server,  lamorel_scoring_module_key)
        
        torch.cuda.empty_cache()
        config_args.rl_script_args.top_p = 0.80  # top_p
        
        run_agent(config_args, envs, lm_server,  lamorel_scoring_module_key)
        
        torch.cuda.empty_cache()
        config_args.rl_script_args.top_p = 0.00  # top_p
        config_args.rl_script_args.epsilon = -2  # argmax
        config_args.rl_script_args.number_episodes = len(envs[0].goal_idxs)
        for env in envs: # decrease the step limit for all evironments
            env.step_limit = 100
        
        run_agent(config_args, envs, lm_server,  lamorel_scoring_module_key)
        
        
        lm_server.close() 
    else:
        run_agent_drrn(config_args, envs)
    
    
if __name__ == "__main__":
    main()