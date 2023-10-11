# load WebEnV
import os
from web_agent_site import WebEnv
import hydra
import csv

from agents.random_agent.random_agent import Random_agent
import logging
import numpy

logger = logging.getLogger(__name__)


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def run_agent(args, envs):
    
    
    algo = Random_agent(envs=envs)
    
    
    logs = algo.generate_trajectories(args.number_episodes)

    success_rates = [1 if r == 100.0 else 0 for r in logs["return_per_episode"]]
    fail_rate = [1 if r == 0.0 else 0 for r in logs["return_per_episode"]]

    average_score = f"avg test score: {numpy.mean(logs['return_per_episode'])}"
    std_score = f"std of test score: {numpy.std(logs['return_per_episode'])}"
    average_success_rate = f"avg test success rate %: {numpy.mean(success_rates) * 100}"
    average_fail_rate = f"avg test fail rate %: {numpy.mean(fail_rate) * 100}"
    number_episodes_done = f"nuumber of done episodes: {logs['episodes_done']}"
    print('------------------------------------------')
    print(average_score)
    print(std_score)
    print(average_success_rate)
    print(average_fail_rate)
    print(number_episodes_done)
    
    
    test_path = os.path.join(os.path.join(args.saving_path_logs, args.id_expe), 'test')
    
    name = f"random_number_episodes_{args.number_episodes}"
    
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
    
    
        
    log_path = os.path.join(config_args.saving_path_logs, config_args.id_expe)
    # create the folder for the tests results and return_per_episode
    test_path = os.path.join(log_path, 'test')
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    
    
    envs = []
    number_envs = config_args.number_envs
    
    test_env = WebEnv(config_args.environment_args, split='test')
    server = test_env.env.server
    for i in range(number_envs):
        env = WebEnv(config_args.environment_args, split='test', server=server, id=f'test{i}_')
        envs.append(env)
    print('envs loaded')
    
    run_agent(config_args, envs)
    
    
if __name__ == "__main__":
    main()