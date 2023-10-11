import numpy as np
from tqdm import tqdm
import logging
from agents.base_agent import BaseAgent

class Random_agent(BaseAgent):
    def __init__(self, envs):
        super().__init__(envs)
        self.subgoals= []
        logging.info("resetting environment")
        for i, env in enumerate(self.env):
            _, info = env.reset(i)
            self.subgoals.append(info['valid'])
        logging.info("reset environment")
        self.returns = [0 for _ in range(len(self.env))]
        self.logs = {
            "return_per_episode": [],
        }

    def generate_trajectories(self, n_tests):
        episodes_done = 0
        reset_index = len(self.env)
        remove_indexes = []
        pbar = tqdm(range(n_tests), ascii=" " * 9 + ">", ncols=100)
        while episodes_done < n_tests:
            
            for i, subgoal in enumerate(self.subgoals):
                action = np.random.randint(low=0, high=len(subgoal))
                
                _, reward, done, info = self.env[i].step(subgoal[int(action)])
                
                self.subgoals[i] = info['valid']
                self.returns[i] += reward
                if done:
                    logging.info(f"reward of '{self.env[i].session['goal']['instruction_text']}' is {reward} ")
                    _, info = self.env[i].reset(reset_index)
                    self.subgoals[i] = info['valid']
                    reset_index += 1
                    episodes_done += 1
                    if len(self.env[0].goal_idxs) == reset_index:
                        reset_index = 0
                    if episodes_done > n_tests:
                        remove_indexes.append(i)
                    pbar.update(1)
                    self.logs["return_per_episode"].append(self.returns[i] * 10)
                    self.returns[i] = 0
                    
            # removing environments
            if len(remove_indexes) > 0:
                for index in sorted(remove_indexes, reverse=True):
                    del self.env[index]
                    del self.subgoals[index]
                    del self.returns[index]
                    
        pbar.close()

        self.logs["episodes_done"] = episodes_done
        return self.logs

    def update_parameters(self):
        pass
