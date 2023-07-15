from agents.base_agent import BaseAgent
from web_agent_site import WebEnv
from collections import deque
from dotmap import DotMap
from tqdm import tqdm, trange
import random
import matplotlib.pyplot as plt
import pandas as pd
from os.path import dirname, abspath, join
from transformers import AutoTokenizer

BASE_DIR = dirname(abspath(__file__))

DEFAULT_PDF_PATH = join(BASE_DIR, 'length_prompts_analysis')

class PromptLength:
    
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.web_env = WebEnv(cfg, split='train', id='train_')
        self.obs_queue = deque([], maxlen=cfg.nbr_obs)
        self.acts_queue = deque([], maxlen=cfg.nbr_obs - 1)
        
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_type)
        # self.tokenizer.add_tokens(['[button], [button_], [clicked button], [clicked button_]'], special_tokens=True)
        
        self.prompts = {"length":[], "page_type":[]}
        
    def add_prompt(self, action, page_type):
        self.acts_queue.append(action)
        observation, _, _, info = self.web_env.step(action)
        self.obs_queue.append(observation)
        prompt = BaseAgent.generate_prompt_webshop(self, goal=info["goal"], subgoals=None, deque_obs=self.obs_queue, deque_actions=self.acts_queue)
        
        self.prompts["length"].append(len(self.tokenizer.encode(prompt)))
        self.prompts["page_type"].append(page_type)
        
        return info["valid"]
        
    def visit_product(self, action):
        
        self.add_prompt(action, "item_page")
        
        self.add_prompt("click[description]", "item_sub_page_description")
        
        self.add_prompt("click[< prev]", "item_page")
        
        self.add_prompt("click[features]", "item_sub_page_features")
        
        self.add_prompt("click[< prev]", "item_page")
        
        self.add_prompt("click[< prev]", "search_results")
        
        
    def visit_products(self, actions, visit_page):
        
        for page in tqdm(range(visit_page)):
        
            if page !=0:
                actions = self.add_prompt("click[next >]", "search_results")
            
            for action in actions:
                if action.startswith("click[item"):
                    self.visit_product(action)
        
        
    def collect_prompts_goal_based(self):
        
        for idx, goal_idx in enumerate(tqdm(self.web_env.goal_idxs)):
            
            self.obs_queue.clear()
            self.acts_queue.clear()
            
            observation, info = self.web_env.reset(idx = goal_idx)
            self.obs_queue.append(observation)
    
            prompt = BaseAgent.generate_prompt_webshop(self, goal=info["goal"], subgoals=None, deque_obs=self.obs_queue, deque_actions=self.acts_queue)
            
            self.prompts["length"].append(len(self.tokenizer.encode(prompt)))
            self.prompts["page_type"].append("index")
        
            actions = self.add_prompt(random.choice(info["valid"]), "search_results")
            
            visit_page = self.cfg.visit_page
            self.visit_products(actions, visit_page)
            
            if idx % 500 == 0 and idx != 0:
                df = pd.DataFrame(self.prompts)
                plot_frequency_histogram(df["length"].values.tolist(), f"Frequency of all kinds of prompts - {idx}")
                plot_frequency_histogram(df[df["page_type"] == "item_page"]["length"].values.tolist(), f"Frequency of item_page prompts - {idx}")
                plot_frequency_histogram(df[df["page_type"] == "index"]["length"].values.tolist(), f"Frequency of index prompts - {idx}")
                plot_frequency_histogram(df[df["page_type"] == "item_sub_page_description"]["length"].values.tolist(), f"Frequency of sub_page_description prompts - {idx}")
                plot_frequency_histogram(df[df["page_type"] == "item_sub_page_features"]["length"].values.tolist(), f"Frequency of sub_page_features prompts - {idx}")
                plot_frequency_histogram(df[df["page_type"] == "search_results"]["length"].values.tolist(), f"Frequency of search_results prompts - {idx}")
        
        print("done")
    

def plot_frequency_histogram(numbers, title):
    plt.hist(numbers, bins='auto', alpha=0.7, rwidth=0.85)
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"{DEFAULT_PDF_PATH}/{title}.pdf", format='pdf')
    plt.clf()


if __name__ == "__main__":
    PL = PromptLength(
        DotMap(
            num=None,
            get_image = 1,
            state_format='text_rich',
            human_goals=1,
            step_limit=None,
            click_item_name=1,
            num_prev_actions=0,
            num_prev_obs=0,
            extra_search_path="/u/spa-d2/grad/mfe261/Projects/Grounding_LLMs_with_online_RL/web_agent_site/data/goal_query_predict.json",
            harsh_reward=0,
            go_to_search=0,
            go_to_item=0,
            ban_buy=0,
            button_version=0,
            nbr_obs=1,
            visit_page=2,
            tokenizer_type="google/flan-t5-base"
        )
    )
    
    PL.collect_prompts_goal_based()
    df = pd.DataFrame(PL.prompts)
    plot_frequency_histogram(df["length"].values.tolist(), "Frequency of all kinds of prompts - all")
    plot_frequency_histogram(df[df["page_type"] == "item_page"]["length"].values.tolist(), "Frequency of item_page prompts - all")
    plot_frequency_histogram(df[df["page_type"] == "index"]["length"].values.tolist(), "Frequency of index prompts - all")
    plot_frequency_histogram(df[df["page_type"] == "item_sub_page_description"]["length"].values.tolist(), "Frequency of sub_page_description prompts - all")
    plot_frequency_histogram(df[df["page_type"] == "item_sub_page_features"]["length"].values.tolist(), "Frequency of sub_page_features prompts - all")
    plot_frequency_histogram(df[df["page_type"] == "search_results"]["length"].values.tolist(), "Frequency of search_results prompts - all")
    
    