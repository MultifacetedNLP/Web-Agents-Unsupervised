# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import json
import random
import torch
import os
import bitsandbytes as bnb
from transformers.trainer_pt_utils import get_parameter_names
from torch import nn

from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from transformers.utils.versions import require_version
from collections import deque
from tqdm import tqdm

from datasets import Dataset
import hydra

import numpy as np


require_version("datasets>=1.8.0",
                "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

IGNORE_INDEX = -100
PAD_TOKEN = 0


def generate_prompt_webshop_v2(goal, subgoals, deque_obs, deque_actions):
    ldo = len(deque_obs)
    lda = len(deque_actions)

    obs = ""
    for i in range(ldo):
        obs += f" \n Observation {i}: {deque_obs[i]}"
            
        if i < lda:
            obs += f"\n Action {i}: search for {deque_actions[i][7:-1]}" if deque_actions[i].startswith('search[') \
                else f"\n Action {i}: click on {deque_actions[i][6:-1]}"
        else:
            obs += f"\n Action {i}: search for " if subgoals[0].startswith('search[') \
                else f"\n Action {i}: click on "
    
                
    return goal + "," + obs


def remove_action_name(action):
    return action[7:-1] if action.startswith('search[') else action[6:-1]

def extract_instruction(state):
    instruction_start_id = state.find('Instruction:')
    instruction_end_id = state.find('\n[button]')
    return state[instruction_start_id:instruction_end_id].strip()

def process(state): # TODO: I may add special tokens like <clicked button>
    instruction_start_id = state.find('Instruction:')
    instruction_end_id = state.find('[button]')
    
    state = state[instruction_end_id:] if instruction_start_id == 0 else state[:instruction_start_id] + state[instruction_end_id:]
    
    return state.lower().replace('amazon shopping game\n', '').replace('webshop\n', '').strip()

def process_goal(state):
    state = state.lower().replace('"', '').replace("'", "")
    state = state.replace('amazon shopping game\ninstruction:', '').replace('webshop\ninstruction:', '')
    state = state.replace('\n[button] search [button_]', '').strip()
    if ', and price lower than' in state:
        state = state.split(', and price lower than')[0]
    return state

def process_str(s):
    s = s.lower().strip('.').replace('"', '').replace("'", "")
    return s

def create_instrunction_to_category_dict(default_file_path, items_human_ins):
    print('Creating instrunction to category dictionary...')
    with open(default_file_path) as f:
        products = json.load(f)
    asin_to_category = {}
    for product in tqdm(products):
        asin = product['asin']
        if asin == 'nan' or len(asin) > 10:
            continue
        asin_to_category[asin] = product['category']
        
    with open(items_human_ins) as f:
        items_human_ins_dict = json.load(f)
    instrunction_to_category = {}
    for _, value in items_human_ins_dict.items():
        for val in value:
            instrunction_to_category[process_str(val['instruction'])] = asin_to_category[val['asin']]
    print('Instrunction to category dictionary created.')
    return instrunction_to_category


def get_data(split, trajectories_path, human_goal_path, default_file_path, items_human_ins, nbr_obs, filter_search=False, category='all'):
    
    instrunction_to_category = {}
    if category != 'all':
        instrunction_to_category = create_instrunction_to_category_dict(default_file_path, items_human_ins)
    
    print('Loading data from {}'.format(trajectories_path))
    with open(trajectories_path, 'r') as json_file:
        json_list = list(json_file)

    human_goals = json.load(open(human_goal_path, 'r'))

    random.seed(233)
    random.shuffle(json_list)

    # split by human goal index
    goal_range = range(len(human_goals))
    if split == 'train':
        goal_range = range(1500, len(human_goals))
    elif split == 'eval':
        goal_range = range(500, 1500)
    elif split == 'test':
        goal_range = range(0, 500)

    cnt = 0
    observation_list = deque([], maxlen=nbr_obs)
    chosen_action_list = deque([], maxlen=nbr_obs-1)
    context_list, final_chosen_action_list = [], []
    num_trajs = 0
    for json_str in json_list:
        result = json.loads(json_str)
        instruction = extract_instruction(result['states'][0])
        goal = process_goal(result['states'][0])
        assert goal in human_goals, goal
        goal_idx = human_goals.index(goal)
        if goal_idx not in goal_range:
            continue
        if category != 'all' and instrunction_to_category.get(goal, "") != category:
            continue
        num_trajs += 1
        observation_list.clear()
        chosen_action_list.clear()
        for state, valid_acts, chosen_act in zip(result['states'], result['available_actions'], result['actions']): #  result['images']
            cnt += 1
            if filter_search and chosen_act.startswith('search['):
                continue
            
            observation_list.append(process(state))
            if len(valid_acts) == 0:
                valid_acts = [chosen_act]
            
            context_list.append(generate_prompt_webshop_v2(instruction, valid_acts, observation_list, chosen_action_list))
            chosen_action_list.append(chosen_act)
            
            chosen_act = remove_action_name(chosen_act)
            final_chosen_action_list.append(chosen_act)

        
    print('num of {} trajs: {}'.format(split, num_trajs))
    print('total transitions: {}'.format(cnt))
    if category == 'all':
        print('all categories are included')
    else:  
        print('category {} is only included'.format(category))
    return context_list, final_chosen_action_list


def truncate_sequence(sequence, truncate_size, save_token):
            
    if truncate_size > save_token:
        first_ids = sequence["input_ids"][:save_token]
        last_ids = sequence["input_ids"][-truncate_size+save_token:]
        ids = first_ids + last_ids
    else:
        ids = sequence["input_ids"][-truncate_size:]
        
    mask = sequence["attention_mask"][-truncate_size:]
        
    if truncate_size == 0:
        ids = []
        mask = []

    sequence["input_ids"] = ids
    sequence["attention_mask"] = mask
    return sequence


def pad_sequence(sequence, size):
    sequence_size = len(sequence["input_ids"])
    ids = sequence["input_ids"] + [
        PAD_TOKEN
        for _ in range(size - sequence_size)]
    mask = sequence["attention_mask"] + [0 for _ in range(size - sequence_size)]
    sequence["input_ids"] = ids
    sequence["attention_mask"] = mask
    return sequence
    
    
def truncate_obs_sequences(sequences, truncate_size):

    sequence = sequences[0] # instruction sequence
    instruction_len = len(sequence["input_ids"])
    num_observations = len(sequences) - 1
        
    allowed_sizes = {}
    observation_size = truncate_size - instruction_len
        
    obs_lengths = [(index, len(seq["input_ids"])) for index, seq in enumerate(sequences[1:])]
    obs_lengths.sort(key=lambda x:x[1])
    for index, obs_len in enumerate(obs_lengths):
        truncate_size_per_ob = observation_size // (num_observations - index)
        if obs_len[1] <= truncate_size_per_ob:
            allowed_sizes[obs_len[0]] = obs_len[1]
            observation_size -= obs_len[1]
        else:
            allowed_sizes[obs_len[0]] = truncate_size_per_ob
            observation_size -= truncate_size_per_ob
        
    for ob_index, seq in enumerate(sequences[1:]):
            
        if len(seq["input_ids"]) > allowed_sizes[ob_index]:
            truncated_sequence = truncate_sequence(seq, allowed_sizes[ob_index], save_token=4) # save_token=4 saves the observation

        else:
            truncated_sequence = seq
                
        sequence["input_ids"].extend(truncated_sequence["input_ids"])
        sequence["attention_mask"].extend(truncated_sequence["attention_mask"])
            
            
    return sequence
    
    
def split_sequence(sequence, observation_ids = [16018, 257, 10]):
    sequence_input_ids = np.array(sequence["input_ids"])
    sequence_attention_mask = np.array(sequence["attention_mask"])
        
    indices = np.where((sequence_input_ids[:-4] == observation_ids[0])
                           & (sequence_input_ids[1:-3] == observation_ids[1]) 
                           & (sequence_input_ids[4:] == observation_ids[2]))[0]
        
    sub_sequence_input_ids = np.split(sequence_input_ids, indices)
    sub_sequence_attention_mask = np.split(sequence_attention_mask, indices)
        
    sequences = []
    for sub_input_ids, sub_attention_mask in zip(sub_sequence_input_ids, sub_sequence_attention_mask):
        sequences.append({"input_ids": sub_input_ids.tolist(), "attention_mask":sub_attention_mask.tolist()})
        
    return sequences


def pad_or_truncate_sequence(sequence, size, max_size, encoder=False):
    sequence_size = len(sequence["input_ids"])
        
    if sequence_size > max_size:
        if not encoder:
            return truncate_sequence(sequence, max_size, save_token=1) # save_token=1 saves the pad token of the decoder's input
        else:
            sequences = split_sequence(sequence)
            return truncate_obs_sequences(sequences, max_size)
    else:
        return pad_sequence(sequence, min(size, max_size))


def get_dataset(split, trajectories_path, human_goal_path, default_file_path, items_human_ins, 
                nbr_obs, tokenizer, encoder_max_size, decoder_max_size, category='all'):
    
    input_text, output_text = get_data(split, trajectories_path, human_goal_path, default_file_path, 
                                       items_human_ins, nbr_obs=nbr_obs, category=category)
    
    tokenized_inputs = [tokenizer(input) for input in input_text]
    contexts_max_size = max([len(i['input_ids']) for i in tokenized_inputs])
    
    input_encodings = {'input_ids':[], 'attention_mask':[]}
    for tokenized_input in tokenized_inputs:
        result = pad_or_truncate_sequence(tokenized_input, contexts_max_size,
                        max_size=encoder_max_size, encoder=True)
        input_encodings['input_ids'].append(result['input_ids'])
        input_encodings['attention_mask'].append(result['attention_mask'])
        
    input_encodings['input_ids'] = torch.tensor(input_encodings['input_ids'])
    input_encodings['attention_mask'] = torch.tensor(input_encodings['attention_mask'])
    
    
    # input_encodings = tokenizer(
    #    input_text, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
    
    tokenizer.truncation_side='left'
    output_encodings = tokenizer(
        output_text, padding='max_length', max_length=decoder_max_size, truncation=True, return_tensors='pt')
    
    labels = output_encodings['input_ids']
    labels[labels == tokenizer.pad_token_id] = IGNORE_INDEX # replace padding token id's of the labels by -100 so it's ignored by the loss
    
    dataset = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': labels,
    }
    return Dataset.from_dict(dataset)


def data_collator(batch):
    input_ids, attention_mask, labels,  = [], [], []
    for sample in batch:
        input_ids.append(sample['input_ids'])
        attention_mask.append(sample['attention_mask'])
        labels.append(sample['labels'])
    max_encoder_len = max(sum(x) for x in attention_mask)
    max_decoder_len = max(sum([0 if item == IGNORE_INDEX else 1 for item in x]) for x in labels)
    return {
        'input_ids': torch.tensor(input_ids)[:, :max_encoder_len],
        'attention_mask': torch.tensor(attention_mask)[:, :max_encoder_len],
        'labels': torch.tensor(labels)[:, :max_decoder_len]
    }


def get_training_args(args, output_dir, logging_dir) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        metric_for_best_model="eval_loss",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        report_to="wandb",
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=args.tf32,
        # optim=args.optim,
        gradient_checkpointing = args.gradient_checkpointing,
        dataloader_drop_last=True,
        run_name=args.run_name,
        # dataloader_num_workers=cfg.num_workers,
        # sharded_ddp="simple",
    )


@hydra.main(config_path='configs', config_name='supervised_train_config')
def main(args):

    # Load pretrained model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path = args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path) # truncation_side='left'
    print(len(tokenizer))

    train_dataset = get_dataset("train", args.trajectories_file, args.human_goal_file, 
                                args.default_file_path, args.items_human_ins,
                                args.nbr_obs, tokenizer, args.encoder_max_size, 
                                args.decoder_max_size, args.category)
    
    eval_dataset = get_dataset("eval", args.trajectories_file, args.human_goal_file,
                               args.default_file_path, args.items_human_ins,
                               args.nbr_obs, tokenizer, args.encoder_max_size,
                               args.decoder_max_size)
    
    output_dir = os.path.join(args.output_dir, args.run_name + "_" + args.category)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    logging_dir = os.path.join(args.logging_dir, args.run_name + "_" + args.category)
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    
    training_args = get_training_args(args, output_dir, logging_dir)
    
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer_kwargs = {
        "betas": (training_args.adam_beta1, training_args.adam_beta2),
        "eps": training_args.adam_epsilon,
    }
    optimizer_kwargs["lr"] = training_args.learning_rate
    adam_bnb_optim = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        lr=training_args.learning_rate,
    )

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        optimizers=(adam_bnb_optim, None)
    )
    
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
