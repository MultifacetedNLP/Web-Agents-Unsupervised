import json

import torch
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import hydra
from datasets import Dataset
import os


# Instruction: \nlarge size white  colored 1/4 zip golf shirt long sleeve athletic pullover with brushed fleece lining, and price lower than 30.00 dollars, \n Observation 0: [button] search [button_]\n Action 0: search for 
def generate_prompt_webshop_v2(goal):

    obs = " \n Observation 0: [button] Search [button_]\n Action 0: search for "
    
    return goal + "," + obs


def get_data(goal_path):
    all_data = json.load(open(goal_path))
    all_goals = []
    all_prompts = []
    for ins_list in all_data.values():
        for ins in ins_list:
            ins = ins['instruction']
            all_goals.append(ins)
            all_prompts.append(generate_prompt_webshop_v2(ins))
    return all_prompts, all_goals

def get_dataset(input, tokenizer, encoder_max_size):
    input_encodings = tokenizer(input, padding=True,
                                max_length=encoder_max_size, truncation=True, return_tensors='pt')
    dataset = Dataset.from_dict({
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
    })
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset


@hydra.main(config_path='configs', config_name='generate_search_config')
def main(args):
    # Load pretrained model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path = args.model_path, cache_dir=args.cache_dir)
    model.eval()
    model = model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path) # truncation_side='left'
    
    all_prompts, all_goals = get_data(args.goal_path)
    
    dataset = get_dataset(all_prompts, tokenizer, args.encoder_max_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    all_queries = []
    for batch in tqdm(dataloader):
        output = model.generate(
            input_ids=batch["input_ids"].to('cuda'),
            attention_mask=batch["attention_mask"].to('cuda'),
            num_beams=args.num_beams, num_return_sequences=args.num_return_sequences,
            max_length=args.max_length, early_stopping=True
        )
        queries = tokenizer.batch_decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        assert len(queries) % args.num_return_sequences == 0
        
        for i in range(len(queries) // args.num_return_sequences):
            all_queries.append(queries[i*args.num_return_sequences : (i+1)*args.num_return_sequences])
            
    assert len(all_goals) == len(all_queries)
    
    data = {goal: queries for goal, queries in zip(all_goals, all_queries)}
    
    output_dir = os.path.join(args.output_dir, args.output_name)
    with open(output_dir, 'w') as f:
        json.dump(data, f, indent = 6)

if __name__ == "__main__":
    main()