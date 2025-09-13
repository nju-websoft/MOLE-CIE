import torch
import json
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM)
from llama_prompt import LlamaForCausalLM
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_task_data(task_idx, task):
    data_root = f'data/{task_idx}/{task}/'
    with open(data_root+'label.json', 'r') as f:
        labels = json.load(f)
    with open(data_root+'train.json', 'r') as f:
        origin_data = json.load(f)
    ins = origin_data['Instances']
    task_data = {}
    for label in labels:
        task_data[label] = []
    for one_ins in ins:
        inp = one_ins['input']
        la = one_ins['output']
        task_data[la].append(one_ins)
    return labels, task_data

def init_model():
    config = AutoConfig.from_pretrained('./pretrain_model/Llama-2-7b-chat-hf/')
    config.bos_token_id = 1
    config.eos_token_id = 2
    config.pad_token_id = 1
    tokenizer = AutoTokenizer.from_pretrained(
        './pretrain_model/Llama-2-7b-chat-hf/',
        use_fast = True)
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 1

    model = AutoModelForCausalLM.from_pretrained(
        './pretrain_model/Llama-2-7b-chat-hf/',
        config=config,
        trust_remote_code=True,
        revision='main',
        use_safetensors=True).to(device)
    
    model.resize_token_embeddings(len(tokenizer))

    model.generation_config.bos_token_id = 1
    model.generation_config.eos_token_id = 2
    model.generation_config.pad_token_id = 1

    for name, param in model.named_parameters():
        param.requires_grad = False

    return tokenizer, model

def select_data(tokenizer, model, task_label, task_data, memory_size, norm):
    replay_data = []
    for label in tqdm(task_label):
        data = task_data[label]
        features = []
        num_clusters = min(memory_size, len(data))
        if num_clusters == len(data):
            memory = []
            for i in data:
                memory.append(i)
        else:
            for one_ins in data:
                inp = one_ins['input']
                outp = one_ins['output']
                ins = inp + ' ' + outp
                tokens = tokenizer(ins, add_special_tokens=True, padding='max_length', truncation=True,
                                            max_length=512, return_tensors='pt')
                input_ids = tokens['input_ids']
                attention_mask = tokens['attention_mask']
                with torch.no_grad():
                    output = model(input_ids.to(device), attention_mask, output_hidden_states = True)
                    feature = output.hidden_states[-1].cpu()
                    if norm:
                        feature = feature.mean(dim=(0, 1), keepdim=True)
                    feature = feature.view((1,-1))
                    features.append(feature)
            features = np.concatenate(features)
            distances = KMeans(n_clusters = num_clusters, random_state = 0).fit_transform(features)
            memory = []
            for k in range(num_clusters):
                select_index = np.argmin(distances[:, k])
                ins = data[select_index]
                memory.append(ins)
        replay_data += memory
    return replay_data





                
                


def main():
    memory_size = 10
    task_order = {
        'EE': ['EE-0', 'EE-1', 'EE-2'],
        'NER': ['NER-0', 'NER-1', 'NER-2'],
        'RE': ['RE-0', 'RE-1', 'RE-2']}
    tokenizer, model = init_model()
    task_replay_data = {'EE':[[], [], []], 'NER':[[], [], []], 'RE':[[], [], []]}
    for task_idx, tasks in task_order.items():
        for t in tasks:
            print(f'Now is task {t}')
            _task = t.split('-')[0]
            _idx = int(t.split('-')[1])
            if '0' not in t:
                task_replay_data[_task][_idx] += task_replay_data[_task][_idx-1]
            replay_data = []
            task_label, task_data = read_task_data(task_idx, t)
            norm = True if 'NER' in t else False
            replay_data = select_data(tokenizer, model, task_label, task_data, memory_size, norm)
            task_replay_data[_task][_idx] += replay_data
    for task in task_replay_data:
        for idx, data in enumerate(task_replay_data[task]):
            file_root = f'replay_data/{task}-{idx}_{memory_size}'
            os.makedirs(file_root, exist_ok=True)
            file_root = f'replay_data/{task}-{idx}_{memory_size}/train.json'
            if 'RE' in task:
                dump_data = {
                    "Definition": ["Please provide the relationship type between these two entities in the sentence."],
                    "Positive Examples": [],
                    "Negative Examples": [],
                    "Instances": data,
                }
            elif 'NER' in task:
                dump_data = {
                    "Definition": ["Please determine the named entity type of the entity based on a sentence and the entity."],
                    "Positive Examples": [],
                    "Negative Examples": [],
                    "Instances": data,
                }
            elif 'EE' in task:
                dump_data = {
                    "Definition": [ "Please determine the type of event that appears in the sentence based on the trigger words."],
                    "Positive Examples": [],
                    "Negative Examples": [],
                    "Instances": data,
                }
            
            with open(file_root, 'w') as f:
                json.dump(dump_data, f, indent=4)
    



if __name__ == "__main__":
    main()