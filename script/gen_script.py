import json
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False,indent=2)

def load_jsonline(path):
    with open(path, 'r', encoding='utf-8') as f:
        result=[]
        for line_s in f:
            line=json.loads(line_s)
            result.append(line)
    return result

def write_jsonline(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            line_s=json.dumps(line, ensure_ascii=False)
            f.write(line_s)
            f.write('\n')

order_seed = 66

task_seq=0

seqs = [["EE-0","RE-0","NER-0","EE-1","RE-1","NER-1","EE-2","RE-2","NER-2","EE-3","RE-3","NER-3",],
["EE-0","EE-1","EE-2","EE-3","RE-0","RE-1","RE-2","RE-3","NER-0","NER-1","NER-2","NER-3",]]
kls = [[0, 1.755635097223566, 1.4320546981046705, 2.7247944125267547, 6.5300681346894445, 3.659165067178503, 4.873368801445493, 11.537119195504289, 4.037208711664233, 8.720567219962625, 17.200544641120405, 8.659787135585377],
    [0, 1.0221921820434832, 1.8019474001204578, 3.0683741892931735, 7.1606849182548125, 8.329910141206675, 9.315192743764172, 10.17574401867341, 5.776934291890805, 6.468330134357005, 4.9412661121341035, 8.659787135585377]]

kl_ratio = kls[task_seq]
all_tasks= seqs[task_seq]

dataset_list = all_tasks
task_order = ','.join(all_tasks)

config_template={'EE':{"EE": [],},
    'RE':{"RE": [],},
    'NER':{"NER": [],}}

import os
import pathlib
import numpy as np
from copy import deepcopy

lora_r = 4
lora_alpha = 32
lora_dropout = 0.
learning_rate = 5e-5
num_train_epochs = 2
attn_lr = 0.
replay_after_n_epoch = 0
llama_path = 'YOUR_PRETRAIN_MODEL_PATH'
experts_num = 12
experts_pool_num = 8
task_experts_num = 1
fixed_experts_num = 1
select_experts_num = 2
task_num = 3
memory_size = 10


run_name = f"MoLE-CIE"
history_config={"EE":[], "RE":[], "NER":[]}
for one_data_name in dataset_list:

    pathlib.Path(f'./configs/{run_name}_configs/{one_data_name}').mkdir(parents=True, exist_ok=True)

    config={
        "sampling strategy": "full",
        "dataset name": f"{one_data_name}"
    } 
    data_type = one_data_name.split('-')[0]
    history_config[data_type].append(config)
    
    
    dev_config=deepcopy(config_template[data_type])
    dev_config[data_type].append(config)
    write_json(f'./configs/{run_name}_configs/{one_data_name}/dev_tasks.json', dev_config)
    
    train_config=deepcopy(config_template[data_type])
    train_config[data_type].append(config)
    write_json(f'./configs/{run_name}_configs/{one_data_name}/train_tasks.json', train_config)

    test_config=deepcopy(config_template[data_type])
    test_config=history_config
    write_json(f'./configs/{run_name}_configs/{one_data_name}/test_tasks.json', test_config)


start_str = rf'''#!/bin/bash
#SBATCH -J cl                           
#SBATCH -o cl-%j.out                       
#SBATCH -p compute 
#SBATCH -N 1

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

port=$(shuf -i25000-30000 -n1)  

deepspeed src/main.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path {llama_path} \
   --data_dir data \
   --task_order {task_order} \
   --task_config_dir configs/{run_name}_configs/{dataset_list[0]} \
   --output_dir logs_and_outputs/{run_name}/outputs/1-{dataset_list[0]} \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 4 \
   --learning_rate {learning_rate} \
   --num_train_epochs {num_train_epochs} \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name {run_name} \
   --max_source_length 1024 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match \
   --evaluation_strategy epoch \
   --save_strategy epoch \
   --save_total_limit 1 \
   --lora_r {lora_r} \
   --lora_alpha {lora_alpha} \
   --lora_dropout {lora_dropout} \
   --experts_num {experts_num} \
   --experts_pool_num {experts_pool_num} \
   --task_experts_num {task_experts_num} \
   --fixed_experts_num {fixed_experts_num} \
   --select_experts_num {select_experts_num} \
   --task_num {task_num} \
   --new_task True \
   --memory_size {memory_size} \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 0 

rm -rf logs_and_outputs/{run_name}/outputs/1-{dataset_list[0]}/checkpoint*
   
sleep 5
'''

def epoch_str(idx, previous_lora_path, new_task, data_replay_freq, klr):
    return rf'''

deepspeed src/main.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path {llama_path} \
   --previous_lora_path {previous_lora_path} \
   --previous_lora_router_path {previous_lora_path} \
   --data_dir data \
   --task_order {task_order} \
   --gen_data_dir generated_data/lora_gen_long_llama \
   --task_config_dir configs/{run_name}_configs/{dataset_list[idx+1]} \
   --output_dir logs_and_outputs/{run_name}/outputs/{idx+2}-{dataset_list[idx+1]} \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 4 \
   --learning_rate {learning_rate} \
   --num_train_epochs {num_train_epochs} \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name {run_name} \
   --max_source_length 1024 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_{dataset_list[idx+1]} \
   --evaluation_strategy epoch \
   --save_strategy epoch \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r {lora_r} \
   --lora_alpha {lora_alpha} \
   --lora_dropout {lora_dropout} \
   --experts_num {experts_num} \
   --experts_pool_num {experts_pool_num} \
   --task_experts_num {task_experts_num} \
   --fixed_experts_num {fixed_experts_num} \
   --select_experts_num {select_experts_num} \
   --task_num {task_num} \
   --new_task {new_task} \
   --memory_size {memory_size} \
   --data_replay_freq {data_replay_freq} \
   --replay_after_n_epoch {replay_after_n_epoch} \
   --kl_ratio {klr} 

rm -rf logs_and_outputs/{run_name}/outputs/{idx+2}-{dataset_list[idx+1]}/checkpoint*

sleep 5
'''


sh_str=start_str
for idx in range(len(dataset_list)-1):
    previous_lora_path = f"logs_and_outputs/{run_name}/outputs/{idx+1}-{dataset_list[idx]}/saved_weights"
    new_task = False
    data_replay_freq = 1 
    klr = kl_ratio[idx+1]
    sh_str+=epoch_str(idx, previous_lora_path, new_task, data_replay_freq, klr)

    
with open(f'{run_name}.sh', 'w') as f:
    f.write(sh_str)