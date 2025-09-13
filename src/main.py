import logging
import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Dict
import math
import torch
from torch import nn

import datasets
import numpy as np
import pickle
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed, )
from transformers.trainer_utils import get_last_checkpoint

from dataloader import DataCollator
from llama_prompt import LlamaForCausalLM, LlamaLoraRouter
from assets import lora_state_dict_A, lora_state_dict_B, router_network_dict, task_key_dict, task_config
from trainer import Trainer, DenserEvalCallback, skip_instructions
from compute_metrics import compute_metrics, compute_grouped_metrics

CURRENT_DIR = os.path.dirname(__file__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                    "the model's position embeddings."
        },
    )
    lora_dim: Optional[int] = field(
        default=8,
        metadata={
            "help": "Intrinsic dimension of the latent space."
        },
    )
    lora_r: Optional[int] = field(
        default=8,
        metadata={
            "help": "Temperature to control attention weights."
        },
    )
    lora_alpha: Optional[int] = field(
        default=1,
        metadata={
            "help": "Temperature to control attention weights."
        },
    )
    lora_dropout: Optional[float] = field(
        default=0.,
        metadata={
            "help": "Temperature to control attention weights."
        },
    )
    experts_num: Optional[int] = field(
        default=12,
        metadata={
            "help": "All_experts_num."
        },
    )
    experts_pool_num: Optional[int] = field(
        default=8,
        metadata={
            "help": "LoRA_experts_num."
        },
    )
    task_experts_num: Optional[int] = field(
        default=1,
        metadata={
            "help": "allocate how many experts for one task."
        },
    )
    select_experts_num: Optional[int] = field(
        default=2,
        metadata={
            "help": "Lora_select_experts_num."
        },
    )
    fixed_experts_num: Optional[int] = field(
        default=1,
        metadata={
            "help": "Lora_fixed_experts_num."
        },
    )
    task_num: Optional[int] = field(
        default=3,
        metadata={
            "help": "task num."
        },
    )
    new_task: bool = field(
        default=False,
        metadata={
            "help": "True if task is new."
        },
    )
    memory_size: Optional[int] = field(
        default=10,
    )
    previous_lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "the path to load previous prompts."}
    )
    previous_lora_router_path: Optional[str] = field(    
        default=None,
        metadata={"help": "the path to load previous lora router."}
    )
    load_checkpoint_from: str = field(
        default=None,
        metadata={"help": "Path to load previous checkpoints"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the CL_Benchmark train/dev/test splits."}
    )
    gen_data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the generated train/dev/test splits."}
    )
    task_order: str = field(
        default=None, metadata={"help": "order of the tasks"}
    )
    task_config_dir: str = field(
        default=None, metadata={"help": "The json file for config training and testing tasks"}
    )
    replay_task_list: Optional[str] = field(
        default='', metadata={
            "help": "Different tasks to replay"
        }
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    input_record_file: str = field(
        default=None, metadata={"help": "file to record model input"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    # for decoder model, it means max_new_tokens
    max_target_length: Optional[int] = field(
        default=50,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Penalty for repeat tokens in decode stage."
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    max_num_instances_per_task: int = field(
        default=10000, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=200,
        metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_dataset_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend dataset name before the task input."}
    )
    add_instruction_replay: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to preappend definition and few-shot cases before the task input during replay."}
    )

@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use computing time to gain more memory"}
    )
    denser_evaluation: Optional[bool] = field(
        default=False,
        metadata={"help": "If specifid, the model will do more evaluation at the beginning of training."}
    )
    do_demo: bool = field(default=False, metadata={"help": "Whether to run the model as a demo in the terminal."})
    kl_ratio: Optional[float] = field(
        default=0.5,
        metadata={"help": "ratio of the replay kl loss"}
    )
    data_replay_freq: Optional[int] = field(
        default=-1,
        metadata={"help": "replay frequency"}
    )
    replay_after_n_epoch: Optional[int] = field(
        default=0,
        metadata={"help": "replay after n epoch"}
    )
    attn_lr: Optional[float] = field(
        default=0,
        metadata={"help": "learning rate of the attention module"}
    )
    eval_every_n_epoch: Optional[int] = field(
        default=5,
        metadata={"help": "replay frequency"}
    )

def cal_total_para(model):
    total_params, params = 0, 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)
            total_params += p.numel()
        params += p.numel()

    print(
        "Total number of parameters: {}M, rate: {}%".format(
            total_params // 1000 / 1000, round(total_params / params * 100, 2)
        )
    )

def init_model(model_args, data_args, training_args):
    # Set seed before initializing model.
    set_seed(training_args.seed)
    task_order = data_args.task_order.split(',')
    cur_task = data_args.task_config_dir.split('/')[-1]
    cur_task_id = int(task_order.index(cur_task))

    replay_data_path = []
    for task_id in range(0, cur_task_id):
        replay_data_task = task_order[task_id]
        replay_data_path.append(f'./replay_data/{replay_data_task}_10/train_tasks.json')

    # Get the CIE dataset
    raw_datasets = load_dataset(
        os.path.join(CURRENT_DIR, "cie_dataset.py"),
        data_dir=data_args.data_dir,
        replay_data_dir = replay_data_path,
        task_config_dir=data_args.task_config_dir,
        max_num_instances_per_task=data_args.max_num_instances_per_task,
        max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
        num_examples=data_args.num_examples,
    )
    raw_datasets.cleanup_cache_files()
    print(raw_datasets)

    # Load the LLM model 
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # config.bos_token_id = 1
    # config.eos_token_id = 2
    config.pad_token_id = config.bos_token_id
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir = model_args.cache_dir,
        use_fast = model_args.use_fast_tokenizer,
        revision = model_args.model_revision,
        use_auth_token = True if model_args.use_auth_token else None,
    )
    # tokenizer.bos_token_id = 1
    # tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = tokenizer.bos_token_id

    prompt_config = {
        'seq_len': data_args.max_source_length,
        'previous_lora_path': model_args.previous_lora_path,
        'task_id': cur_task_id,
        'lora_r': model_args.lora_r,
        'lora_alpha': model_args.lora_alpha,
        'lora_dropout': model_args.lora_dropout,
        'experts_num': model_args.experts_num,
        'experts_pool_num': model_args.experts_pool_num,
        'task_experts_num': model_args.task_experts_num,
        'fixed_experts_num': model_args.fixed_experts_num,
        'select_experts_num': model_args.select_experts_num,
        'task_num': model_args.task_num,
        'new_task': model_args.new_task, # True if this task is a new task
    }

    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        prompt_config,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        use_safetensors=True,
    )
    
    model.resize_token_embeddings(len(tokenizer))

    if 'llama' in model_args.model_name_or_path.lower():
        model.generation_config.bos_token_id = config.bos_token_id
        model.generation_config.eos_token_id = config.eos_token_id
        model.generation_config.pad_token_id = config.pad_token_id

    if model_args.previous_lora_path:  
        path = model_args.previous_lora_path
        print("----------Loading Previous LoRA Weights----------")
        lora_A = torch.load(os.path.join(path, "lora_experts_A.pt"))
        lora_B = torch.load(os.path.join(path, "lora_experts_B.pt"))
        ## Loading LoRA weights for LLaMA
        for j in range(config.num_hidden_layers):
            for k in range(model_args.experts_num):
                model.model.layers[j].self_attn.lora_experts_q[k].lora_A.data.copy_(
                    lora_A[f"model.layers.{j}.self_attn.lora_experts_q.{k}.lora_A"]
                )
                model.model.layers[j].self_attn.lora_experts_q[k].lora_B.data.copy_(
                    lora_B[f"model.layers.{j}.self_attn.lora_experts_q.{k}.lora_B"]
                )
                model.model.layers[j].self_attn.lora_experts_v[k].lora_A.data.copy_(
                    lora_A[f"model.layers.{j}.self_attn.lora_experts_v.{k}.lora_A"]
                )
                model.model.layers[j].self_attn.lora_experts_v[k].lora_B.data.copy_(
                    lora_B[f"model.layers.{j}.self_attn.lora_experts_v.{k}.lora_B"]
                )
    # Load LoRA Router
    if model_args.previous_lora_router_path:
        with torch.no_grad():
            path = model_args.previous_lora_router_path
            print("----------Loading Previous LoRA Router Weights----------")
            lora_router = torch.load(os.path.join(path, "router_network.pt"))
            for j in range(config.num_hidden_layers):
                model.model.layers[j].self_attn.lora_router.router_network[0].weight.copy_(
                    lora_router[f"model.layers.{j}.self_attn.lora_router.router_network.0.weight"]
                )
                model.model.layers[j].self_attn.lora_router.router_network[2].weight.copy_(
                    lora_router[f"model.layers.{j}.self_attn.lora_router.router_network.2.weight"]
                )
                model.model.layers[j].self_attn.lora_router.task_keys.copy_(
                    lora_router[f"model.layers.{j}.self_attn.lora_router.task_keys"]
                )

    # Frozen parameters
    task_id = cur_task_id%3
    for name, param in model.named_parameters():
        param.requires_grad = False
        if f'lora_experts_v' in name or \
        f'lora_experts_q' in name or \
        f'lora_router' in name:
            param.requires_grad = True
        
    
    return raw_datasets, config, tokenizer, prompt_config, model, cur_task_id, task_order, cur_task



def main():
    # Load argument.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args._frozen = False

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")

    raw_datasets, config, tokenizer, prompt_config, model, cur_task_id, task_order, cur_task = init_model(model_args, data_args, training_args)
    cal_total_para(model)

    if hasattr(model.config, "max_position_embeddings") and model.config.max_position_embeddings < data_args.max_source_length:
        if model_args.resize_position_embeddings is None:
            print(f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} to {data_args.max_source_length}.")
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollator(
        tokenizer,
        model=model,
        padding="longest",
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        add_task_name=data_args.add_task_name,
        add_dataset_name=data_args.add_dataset_name,
        num_examples=data_args.num_examples,
        input_record_file=data_args.input_record_file
    )
    training_args.remove_unused_columns = False
    previous_lora_router = None

    if cur_task_id!=0:
        previous_lora_router = []
        path = model_args.previous_lora_path
        lora_router = torch.load(os.path.join(path, 'router_network.pt'))
        with torch.no_grad():
            for j in range(config.num_hidden_layers):
                # TODO
                lr = LlamaLoraRouter(4096, experts_num=model_args.experts_num, experts_pool_num=model_args.experts_pool_num, fixed_experts_num=model_args.fixed_experts_num, task_experts_num=model_args.task_experts_num, select_experts_num=model_args.select_experts_num, task_num=model_args.task_num)
                lr.router_network[0].weight.copy_(
                    lora_router[f"model.layers.{j}.self_attn.lora_router.router_network.0.weight"]
                )
                lr.router_network[2].weight.copy_(
                    lora_router[f"model.layers.{j}.self_attn.lora_router.router_network.2.weight"]
                )
                lr.task_keys.copy_(
                    lora_router[f"model.layers.{j}.self_attn.lora_router.task_keys"]
                )
                lr.requires_grad = False
                lr.cuda().to(dtype=torch.bfloat16)
                previous_lora_router.append(lr)


    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
    
    
    

    print(f"-----Gradient checkpointing: {training_args.gradient_checkpointing} -----")
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    training_args.step_per_epoch = math.ceil(len(raw_datasets["train"]) / training_args.per_device_train_batch_size / world_size / training_args.gradient_accumulation_steps)
    training_args.eval_steps = training_args.eval_every_n_epoch * training_args.step_per_epoch
    training_args.save_steps = training_args.eval_every_n_epoch * training_args.step_per_epoch
    

    # Metric
    def compute_rouge_metrics(dataset, preds, save_prefix=None):
        decoded_preds = skip_instructions(model, preds, tokenizer)
        references = [e["Instance"]["label"] for e in dataset]
        result = compute_metrics(predictions=decoded_preds, references=references)
        result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                    groups=dataset["Task"])
        result.update(result_per_task)
        categories = dataset["Dataset"]
        result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                        groups=categories)
        result.update(result_per_category)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        if save_prefix is not None:
            with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                for example, pred in zip(dataset, decoded_preds):
                    fout.write(json.dumps({
                        "Task": example["Task"],
                        "Dataset": example["Dataset"],
                        "Instance": example["Instance"],
                        "Prediction": pred
                    }) + "\n")
        return result
    
    trainer = Trainer(
        model=model.cuda(),
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        cur_task_id=cur_task_id,
        task_order=task_order,
        previous_lora_router = previous_lora_router,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_rouge_metrics,
        callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None
    )

    all_metrics = {"run_name": training_args.run_name}

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        save_path = training_args.output_dir + "/saved_weights"
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except:
                pass

        if world_size > 1:
            rank = torch.distributed.get_rank()
            is_main_process = rank == 0
        else:
            is_main_process = 1

        if is_main_process:
            torch.save(lora_state_dict_A(model, task_name=cur_task), os.path.join(save_path, 'lora_experts_A.pt'))
            torch.save(lora_state_dict_B(model, task_name=cur_task), os.path.join(save_path, 'lora_experts_B.pt'))
            torch.save(router_network_dict(model, task_name=cur_task), os.path.join(save_path, 'router_network.pt'))


        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(f"Metrics {metrics}")
        all_metrics.update(metrics)
    
    # Evaluation
    results = {}
    if training_args.do_predict:
        max_new_tokens = (
            training_args.generation_max_length
            if training_args.generation_max_length is not None
            else data_args.max_target_length
        )

        num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
        repetition_penalty = data_args.repetition_penalty
        print("*** Prediction ***")
        print("*** Loading CheckPoint ***")

        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

        if world_size > 1:
            rank = torch.distributed.get_rank()
            is_main_process = rank == 0
        else:
            is_main_process = 1

        

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log(metrics)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        all_metrics.update(metrics)

        with open(os.path.join("logs_and_outputs", training_args.run_name, "outputs", "task_order.txt"), 'w') as f:
            f.write(data_args.task_order)

    return results

if __name__ == "__main__":
    main()