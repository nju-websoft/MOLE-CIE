# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""CL_Benchmark Dataset."""

import json
import os
import random
import datasets
from hashlib import md5

TASK_CONFIG_FILES = {"train": "train_tasks.json", "test": "test_tasks.json", "dev": "dev_tasks.json"}
INSTRUCTION_STRATEGIES = ['single', 'multiple']
ANSWER_PREFIX = "Output:"
SINGLE_QUOTES_SUBSTITUTE = "#$%#"
AUX_PROB = 0.3

def gen_cache_path(cache_dir, data_args):
    hash_str = data_args.data_dir + data_args.task_config_dir + \
               str(data_args.max_num_instances_per_task) + str(data_args.max_num_instances_per_eval_task)
    hash_obj = md5(hash_str.encode("utf-8"))
    hash_id = hash_obj.hexdigest()
    cache_path = os.path.join(cache_dir, str(hash_id))

    return cache_path


def check_path(path):
    if not path or not os.path.exists(path):
        raise ValueError('{} is not valid, please check the input path!'.format(path))


def save_ds(instances, file_name):
    with open(file_name, "w+", encoding='utf-8') as fi:
        json.dump(instances, fi, ensure_ascii=False, indent=2)


class CLConfig(datasets.BuilderConfig):
    """
    Config dataset load procedure.

    Args:
        data_dir: task data dir, which contains the corresponding dataset dirs
        prompt_path: prompt json file, which saves task and its prompts map
        task_file: task config file, save training and testing split config, and sampling strategies.
         Support two sampling strategies: 'random' indicates random sampling, while 'full' means to return all samples.
        max_num_instances_per_task: max training sample size of each task
        max_num_instances_per_eval_task: max eval sample size of each task
    """

    def __init__(
            self,
            *args,
            data_dir=None,
            replay_data_dir = None,
            task_config_dir=None,
            num_examples=None,
            max_num_instances_per_task=None,
            max_num_instances_per_eval_task=None,
            over_sampling=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.replay_data_dir = replay_data_dir
        self.num_examples = num_examples
        self.over_sampling = over_sampling
        self.task_configs = self._parse_task_config(task_config_dir)
        self.max_num_instances_per_task = max_num_instances_per_task
        self.max_num_instances_per_eval_task = max_num_instances_per_eval_task

    def _parse_task_config(self, task_config_dir):

        if not task_config_dir:
            return None

        task_configs = {}
        for task, file_name in TASK_CONFIG_FILES.items():
            task_config_file = os.path.join(task_config_dir, file_name)

            if not os.path.exists(task_config_file):
                raise ValueError('Please check {} config, {} not exists!'.format(task, task_config_file))

            with open(task_config_file, 'r+') as f:
                task_configs[task] = json.loads(f.read())

        return task_configs

task_id = {"EE":0, "RE":1, "NER":2}

class CLInstructions(datasets.GeneratorBasedBuilder):
    """CL Dataset."""

    VERSION = datasets.Version("2.0.0")
    BUILDER_CONFIG_CLASS = CLConfig
    BUILDER_CONFIGS = [
        CLConfig(name="default", description="Default config for NaturalInstructions")
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "Task": datasets.Value("string"),
                    "Dataset": datasets.Value("string"),
                    "subset": datasets.Value("string"),
                    "Samples": [{
                        "id": datasets.Value("string"),
                        "sentence": datasets.Value("string"),
                        "label": datasets.Value("string"),
                        "ground_truth": datasets.Value("string"),
                        "task_id": datasets.Value("string"),
                    }],
                    "Instance": {
                        "id": datasets.Value("string"),
                        "sentence": datasets.Value("string"),
                        "label": datasets.Value("string"),
                        "instruction": datasets.Value("string"),
                        "ground_truth": datasets.Value("string"),
                        "task_id": datasets.Value("string"),
                    }
                }
            ),
            supervised_keys=None
        )


    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        
        # split dir save datasets
        # task config to specify train,dev,test
        split_dir = self.config.data_dir
        task_configs = self.config.task_configs
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": split_dir,
                    "replay_data_path": self.config.replay_data_dir,
                    "task_config": task_configs['train'],
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                    "subset": "train"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path": split_dir,
                    "replay_data_path": None,
                    "task_config": task_configs['dev'],
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "dev"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": split_dir,
                    "replay_data_path": None,
                    "task_config": task_configs['test'],
                    "max_num_instances_per_task": None,  # default load total test samples to test
                    "subset": "test"
                }),
        ]


    def _load_dataset(self, dataset_path):
        with open(dataset_path, encoding="utf-8") as task_f:
            s = task_f.read()
            instances = json.loads(s)

        return instances
    



    def load_ie_dataset(self, dataset_path, replay_data_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
        if dataset_path:
            if 'EE' in dataset_path:
                task_id = 0
            elif 'RE' in dataset_path:
                task_id = 1
            else:
                task_id = 2
            data = self._load_dataset(dataset_path)
            input_mode='zeroshot'
            definition = definition = "Definition: " + data["Definition"][0].strip() + "\n\n"
            sample_template = {"Task": "CL", "Dataset": dataset_name, "Samples": [], "subset": subset}
            ## Change data size
            for idx, instance in enumerate(data['Instances']):
                example = sample_template.copy()
                instruction = ""
                # instruction += "Now complete the following example -\n"
                instruction += "Input: {0}"
                instruction += "\n"
                instruction += "Output: "
                pos_examples = []
                if input_mode=='fewshot':
                    for idx, pos_example in enumerate(data["Examples"]):
                        pos_example_str = f"Example {idx+1} -\n"
                        pos_example_str += f"Input: {pos_example['input'].strip()}"
                        pos_example_str += "\n"
                        pos_example_str += f"Output: {pos_example['output'].strip()}"
                        pos_example_str += "\n" 
                        pos_examples.append(pos_example_str)

                instruction = definition + "".join(pos_examples) + instruction

                label=instance["output"]

                example["Instance"] = {
                    "id": str(idx),
                    "sentence": instance['input'],
                    "label": label,
                    "ground_truth": label,
                    "instruction": instruction,
                    "task_id": task_id,
                }

                yield example

        if replay_data_path:
            for path in replay_data_path:
                if 'EE' in dataset_path:
                    task_id = 0
                elif 'RE' in dataset_path:
                    task_id = 1
                else:
                    task_id = 2
                replay_data = self._load_dataset(path)
                input_mode='zeroshot'
                definition = "Definition: " + replay_data["Definition"][0].strip() + "\n\n"
                sample_template = {"Task": "CL", "Dataset": dataset_name, "Samples": [], "subset": subset}
                ## Change data size
                for idx, instance in enumerate(replay_data['Instances']):
                    example = sample_template.copy()
                    instruction = ""
                    # instruction += "Now complete the following example -\n"
                    instruction += "Input: {0}"
                    instruction += "\n"
                    instruction += "Output: "
                    pos_examples = []
                    if input_mode=='fewshot':
                        for idx, pos_example in enumerate(replay_data["Examples"]):
                            pos_example_str = f"Example {idx+1} -\n"
                            pos_example_str += f"Input: {pos_example['input'].strip()}"
                            pos_example_str += "\n"
                            pos_example_str += f"Output: {pos_example['output'].strip()}"
                            pos_example_str += "\n" 
                            pos_examples.append(pos_example_str)

                    instruction = definition + "".join(pos_examples) + instruction

                    label=instance["output"]

                    example["Instance"] = {
                        "id": str(idx),
                        "sentence": instance['input'],
                        "label": label,
                        "ground_truth": label,
                        "instruction": instruction,
                        "task_id": task_id,
                    }

                    yield example

    def _generate_examples(self, path=None, replay_data_path = None, task_config=None, max_num_instances_per_task=None, subset=None):
        """Yields examples."""
        print(f"Generating tasks from = {path}")

        for task in task_config:
            load_func = self.load_ie_dataset

            # load dataset
            print(task_config)
            for dataset in task_config[task]:
                ds_name = dataset["dataset name"]
                sampling_strategy = dataset.get("sampling strategy", "random")
                ds_path = os.path.join(path, task, ds_name, subset + '.json')

                print(ds_path)
                print(replay_data_path)
                if 'replay' in path:
                    ds_path = None
                labels_path = None
                if ds_path:
                    assert os.path.exists(ds_path)
                if replay_data_path:
                    for path in replay_data_path:
                        assert os.path.exists(path)

                idx = -1
                instances = []
                for sample in load_func(ds_path, replay_data_path, labels_path, ds_name, sampling_strategy, max_num_instances_per_task,
                                        subset):
                    idx += 1
                    instances.append(sample)
                    yield f"{task}##{ds_path}##{idx}", sample
