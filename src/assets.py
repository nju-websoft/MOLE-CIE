from torch import nn
import torch
from typing import Dict

task_config = {
    'EE-0': 'configs/cie/EE-0',
    'EE-1': 'configs/cie/EE-1',
    'EE-2': 'configs/cie/EE-2',
    'EE-3': 'configs/cie/EE-3',
    'RE-0': 'configs/cie/RE-0',
    'RE-1': 'configs/cie/RE-1',
    'RE-2': 'configs/cie/RE-2',
    'RE-3': 'configs/cie/RE-3',
    'NER-0': 'configs/cie/NER-0',
    'NER-1': 'configs/cie/NER-1',
    'NER-2': 'configs/cie/NER-2',
    'NER-3': 'configs/cie/NER-3',

}

def lora_state_dict_A(model: nn.Module, bias: str = 'none', task_name=None) -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_A' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_A' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_A')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError

def lora_state_dict_B(model: nn.Module, bias: str = 'none', task_name=None) -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_B' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_B' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_B')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
    
def router_network_dict(model: nn.Module, bias: str = 'none', task_name=None) -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_router' in k}
    else:
        raise NotImplementedError

def task_key_dict(model: nn.Module, bias: str = 'none', task_name=None) -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'task_key' in k}
    else:
        raise NotImplementedError
