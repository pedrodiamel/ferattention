import torch.nn as nn

def load_state_dict(state_dict_des, state_dict_src):
    """
    Load model for match weight
    """
    own_state = state_dict_des
    for name, param in state_dict_src.items():
        if name in own_state:            
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                print('{} dim model {} vs dim checkpoint{}'.format(name, own_state[name].size(), param.size()))