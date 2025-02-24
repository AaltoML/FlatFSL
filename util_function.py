import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F

def _to_torch(sample):
    for key, val in sample.items():
        sample[key] = torch.from_numpy(val).to(device)
    return sample

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)
