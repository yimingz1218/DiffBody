import torch
from torch import nn
from torch.nn import functional as F

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module