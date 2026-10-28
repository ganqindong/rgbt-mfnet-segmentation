"""Small native-PyTorch equivalents for the upstream encoder's utilities."""
import torch
from torch import nn

trunc_normal_ = nn.init.trunc_normal_


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * mask / keep


def load_state_dict(model, state_dict, strict=False, logger=None):
    return model.load_state_dict(state_dict, strict=strict)
