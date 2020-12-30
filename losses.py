import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityLoss:
    def forward(self, inputs, targets=None):
        return inputs

    def __call__(self, inputs, targets=None):
        return self.forward(inputs)
