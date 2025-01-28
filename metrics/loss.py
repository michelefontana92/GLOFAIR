import torch.nn as nn
from .metrics_factory import register_metric

def _cross_entropy_loss(**kwargs):
    return nn.CrossEntropyLoss()

def _mse_loss(**kwargs):
    return nn.MSELoss()

def _binary_crossentropy_loss(**kwargs):
    return nn.BCELoss()

def _weighted_cross_entropy_loss(**kwargs):
    return nn.CrossEntropyLoss(weight=kwargs.get('weight'))

@register_metric('loss')
class Loss:
    def __init__(self, name):
        self.name = name
        self.loss_fn = None
        self._losses = {
            'cross_entropy': _cross_entropy_loss,
            'mse': _mse_loss,
            'binary_crossentropy': _binary_crossentropy_loss,
            'weighted_cross_entropy': _weighted_cross_entropy_loss
        }
        self._set_loss_fn()

    def _set_loss_fn(self):
        self.loss_fn = self._losses.get(self.name)

    def __call__(self, **kwargs):
        return self.loss_fn(**kwargs)