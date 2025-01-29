from .surrogate_factory import register_surrogate
from torch.nn import CrossEntropyLoss
from functools import partial
import torch
@register_surrogate('performance')
class PerformanceSurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name')
        self.weight = kwargs.get('surrogate_weight',1.0)
        loss_params = kwargs.get('loss_params',{})
        
        self.loss = CrossEntropyLoss(reduction='mean',**loss_params)
    def __call__(self,**kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        final_loss = self.loss(logits,labels.long().view(-1,)).squeeze()
       
        return final_loss
