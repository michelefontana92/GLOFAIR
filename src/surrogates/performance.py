from .surrogate_factory import register_surrogate
from torch.nn import CrossEntropyLoss
from functools import partial
import torch
@register_surrogate('performance')
class PerformanceSurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        loss_params = kwargs.get('loss_params',{})
        
        self.loss = CrossEntropyLoss(reduction='mean',**loss_params)
    def __call__(self,**kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        final_loss = self.loss(logits,labels.long().view(-1,)).squeeze()
       
        # Numero di classi
        #C = logits.size(1)

        # Calcola la Normalized Cross Entropy
        #final_loss = loss.mean()# / torch.log(torch.tensor(C, dtype=torch.float))
        return final_loss
