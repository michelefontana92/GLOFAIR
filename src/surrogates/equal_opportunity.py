from .base_surrogate import BaseSurrogate,BaseBinarySurrogate
from typing import Any
from torch.nn.functional import softmax
import torch 
from .surrogate_factory import register_surrogate


class BinaryEqualOpportunitySurrogate(BaseBinarySurrogate):
    def __init__(self, **kwargs) -> None:
        super(BinaryEqualOpportunitySurrogate, self).__init__(**kwargs)
        
    def _compute_statistic(self, logits, labels, group_ids):
        #print('positive_group_id:',self.positive_group_id)
        #print('negative_group_id:',self.negative_group_id)
        positive_mask = ((group_ids[self.group_name] == self.positive_group_id) & (labels==1)).squeeze()
        negative_mask = ((group_ids[self.group_name] == self.negative_group_id)& (labels==1)).squeeze()
        surrogate = self._calculate(logits, positive_mask, negative_mask)
        return surrogate
    
@register_surrogate('equal_opportunity')
class EqualOpportunitySurrogate(BaseSurrogate):
    def __init__(self,**kwargs) -> None:
        super(EqualOpportunitySurrogate, self).__init__(**kwargs)
        self._init_surrogates(BinaryEqualOpportunitySurrogate)
        
