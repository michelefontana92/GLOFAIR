from .base_surrogate import BaseSurrogate,BaseBinarySurrogate,BaseBinaryWassersteinSurrogate
import torch 
from .surrogate_factory import register_surrogate



class BinaryEqualizedOddsSurrogate(BaseBinarySurrogate):
    def __init__(self, **kwargs) -> None:
        super(BinaryEqualizedOddsSurrogate, self).__init__(**kwargs)
        

    
    
    def _compute_statistic(self, logits, labels, group_ids):
        positive_mask = ((group_ids[self.group_name] == self.positive_group_id) & (labels==1)).squeeze()
        negative_mask = ((group_ids[self.group_name] == self.negative_group_id)& (labels==1)).squeeze()
        surrogate_opportunity = self._calculate(logits, positive_mask, negative_mask)
        positive_mask = ((group_ids[self.group_name] == self.positive_group_id) & (labels==0)).squeeze()
        negative_mask = ((group_ids[self.group_name] == self.negative_group_id)& (labels==0)).squeeze()
        surrogate_equality = self._calculate(logits, positive_mask, negative_mask)
        
        surrogate_odds = torch.max(torch.cat((surrogate_opportunity.view(1,-1),surrogate_equality.view(1,-1)),dim=1))
        return surrogate_odds
    


class BinaryWassersteinEqualizedOddsSurrogate(BaseBinaryWassersteinSurrogate):
    def __init__(self, **kwargs) -> None:
        super(BinaryWassersteinEqualizedOddsSurrogate, self).__init__(**kwargs)
    
    def _compute_statistic(self, logits, labels, group_ids,**kwargs):
        positive_mask = ((group_ids[self.group_name] == self.positive_group_id) & (labels==1)).squeeze()
        negative_mask = ((group_ids[self.group_name] == self.negative_group_id)& (labels==1)).squeeze()
        surrogate_opportunity = self._calculate(logits, positive_mask, negative_mask,**kwargs)
        positive_mask = ((group_ids[self.group_name] == self.positive_group_id) & (labels==0)).squeeze()
        negative_mask = ((group_ids[self.group_name] == self.negative_group_id)& (labels==0)).squeeze()
        surrogate_equality = self._calculate(logits, positive_mask, negative_mask,**kwargs)
        
        surrogate_odds = Approximation.differentiable_max(torch.cat((surrogate_opportunity.view(1,-1),surrogate_equality.view(1,-1)),dim=1))
        return surrogate_odds
    
    
    
@register_surrogate('equalized_odds')
class EqualizedOddsSurrogate(BaseSurrogate):
    def __init__(self,**kwargs) -> None:
        super(EqualizedOddsSurrogate, self).__init__(**kwargs)
        self._init_surrogates(BinaryEqualizedOddsSurrogate)
        
    
@register_surrogate('wasserstein_equalized_odds')
class WassersteinEqualizedOddsSurrogate(BaseSurrogate):
    def __init__(self,**kwargs) -> None:
        super(WassersteinEqualizedOddsSurrogate, self).__init__(**kwargs)
        self._init_surrogates(BinaryWassersteinEqualizedOddsSurrogate)