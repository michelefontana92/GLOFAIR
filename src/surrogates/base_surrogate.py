from typing import Any
from abc import  abstractmethod
import torch
from torch.nn.functional import softmax
from wasserstein import wasserstein_distance_differentiable,wasserstein_distance_threshold

class BaseBinaryWassersteinSurrogate:
    def __init__(self, **kwargs: Any) -> None:
        self.group_name = kwargs.get('group_name')
        self.positive_group_id:int = kwargs.get('positive_group_id')
        self.negative_group_id:int = kwargs.get('negative_group_id')
        assert self.positive_group_id != self.negative_group_id, f'positive_group_id and negative_group_id should be different'
        assert self.positive_group_id is not None, f'positive_group_id should not be None'
        assert self.negative_group_id is not None, f'negative_group_id should not be None'

    def __call__(self, **kwargs) -> Any:
        logits = kwargs.get('logits')
        group_ids_dict: dict = kwargs.get('group_ids')
        labels = kwargs.get('labels')
        use_differentiable = kwargs.get('use_differentiable',True)
        group_ids = group_ids_dict[self.group_name]
        assert logits.shape[0] == labels.shape[0], f'logits and labels should have the same length'
        assert logits.shape[0] == group_ids.shape[0], f'logits and group_ids should have the same length'
        return self._compute_statistic(logits, labels, group_ids_dict,
                                       use_differentiable=use_differentiable)


    def _calculate(self,logits, positive_mask,negative_mask,**kwargs):
        use_differentiable = kwargs.get('use_differentiable',True)
        if positive_mask.sum() == 0 or negative_mask.sum() == 0:
            return 0*(logits.sum())
        probabilities = softmax(logits, dim=1)[:,1]
        positive_probabilities = probabilities[positive_mask]
        negative_probabilities = probabilities[negative_mask]
        if use_differentiable:
            surrogate,_,_= wasserstein_distance_differentiable(positive_probabilities,
                                                               negative_probabilities,
                                                               p=2)
        else:
            surrogate,_,_ = wasserstein_distance_threshold(positive_probabilities,
                                                           negative_probabilities,
                                                           p=1)
        return surrogate


class BaseBinarySurrogate:
    def __init__(self, **kwargs: Any) -> None:
        self.group_name = kwargs.get('group_name')
        self.positive_group_id:int = kwargs.get('positive_group_id')
        self.negative_group_id:int = kwargs.get('negative_group_id')
        self.weight = kwargs.get('weight',1.0)
        assert self.positive_group_id != self.negative_group_id, f'positive_group_id and negative_group_id should be different'
        assert self.positive_group_id is not None, f'positive_group_id should not be None'
        assert self.negative_group_id is not None, f'negative_group_id should not be None'
    
    def __call__(self, **kwargs) -> Any:
        logits = kwargs.get('logits')
        group_ids_dict: dict = kwargs.get('group_ids')
        labels = kwargs.get('labels')
        group_ids =group_ids_dict[self.group_name]
        assert logits.shape[0] == labels.shape[0], f'logits and labels should have the same length'
        assert logits.shape[0] == group_ids.shape[0], f'logits and group_ids should have the same length'
        return self._compute_statistic(logits, labels, group_ids_dict)


    def _calculate(self,logits, positive_mask,negative_mask):
        if positive_mask.sum() == 0 or negative_mask.sum() == 0:
            return 0*(logits.sum())
        probabilities = softmax(logits, dim=1)[:,1]
        positive_probabilities = probabilities[positive_mask]
        negative_probabilities = probabilities[negative_mask]
        positive_surrogate = torch.mean(torch.tanh(torch.relu(positive_probabilities)))
        negative_surrogate = torch.mean(torch.tanh(torch.relu(negative_probabilities)))
        surrogate = torch.abs(positive_surrogate - negative_surrogate)
        return surrogate

class BaseSurrogate:
    def __init__(self, **kwargs: Any) -> None:
       
        self.group_name = kwargs.get('group_name')
        self.unique_group_ids:dict = kwargs.get('unique_group_ids')
        assert isinstance(self.unique_group_ids, dict), f'unique_group_ids should be a dictionary'
        self.group_name:str = kwargs.get('group_name')
        assert self.group_name in self.unique_group_ids.keys(), f'{self.group_name} is not a valid group_name'
        self.reduction = kwargs.get('reduction','mean')
        assert self.reduction in ['min','max','mean'], f'{self.reduction} is not a valid reduction'
        self.reduction_fn = self._init_reduction()
        self.name = kwargs.get('name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        
    def _init_surrogates(self,surrogate_class:BaseBinarySurrogate):
        self.surrogates = []
        current_group_ids = self.unique_group_ids[self.group_name]
        for i in range(len(current_group_ids)):
            for j in range(i+1,len(current_group_ids)):
               self.surrogates.append(surrogate_class(
                   group_name=self.group_name,
                   positive_group_id=current_group_ids[i], 
                   negative_group_id=current_group_ids[j]))
    
    def _init_reduction(self):
        if self.reduction == 'mean':
            reduction_fn = torch.mean
        elif self.reduction == 'max':
            reduction_fn = torch.max
        elif self.reduction == 'min':
            reduction_fn = torch.min  
        else:
            raise ValueError(f'{self.reduction} is not a valid reduction')
        return reduction_fn
    
    def __call__(self, **kwargs: Any) -> Any:
        results = self.surrogates[0](**kwargs).view(1,-1)
        for surrogate in self.surrogates[1:]:
            result = surrogate(**kwargs).view(1,-1)
            results = torch.cat((results,result),dim=1)
        #print(results)
        return self.reduction_fn(results).squeeze()