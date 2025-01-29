from abc import ABC, abstractmethod
from metrics.fairness import GroupFairnessMetric
from metrics.performance import Performance
from metrics.base_metric import BaseMetric

class BaseRequirement(ABC):
    
    def __init__(self,  name:str,metric: BaseMetric, weight:float,**kwargs) -> None:
        self.metric = metric
        self.name = name
        self.weight = weight
        assert self.weight >= 0, f'{self.weight} is not a valid weight'
        assert isinstance(self.metric,BaseMetric), f'{self.metric} is not a valid metric'
        self.performance_metric = kwargs.get('performance_metric','f1')
    
    @abstractmethod
    def evaluate(self,y_pred, y_true, group_ids):
        pass
    
    def _compute(self,y_pred, y_true, group_ids):
        if issubclass(self.metric.__class__,GroupFairnessMetric):
            self.metric.calculate(y_pred, y_true, group_ids)
            metric_value_dict:dict = self.metric.get()
            #print('metric_value_dict:',metric_value_dict)
            metric_value = next(iter(metric_value_dict.values()))
        elif issubclass(self.metric.__class__,Performance):
            self.metric.calculate(y_pred, y_true)
            metric_value_dict:dict = self.metric.get()
            #print('metric_value_dict:',metric_value_dict)
            metric_value = metric_value_dict[self.performance_metric]
        else:
            raise NotImplementedError(f'{self.metric.__class__} is not implemented')
        return metric_value