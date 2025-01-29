from metrics.base_metric import BaseMetric
import numpy as np
from .base_requirement import BaseRequirement

class UnconstrainedRequirement(BaseRequirement):
    
    def __init__(self,  name:str,metric: BaseMetric, weight:float,**kwargs) -> None:
        super(UnconstrainedRequirement, self).__init__(name,metric,weight)
        self.mode:str = kwargs.get('mode','max')
        self.bound:float = kwargs.get('bound',1.0)
        assert self.mode in ['min','max'], f'{self.mode} is not a valid mode'
        assert self.bound >= 0, f'{self.bound} is not a valid bound'
        self.performance_metric = kwargs.get('performance_metric','f1')

    def _reset(self):
        self.metric.reset()

    def _distance(self,metric_value):
        if self.mode == 'max':
            return (self.weight*(self.bound - metric_value)).item()
        elif self.mode == 'min':
            return (self.weight*(metric_value - self.bound)).item()
        else:
            raise NotImplementedError(f'{self.mode} is not implemented')
        
    def evaluate(self,y_pred, y_true, group_ids):
        self._reset()
        metric_value=self._compute(y_pred, y_true, group_ids)
        return {self.name: self._distance(metric_value)}
    
    def __str__(self) -> str:
        return f'{self.name}: {self.mode} {self.metric} with bound {self.bound} performance metric {self.performance_metric} and weight {self.weight}'
  