from metrics.base_metric import BaseMetric
import numpy as np
from .base_requirement import BaseRequirement

class ConstrainedRequirement(BaseRequirement):
    
    def __init__(self,  name:str,metric: BaseMetric, weight:float,**kwargs) -> None:
        super(ConstrainedRequirement, self).__init__(name,metric,weight)
        self.operator = kwargs.get('operator','<=')
        self.threshold = kwargs.get('threshold',0.2)
        self.hard_constraint = kwargs.get('hard_constraint',False)
        self.tolerance = kwargs.get('tolerance', 0.005)
        assert self.operator in ['<=','>=','<','>','==','!='], f'{self.operator} is not a valid operator'
        assert self.threshold >= 0, f'{self.threshold} is not a valid threshold'
        

    def _check(self, metric_value):
        # Aggiungi la tolleranza al controllo
        if self.operator in ['<=', '<']:
            return eval(f'{metric_value} {self.operator} {self.threshold + self.tolerance}')
        elif self.operator in ['>=', '>']:
            return eval(f'{metric_value} {self.operator} {self.threshold - self.tolerance}')
        else:
            return eval(f'{metric_value} {self.operator} {self.threshold}')
    
    def _reset(self):
        self.metric.reset()

    def _distance(self,metric_value):
        if self._check(metric_value):
            return 0
        return self.weight*np.linalg.norm(self.threshold - metric_value)
        
    def evaluate(self,y_pred, y_true, group_ids):
        self._reset()
        metric_value=self._compute(y_pred, y_true, group_ids)
        is_satisfied = self._check(metric_value)
        return {self.name: self._distance(metric_value),'status':is_satisfied}
    
    def __str__(self) -> str:
        return f'{self.name}: {self.metric} {self.operator} {self.threshold} with weight {self.weight} and performance metric {self.performance_metric}'
