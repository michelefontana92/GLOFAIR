from .aggregator_factory import register_aggregator
from .base_aggregator import BaseAggregator
import torch.nn as nn

@register_aggregator("FedAvgAggregator")
class FedAvgAggregator(BaseAggregator):
    def __init__(self,**kwargs):
        super(FedAvgAggregator,self).__init__(**kwargs)
    
    def setup(self,**kwargs):
        pass
    
    def _compute_total_weight(self,params_list):
        return sum([params['weight'] for params in params_list])
        
    def __call__(self,**kwargs):
        params_list = kwargs.get('params')
        model = kwargs.get('model')
        assert isinstance(model,nn.Module), "model must be a torch.nn.Module"
        assert isinstance(params_list,list), "params must be a list"
        for params in params_list:
            assert isinstance(params,dict), "params must be a list of dictionaries"
            assert 'weight' in params.keys(), "params must have a 'weight' key"
            assert 'params' in params.keys(), "params must have a 'params' key"
            assert isinstance(params['params'],dict), "params['params'] must be a dictionary"
            assert params['params'].keys() == model.state_dict().keys(), "params['params'] must have the same keys as the model's state_dict"
        total_weight = self._compute_total_weight(params_list)
        assert total_weight > 0, "total_weight must be greater than 0"
       
        
        new_params = {k: params_list[0]['weight']/total_weight * v for k,v in params_list[0]['params'].items()}
        assert new_params.keys() == model.state_dict().keys(), "new_params must have the same keys as the model's state_dict"
        
        for params in params_list[1:]:
            weight = params['weight']/total_weight
            for k,v in params['params'].items():
                new_params[k] += weight * v
        #model.load_state_dict(new_params)
        return new_params