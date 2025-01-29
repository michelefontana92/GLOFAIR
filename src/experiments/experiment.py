from wrappers import TorchNNWrapper
from abc import ABC, abstractmethod

class Experiment(ABC):
    def compute_groups_ids(self,clients_config):
        group_ids_result = {}
        for client_config in clients_config:
            group_ids = client_config['data'].get_group_ids()
            for group_name in group_ids.keys():
                if group_name not in group_ids_result:
                    group_ids_result[group_name] = set(group_ids[group_name])
                else:
                    group_ids_result[group_name] = group_ids_result[group_name].union(set(group_ids[group_name]))     
        
        group_ids_result = {group_name: list(group_ids_result[group_name]) for group_name in group_ids_result.keys()}
        return group_ids_result
    
    @abstractmethod
    def run(self,**kwargs):
        pass

    @abstractmethod
    def setup(self,**kwargs):
        pass

    @abstractmethod
    def teardown(self,**kwargs):
        pass
      
