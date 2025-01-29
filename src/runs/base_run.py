from abc import ABC, abstractmethod
class BaseRun(ABC):
    def __init__(self,**kwargs):
        
        self.model = kwargs.get('model')
        self.dataset = kwargs.get('dataset')
        self.sensitive_attributes = kwargs.get('sensitive_attributes')
        self.project_name = kwargs.get('project_name')
        self.data_root = kwargs.get('data_root')

    def compute_group_cardinality(self,group_name):
        for name,group_dict in self.sensitive_attributes:
            if name == group_name:
                total = 1
                for key in group_dict.keys():
                    total *= len(group_dict[key])
                return total 
        raise KeyError(f'Group {group_name} not found in sensitive attributes') 
    
    @abstractmethod
    def setUp(self):
        pass 
    
    @abstractmethod
    def tearDown(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def __call__(self):
        self.setUp()
        self.run()
        self.tearDown()
