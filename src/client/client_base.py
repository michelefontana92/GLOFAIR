from abc import ABC,abstractmethod



class BaseClient(ABC):
    def __init__(self,**kwargs):
        self.config:dict = kwargs['config']
        assert isinstance(self.config,dict), "config must be a dictionary"
        
    @abstractmethod
    def update(self,**kwargs):
        pass
    
    @abstractmethod
    def setup(self,**kwargs):
        pass 
    
    @abstractmethod
    def evaluate(self,**kwargs):
        pass
    
    @abstractmethod
    def fine_tune(self,**kwargs):
        pass
    
    @abstractmethod
    def shutdown(self,**kwargs):
        pass
