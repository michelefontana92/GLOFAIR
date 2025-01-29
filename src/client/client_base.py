from abc import ABC,abstractmethod



class BaseClient(ABC):
    """
    BaseClient is an abstract base class that defines the interface for client operations.
    Attributes:
        config (dict): Configuration dictionary passed during initialization.
    Methods:
        update(**kwargs):
            Abstract method to update the client state.
        setup(**kwargs):
            Abstract method to set up the client.
        evaluate(**kwargs):
            Abstract method to evaluate the client.
        fine_tune(**kwargs):
            Abstract method to fine-tune the client.
        shutdown(**kwargs):
            Abstract method to shut down the client.
    """
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
