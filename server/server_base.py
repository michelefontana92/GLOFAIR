from abc import ABC,abstractmethod
from .server_factory import register_server
from client import BaseClient

@register_server("BaseServer")
class BaseServer(ABC):
    def __init__(self,**kwargs):
        self.config:dict = kwargs['config']
        self.children_list:list = kwargs['children_list']
        assert isinstance(self.config,dict), "config must be a dictionary"
        assert isinstance(self.children_list,list), "children_list must be a list"
    
    @abstractmethod
    def setup(self,**kwargs):
        pass 
    
    @abstractmethod
    def step(self,**kwargs):
        pass
    
    @abstractmethod
    def execute(self,**kwargs):
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
