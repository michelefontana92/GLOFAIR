from abc import ABC, abstractmethod
class BaseAggregator(ABC):
    @abstractmethod
    def __call__(self,**kwargs):
        pass