from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    @abstractmethod
    def __call__(self,**kwargs):
        pass