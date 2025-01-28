from abc import ABC, abstractmethod
class BaseMetric(ABC):
    @abstractmethod
    def calculate(self, y_pred, y_true):
        pass
    
    @abstractmethod
    def get(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass