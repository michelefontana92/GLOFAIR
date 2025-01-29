# Define the BaseWrapper abstract class, which is a wrapper for a black box ML model
from abc import ABC, abstractmethod
class BaseWrapper(ABC):
    
    @abstractmethod
    def predict(self, data_loader):
        pass

    @abstractmethod
    def predict_proba(self, data_loader):
        pass
    
    @abstractmethod
    def score(self, data_loader,metrics):
        pass

    @abstractmethod    
    def save(self, path):
        pass
    
    @abstractmethod
    def load(self, path):
        pass

   

    