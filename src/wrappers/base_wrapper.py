# Define the BaseWrapper abstract class, which is a wrapper for a black box ML model
from abc import ABC, abstractmethod
class BaseWrapper(ABC):
    """
    BaseWrapper is an abstract base class that defines the interface for a machine learning model wrapper.
    Methods
    -------
    predict(data_loader)
        Abstract method to make predictions using the model.
    predict_proba(data_loader)
        Abstract method to predict class probabilities using the model.
    score(data_loader, metrics)
        Abstract method to evaluate the model using specified metrics.
    save(path)
        Abstract method to save the model to the specified path.
    load(path)
        Abstract method to load the model from the specified path.
    """
    
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

   

    