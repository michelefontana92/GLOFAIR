from abc import ABC, abstractmethod

class BaseDataLoader(ABC):
    @abstractmethod
    def train_loader(self):
        pass 

    @abstractmethod
    def val_loader(self):
        pass 
    
    @abstractmethod
    def test_loader(self):
        pass 
    