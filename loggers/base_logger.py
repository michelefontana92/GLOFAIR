from abc import ABC, abstractmethod

class BaseLogger(ABC):
    @abstractmethod
    def log(self, message):
        pass

    @abstractmethod
    def error(self, message):
        pass

    @abstractmethod
    def info(self, message):
        pass

    @abstractmethod
    def debug(self, message):
        pass
    
    @abstractmethod
    def close(self): 
        pass