from abc import ABC, abstractmethod


class Guard(ABC):
    
    @abstractmethod
    def reset(self) -> None:
        pass
    
    @abstractmethod
    def inform(self, data) -> None:
        pass
    
    @abstractmethod
    def __bool__(self):
        pass
    
    
    
class DummyGuard(Guard):
    
    def reset(self) -> None:
        pass
    
    def inform(self, data):
        pass
    
    def __bool__(self):
        return False