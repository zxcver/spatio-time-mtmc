from abc import ABC,abstractmethod

class BaseTrack(ABC):
    @abstractmethod
    def update(self,croped_imgs):
        """
        an abstract method need to be implemented
        """