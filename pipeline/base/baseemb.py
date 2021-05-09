from abc import ABC,abstractmethod

class BaseEmb(ABC):
    @abstractmethod
    def extract(self,croped_imgs):
        """
        an abstract method need to be implemented
        """