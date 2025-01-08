from abc import ABC, abstractmethod
from rumdpy import Configuration

class Interaction(ABC):
    """
    Abstract Base Class specifying the requirements for an integrator
    """

    @abstractmethod   
    def get_kernel(self, configuration: Configuration, compute_plan: dict, compute_flags: dict[str,bool]):
        """
        Get a kernel (or python function depending on compute_plan["gridsync"]) that implements calculation of the interaction
        """

        pass

    @abstractmethod
    def get_params(self, configuration: Configuration, compute_plan: dict) -> tuple :
        """
        Get a tuple with the parameters expected by the associated kernel
        """

        pass

    