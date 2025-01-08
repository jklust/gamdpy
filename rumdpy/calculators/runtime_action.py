from abc import ABC, abstractmethod
from rumdpy import Configuration

class RuntimeAction(ABC):
    """
    Abstract Base Class specifying the requirements for a runtime_action, i.e. an action to compiled into the inner MD kernel
    """

    @abstractmethod   
    def get_kernel(self, configuration: Configuration, compute_plan: dict):
        """
        Get a kernel (or python function depending on compute_plan["gridsync"]) that implements the runtime_action
        """

        pass

    @abstractmethod
    def get_params(self, configuration: Configuration, compute_plan: dict) -> tuple :
        """
        Get a tuple with the parameters expected by the associated kernel
        """

        pass

