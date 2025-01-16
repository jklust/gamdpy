from numba import cuda
from abc import ABC, abstractmethod
from rumdpy import Configuration
from typing import Callable


class RuntimeAction(ABC):
    """
    Abstract Base Class specifying the requirements for a runtime_action, i.e. an action to compiled into the inner MD kernel
    """

    @abstractmethod   
    def get_kernel(self, configuration: Configuration, compute_plan: dict) -> Callable:
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

    def update_at_end_of_timeblock(self, timeblock: int, output_reference):
        """
        Method to be called at the end of a timeblock, for e.g. saving data to a file if needed
        """

        pass

def merge_runtime_actions(configuration: Configuration, kernelA: Callable, paramsA: tuple, actionB: RuntimeAction, compute_plan: dict) -> tuple[Callable, tuple] :
    paramsB = actionB.get_params(configuration, compute_plan)
    kernelB = actionB.get_kernel(configuration, compute_plan)

    if compute_plan['gridsync']:
        # A device function, calling a number of device functions, using gridsync to syncronize
        @cuda.jit( device=compute_plan['gridsync'])
        def kernel(grid, vectors, scalars, r_im, sim_box, step, runtime_actions_params):
            kernelA(grid, vectors, scalars, r_im, sim_box, step, runtime_actions_params[0])
            grid.sync() # Not always necessary !!!
            kernelB(grid, vectors, scalars, r_im, sim_box, step, runtime_actions_params[1])
            return
        return kernel, (paramsA, paramsB, )
    else:
        # A python function, making several kernel calls to syncronize
        def kernel(grid, vectors, scalars, r_im, sim_box, step, runtime_actions_params):
            kernelA(grid, vectors, scalars, r_im, sim_box, step, runtime_actions_params[0])
            kernelB(grid, vectors, scalars, r_im, sim_box, step, runtime_actions_params[1])
            return
        return kernel, (paramsA, paramsB, )


def add_runtime_actions_list(configuration: Configuration, runtime_actions_list: list[RuntimeAction], compute_plan: dict, verbose: bool = False) -> tuple[Callable, tuple]:

    # Setup first interaction and cuda.jit it if gridsync is used for syncronization
    params = runtime_actions_list[0].get_params(configuration, compute_plan)
    kernel = runtime_actions_list[0].get_kernel(configuration, compute_plan)
    if compute_plan['gridsync']:
        kernel: Callable = cuda.jit( device=compute_plan['gridsync'] )(kernel)

    # Merge in the rest of the runtime_actions (maximum recursion depth might set a maximum for number of interactions)
    for i in range(1, len(runtime_actions_list)):
        kernel, params = merge_runtime_actions(configuration, kernel, params, runtime_actions_list[i], compute_plan)

    return kernel, params
