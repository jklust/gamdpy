import numpy as np
import numba
import math
from numba import cuda

# rumdpy
import rumdpy as rp

# IO
import h5py


class Simulation():
    """ Class for running a simulation.

    Parameters
    ----------

    configuration : rumdpy.Configuration
        The configuration to simulate.

    interactions : an interaction or list of interactions
        Interactions such as pair potentials, bonds, external fields, etc.

    integrator : an integrator
        The integrator to use for the simulation.

    num_steps : int
        Number of steps to run the simulation. If 0, num_timeblocks and steps_per_timeblock should be set.

    num_timeblocks : int
        Number of timeblocks to run the simulation. If not 0, then steps_per_timeblock should be set.

    steps_per_timeblock : int
        Number of steps per timeblock.

    compute_plan : dict
        A dictionary with the compute plan for the simulation. If None, a default compute plan is used.

    storage : str
        Storage for the simulation output. Can be 'memory' or a filename with extension '.h5'.

    scalar_output : str or int
        How often to save scalar output. If 'default', then a default value is used.

    conf_output : str or None
        If 'default', then a default method is used (logarithmic spacing).
        If None, no configuration output is saved.

    steps_between_momentum_reset : int
        Number of steps between momentum reset. If 'default', then a default value is used.

    compute_stresses : bool
        If True, stresses are computed.

    verbose : bool
        If True, print verbose output.

    timing : bool
        If True, timing information is saved.


    See also
    --------

    :func:`rumdpy.get_default_sim`

    """

    def __init__(self, configuration: rp.Configuration, interactions, integrator, 
                 num_steps=0, num_timeblocks=0, steps_per_timeblock=0,
                 compute_plan=None, storage='output.h5', scalar_output: int='default', conf_output='default',
                 steps_between_momentum_reset: int='default', compute_stresses=False, verbose=False, timing=True, include_simbox_in_output=False):

        self.configuration = configuration
        if compute_plan == None:
            self.compute_plan = rp.get_default_compute_plan(self.configuration)
        else:
            self.compute_plan = compute_plan

        self.compute_stresses = compute_stresses
        if self.compute_stresses and not configuration.compute_stresses:
            raise ValueError("Configuration must have compute_stresses set as well!")

        # Integrator
        if type(interactions) == list:
            self.interactions = interactions
        else:
            self.interactions = [interactions, ]
        self.integrator = integrator
        self.dt = self.integrator.dt

        if num_timeblocks == 0:
            if num_steps == 0:
                raise ValueError("Either num_steps or num_timeblocks must be non-zero")
            num_timeblocks = 32
            steps_per_timeblock = 2 ** int(math.log2(math.ceil(num_steps / num_timeblocks)))
            num_timeblocks = math.ceil(num_steps / steps_per_timeblock)
            print('num_steps: ', num_steps)
            print('num_blocks: ', num_timeblocks)
            print('steps_per_block: ', steps_per_timeblock)
        elif steps_per_timeblock == 0:
            raise ValueError("If num_timeblocks is non-zero then steps_per_timeblock must be too (num_steps is ignored in this case)")
        # we do not use nsteps if num_timeblocks is non-zero, because it's
        # not guaranteed to be a multiple of the latter
        self.num_blocks = num_timeblocks
        self.current_block = -1

        self.steps_per_block = steps_per_timeblock
        self.storage = storage
        self.timing = timing

        # Create output objects
        if self.storage[-3:] == '.h5':  # Saving in hdf5 format
            with h5py.File(self.storage, "w") as f:
                # Attributes for simulation (maybe save full configurations)
                f.attrs['dt'] = self.dt
                f.attrs['simbox_initial'] = self.configuration.simbox.lengths
                ds = f.create_dataset("ptype", shape=(self.configuration.N), dtype=np.int32)
                ds[:] = configuration.ptype
        elif self.storage == 'memory':
            # Set up a dictionary that exactly mirrors hdf5 file, so analysis programs can be the same
            self.output = {}
            self.output['attrs'] = {'dt': self.dt, 'simbox_initial': self.configuration.simbox.lengths.copy()}
            self.output['ptype'] = configuration.ptype.copy()

        # Momentum reset
        if steps_between_momentum_reset == 'default':
            steps_between_momentum_reset = 100

        if steps_between_momentum_reset > 0:
            self.momentum_reset = rp.MomentumReset(steps_between_momentum_reset)
        else:
            self.momentum_reset = None

        # Scalar saving
        if scalar_output == 'default':
            scalar_output = 16

        if scalar_output == None or scalar_output == 'none' or scalar_output < 1:
            self.output_calculator = None
        else:
            self.output_calculator = rp.ScalarSaver(configuration, scalar_output, num_timeblocks, steps_per_timeblock,
                                                    storage)

        # Saving of configurations
        if conf_output == 'default':
            self.conf_saver = rp.ConfSaver(self.configuration, num_timeblocks, steps_per_timeblock, storage, include_simbox=include_simbox_in_output)
        elif conf_output == None or conf_output == 'none':
            self.conf_saver = None
        else:
            raise RuntimeError('Did not understand conf_output = ', conf_output)

        # Update state in case of memory
        if self.storage[-3:] != '.h5':
            if not self.output_calculator == None: self.output.update(self.output_calculator.output)
            if not self.conf_saver        == None: self.output.update(self.conf_saver.output)

        self.vectors_list = []
        self.scalars_list = []
        self.simbox_data_list = []

        self.JIT_and_test_kernel()

    def JIT_and_test_kernel(self):
        while True:
            try:
                self.get_kernels_and_params()
                self.integrate = self.make_integrator(self.configuration, self.integrator_kernel, self.interactions_kernel,
                                                self.output_calculator_kernel, self.conf_saver_kernel,
                                                self.momentum_reset_kernel,
                                                self.compute_plan, True)
        
                self.configuration.copy_to_device() # By _not_ copying back to host later we dont change configuration
                self.integrate_self(0.0, 1)
                break
            except numba.cuda.cudadrv.driver.CudaAPIError as e:
                #print('Failed compute_plan : ', self.compute_plan)
                if self.compute_plan['tp'] > 1:             # Most common problem tp is too big
                    self.compute_plan['tp'] -= 1            # ... so we reduce it and try again
                elif self.compute_plan['gridsync'] == True: # Last resort: turn off gridsync
                    self.compute_plan['gridsync'] = False
                else:
                    print(f'FAILURE. Can not handle cuda error {e}')
                    exit()
                print('Trying adjusted compute_plan :', self.compute_plan)

    def get_kernels_and_params(self, verbose=False):
        # Interactions
        self.interactions_kernel, self.interactions_params = rp.add_interactions_list(self.configuration,
                                                                                      self.interactions,
                                                                                      compute_plan=self.compute_plan,
                                                                                      compute_stresses=self.compute_stresses,
                                                                                      verbose=verbose)

        # Momentum reset 
        if self.momentum_reset != None:
            self.momentum_reset_params = self.momentum_reset.get_params(self.configuration, self.compute_plan)
            self.momentum_reset_kernel = self.momentum_reset.get_kernel(self.configuration, self.compute_plan)
        else:
            self.momentum_reset_kernel = None
            self.momentum_reset_params = (0,)

        # Scalar saving
        if self.output_calculator != None:
            self.output_calculator_params = self.output_calculator.get_params(self.configuration, self.compute_plan)
            self.output_calculator_kernel = self.output_calculator.get_kernel(self.configuration, self.compute_plan)
        else:
            self.output_calculator_kernel = None
            self.output_calculator_params = (0,)

        # Configuration saving
        if self.conf_saver != None:
            self.conf_saver_kernel = self.conf_saver.get_kernel(self.configuration, self.compute_plan)
            self.conf_saver_params = self.conf_saver.get_params(self.configuration, self.compute_plan)
        else:
            self.conf_saver_kernel = None
            self.conf_saver_params = (0,)

        # Integrator
        self.integrator_params = self.integrator.get_params(self.configuration, verbose)
        self.integrator_kernel = self.integrator.get_kernel(self.configuration, self.compute_plan, verbose)

        return

    def update_params(self, verbose=False):
        # Interactions
        _, self.interactions_params = rp.add_interactions_list(self.configuration,
                                                                self.interactions,
                                                                compute_plan=self.compute_plan,
                                                                compute_stresses=self.compute_stresses,
                                                                verbose=verbose)

        # Momentum reset 
        if self.momentum_reset != None:
            self.momentum_reset_params = self.momentum_reset.get_params(self.configuration, self.compute_plan)
            #self.momentum_reset_kernel = self.momentum_reset.get_kernel(self.configuration, self.compute_plan)
        else:
            #self.momentum_reset_kernel = None
            self.momentum_reset_params = (0,)

        # Scalar saving
        if self.output_calculator != None:
            self.output_calculator_params = self.output_calculator.get_params(self.configuration, self.compute_plan)
            #self.output_calculator_kernel = self.output_calculator.get_kernel(self.configuration, self.compute_plan)
        else:
            #self.output_calculator_kernel = None
            self.output_calculator_params = (0,)

        # Configuration saving
        if self.conf_saver != None:
            #self.conf_saver_kernel = self.conf_saver.get_kernel(self.configuration, self.compute_plan)
            self.conf_saver_params = self.conf_saver.get_params(self.configuration, self.compute_plan)
        else:
            #self.conf_saver_kernel = None
            self.conf_saver_params = (0,)

        # Integrator
        self.integrator_params = self.integrator.get_params(self.configuration, verbose)
        #self.integrator_kernel = self.integrator.get_kernel(self.configuration, self.compute_plan, verbose)

        return

    def integrate_self(self, time_zero, steps):
        self.integrate(self.configuration.d_vectors,
                           self.configuration.d_scalars,
                           self.configuration.d_ptype,
                           self.configuration.d_r_im,
                           self.configuration.simbox.d_data,
                           self.interactions_params,
                           self.integrator_params,
                           self.conf_saver_params,
                           self.momentum_reset_params,
                           self.output_calculator_params,
                           np.float32(time_zero),
                           steps)
        return


    def make_integrator(self, configuration, integration_step, compute_interactions, output_calculator_kernel,
                        conf_saver_kernel, momentum_reset_kernel, compute_plan, verbose=True):
        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']]
        num_blocks = (num_part - 1) // pb + 1

        if gridsync:
            # Return a kernel that does 'steps' timesteps, using grid.sync to syncronize   
            @cuda.jit
            def integrator(vectors, scalars, ptype, r_im, sim_box, interaction_params, integrator_params,
                           conf_saver_params, momentum_reset_params, output_calculator_params, time_zero, steps):
                grid = cuda.cg.this_grid()
                for step in range(steps):
                    compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_params)
                    if conf_saver_kernel != None:
                        conf_saver_kernel(grid, vectors, scalars, r_im, sim_box, step, conf_saver_params)
                    grid.sync()
                    time = time_zero + step * integrator_params[0]
                    integration_step(grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                    if momentum_reset_kernel != None:
                        momentum_reset_kernel(grid, vectors, scalars, r_im, sim_box, step, momentum_reset_params)
                    if output_calculator_kernel != None:
                        output_calculator_kernel(grid, vectors, scalars, r_im, sim_box, step, output_calculator_params)

                    grid.sync()
                if conf_saver_kernel != None:
                    conf_saver_kernel(grid, vectors, scalars, r_im, sim_box, steps,
                                      conf_saver_params)  # Save final configuration (if conditions fullfiled)
                return

            return integrator[num_blocks, (pb, tp)]

        else:

            # Return a Python function that does 'steps' timesteps, using kernel calls to syncronize  
            def integrator(vectors, scalars, ptype, r_im, sim_box, interaction_params, integrator_params,
                           conf_saver_params, momentum_reset_params, output_calculator_params, time_zero, steps):
                for step in range(steps):
                    compute_interactions(0, vectors, scalars, ptype, sim_box, interaction_params)
                    if conf_saver_kernel != None:
                        conf_saver_kernel(0, vectors, scalars, r_im, sim_box, step, conf_saver_params)
                    time = time_zero + step * integrator_params[0]
                    integration_step(0, vectors, scalars, r_im, sim_box, integrator_params, time)
                    if output_calculator_kernel != None:
                        output_calculator_kernel(0, vectors, scalars, r_im, sim_box, step, output_calculator_params)
                    if momentum_reset_kernel != None:
                        momentum_reset_kernel(0, vectors, scalars, r_im, sim_box, step, momentum_reset_params)

                if conf_saver_kernel != None:
                    conf_saver_kernel(0, vectors, scalars, r_im, sim_box, steps,
                                      conf_saver_params)  # Save final configuration (if conditions fullfiled)
                return

            return integrator
        return

        # simple run function

    def run(self, verbose=True):
        """ Run the simulation.

        See also
        --------

        :func:`rumdpy.Simulation.timeblocks`

        """
        for _ in self.timeblocks():
            if verbose:
                print(self.status(per_particle=True))
        if verbose:
            print(self.summary())

    # generator for running simulation one block at a time
    def timeblocks(self, num_timeblocks=-1):
        """ Generator for running the simulation one block at a time.

        Parameters
        ----------

        num_timeblocks : int
            Number of blocks to run. If -1, all blocks are run.

        Examples
        --------

        >>> import rumdpy as rp
        >>> sim = rp.get_default_sim()
        >>> for block in sim.timeblocks(num_timeblocks=3):
        ...     print(f'{block=}')  # Replace with code to analyze the current configuration
        block=0
        block=1
        block=2

        See also
        --------

        :func:`rumdpy.Simulation.run`

        """
        if num_timeblocks == -1:
            num_timeblocks = self.num_blocks
        self.last_num_blocks = num_timeblocks
        assert (num_timeblocks <= self.num_blocks)  # Could be made OK with more blocks

        self.configuration.copy_to_device()
        self.vectors_list = []
        self.scalars_list = []
        self.simbox_data_list = []
        self.scalars_t = []

        if self.timing:
            start = cuda.event()
            end = cuda.event()
            start_block = cuda.event()
            end_block = cuda.event()
            block_times = []
            start.record()

        zero = np.float32(0.0)

        for block in range(num_timeblocks):
            if self.timing: start_block.record()
            self.current_block = block
            if self.output_calculator != None:
                self.output_calculator.initialize_before_timeblock()
            self.integrate_self(np.float32(block * self.steps_per_block * self.dt),self.steps_per_block)
            
            self.configuration.copy_to_host()
            self.vectors_list.append(self.configuration.vectors.copy())  # Needed for 3D viz, should use memory/hdf5
            self.scalars_list.append(self.configuration.scalars.copy())  # same
            self.simbox_data_list.append(self.configuration.simbox.lengths.copy())  # same
            #self.scalars_t.append(self.d_output_array.copy_to_host())              # same

            if self.output_calculator != None:
                self.output_calculator.update_at_end_of_timeblock(block)
                #self.output_calculator.initialize_before_timeblock()
            
            if self.conf_saver != None:
                self.conf_saver.update_at_end_of_timeblock(block)

            if self.timing:
                end_block.record()
                end_block.synchronize()
                block_times.append(cuda.event_elapsed_time(start_block, end_block))
            yield block

        # Finalizing run
        if self.timing:
            end.record()
            end.synchronize()

            self.timing_numba = cuda.event_elapsed_time(start, end)
            self.timing_numba_blocks = np.array(block_times)
        self.nbflag = self.interactions[0].nblist.d_nbflag.copy_to_host()
        self.scalars_list = np.array(self.scalars_list)

    def status(self, per_particle=False) -> str:
        """ String with the current status
        Should be executed during the simulation run, see :func:`rumdpy.Simulation.timeblocks`

        Parameters
        ----------

        per_particle : bool
            If True, the values are divided by the number of particles in the configuration.

        Returns
        -------

        str
            A string with the current status of the simulation.

        """
        time = self.current_block * self.steps_per_block * self.dt
        st = f'{time= :<10.3f}'
        for name in self.configuration.sid:
            data = np.sum(self.configuration[name])
            if per_particle:
                data /= self.configuration.N
            st += f'{name}= {data:<10.3f}'
        return st

    def summary(self) -> str:
        """ Returns a summary string of the simulation run.
         Should be called after the simulation has been run,
         see :func:`rumdpy.Simulation.timeblocks` or :func:`rumdpy.Simulation.run`
         """
        if self.timing:
            time_total = self.timing_numba / 1000
            tps_total = self.last_num_blocks * self.steps_per_block / time_total
            time_sim = np.sum(self.timing_numba_blocks) / 1000
            tps_sim = self.last_num_blocks * self.steps_per_block / time_sim

            if self.timing_numba_blocks.shape[0] > 1:
                extratime_firstblock = (self.timing_numba_blocks[0]
                                        - np.mean(self.timing_numba_blocks[1:])) / 1000
                time_sim_minus_extra = time_sim - extratime_firstblock
                tps_sim_minus_extra = self.last_num_blocks * self.steps_per_block / time_sim_minus_extra

        st = f'Particles : {self.configuration.N} \n'
        st += f'Steps : {self.last_num_blocks * self.steps_per_block} \n'
        st += f'nbflag : {self.nbflag} \n'
        if self.timing:
            st += f'Total time (incl. time spent between blocks): {time_total:.2f} s \n'
            st += f'Simulation time : {time_sim:.2f} s \n'
            st += f'Extra time 1.st block (presumably JIT): {extratime_firstblock:.2f} s \n'
            st += f'TPS_total : {tps_total:.2e} \n'
            st += f'TPS_sim : {tps_sim:.2e} \n'
            if self.timing_numba_blocks.shape[0] > 1:
                st += f'TPS_sim_minus_extra : {tps_sim_minus_extra:.2e} \n'
        return st

    def autotune_bruteforce(self, pbs='auto', skins='auto', tps='auto', timesteps=0, repeats=1):
        print('compute_plan :', self.compute_plan)
        if timesteps==0: 
            timesteps = self.steps_per_block
        assert timesteps<=self.steps_per_block
        
        pb = self.compute_plan['pb']
        if pbs=='auto':
            pbs = [pb//2, pb, pb*2]
        if pbs=='default':
            pbs = [pb,]
        print('pbs :', pbs)

        tp = self.compute_plan['tp']
        if tps=='auto': 
            tps = [tp - 3, tp - 2, tp - 1, tp, tp + 1, tp + 2, tp + 3,]
        if tps=='default':
            tps = [tp,]
        print('tps :', tps)

        skin = self.compute_plan['skin']
        if skins=='auto':
            skins = [skin - 0.3, skin - 0.2, skin - 0.1, skin - 0.05, skin, skin + 0.05, skin + 0.1, skin + 0.2, skin + 0.3]
        elif skins=='default':
            skins = [skin, ]
        print('skins :', skins)
            
        
        flag = cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS
        cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = False
        
        skin_times = []
        total_min_time = 1e9
        for pb in pbs:
            if pb <= 256:
                self.compute_plan['pb'] = pb
                for tp in tps:
                    if tp>0:
                        self.compute_plan['tp'] = tp
                        gridsync = self.compute_plan['gridsync']
                        #print(f'Trying: pb={pb}, tp={tp}')
                        self.JIT_and_test_kernel()
                        # does kernel run without adjustment?
                        if self.compute_plan['tp'] != tp or self.compute_plan['gridsync'] != gridsync: 
                            break
                        #print('Seems to work, so looping over skins...')
                        min_time = 1e9
                        for skin in skins:
                            if skin>0:
                                self.compute_plan['skin'] = skin
                                self.update_params()
                                start = cuda.event()
                                end = cuda.event()
                                start.record()
                                self.configuration.copy_to_device() # By _not_ copying back to host later we dont change configuration
                                for i in range(repeats):
                                    self.integrate_self(0.0, timesteps)
                                end.record()
                                end.synchronize()
                                time_elapsed = cuda.event_elapsed_time(start, end)
                                if time_elapsed < min_time:
                                    min_time = time_elapsed
                                    min_skin = skin
                                skin_times.append(time_elapsed)
                                #print(self.compute_plan['tp'], skin, skin_times[-1])
                        max_TPS = repeats * timesteps / min_time * 1000
                        print(pb, tp, min_skin, min_time, max_TPS)
                    if min_time < total_min_time:
                        total_min_time = min_time
                        total_min_skin = min_skin
                        total_min_pb = pb
                        total_min_tp = tp    

        self.compute_plan['pb'] = total_min_pb
        self.compute_plan['tp'] = total_min_tp
        self.compute_plan['skin'] = total_min_skin
        print('Final compute_plan :', self.compute_plan)
        
        cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = flag



