""" In this example we use the Widom's particle insertion method
to calculate the chemical potential of a Lennard-Jones fluid.

The excess chemical potential, μᵉˣ, of a fluid is given by:

  μᵉˣ = -kT ln 〈exp(-ΔU/kT)〉

where ΔU_i is the energy difference between the system with and without a ghost particle.

"""

import numpy as np

import rumdpy as rp


###########################################
#  Widom's particle insertion calculator  #
###########################################


class CalculatorWidomsParticleInsertion:
    def __init__(self,
                 simulation: rp.Simulation,
                 ):
        self.sim = simulation
        self.insertions_per_update = 100  # Number of insertions for each update
        self.ptype_ghost = 0  # The ghost particle of this type

        # Expect an NVT integrator with a target temperature
        if not hasattr(sim.integrator, 'temperature'):
            raise AttributeError('The integrator object must have a temperature attribute. Pick an NVT integrator')
        self.temperature = sim.integrator.temperature

        self.sum_exp_du: float = 0.0
        self.num_insertions: int = 0


    def update(self):
        # Raise error if more than one interaction
        if len(self.sim.interactions) != 1:
            raise AttributeError('The simulation must have a pair interaction')
        pair_pot: rp.PairPotential = self.sim.interactions[0]  # Assume only one pair potential
        # Assert that it is a pair potential
        if not isinstance(pair_pot, rp.PairPotential):
            raise AttributeError('The pair potential must be a PairPotential object')
        pair_pot_func = pair_pot.pairpotential_function

        lengths = self.sim.configuration.simbox.lengths
        D = self.sim.configuration.D
        N = self.sim.configuration.N
        random_positions = np.random.rand(self.insertions_per_update, D) * lengths
        params, max_cut = pair_pot.convert_user_params()
        func_dr2 = self.sim.configuration.simbox.dist_sq_function
        for ghost_pos in random_positions:
            this_u = 0.0
            for n in range(N):
                ptype = self.sim.configuration.ptype[n]
                r = self.sim.configuration.vectors['r'][n]
                dr2 = func_dr2(r, ghost_pos, lengths)
                dr = np.sqrt(dr2)
                if dr < max_cut:
                    this_u += pair_pot_func(dr, params[ptype, self.ptype_ghost])[0]
            self.sum_exp_du += np.exp(-this_u / self.temperature)  # Update the sum of exp(-ΔU/kT) for the current configuration
            self.num_insertions += 1

    def read(self) -> float:
        return -np.log(self.sum_exp_du / self.num_insertions)


##################################
#  Use calculator in simulation  #
##################################

# Setup configuration: FCC Lattice
configuration = rp.Configuration(D=3)
configuration.make_lattice(rp.unit_cells.FCC, cells=[5, 5, 5], rho=0.1)
configuration['m'] = 1.0
configuration.randomize_velocities(T=0.7)

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = rp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator: NVT
integrator = rp.integrators.NVT(temperature=1.0, tau=0.2, dt=0.005)

# Setup Simulation
sim = rp.Simulation(configuration, pair_pot, integrator,
                    steps_between_momentum_reset=100,
                    num_timeblocks=2,
                    steps_per_timeblock=128,
                    storage='memory'
                    )

# Equilibrate the system
sim.run()

# Setup the Widom's particle insertion calculator
calc_widom = CalculatorWidomsParticleInsertion(sim)
for block in sim.timeblocks():
    print(f'Block {block}')
    calc_widom.update()
    print(f'Excess chemical potential: {calc_widom.read():.3f}')

mu_ex = calc_widom.read()
print(f'Excess chemical potential: {mu_ex:.3f}')
