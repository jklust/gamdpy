import rumdpy as rp

# Set up a 3d configuration for a single component system
configuration_SC = rp.Configuration(D=3, compute_flags={'W':True})
configuration_SC.make_positions(N=1000, rho=0.754)
configuration_SC['m'] = 1.0  # Set all masses to 1.0
configuration_SC.randomize_velocities(temperature=2.0)

# Set up a LJ pair potential
pairfunc = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pairpot_LJ = rp.PairPotential(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

