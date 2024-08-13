import pytest

@pytest.mark.slow
def test_JIT():
        import rumdpy as rp
        import numpy as np
        
        # Generate configurations with a FCC lattice
        configuration1 = rp.make_configuration_fcc(nx= 8, ny= 8, nz=8,  rho=0.8442)
        configuration1.randomize_velocities(T=1.44, seed=1234)
        configuration2 = rp.make_configuration_fcc(nx= 5, ny= 5, nz=13, rho=1.2000)
        configuration2.randomize_velocities(T=0.44, seed=4123)
        configuration3 = rp.make_configuration_fcc(nx=16, ny=16, nz=32, rho=0.8442)
        configuration3.randomize_velocities(T=2.44, seed=3412)

        # Make pair potentials
        pairfunc = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
        sig, eps, cut = 1.0, 1.0, 2.5
        pairpot1 = rp.PairPotential(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

        pairfunc = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
        sig = [[1.00, 0.80],
               [0.80, 0.88]]
        eps = [[1.00, 1.50],
               [1.50, 0.50]]
        cut = np.array(sig)*2.5
        pairpot2 = rp.PairPotential(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

        # Make integrators
        dt = 0.001 # timestep. Conservative choice
        temperature = 0.7 # Not used for NVE
        pressure = 1.2 # Not used for NV*

        integrators = [rp.integrators.NVE(dt=dt),
                       rp.integrators.NVE_Toxvaerd(dt=dt), 
                       rp.integrators.NVT(temperature=temperature, tau=0.2, dt=dt), 
                       rp.integrators.NVT_Langevin(temperature=temperature, alpha=0.2, dt=dt, seed=2023), 
                       rp.integrators.NPT_Atomic  (temperature=temperature, tau=0.4, pressure=pressure, tau_p=20, dt=dt),
                       rp.integrators.NPT_Langevin(temperature=temperature, pressure=pressure, 
                                                    alpha=0.1, alpha_baro=0.0001, mass_baro=0.0001, 
                                                    volume_velocity=0.0, barostatModeISO = True , boxFlucCoord = 2,
                                                    dt=dt, seed=2023)]

        for configuration in [configuration1, configuration2, configuration3]:
            print("conf #:", [configuration1, configuration2, configuration3].index(configuration))
            for pairpot in [pairpot1, pairpot2]:
                print("pairpot #:", [pairpot1, pairpot2].index(pairpot))
                ev = rp.Evaluater(configuration, pairpot)
                for integrator in integrators:
                    print(integrator)
                    sim = rp.Simulation(configuration, pairpot, integrator,
                                        num_timeblocks=2, steps_per_timeblock=1024, 
                                        steps_between_momentum_reset=100,
                                        storage='memory')
                    print(sim.compute_plan, "\n")
                    sim.output.close()
                    
if __name__ == '__main__':
     test_JIT()

