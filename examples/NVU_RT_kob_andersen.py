"""
Application of the NVU Ray Tracing integrator.
Comparision with NVE inspired by article:
`NVU dynamics. II. Comparing to four other dynamics.`
"""
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import rumdpy as rp
from numba import cuda

RHO = 1.2
TEMPERATURE = .4
SEED = 0

DO_NVE_EQ = False
NVE_EQ_STEPS = 2**20
NVE_EQ_STEPS_PER_TIMEBLOCK = 2**15
NVE_EQ_OUTPUT = "examples/data/NVU_RT_NVE_EQ_OUTPUT.h5"
NVE_DT = .005

DO_NVE_PROD = False
NVE_PROD_STEPS = 2**25
NVE_PROD_STEPS_PER_TIMEBLOCK = 2**15
NVE_PROD_OUTPUT = "examples/data/NVU_RT_NVE_PROD_OUTPUT.h5"

DO_NVU_PROD = True
NVU_PROD_STEPS = 2**25
NVU_PROD_STEPS_PER_TIMEBLOCK = 2**18
NVU_EQ_BLOCKS = NVU_PROD_STEPS//NVU_PROD_STEPS_PER_TIMEBLOCK//3
NVU_SCALAR_OUTPUT = 64
NVU_PROD_OUTPUT = "examples/data/NVU_RT_NVU_PROD_OUTPUT.h5"


def run_simulations():
    # Setup configuration: FCC crystal
    conf = rp.Configuration(D=3)
    conf.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=RHO)
    conf['m'] = 1.0
    conf.randomize_velocities(T=TEMPERATURE)
    conf.ptype[::5] = 1     # Every fifth particle set to type 1 (4:1 mixture)
    #configuration['r'][27,2] += 0.01 # Perturb z-coordinate of particle 27

    # Setup pair potential: Binary Kob-Andersen LJ mixture.
    pair_func = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
    sig = [[1.00, 0.80],
           [0.80, 0.88]]
    eps = [[1.00, 1.50],
           [1.50, 0.50]]
    # sig, eps = 1, 1
    cut = np.array(sig)*2.5
    pair_pot = rp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

    nve_integrator = rp.integrators.NVE(dt=NVE_DT)
    if DO_NVE_EQ:
        print(f"========== NVE EQ ({NVE_EQ_STEPS//NVE_EQ_STEPS_PER_TIMEBLOCK} blocks) ==========")

        sim = rp.Simulation(
            conf, pair_pot, nve_integrator,
            num_timeblocks=NVE_EQ_STEPS//NVE_EQ_STEPS_PER_TIMEBLOCK, 
            steps_per_timeblock=NVE_EQ_STEPS_PER_TIMEBLOCK,
            storage=NVE_EQ_OUTPUT)

        for block in sim.timeblocks():
            if block % 10 == 0:
                print(f'{block=:4}  {sim.status(per_particle=True)}')
        print(sim.summary())

    nve_eq_output = rp.tools.load_output(NVE_EQ_OUTPUT)
    _eq_nblocks, _eq_nconfs, _, n, d = nve_eq_output["block"].shape

    if DO_NVE_PROD:
        print(f"========== NVE PROD ({NVE_PROD_STEPS//NVE_PROD_STEPS_PER_TIMEBLOCK} blocks) ==========")
        conf = rp.Configuration(D=d, N=n)
        conf['m'] = 1
        conf.simbox = rp.Simbox(D=d, lengths=nve_eq_output["attrs"]["simbox_initial"])
        conf["r"] = nve_eq_output["block"][-1, -1, 0, :, :]
        conf.randomize_velocities(T=TEMPERATURE, seed=SEED)

        sim = rp.Simulation(
            conf, pair_pot, nve_integrator,
            num_timeblocks=NVE_PROD_STEPS//NVE_PROD_STEPS_PER_TIMEBLOCK, 
            steps_per_timeblock=NVE_PROD_STEPS_PER_TIMEBLOCK,
            storage=NVE_PROD_OUTPUT)

        for block in sim.timeblocks():
            if block % 5 == 0:
                print(f'{block=:4}  {sim.status(per_particle=True)}')
        print(sim.summary())

    nvu_integrator = rp.integrators.NVU_RT(
        max_abs_val=200,
        # In various runs i got no convergence because float64 starts 
        # numerical inaccuraciues in 2.5e-7 so i put eps to even lower
        # and make sure the threshold is above
        threshold=5e-7,
        eps=3e-7,
        max_steps=1000,
        max_initial_step_corrections=6,
        initial_step=0.0001,
        step=0.0001,
        debug_print=False,
    )

    if DO_NVU_PROD:
        print(f"========== NVU PROD ({NVU_PROD_STEPS//NVU_PROD_STEPS_PER_TIMEBLOCK} blocks) ==========")
        conf = rp.Configuration(D=d, N=n)
        conf['m'] = 1
        conf.simbox = rp.Simbox(D=d, lengths=nve_eq_output["attrs"]["simbox_initial"])
        conf["r"] = nve_eq_output["block"][-1, -1, 0, :, :]
        conf.randomize_velocities(T=TEMPERATURE, seed=SEED)

        sim = rp.Simulation(
            conf, pair_pot, nvu_integrator,
            num_timeblocks=NVU_PROD_STEPS//NVU_PROD_STEPS_PER_TIMEBLOCK, 
            steps_per_timeblock=NVU_PROD_STEPS_PER_TIMEBLOCK,
            storage=NVU_PROD_OUTPUT,
            scalar_output=NVU_SCALAR_OUTPUT,
            save_time=True,
        )

        for block in sim.timeblocks():
            if block % 5 == 0:
                print(f'{block=:4}  {sim.status(per_particle=True)}')
        print(sim.summary())


def plot_output(output_path, name, steps, first_block=0):
    output = rp.tools.load_output(output_path)
    positions = output["block"][:, :, 0, :, :]
    nblocks, nconfs, _, n, d = output["block"].shape
    conf = rp.Configuration(D=d, N=n)
    conf['m'] = 1
    conf.simbox = rp.Simbox(D=d, lengths=output["attrs"]["simbox_initial"])

    cal_rdf = rp.CalculatorRadialDistribution(conf, num_bins=1000)
    for i in range(positions.shape[0]):
        if i < first_block:
            continue
        pos = positions[i, -1, :, :]
        conf["r"] = pos
        conf.copy_to_device()
        cal_rdf.update()

    cal_rdf.save_average()
    rdf = cal_rdf.read()

    msd = rp.tools.calc_dynamics(output, first_block=first_block)["msd"][:, 0]

    fig = plt.figure(figsize=(5, 5))
    fig.suptitle(rf"{name} 2^{np.log2(steps)} steps")
    ax = fig.add_subplot()
    ax.plot(rdf['distances'], np.mean(rdf['rdf'], axis=0), linewidth=1, color="black")
    ax.grid()
    ax.set_title(rf"$g(r)$")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$g(r)$")

    time = output["attrs"]['dt'] * 2 ** np.arange(len(msd))

    fig = plt.figure(figsize=(5, 5))
    fig.suptitle(rf"{name} 2^{np.log2(steps)} steps")
    ax = fig.add_subplot()
    ax.loglog(time, msd, linewidth=1, color="black")
    ax.grid()
    ax.set_title(rf"$MSD$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$MSD$")

    if name == "NVU":

        dt, = rp.extract_scalars(output, ["dt"], first_block=first_block)

        fig = plt.figure(figsize=(5, 5))
        fig.suptitle(rf"{name} 2^{np.log2(steps)} steps")
        ax = fig.add_subplot()
        ax.hist(dt, bins=20, linewidth=1, color="black")
        ax.grid()
        ax.set_title(r"$\Delta t$")
        ax.set_xlabel(r"$\Delta t$")

    plt.show(block=False)


if __name__ == "__main__":
    run_simulations()
    plot_output(NVE_EQ_OUTPUT, "NVE EQ", NVE_EQ_STEPS, first_block=NVE_EQ_STEPS//NVE_EQ_STEPS_PER_TIMEBLOCK//2)
    plot_output(NVE_PROD_OUTPUT, "NVE", NVE_PROD_STEPS)
    plot_output(NVU_PROD_OUTPUT, "NVU", NVU_PROD_STEPS, first_block=NVU_EQ_BLOCKS)
    plt.show()
