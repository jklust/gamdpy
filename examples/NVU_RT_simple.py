from matplotlib.backends.backend_pdf import PdfPages
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import rumdpy as rp
from numba import cuda


RHO = 1.2
NVE_EQ_INITIAL_TEMPERATURE = 1
NVE_DT = 0.005

DO_NVE_EQ = True
NVE_EQ_STEPS = 2**23
NVE_EQ_STEPS_PER_BLOCK = 2**18
NVE_EQ_SCALAR_OUTPUT = 64
NVE_EQ_OUTPUT = "nvu_LJ_test_nve_eq.h5"

DO_NVU_PROD = True
NVU_EQ_STEPS = 2**15
NVU_EQ_STEPS_PER_BLOCK = 2**10
NVU_EQ_SCALAR_OUTPUT = 8

NVU_PROD_STEPS = 2**15
NVU_PROD_STEPS_PER_BLOCK = 2**10
NVU_PROD_OUTPUT = "nvu_LJ_test_nvu_prod.h5"
NVU_PROD_SCALAR_OUTPUT = 8

DO_NVE_PROD = True
NVE_PROD_STEPS = 2*17
NVE_PROD_STEPS_PER_BLOCK = 2**10
NVE_PROD_OUTPUT = "nvu_LJ_test_nve_prod.h5"
NVE_PROD_SCALAR_OUTPUT = 8

OUTPUT_FIGS = "nvu_LJ_test_figs.pdf"


PAIR_F = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
SIG, EPS, CUTOFF = 1, 1, 2.5
PAIR_P = rp.PairPotential(PAIR_F, [SIG, EPS, CUTOFF], max_num_nbs=1000)


def test_nvu_reflection(
    verbose: bool = False, 
    plot_nve_eq_output: bool = False,
    plot_figures: bool = False, 
    show_figures: bool = False, 
    use_last_output: bool = False
) -> None:
    """ Test the nvu integrator that uses relfection on Omega """
    global dt, output
    # cuda.select_device(1)
    if use_last_output:
        if verbose:
            print(f"=========== USING OUTPUT {NVU_PROD_OUTPUT} ===========")
        output = rp.tools.load_output(NVU_PROD_OUTPUT)
        nblocks, nconfs, _, n, d = output["block"].shape
        conf = rp.Configuration(D=d, N=n)
        conf['m'] = 1
        conf.simbox = rp.Simbox(D=d, lengths=output["attrs"]["simbox_initial"])

        nve_eq_output = rp.tools.load_output(NVE_EQ_OUTPUT)
        u, = rp.extract_scalars(nve_eq_output, ["U"], first_block=NVE_EQ_STEPS//NVE_EQ_STEPS_PER_BLOCK//2)
        target_u = np.mean(u)
    else:
        nve_intg = rp.integrators.NVE(dt=NVE_DT)

        if DO_NVE_EQ:
            if verbose:
                print("=========== NVE EQ ===========")
            conf = rp.Configuration(D=3)
            conf.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=RHO)
            conf['m'] = 1
            conf.randomize_velocities(T=NVE_EQ_INITIAL_TEMPERATURE)

            sim = rp.Simulation(
                conf, PAIR_P, nve_intg, storage=NVE_EQ_OUTPUT,
                scalar_output=NVE_EQ_SCALAR_OUTPUT, num_timeblocks=NVE_EQ_STEPS//NVE_EQ_STEPS_PER_BLOCK, 
                steps_per_timeblock=NVE_EQ_STEPS_PER_BLOCK
            )
            sim.run(verbose=verbose)
        else:
            if verbose:
                print(f"=========== USING OUTPUT {NVE_EQ_OUTPUT} ===========")

        nve_eq_output = rp.tools.load_output(NVE_EQ_OUTPUT)
        _eq_nblocks, _eq_nconfs, _, n, d = nve_eq_output["block"].shape
        conf = rp.Configuration(D=d, N=n)
        conf['m'] = 1
        conf.simbox = rp.Simbox(D=d, lengths=nve_eq_output["attrs"]["simbox_initial"])
        conf["r"] = nve_eq_output["block"][-1, -1, 0, :, :]
        conf.randomize_velocities(T=NVE_EQ_INITIAL_TEMPERATURE)

        u, = rp.extract_scalars(nve_eq_output, ["U"], first_block=NVE_EQ_STEPS//NVE_EQ_STEPS_PER_BLOCK//2)
        target_u = np.mean(u)

        if plot_nve_eq_output:
            du_rel = (u - target_u) / abs(target_u)
            print(f"Mean potential energy after NVE (2^{np.log2(NVE_EQ_STEPS//2)} steps): {np.mean(u)}")
            print(f"Std potential energy after NVE (2^{np.log2(NVE_EQ_STEPS//2)} steps): {np.std(u)}")

            fig = plt.figure(figsize=(12, 7))
            ax = fig.add_subplot()
            ax.axhline(0, alpha=0.5)
            ax.plot(du_rel, linewidth=1, color='black', alpha=0.8)
            ax.grid()
            ax.set_title(r"relative u with respect to u0")
            ax.set_xlabel(r"$\frac{U - U_0}{|U_0|}$")

            fig = plt.figure(figsize=(12, 7))
            ax = fig.add_subplot()
            ax.axhline(0, alpha=0.5)
            ax.hist(du_rel, bins=30)
            ax.grid()
            ax.set_title(r"relative u with respect to u0")
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$\frac{U - U_0}{|U_0|}$")
            
            positions = nve_eq_output["block"][:, :, 0, :, :]
            cal_rdf = rp.CalculatorRadialDistribution(conf, num_bins=1000)
            for i in range(positions.shape[0]):
                pos = positions[i, -1, :, :]
                conf["r"] = pos
                conf.copy_to_device()
                cal_rdf.update()

            cal_rdf.save_average()
            rdf = cal_rdf.read()

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(rdf['distances'], np.mean(rdf['rdf'], axis=0), linewidth=1, color="black")
            ax.grid()
            ax.set_title(r"$g(r)$")
            ax.set_xlabel(r"$r$")
            ax.set_ylabel(r"$g(r)$")

            plt.show(block=False)

        if DO_NVE_PROD:
            if verbose: 
                print("=========== NVE PROD ===========")
            conf = rp.Configuration(D=d, N=n)
            conf['m'] = 1
            conf.simbox = rp.Simbox(D=d, lengths=nve_eq_output["attrs"]["simbox_initial"])
            conf["r"] = nve_eq_output["block"][-1, -1, 0, :, :]
            conf.randomize_velocities(T=NVE_EQ_INITIAL_TEMPERATURE)
            sim = rp.Simulation(
                conf, PAIR_P, nve_intg, 
                num_timeblocks=NVE_PROD_STEPS//NVE_PROD_STEPS_PER_BLOCK, 
                steps_per_timeblock=NVE_PROD_STEPS_PER_BLOCK,
                scalar_output=NVE_PROD_SCALAR_OUTPUT,
                storage=NVE_PROD_OUTPUT, 
            )

            for block in sim.timeblocks():
                if verbose: 
                    print(sim.status(per_particle=True))
            if verbose: 
                print(sim.summary())

        if DO_NVU_PROD:
            if verbose:
                print("=========== NVU EQ ===========")
            nvu_intg = rp.integrators.NVU_RT(
                target_u=target_u,
                max_abs_val=200,
                # In various runs i got no convergence because float64 starts 
                # numerical inaccuraciues in 2.5e-7 so i put eps to even lower
                # and make sure the threshold is above
                threshold=5e-7,
                eps=3e-7,
                max_steps=1000,
                max_initial_step_corrections=6,
                initial_step=0.0001,
                initial_step_if_high=0.001,
                step=0.0001,
                debug_print=True,
            )

            conf = rp.Configuration(D=d, N=n)
            conf['m'] = 1
            conf.simbox = rp.Simbox(D=d, lengths=nve_eq_output["attrs"]["simbox_initial"])
            conf["r"] = nve_eq_output["block"][-1, -1, 0, :, :]
            conf.randomize_velocities(T=NVE_EQ_INITIAL_TEMPERATURE)
            sim = rp.Simulation(
                conf, PAIR_P, nvu_intg, storage="memory",
                scalar_output=NVU_EQ_SCALAR_OUTPUT, 
                num_timeblocks=NVU_EQ_STEPS//NVU_EQ_STEPS_PER_BLOCK, 
                steps_per_timeblock=NVU_EQ_STEPS_PER_BLOCK, 
                save_time=True,
            )
            sim.run(verbose=verbose)

            if verbose: 
                print("=========== NVU PROD ===========")
            sim = rp.Simulation(
                conf, PAIR_P, nvu_intg, 
                num_timeblocks=NVU_PROD_STEPS//NVU_PROD_STEPS_PER_BLOCK, 
                steps_per_timeblock=NVU_PROD_STEPS_PER_BLOCK,
                scalar_output=NVU_PROD_SCALAR_OUTPUT,
                storage=NVU_PROD_OUTPUT, 
                save_time=True,
            )

            for block in sim.timeblocks():
                if verbose: 
                    print(sim.status(per_particle=True))
            if verbose: 
                print(sim.summary())

    if plot_figures:
        nvu_prod_output = rp.tools.load_output(NVU_PROD_OUTPUT)
        nve_prod_output = rp.tools.load_output(NVE_PROD_OUTPUT)
        nvu_positions = nvu_prod_output["block"][:, :, 0, :, :]
        nve_positions = nve_prod_output["block"][:, :, 0, :, :]
        nblocks, nconfs, _, n, d = nvu_prod_output["block"].shape

        cal_rdf = rp.CalculatorRadialDistribution(conf, num_bins=1000)
        for i in range(nvu_positions.shape[0]):
            pos = nvu_positions[i, -1, :, :]
            conf["r"] = pos
            conf.copy_to_device()
            cal_rdf.update()

        cal_rdf.save_average()
        nvu_rdf = cal_rdf.read()

        cal_rdf = rp.CalculatorRadialDistribution(conf, num_bins=1000)
        for i in range(NVU_EQ_STEPS//NVE_PROD_STEPS_PER_BLOCK, nve_positions.shape[0]):
            pos = nve_positions[i, -1, :, :]
            conf["r"] = pos
            conf.copy_to_device()
            cal_rdf.update()

        cal_rdf.save_average()
        nve_rdf = cal_rdf.read()

        u, dt, = rp.extract_scalars(nvu_prod_output, ["U", "dt"], first_block=0)
        du_rel = (u - target_u) / abs(target_u)
        step = np.arange(len(u)) * nvu_prod_output["steps_between_output"]

        fig = plt.figure(figsize=(10, 12))
        ax0 = fig.add_subplot(2, 1, 1)
        ax1 = fig.add_subplot(2, 1, 2)
        ax0.semilogy(step, du_rel, linewidth=0, marker='o', color="black")
        ax0.set_xlabel(r"$step$")
        ax0.set_ylabel(r"$\frac{U - U_0}{|U_0|}$")
        ax0.grid()
        ax1.hist(du_rel, bins=30, color="black", alpha=.8)
        ax1.set_xlabel(r"$(U - U_0) / |U_0|$")
        ax1.grid()
        fig.suptitle(r"Relative potential energy in each step NVU")

        fig = plt.figure(figsize=(10, 12))
        ax0 = fig.add_subplot(2, 1, 1)
        ax1 = fig.add_subplot(2, 1, 2, sharex=ax0, sharey=ax0)
        ax0.plot(nvu_rdf['distances'], np.mean(nvu_rdf['rdf'], axis=0), linewidth=1, color="black")
        ax0.grid()
        ax0.set_title(r"NVU $g(r)$")
        ax0.set_xlabel(r"$r$")
        ax0.set_ylabel(r"$g(r)$")
        ax1.plot(nve_rdf['distances'], np.mean(nve_rdf['rdf'], axis=0), linewidth=1, color="black")
        ax1.grid()
        ax1.set_title(r"NVE $g(r)$")
        ax1.set_xlabel(r"$r$")
        ax1.set_ylabel(r"$g(r)$")

        nvu_msd = rp.tools.calc_dynamics(nvu_prod_output, first_block=0)["msd"][:, 0]
        nve_msd = rp.tools.calc_dynamics(nve_prod_output, first_block=NVU_EQ_STEPS//NVE_PROD_STEPS_PER_BLOCK)["msd"][:, 0]
        nvu_time = nvu_prod_output["attrs"]['dt'] * 2 ** np.arange(len(nvu_msd))
        nve_time = nvu_prod_output["attrs"]['dt'] * 2 ** np.arange(len(nve_msd))

        fig = plt.figure(figsize=(10, 12))
        ax0 = fig.add_subplot(2, 1, 1)
        ax1 = fig.add_subplot(2, 1, 2, sharex=ax0, sharey=ax0)
        ax0.loglog(nvu_time, nvu_msd, linewidth=1, color="black")
        ax0.grid()
        ax0.set_title(r"NVU $MSD$")
        ax0.set_xlabel(r"$t$")
        ax0.set_ylabel(r"$MSD$")
        ax1.loglog(nve_time, nve_msd, linewidth=1, color="black")
        ax1.grid()
        ax1.set_title(r"NVE $MSD$")
        ax1.set_xlabel(r"$t$")
        ax1.set_ylabel(r"$MSD$")

        step_conf = np.concatenate([NVU_PROD_STEPS_PER_BLOCK * i + 2 ** np.concatenate([[0], np.arange(nconfs-1)]) for i in range(nblocks)])
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        ax.plot(step_conf, nvu_prod_output["time"].flatten(), color="black", alpha=.8)
        ax.grid()
        ax.set_title(r"$t(step)$")
        ax.set_xlabel(r"$step$")
        ax.set_ylabel(r"$t$")

        fig = plt.figure(figsize=(10, 12))
        ax0 = fig.add_subplot(2, 1, 1)
        ax1 = fig.add_subplot(2, 1, 2)
        ax0.plot(step, dt, linewidth=1, color="black", alpha=.8)
        ax0.set_ylabel(r"$\Delta t$")
        ax0.set_xlabel(r"$step$")
        ax0.grid()
        ax1.hist(dt, bins=20, color="black", alpha=.8)
        ax1.set_xlabel(r"$\Delta t$")
        ax1.grid()
        fig.suptitle(r"$\Delta t$")

    print(f"Saving figures to {OUTPUT_FIGS}")
    pages = PdfPages(OUTPUT_FIGS)
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pages, format='pdf')
    pages.close()

    if show_figures:
        plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(
        "Run test for NVU Ray Tracing simulation with initial coonfiguration form an "
        f"NVE simulation ({NVE_EQ_STEPS} steps) and equilibrium stage ({NVU_EQ_STEPS}). "
        f"Production steps: {NVU_PROD_STEPS}."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-e", "--plot-nve-eq-output", action="store_true")
    parser.add_argument("-p", "--plot-figures", action="store_true")
    parser.add_argument("-s", "--show-figures", action="store_true")
    parser.add_argument("-u", "--use-last-output", action="store_true", help="Reuses last output run using and does not run any simulations")
    parser.add_argument("-d", "--device", help="Choose nvidia device", default=0, type=int)
    args = parser.parse_args()
    cuda.select_device(args.device)
    test_nvu_reflection(
        verbose=args.verbose, 
        plot_nve_eq_output=args.plot_nve_eq_output,
        plot_figures=args.plot_figures, 
        show_figures=args.show_figures, 
        use_last_output=args.use_last_output,
    )


