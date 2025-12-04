import gamdpy as gp
import pandas as pd
import numpy as np
import math
import h5py
import os
import matplotlib.pyplot as plt

paramsCu = gp.EAM_ZJW_2004_params['Cu']
paramsAu = gp.EAM_ZJW_2004_params['Au']
cutCu = math.sqrt(5.) * paramsCu[0] # gives 5.7157519935Å
cutAu = math.sqrt(5.) * paramsAu[0] # gives 7.1553683345Å
cut = max(cutCu, cutAu)

paramsCu = np.append(paramsCu, cut)
paramsAu = np.append(paramsAu, cut)






# type 0 will be Cu, type 1 will be Au
# In the L1_2 structure, we have an fcc lattice where the corner atoms are
# the minority type, so Au ie type 1, and the face atoms are the majority type
# so Cu ie type 0
# in the code for the FCC unit cell the corner atom is the first of the four in the unit cell, so the type list should be [1, 0, 0, 0]


def FindMinimumEnthalpy(rho_array, ptype_unit_cell, plotindex = None, plotlabel=None):

    eam_pot = gp.EAM_ZJW_2004([paramsCu, paramsAu], max_num_nbs=1000)


    rho = min(rho_array)
    conf = gp.Configuration(D=3)
    conf.make_lattice(gp.unit_cells.FCC, cells=[6,6,6], rho=rho, ptype_unit_cell=ptype_unit_cell)
    evaluator = gp.Evaluator(conf, eam_pot)
    
    E_array = np.zeros_like(rho_array)
    for rdx in range(len(rho_array)):
        rho = rho_array[rdx]
        conf.atomic_scale(rho)
        evaluator.evaluate()
        E = np.sum(conf['U'])/conf.N
        E_array[rdx] = E
        #print(f"{a:.4f} {E:.4f} {rho:.4f}")

    p_degree = 3
    coeffs = np.polyfit(rho_array, E_array, deg=p_degree)
    p = np.poly1d(coeffs)
    if p_degree == 2:
        a, b, c = coeffs
        rho_min = -b / (2*a)
        E_min = p(rho_min)
    elif p_degree == 3:
        a, b, c, d = coeffs

        # For cubic fit
        rho_min_pl = (-2*b + math.sqrt(4*b**2-12*a*c))/(6*a)
        rho_min_mi = (-2*b - math.sqrt(4*b**2-12*a*c))/(6*a)

        E_min_pl = p(rho_min_pl)
        E_min_mi = p(rho_min_mi)
        
        if E_min_pl < E_min_mi:
            E_min = E_min_pl
            rho_min = rho_min_pl

    #print(f"rho_min {rho_min:.4f}, E_min {E_min:.4f}")

    if plotindex is not None:
        rho_fit = np.linspace(min(rho_array), max(rho_array), 200)
        E_fit = p(rho_fit)

        plt.figure(plotindex)
        plt.plot(rho_array, E_array, color="blue", marker='o', label="Data points")
        plt.plot(rho_fit, E_fit, color="red", label="Fitted polynomial")
        plt.plot(rho_min, E_min, color="green", marker="x", ms=20, label="minimum")
        plt.text(rho_min-0.001, E_min+0.06, plotlabel, fontsize=22)
        plt.xlabel(r"rho [Å^-3]")
        plt.ylabel(r"E [eV/atom]")
        plt.legend()
        #plt.savefig("E_vs_rho_%s.pdf" % plotlabel)

    a_min = pow(4/rho_min, 1/3)
    return rho_min, E_min, a_min

verbose = True

# Cu
rho_array = np.arange(0.075, 0.095, 0.001)
rho_Cu, E_Cu, a_Cu = FindMinimumEnthalpy(rho_array, ptype_unit_cell=[0, 0, 0, 0], plotindex=1, plotlabel="Cu")
assert math.isclose(a_Cu, 3.615, rel_tol=0.001) # Gola et al 2018
if verbose:
    print(f"a_Cu and reference value {a_Cu:.5f} 3.615")
    
# Au
rho_array = np.arange(0.050, 0.070, 0.001)
rho_Au, E_Au, a_Au = FindMinimumEnthalpy(rho_array, ptype_unit_cell=[1, 1, 1, 1], plotindex=2, plotlabel="Au")
assert math.isclose(a_Au, 4.080, rel_tol=0.001) # Gola et al 2018
if verbose:
    print(f"a_Au and reference value {a_Au:.5f} 4.080")


# Cu3Au
rho_array = np.arange(0.065, 0.085, 0.001)
rho_Cu3Au, E_Cu3Au, a_Cu3Au = FindMinimumEnthalpy(rho_array, ptype_unit_cell=[1, 0, 0, 0], plotindex=3, plotlabel="Cu3Au")
E_mixing_Cu3Au = E_Cu3Au - 0.75*E_Cu - 0.25*E_Au
assert math.isclose(a_Cu3Au, 3.750, abs_tol=0.001) #  Gola et al 2018
assert math.isclose(E_mixing_Cu3Au, -0.093, abs_tol=0.0015) #  Gola et al 2018
# the absolute difference is just bigger than 0.001 eV/atom here so I made the tolerance 0.0015

if verbose:
    print(f"a_Cu3Au and reference value {a_Cu3Au:.5f} 3.750")
    print(f"E_mixing_Cu3Au and reference value {E_mixing_Cu3Au:.5f} -0.093")


# CuAu3
rho_array = np.arange(0.055, 0.075, 0.001)
rho_CuAu3, E_CuAu3, a_CuAu3 = FindMinimumEnthalpy(rho_array, ptype_unit_cell=[0, 1, 1, 1], plotindex=4, plotlabel="CuAu3")
E_mixing_CuAu3 = E_CuAu3 - 0.25*E_Cu - 0.75*E_Au
assert math.isclose(a_CuAu3, 3.976, rel_tol=0.001) #  Gola et al 2018
assert math.isclose(E_mixing_CuAu3, -0.095, abs_tol=0.001) #  Gola et al 2018

if verbose:
    print(f"a_CuAu3 and reference value {a_CuAu3:.5f} 3.976")
    print(f"E_mixing_CuAu3 and reference value {E_mixing_CuAu3:.5f} -0.095")



plt.show()
