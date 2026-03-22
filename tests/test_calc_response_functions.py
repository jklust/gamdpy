import gamdpy as gp
import numpy as np
import pytest

def test_calculate_response_functions_NpT():

    np.random.seed(2026)  # Set seed for reproducibility
    n = 10_000  # Number of datapoints in synthetic data

    out_arg = gp.tools.calculate_response_functions_NpT(
        1254,
        1251,
        2449+np.random.randn(n)*4.1,
        1244+np.random.randn(n)*2.2,
        3674+np.random.randn(n)*6.5,
        1000+np.random.randn(n)*8.2,
        1.23,
        0.876,
        3.432,
        False
    )

    out = gp.tools.calculate_response_functions_NpT(
        N=1254,
        dof=1251,
        U=2449+np.random.randn(n)*4.1,
        W=1244+np.random.randn(n)*2.2,
        K=3674+np.random.randn(n)*6.5,
        Vol=1000+np.random.randn(n)*8.2,
        k_B=1.23,
        T_ext=0.876,
        p_ext=3.432,
        per_particle=False
    )
    assert type(out) == dict

    # print(out)
    ## Generated with above
    expected = {
        'density': 1.2541207459863282, 'specific_volume': 0.7973713880424887,
        'potential_energy': 1.9529325489435667, 'kinetic_energy': 2.9298631269371653,
        'internal_energy': 4.8827956758807325, 'external_temperature': 0.876,
        'kinetic_temperature': 4.775429557075258, 'external_pressure': 3.432,
        'pressure': 8.61112915846658, 'compressibility_factor': 2.5397952665124377,
        'enthalpy': 7.619374279642554, 'isobaric_heat_capacity': 0.719717102182806,
        'isothermal_compressibility': 0.061953206277420306, 'isothermal_bulk_modulus': 16.141214637416816,
        'isobaric_expansion_coefficient': 0.24365934735003494, 'isochoric_heat_capacity': 0.05034575249164963,
        'isochoric_heat_capacity_excess': -0.5631829556423217, 'adiabatic_index': 14.295488031531134,
        'adiabatic_compressibility': 0.004333759444992151, 'adiabatic_bulk_modulus': 230.74654066356723,
        'thermal_pressure_coefficient': 3.9329578239898124, 'adiabatic_pressure_coefficient': 4.228769291358482,
        'adiabatic_expansion_coefficient': -0.018326468857117556, 'thermodynamic_gruneisen_parameter': 62.289823550605696,
        'joule_thomson_coefficient': -0.8714201470870574}

    for key in expected.keys():
        pytest.approx(expected[key]) == out[key], f"Key {key} not equal to expected values"

    for key in out.keys():
        assert key in out_arg.keys()
        pytest.approx(out_arg[key]) == out[key]

    # print(out.keys())
    expected_keys = [
        'density', 'specific_volume', 'potential_energy', 'kinetic_energy', 'internal_energy',
        'external_temperature', 'kinetic_temperature', 'external_pressure', 'pressure', 'compressibility_factor',
        'enthalpy', 'isobaric_heat_capacity', 'isothermal_compressibility', 'isothermal_bulk_modulus', 'isobaric_expansion_coefficient',
        'isochoric_heat_capacity', 'isochoric_heat_capacity_excess', 'adiabatic_index', 'adiabatic_compressibility', 'adiabatic_bulk_modulus',
        'thermal_pressure_coefficient', 'adiabatic_pressure_coefficient', 'adiabatic_expansion_coefficient', 'thermodynamic_gruneisen_parameter',
        'joule_thomson_coefficient']

    for key in expected_keys:
        assert key in out.keys(), f"Key {key} not found in output of calculate_response_functions_NpT"

if __name__ == "__main__":
    test_calculate_response_functions_NpT()
