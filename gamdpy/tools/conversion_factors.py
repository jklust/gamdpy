""" Function that return dictionary with conversion factors from reduced units to real units """
from scipy.constants import Boltzmann


def conversion_factors(**kwargs):
    """ Function that return dictionary with conversion factors from reduced units to real units

    Parameters
    ----------
    **kwargs: Keyword arguments with unit length, energy and mass.

    Examples
    --------

    >>> from pprint import pprint
    >>> from gamdpy.tools import conversion_factors
    >>> cf_SI = conversion_factors(unit_length=1.0, unit_energy=1.0, unit_mass=1.0)  # Standard SI units
    >>> cf_SI = conversion_factors()  # Standard SI units (same as above)
    >>> cf_Argon = conversion_factors(unit_length_in_Angstrom=3.4, unit_energy_in_K=120.0, unit_mass_in_u = 39.948) # Argon units
    >>> cf_cgs = conversion_factors(unit_length_in_cm=1.0, unit_energy_in_erg=1.0, unit_mass_in_g=1.0)  # centimetre–gram–second units
    >>> cf_molar = conversion_factors(unit_length_in_nm=1.0, unit_energy_in_kJ_per_mol=1.0, unit_mass_in_u = 1.0)  # Molar units
    >>> cf_cal = conversion_factors(unit_length_in_nm=1.0, unit_energy_in_kcal_per_mol=1.0, unit_mass_in_u = 1.0)  # Calometric molar units
    >>> cf_metal = conversion_factors(unit_length_in_Angstrom=1.0, unit_energy_in_eV=1.0, unit_mass_in_u = 1.0)  # Metallic units
    >>> pprint(conversion_factors(get_possible_inputs=True))  # name, conversion_factor_to_SI_unit
    {'unit_energy': 1.0,
     'unit_energy_in_K': 1.380649e-23,
     'unit_energy_in_Kelvin': 1.380649e-23,
     'unit_energy_in_eV': 1.602176634e-19,
     'unit_energy_in_erg': 1e-07,
     'unit_energy_in_hartree': 4.359744722206e-18,
     'unit_energy_in_kJ_per_mol': 1.6605390671738467e-21,
     'unit_energy_in_kcal_per_mol': 6.947695457055374e-21,
     'unit_energy_in_zeptojoule': 1e-21,
     'unit_length': 1.0,
     'unit_length_in_AU': 149597870700.0,
     'unit_length_in_Angstrom': 1e-10,
     'unit_length_in_bohr_radius': 5.29177210544e-11,
     'unit_length_in_cm': 0.01,
     'unit_length_in_nm': 1e-09,
     'unit_length_in_parsec': 3.085677581491367e+16,
     'unit_length_in_Å': 1e-10,
     'unit_mass': 1.0,
     'unit_mass_in_amu': 1.66053906892e-27,
     'unit_mass_in_attograms': 1e-21,
     'unit_mass_in_g': 0.001,
     'unit_mass_in_gram_per_mol': 1.6605390671738466e-27,
     'unit_mass_in_u': 1.66053906892e-27}
    """

    # Check that one, and only one energy, length and mass is given
    for prefix in 'unit_length', 'unit_energy', 'unit_mass':
        matches = [k for k in kwargs if k.startswith(prefix)]
        if len(matches) > 1:
            raise ValueError(f'Expected only one {prefix} key, but got {len(matches)}: {matches}')

    # Fundamental constants (same format as scipy.constants)
    Avogadro = 6.02214076e+23  # 1 / mol
    atomic_mass = 1.66053906892e-27  # kg
    Boltzmann = 1.380649e-23  # m² · kg / ( s² · K ) = J / K
    calorie = 4.184  # J
    electron_volt = 1.602176634e-19  # J
    astronomical_unit = 149597870700.0  # m
    parsec = 3.085677581491367e+16  # m
    atomic_mass = 1.66053906892e-27  # kg
    solar_masses = 1.988416e30  # kg
    hartree = 4.3597447222060e-18  # J
    bohr_radius = 5.29177210544e-11 # m

    # What unit one in reduced units corresponds to in SI units
    unit_length, unit_energy, unit_mass = None, None, None
    if not kwargs:
        unit_length, unit_energy, unit_mass = 1.0, 1.0, 1.0  # Assume SI units of no kwargs are given

    possible_lengths = {
        # Name         # Conversion factor to SI
        'unit_length': 1.0,
        'unit_length_in_cm': 1e-2,
        'unit_length_in_nm': 1e-9,
        'unit_length_in_Å': 1e-10,
        'unit_length_in_Angstrom': 1e-10,
        'unit_length_in_bohr_radius': bohr_radius,
        'unit_length_in_AU': astronomical_unit,
        'unit_length_in_parsec': parsec
    }
    for key, val in possible_lengths.items():
        if key in kwargs:
            unit_length = kwargs[key]*val

    possible_energies = {
        # Name         # Conversion factor to SI
        'unit_energy': 1.0,
        'unit_energy_in_K': Boltzmann,
        'unit_energy_in_Kelvin': Boltzmann,
        'unit_energy_in_kJ_per_mol': 1e3 / Avogadro,
        'unit_energy_in_kcal_per_mol': 1e3 * calorie / Avogadro,
        'unit_energy_in_erg': 1e-7,
        'unit_energy_in_zeptojoule': 1e-21,
        'unit_energy_in_hartree': hartree,
        'unit_energy_in_eV': electron_volt,
    }
    for key, val in possible_energies.items():
        if key in kwargs:
            unit_energy = kwargs[key]*val

    possible_masses = {
        # Name         # Conversion factor to SI
        'unit_mass': 1.0,
        'unit_mass_in_g': 1e-3,
        'unit_mass_in_u': atomic_mass,
        'unit_mass_in_amu': atomic_mass,
        'unit_mass_in_gram_per_mol': 1e-3 / Avogadro,
        'unit_mass_in_attograms': 1e-21
    }
    for key, val in possible_masses.items():
        if key in kwargs:
            unit_mass = kwargs[key]*val

    # Return a dict with possible inputs and their
    if 'get_possible_inputs' in kwargs:
        if kwargs['get_possible_inputs']:
            return possible_lengths | possible_energies | possible_masses

    # Check that a length, energy and mass have been given in keyword arguments
    if unit_length == None or unit_energy == None or unit_mass == None:
        raise KeyError(f'Some unit_length, unit_energy and unit_mass needs to be specified. Got {kwargs}')

    # Derived units. N.B. Below we assume D=3 spatial dimensions
    unit_time = unit_length * (unit_mass / unit_energy) ** 0.5  # s
    unit_pressure = unit_energy / unit_length ** 3  # Pa
    unit_density = unit_mass / unit_length ** 3  # kg/m^3

    return {
        # Length
        "unit_length": unit_length,  # m
        "in_parsec": unit_length / parsec,
        "AU": unit_length / astronomical_unit,
        "km": unit_length * 1e-3,
        "m": unit_length,
        "cm": unit_length * 1e2,
        "mm": unit_length * 1e3,
        "micrometer": unit_length * 1e6,
        "μm": unit_length * 1e6,
        "micron": unit_length * 1e6,
        "nm": unit_length * 1e9,
        "Å": unit_length * 1e10,
        "Angstrom": unit_length * 1e10,
        "Ångstrom": unit_length * 1e10,
        "Bohr_radius": unit_length / bohr_radius,

        # Energy
        "unit_energy": unit_energy,  # J
        "J": unit_energy,
        "zeptojoule": unit_energy*1e21,
        "zJ": unit_energy*1e21,
        "kJ": unit_energy * 1e-3,
        "kJ/mol": unit_energy * 1e-3 * Avogadro,
        "cal": unit_energy / calorie,
        "kcal": unit_energy * 1e-3 / calorie,
        "kcal/mol": unit_energy * 1e-3 / calorie * Avogadro,
        "K": unit_energy / Boltzmann, # K
        "Kelvin": unit_energy / Boltzmann,
        "electron_volt": unit_energy / electron_volt,
        "eV": unit_energy / electron_volt,
        "keV": unit_energy / electron_volt * 1e-3,
        "MeV": unit_energy / electron_volt * 1e-6,
        "GeV": unit_energy / electron_volt * 1e-9,
        "erg":  unit_energy * 1e7,
        "hartree": unit_energy / hartree,

        # Mass
        "unit_mass": unit_mass,  # kg
        "solar_masses": unit_mass/solar_masses,
        "M☉": unit_mass/solar_masses,
        "kg": unit_mass,
        "gram": unit_mass*1e3,
        "g": unit_mass*1e3,
        "mg": unit_mass*1e6,
        "mikrogram": unit_mass*1e9,
        "μg": unit_mass*1e9,
        "nanogram": unit_mass*1e12,
        "ng": unit_mass*1e12,
        "picogram": unit_mass*1e15,
        "pg": unit_mass*1e15,
        "attogram": unit_mass*1e21,
        "ag": unit_mass*1e21,
        "u": unit_mass/atomic_mass,
        "dalton": unit_mass/atomic_mass,
        "kg/mol": unit_mass * Avogadro,
        "g/mol": unit_mass * 1e3 * Avogadro,

        # Time
        'unit_time': unit_time,  # s
        'years': unit_time / 31556926,
        'weeks': unit_time / 604800,
        'days': unit_time / 86400,
        'hours': unit_time / 3600,
        'minutes': unit_time / 60,
        's': unit_time,
        'seconds': unit_time,
        'milliseconds': unit_time * 1e3,
        'ms': unit_time * 1e3,
        'microseconds': unit_time * 1e6,
        'μs': unit_time * 1e6,
        'nanoseconds': unit_time * 1e9,
        'ns': unit_time * 1e9,
        'picoseconds': unit_time * 1e12,
        'ps': unit_time * 1e12,
        'femtoseconds': unit_time * 1e15,
        'fs': unit_time * 1e15,

        # Pressure
        'unit_pressure': unit_pressure,
        'Pascal': unit_pressure,
        'pa': unit_pressure,
        'hectopascal': unit_pressure * 1e-2,
        'kilopascal': unit_pressure * 1e-3,
        'kpa': unit_pressure * 1e-3,
        'megapascal': unit_pressure * 1e-6,
        'MPa': unit_pressure * 1e-6,
        'gigapascal': unit_pressure * 1e-9,
        'GPa': unit_pressure * 1e-9,
        'bar': unit_pressure * 1e-5,
        'mbar': unit_pressure * 1e-2,
        'psi': unit_pressure / 6894.757293168361,
        'torr': unit_pressure / 133,
        'atm': unit_pressure / 101325,
        'atmospheres': unit_pressure / 101325,

        # Density (3D)
        'unit_density': unit_density,  # kg/m³
        'kg/m3': unit_density,
        'g/ml': unit_density * 1e-3,
        'g/cm3': unit_density * 1e-3,
        'grams_per_cubic_centimeter': unit_density * 1e-3,

    }