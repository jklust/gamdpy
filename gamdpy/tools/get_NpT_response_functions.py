import numpy as np

def get_NpT_response_functions(N, dof, V, U, W, K, k_B=1.0, T_ext=None, p_ext=None):
    """ Compute thermodynamic response functions of an isotropic NpT simulation from thermal equilibrium fluctuations
    EXPERIMENTAL """
    if not all(np.isscalar(x) for x in (N, dof, k_B)):
        raise TypeError("N, dof and k_B must be scalars")
    for name, value in (("V", V), ("U", U), ("W", W), ("K", K)):
        if np.isscalar(value):
            raise TypeError(f"{name} must be array-like, not a scalar")
        try:
            np.asarray(value)
        except Exception as exc:
            raise TypeError(f"{name} must be array-like") from exc

    output = {}

    # Density
    mV = np.mean(V)
    rho = N/mV
    output.update(dict(density=float(rho)))
    output.update(dict(specific_volume=float(1/rho)))

    # Energies
    mU = np.mean(U)
    output.update(dict(potential_energy=float(mU / N)))
    mK = np.mean(K)
    output.update(dict(kinetic_energy=float(mK / N)))
    E = U + K
    mE = np.mean(E)
    output.update(dict(internal_energy=float(mE / N)))
    T_inst = 2.0 * K / (k_B * dof)
    mT = np.mean(T_inst)
    if T_ext is None:
        T_ext = mT
    else:
        output.update(dict(external_temperature=float(T_ext)))
    output.update(dict(kinetic_temperature=float(mT)))
    T = T_ext  # To simplify below formulas

    # Pressure
    P = (N * k_B * T_inst + W) / V  # Instantaneous pressure
    mP = np.mean(P)  # Below, we assume that this is the "external pressure", p_ext
    if p_ext is None:
        p_ext = mP
    else:
        output.update(dict(external_pressure=float(p_ext)))
    output.update(dict(pressure=float(mP)))

    # Compressibility Factor
    Z = p_ext/(rho*k_B*T)
    output.update(dict(compressibility_factor=float(Z)))

    # Enthalpy
    H = U + K + p_ext*V
    mH = np.mean(H)
    output.update(dict(enthalpy=float(mH / N)))

    # Isobaric heat capacity,
    c_p = np.var(H, ddof=1) / (N * k_B * T ** 2)
    output.update(dict(isobaric_heat_capacity=float(c_p)))

    # Isothermal compressibility, κ_T = -(∂V/∂p)_T / V
    kappa_T = np.var(V, ddof=1) / (np.mean(V) * k_B * T )
    output.update(dict(isothermal_compressibility=float(kappa_T)))
    output.update(dict(isothermal_bulk_modulus=float(1/kappa_T)))

    # Isobaric expansion coefficient, αₚ = (δV/δT)ₚ / V
    cov_VH = np.cov(V, H, ddof=1)[0, 1]
    alpha_p = cov_VH/(k_B * T**2 * mV)
    output.update(dict(isobaric_expansion_coefficient=float(alpha_p)))

    # Isochoric heat capacity
    c_V = c_p - T * alpha_p**2 / ( rho * kappa_T * k_B )
    output.update(dict(isochoric_heat_capacity=float(c_V)))

    # Adiabatic index
    gamma = c_p/c_V
    output.update(dict(adiabatic_index=gamma))

    # Adiabatic compressibility, κₛ = -(∂V/∂p)ₛ / V
    kappa_s = kappa_T*c_V/c_p
    output.update(dict(adiabatic_compressibility=float(kappa_s)))
    output.update(dict(adiabatic_bulk_modulus=float(1/kappa_s)))

    # Thermal pressure coefficient: βᵥ = (∂P/∂T)ᵥ
    beta_v = alpha_p/kappa_T
    output.update(dict(thermal_pressure_coefficient=float(beta_v)))

    # Adiabatic pressure coefficient: βₛ = (∂P/∂T)ₛ
    beta_s = rho*k_B*c_p/(T*alpha_p)
    output.update(dict(adiabatic_pressure_coefficient=float(beta_s)))

    # Adiabatic expansion coefficient, αₛ = (δV/δT)ₛ/V
    alpha_s = alpha_p - kappa_T * beta_s
    output.update(dict(adiabatic_expansion_coefficient=float(alpha_s)))

    # Thermodynamic Grüneisen parameter (dimensionless)
    gamma_G = beta_v/(k_B*c_V*rho)
    output.update(dict(thermodynamic_gruneisen_parameter=float(gamma_G)))

    # Joule–Thomson coefficient, μ_JT = (δT/δp)_H
    # REF: https://en.wikipedia.org/wiki/Joule%E2%80%93Thomson_effect#The_Joule%E2%80%93Thomson_(Kelvin)_coefficient
    mu_JT = (alpha_p*T-1.0)/(c_p*rho*k_B)
    output.update(dict(joule_thomson_coefficient=mu_JT))

    return output
