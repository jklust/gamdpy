import numpy as np
from collections.abc import Iterable

def calculate_response_functions_NpT(
        N: int, dof: int,
        U: Iterable[float], W: Iterable[float],
        K: Iterable[float], Vol: Iterable[float],
        k_B=1.0, T_ext: float=None,
        p_ext: float=None, per_particle=True
):
    r"""
    Calculate thermodynamic response functions from equilibrium fluctuations in an isotropic NpT simulation.

    This function takes time series of extensive thermodynamic observables from an
    NpT (isothermal–isobaric) ensemble—potential energy :math:`U`, configurational
    virial :math:`W`, kinetic energy :math:`K`, and volume :math:`V`—and computes a
    suite of response functions using fluctuation relations and standard
    thermodynamic identities.

    **Ensemble and conventions**

    - Isotropic NpT ensemble at thermal equilibrium.
    - Time series are assumed stationary and sampled at equal intervals.
    - If ``per_particle=True``, the inputs ``U``, ``W``, ``K``, ``Vol`` are *per-particle*
      values and if ``per_particle=False``, the inputs are treated as extensive.
    - The instantaneous temperature is computed from equipartition,
      :math:`T_\mathrm{inst} = \frac{2K}{k_B\, \mathrm{dof}}`.
      The external temperature ``T_ext`` defaults to :math:`\langle T_\mathrm{inst}\rangle` if not provided.
    - The instantaneous pressure follows the microscopic virial expression,
      :math:`P = \frac{Nk_B T_\mathrm{inst} + W}{V}`.
      The external pressure ``p_ext`` defaults to :math:`\langle P\rangle` if not provided.

    **Primary estimators (ensemble averages)**

    - Number density: :math:`\rho = \dfrac{N}{\langle V\rangle}`, specific volume: :math:`v = 1/\rho`.
    - Internal energy (per particle): :math:`\langle U + K\rangle / N`.
    - Enthalpy (per particle): :math:`\langle H\rangle/N` with :math:`H = U + K + p_\mathrm{ext} V`.
    - Compressibility factor: :math:`Z = \dfrac{p_\mathrm{ext}}{\rho\, k_B T}`.

    **Fluctuation formulas in NpT**

    Using:

    .. math::

        c_p &= \frac{\mathrm{Var}(H)}{N\, k_B\, T^2}, \\
        \kappa_T &= \frac{\mathrm{Var}(V)}{\langle V\rangle\, k_B\, T}, \\
        \alpha_p &= \frac{\mathrm{Cov}(V,H)}{\langle V\rangle\, k_B\, T^2}.

    Here :math:`c_p` is the isobaric heat capacity per particle, :math:`\kappa_T`
    the isothermal compressibility, and :math:`\alpha_p` the isobaric expansion
    coefficient.

    **Thermodynamic identities used**

    From standard identities:

    .. math::

        c_V &= c_p - \frac{T\, \alpha_p^2}{\rho\, \kappa_T}, \\
        \gamma &= \frac{c_p}{c_V}, \\
        \kappa_S &= \kappa_T \frac{c_V}{c_p}, \qquad K_T = \kappa_T^{-1}, \quad K_S = \kappa_S^{-1}, \\
        \beta_v &= \left(\frac{\partial P}{\partial T}\right)_V = \frac{\alpha_p}{\kappa_T}, \\
        \beta_s &= \left(\frac{\partial P}{\partial T}\right)_S = \frac{\rho\, c_p}{T\, \alpha_p}, \\
        \alpha_s &= \left(\frac{\partial V}{\partial T}\right)_S \frac{1}{V} = \alpha_p - \kappa_T\, \beta_s, \\
        \gamma_G &= \frac{\beta_v}{\rho\, c_V}, \\
        \mu_{JT} &= \left(\frac{\partial T}{\partial p}\right)_H = \frac{T\, \alpha_p - 1}{\rho\, c_p}.

    Parameters
    ----------
    N : int
        Number of particles.
    dof : int
        Total number of quadratic degrees of freedom of the system.
    U : Iterable[float]
        Time series of potential energy.
    W : Iterable[float]
        Time series of configurational virial.
    K : Iterable[float]
        Time series of kinetic energy.
    Vol : Iterable[float]
        Time series of volume.
    k_B : float, optional
        Boltzmann constant of used unit system (default 1.0).
    T_ext : float, optional
        External (bath) temperature. If ``None``, the estimator uses :math:`\langle T_\mathrm{inst}\rangle`.
    p_ext : float, optional
        External (bath) pressure. If ``None``, the estimator uses :math:`\langle P\rangle`.
    per_particle : bool, optional
        If ``True`` (default), inputs are interpreted as per-particle series.
        If ``False``, inputs are treated as extensive already.

    Returns
    -------
    dict
        A dictionary with the following keys (scalars):

        - ``density`` (:math:`\rho`)
        - ``specific_volume`` (:math:`v`)
        - ``potential_energy`` (:math:`\langle U\rangle/N`)
        - ``kinetic_energy`` (:math:`\langle K\rangle/N`)
        - ``internal_energy`` (:math:`\langle U+K\rangle/N`)
        - ``kinetic_temperature`` (:math:`\langle T_\mathrm{inst}\rangle`)
        - ``external_temperature`` (if provided)
        - ``pressure`` (:math:`\langle P\rangle`)
        - ``external_pressure`` (if provided)
        - ``compressibility_factor`` (:math:`Z`)
        - ``enthalpy`` (:math:`\langle H\rangle/N`)
        - ``isobaric_heat_capacity`` (:math:`c_p`)
        - ``isothermal_compressibility`` (:math:`\kappa_T`)
        - ``isothermal_bulk_modulus`` (:math:`K_T`)
        - ``isobaric_expansion_coefficient`` (:math:`\alpha_p`)
        - ``isochoric_heat_capacity`` (:math:`c_V`)
        - ``isochoric_heat_capacity_excess`` (:math:`c_V^{\mathrm{ex}}`)
        - ``adiabatic_index`` (:math:`\gamma`)
        - ``adiabatic_compressibility`` (:math:`\kappa_S`)
        - ``adiabatic_bulk_modulus`` (:math:`K_S`)
        - ``thermal_pressure_coefficient`` (:math:`\beta_v`)
        - ``adiabatic_pressure_coefficient`` (:math:`\beta_s`)
        - ``adiabatic_expansion_coefficient`` (:math:`\alpha_s`)
        - ``thermodynamic_gruneisen_parameter`` (:math:`\gamma_G`)
        - ``joule_thomson_coefficient`` (:math:`\mu_{JT}`)

    References
    ----------
    .. [AllenTildesley2017] M. P. Allen and D. J. Tildesley, *Computer Simulation of Liquids*, 2017.
    .. [FrenkelSmit2002] D. Frenkel and B. Smit, *Understanding Molecular Simulation*, 2002.
    .. [Callen1985] H. B. Callen, *Thermodynamics and an Introduction to Thermostatistics*, 1985.
    """

    if not all(np.isscalar(x) for x in (N, dof, k_B)):
        raise TypeError("N, dof and k_B must be scalars")
    for name, value in (("Vol", Vol), ("U", U), ("W", W), ("K", K)):
        if np.isscalar(value):
            raise TypeError(f"{name} must be array-like, not a scalar")
        try:
            np.asarray(value)
        except Exception as exc:
            raise TypeError(f"{name} must be array-like") from exc

    # Convert to numpy arrays
    if type(U) is not np.ndarray:
        U = np.array(U)
    if type(W) is not np.ndarray:
        W = np.array(W)
    if type(K) is not np.ndarray:
        K = np.array(K)
    if type(Vol) is not np.ndarray:
        Vol = np.array(Vol)

    # Scale to values
    if per_particle:
        U = U * N
        W = W * N
        K = K * N
        Vol = Vol * N

    # Dictionary that is returned by this function
    output = {}

    # Density
    V = Vol
    mV = float(np.mean(V))
    rho = N/mV
    output.update(dict(density=float(rho)))
    output.update(dict(specific_volume=float(1/rho)))

    # Energies
    mU = float(np.mean(U))
    output.update(dict(potential_energy=float(mU / N)))
    mK = float(np.mean(K))
    output.update(dict(kinetic_energy=float(mK / N)))
    E = U + K
    mE = float(np.mean(E))
    output.update(dict(internal_energy=float(mE / N)))
    T_inst = 2.0 * K / (k_B * dof)
    mT = float(np.mean(T_inst))
    if T_ext is None:
        T_ext = mT  # Use ensemble temperature to estimate external temperature
    else:
        output.update(dict(external_temperature=float(T_ext)))
    output.update(dict(kinetic_temperature=float(mT)))
    T = T_ext  # To simplify below formulas

    # Pressure
    P = (N * k_B * T_inst + W) / V  # Instantaneous pressure
    mP = float(np.mean(P))  # Below, we assume that this is the "external pressure", p_ext
    if p_ext is None:
        p_ext = mP  # Use ensemble pressure to estimate external pressure
    else:
        output.update(dict(external_pressure=float(p_ext)))
    output.update(dict(pressure=float(mP)))

    # Compressibility Factor
    Z = p_ext/(rho*k_B*T)
    output.update(dict(compressibility_factor=float(Z)))

    # Enthalpy
    H = U + K + p_ext*V
    mH = float(np.mean(H))
    output.update(dict(enthalpy=float( mH / N )))

    # Isobaric heat capacity,
    dHH = float(np.var(H, ddof=1))
    c_p = dHH / ( N * k_B * T ** 2)
    output.update(dict(isobaric_heat_capacity=float(c_p)))

    # Isothermal compressibility, κ_T = -(∂V/∂p)_T / V
    dVV = float(np.var(V, ddof=1))
    kappa_T = dVV / ( mV * k_B * T )
    output.update(dict(isothermal_compressibility=float(kappa_T)))
    output.update(dict(isothermal_bulk_modulus=float(1/kappa_T)))

    # Isobaric expansion coefficient, αₚ = (δV/δT)ₚ / V
    cov_VH = np.cov(V, H, ddof=1)[0, 1]
    alpha_p = cov_VH/(k_B * T**2 * mV)
    output.update(dict(isobaric_expansion_coefficient=float(alpha_p)))

    # Isochoric heat capacity,
    c_V = c_p - T * alpha_p**2 / ( rho * kappa_T )
    output.update(dict(isochoric_heat_capacity=float(c_V)))

    # Ideal gas heat capacity, c_V_ex = c_V - c_V_id
    c_V_id = dof/N*k_B/2
    c_V_ex = c_V - c_V_id
    output.update(dict(isochoric_heat_capacity_excess=float(c_V_ex)))

    # Adiabatic index,
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
    beta_s = rho*c_p/(T*alpha_p)
    output.update(dict(adiabatic_pressure_coefficient=float(beta_s)))

    # Adiabatic expansion coefficient, αₛ = (δV/δT)ₛ/V
    alpha_s = alpha_p - kappa_T * beta_s
    output.update(dict(adiabatic_expansion_coefficient=float(alpha_s)))

    # Thermodynamic Grüneisen parameter (dimensionless)
    gamma_G = beta_v/(c_V*rho)
    output.update(dict(thermodynamic_gruneisen_parameter=float(gamma_G)))

    # Joule–Thomson coefficient, μ_JT = (δT/δp)_H
    mu_JT = (alpha_p*T-1.0)/(c_p*rho)
    output.update(dict(joule_thomson_coefficient=mu_JT))

    return output


def calculate_response_functions_NVT(N, dof, V, U, W, K, k_B=1.0):
    """ Compute thermodynamic response functions of a NVT simulation """
    if not all(np.isscalar(x) for x in (N, dof , V)):
        raise TypeError("N, D and V must be scalars")
    for name, value in (("U", U), ("W", W), ("K", K)):
        if np.isscalar(value):
            raise TypeError(f"{name} must be array-like, not a scalar")
        try:
            np.asarray(value)
        except Exception as exc:
            raise TypeError(f"{name} must be array-like") from exc

    output = {}
    rho = N / V
    output.update(dict(density=float(rho)))
    mU = np.mean(U)
    output.update(dict(potential_energy=float(mU / N)))
    mK = np.mean(K)
    output.update(dict(kinetic_energy=float(mK / N)))
    E = U + K
    mE = np.mean(E)
    output.update(dict(internal_energy=float(mE / N)))
    T_kin = 2.0 * mK / k_B / dof
    T = np.mean(T_kin)
    output.update(dict(kinetic_temperature=float(T_kin)))
    P = rho * k_B * T + W / V  # Instantaneous pressure
    output.update(dict(pressure=float(np.mean(P))))
    c_V = np.var(E, ddof=1) / (k_B * T ** 2 * N)
    output.update(dict(isochoric_heat_capacity=float(c_V)))
    cov_PE = np.cov(P, E)[0, 1]
    beta_V = cov_PE / k_B / T ** 2  # Thermal pressure coefficient: βᵥ = (∂P/∂T)ᵥ
    output.update(dict(thermal_pressure_coefficient=float(beta_V)))

    # We need hyper-virial to compute K_T. Then we would have a complete set of responce function,
    # and could compute everything

    return output
