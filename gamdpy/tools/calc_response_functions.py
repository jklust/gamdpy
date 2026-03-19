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

    The implementation analyses time series of thermodynamic observables from a constant :math:`NpT` (isothermal–isobaric) ensemble,
    specifically
    potential energy :math:`U`,
    configurational virial :math:`W= \left. \frac{\partial U\!\left(\lambda\mathbf{r}^N\right)}{\partial \lambda} \right|_{\lambda=1}`,
    kinetic energy :math:`K`, and volume :math:`V`, and computes thermodynamic response functions.
    The implementation assumes:

    - An isotropic :math:`NpT` ensemble at equilibrium.
    - If ``per_particle=True`` (default), the inputs ``U``, ``W``, ``K``, ``Vol`` are *per-particle*
      values and if ``per_particle=False``, the inputs are treated as values for the entire system. ``N`` is used in the conversion.
    - The external temperature, :math:`T`, defaults to :math:`\langle T_\mathrm{inst}\rangle` if not provided, where
      the instantaneous temperature is computed from equipartition,
      :math:`T_\mathrm{inst} = \frac{2K}{k_B\, N_d}` and :math:`\mathrm{N_d}` is the number of degrees of freedom (``dof``).
    - The external pressure, :math:`p`, defaults to :math:`\langle p_\mathrm{inst}\rangle` if not provided, where
      the instantaneous pressure follows from,
      :math:`p_\mathrm{inst} = \frac{Nk_B T_\mathrm{inst} + W}{V}`.


    The isobaric heat capacity (per particle), :math:`c_p=\frac{1}{N}\left(\frac{\partial H}{\partial T}\right)_p`,
    the isothermal compressibility, :math:`\kappa_T=-\frac{1}{V}\left(\frac{\partial V}{\partial p}\right)_T`, and
    the isobaric expansion coefficient, :math:`\alpha_p = \frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_p`
    are computed using fluctuation-disipation formulas of the NpT ensemble:

    .. math::

        c_p &= \frac{\mathrm{Var}(H)}{N\, k_B\, T^2}, \\
        \kappa_T &= \frac{\mathrm{Var}(V)}{\langle V\rangle\, k_B\, T}, \\
        \alpha_p &= \frac{\mathrm{Cov}(V,H)}{\langle V\rangle\, k_B\, T^2}.

    Thermodynamic identities are used to estimate other quantities such as:

    .. math::

        \rho &= N/\langle V \rangle \\
        c_V &= \frac{1}{N}\left(\frac{\partial U}{\partial T}\right)_V = c_p - \frac{T\, \alpha_p^2}{\rho\, \kappa_T}, \\
        \gamma &= \frac{c_p}{c_V}, \\
        \kappa_S &= -\frac{1}{V}\left(\frac{\partial V}{\partial p}\right)_S = \kappa_T \frac{c_V}{c_p}, \\
        K_T &= \kappa_T^{-1}, \quad K_S = \kappa_S^{-1}, \\
        \beta_v &= \left(\frac{\partial p}{\partial T}\right)_V = \frac{\alpha_p}{\kappa_T}, \\
        \beta_s &= \left(\frac{\partial p}{\partial T}\right)_S = \frac{\rho\, c_p}{T\, \alpha_p}, \\
        \alpha_s &= \frac{1}{V} \left(\frac{\partial V}{\partial T}\right)_S  = \alpha_p - \kappa_T\, \beta_s, \\
        \gamma_G &= V\left(\frac{\partial p}{\partial E}\right)_V = -\left(\frac{\partial \ln T}{\partial \ln V}\right)_S = \frac{\alpha_p}{\rho \, \kappa_T \, c_V}, \\
        \mu_{JT} &= \left(\frac{\partial T}{\partial p}\right)_H = \frac{T\, \alpha_p - 1}{\rho\, c_p}.

    Parameters
    ----------
    N : int
        Number of particles.
    dof : int
        Total number of degrees of freedom of the system.
    U : Iterable[float]
        Time series of potential energy.
    W : Iterable[float]
        Time series of configurational virial.
    K : Iterable[float]
        Time series of kinetic energy.
    Vol : Iterable[float]
        Time series of volume.
    k_B : float, optional
        Boltzmann constant of the unit system (default 1.0).
    T_ext : float, optional
        External (bath) temperature. If ``None``, the estimator uses :math:`\langle T_\mathrm{inst}\rangle`.
    p_ext : float, optional
        External (bath) pressure. If ``None``, the estimator uses :math:`\langle P\rangle`.
    per_particle : bool, optional
        If ``True`` (default), the ``U``, ``W``, ``K``, ``Vol`` inputs are interpreted as *per-particle* values.
        If ``False``, they are treated as extensive.

    Returns
    -------
    dict
        A dictionary with scalar properties and response functions reported as *intensive* (per particle).

        - ``density``: :math:`\rho = \dfrac{N}{\langle V \rangle}`.
        - ``specific_volume``: :math:`v = \dfrac{1}{\rho}`.
        - ``potential_energy``: :math:`\langle U \rangle / N`.
        - ``kinetic_energy``: :math:`\langle K \rangle / N`.
        - ``internal_energy``: :math:`\langle U + K \rangle / N`.
        - ``kinetic_temperature``: :math:`\langle T_\mathrm{inst} \rangle`, with :math:`T_\mathrm{inst} = \dfrac{2K}{k_B\,\mathrm{dof}}`.
        - ``external_temperature``: Provided external :math:`T` (if given).
        - ``pressure``: :math:`\langle P \rangle`, with :math:`P = \dfrac{N k_B T_\mathrm{inst} + W}{V}`.
        - ``external_pressure``: Provided external :math:`p` (if given).
        - ``compressibility_factor``: :math:`Z = \dfrac{p_\mathrm{ext}}{\rho\, k_B T}`.
        - ``enthalpy``: :math:`\langle H \rangle / N`, with :math:`H = U + K + p_\mathrm{ext} V`.
        - ``isobaric_heat_capacity``: :math:`c_p = \dfrac{\mathrm{Var}(H)}{N\, k_B\, T^2}`.
        - ``isothermal_compressibility``: :math:`\kappa_T = \dfrac{\mathrm{Var}(V)}{\langle V \rangle\, k_B\, T}`.
        - ``isothermal_bulk_modulus``: :math:`K_T = \kappa_T^{-1}`.
        - ``isobaric_expansion_coefficient``: :math:`\alpha_p = \dfrac{\mathrm{Cov}(V,H)}{\langle V \rangle\, k_B\, T^2}`.
        - ``isochoric_heat_capacity``: :math:`c_V = c_p - \dfrac{T\, \alpha_p^2}{\rho\, \kappa_T}`.
        - ``isochoric_heat_capacity_excess``: :math:`c_V^{\mathrm{ex}} = c_V - c_V^{\mathrm{id}}`, with :math:`c_V^{\mathrm{id}} = \dfrac{\mathrm{dof}}{2N} k_B`.
        - ``adiabatic_index``: :math:`\gamma = \dfrac{c_p}{c_V}`.
        - ``adiabatic_compressibility``: :math:`\kappa_S = \kappa_T \dfrac{c_V}{c_p}`.
        - ``adiabatic_bulk_modulus``: :math:`K_S = \kappa_S^{-1}`.
        - ``thermal_pressure_coefficient``: :math:`\beta_v = \left(\dfrac{\partial P}{\partial T}\right)_V = \dfrac{\alpha_p}{\kappa_T}`.
        - ``adiabatic_pressure_coefficient``: :math:`\beta_s = \left(\dfrac{\partial P}{\partial T}\right)_S = \dfrac{\rho\, c_p}{T\, \alpha_p}`.
        - ``adiabatic_expansion_coefficient``: :math:`\alpha_s = \alpha_p - \kappa_T\, \beta_s`.
        - ``thermodynamic_gruneisen_parameter``: :math:`\gamma_G = \dfrac{\alpha_p}{\rho \, \kappa_T \, c_V}`.
        - ``joule_thomson_coefficient``: :math:`\mu_{JT} = \left(\dfrac{\partial T}{\partial p}\right)_H = \dfrac{T\, \alpha_p - 1}{\rho\, c_p}`.

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
