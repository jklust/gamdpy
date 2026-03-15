import numpy as np

def get_NVT_response_functions(N, dof, V, U, W, K, k_B=1.0):
    """ Compute thermodynamic response functions of a NVT simulation
    EXPERIMENTAL
    """
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