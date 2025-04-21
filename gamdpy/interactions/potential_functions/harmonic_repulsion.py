import numba

def harmonic_repulsion(dist, params):
    """ The harmonic repulsion pair potential

    .. math::

        u(r) = ½\\epsilon(1-r/\\sigma)^2

    for :math:`r<\\sigma` and zero otherwise.
    Parameters: ε=epsilon, σ=cut.
    Note that this potential is naturally truncated at r=σ.

    Parameters
    ----------

    dist : float
        Distance between particles

    params : array-like
        ε, σ

    """

    eps = numba.float32(params[0])
    sigma = numba.float32(params[1])
    inv_sigma = numba.float32(1.0/sigma)  # 1/σ
    one = numba.float32(1.0)
    one_half = numba.float32(1.0/2.0)

    delta = one - dist * inv_sigma  # 1 - r/σ
    u = one_half * eps * delta * delta  # ½ ε (1 - r/σ)²
    s = eps * delta * inv_sigma / dist  # s(r) = -u'(r)/r
    d2u_dr2 = eps * inv_sigma * inv_sigma  # u''(r)

    return u, s, d2u_dr2
