import numba

def harmonic_repulsion(dist, params):
    """ The harmonic repulsion potential: u(r) = ½ ε (1 - r/σ)²  for r/σ < 1 and 0 otherwise.
    u(r) = 0.5 eps (1 - r/sigma)^2 for r < sigma and zero otherwise.
    parameters: ε=eps, σ=cut

    Note that this potential is naturally truncated at r=sigma.
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
