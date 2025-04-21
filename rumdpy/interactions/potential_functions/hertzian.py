import numba

def hertzian(dist, params):
    """ Hertzian potential

    .. math::

        u(r) = \\epsilon(1-r/\\sigma)^\\alpha

    for :math:`r<\\sigma` and zero otherwise.
    Parameters: ε=epsilon, α=alpha, σ=cut.
    Note that this potential is naturally truncated at r=σ.
    For Hertzian disks chose α=7/2, and for Hertzian spheres chose α=5/2

    Parameters
    ----------

    dist : float
        Distance between particles

    params : array-like
        ε, α, σ

    """

    eps = numba.float32(params[0])
    alpha = numba.float32(params[1])
    sigma = numba.float32(params[2])
    inv_sigma = numba.float32(1.0/sigma)  # 1/σ
    one = numba.float32(1.0)
    two = numba.float32(2.0)

    delta = one - dist * inv_sigma  # 1 - r/σ
    u = eps * delta ** alpha   # ε (1 - r/σ)^α
    s = eps * alpha * delta ** (alpha - one) * inv_sigma / dist  # s(r) = -u'(r)/r
    d2u_dr2 = eps * alpha * (alpha - one) * delta ** (alpha - two) * inv_sigma * inv_sigma  # u''(r)

    return u, s, d2u_dr2
