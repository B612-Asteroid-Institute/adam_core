import jax.numpy as jnp
from jax import config, jit

from ..constants import Constants as c
from .lagrange import apply_lagrange_coefficients, calc_lagrange_coefficients

config.update("jax_enable_x64", True)


MU = c.MU


@jit
def _propagate_2body(
    orbit: jnp.ndarray,
    t0: float,
    t1: float,
    mu: float = MU,
    max_iter: int = 1000,
    tol: float = 1e-14,
) -> jnp.ndarray:
    """
    Propagate an orbit from t0 to t1.

    Parameters
    ----------
    orbit : `~jax.numpy.ndarray` (6)
        Cartesian orbit with position in units of au and velocity in units of au per day.
    t0 : float (1)
        Epoch in MJD at which the orbit are defined.
    t1 : float (N)
        Epochs to which to propagate the given orbit.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.
    max_iter : int, optional
        Maximum number of iterations over which to converge. If number of iterations is
        exceeded, will return the value of the universal anomaly at the last iteration.
    tol : float, optional
        Numerical tolerance to which to compute universal anomaly using the Newtown-Raphson
        method.

    Returns
    -------
    orbits : `~jax.numpy.ndarray` (N, 6)
        Orbit propagated to each MJD with position in units of au and velocity in units
        of au per day.
    """
    r = orbit[0:3]
    v = orbit[3:6]
    dt = t1 - t0

    lagrange_coeffs, stumpff_coeffs, chi = calc_lagrange_coefficients(
        r, v, dt, mu=mu, max_iter=max_iter, tol=tol
    )
    r_new, v_new = apply_lagrange_coefficients(r, v, *lagrange_coeffs)

    return jnp.array([r_new[0], r_new[1], r_new[2], v_new[0], v_new[1], v_new[2]])
