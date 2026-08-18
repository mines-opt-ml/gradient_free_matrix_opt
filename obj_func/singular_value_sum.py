"""Singular-value-sum objective."""

import jax
import jax.numpy as jnp

from obj_func.obj_func_class import MatrixObjectiveFunction


class SingularValueSum(MatrixObjectiveFunction):
    """sum the squares of the first ``rank`` singular values and add noise."""

    def __init__(self, rank: int, noise: float = 0.0):
        super().__init__(rank=rank, noise=noise)
        self.is_convex = True  # follows by similar arguments to Example 3.6 in Boyd and Vandenberghe, Convex Optimization, 2004, which shows that the sum of the first k singular values is convex.

    def _evaluate(self, X: jnp.ndarray, key: jax.Array = None) -> jnp.ndarray:
        singular_values = jnp.linalg.svdvals(X)
        if key is None:
            return 0.5 * jnp.sum(singular_values[: self.rank] ** 2)
        else:
            noise_term = self.noise * jax.random.normal(key, shape=())
            return 0.5 * jnp.sum(singular_values[: self.rank] ** 2) + noise_term


# Not yet implemented
# def gradient(self, X: jnp.ndarray, key: jax.Array = None) -> jnp.ndarray:
#     """Compute the gradient of the objective function at X."""
#     # Not yet worked out. Need to look at the refs below:
#     # See https://www.stats.uwo.ca/faculty/hssendov/Publications/NASVT.pdf#page=22.29
#     # "Nonsmooth analysis of singular values. Part I: Theory" by Adrian S. Lewis
#     # and Hristo S. Sendov, January 25, 2005
#     # (somewhat similar to Watson's 1993 "On matrix approximation problems with Ky Fank norms" though
#     # that is for norms only, so we'd have to do matrix-valued chain-rule)
#     if key is not None:
#         raise ValueError("Gradient computation does not support noise.")
#     U, S, Vh = jnp.linalg.svd(X, full_matrices=False)
#     grad = TBD
#     return grad
