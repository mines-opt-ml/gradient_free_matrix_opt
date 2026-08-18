import jax
import jax.numpy as jnp
from jax import random

from obj_func.obj_func_class import MatrixObjectiveFunction


class KyFanRegression(MatrixObjectiveFunction):
    """Regression in the r Ky Fan norm. That is:
        ||A|| = sigma_1(A) + ... + sigma_r(A).
    At init the target matrix is randomly initialized.
    """

    def __init__(self, rank: int, m: int, n: int, seed: int = 0, noise: float = 0.0):
        key = random.PRNGKey(seed)
        k1, k2 = random.split(key, 2)
        self.A = random.normal(k2, shape=(m, n))
        super().__init__(rank=rank, noise=noise)

    def _kyfan(self, B: jnp.ndarray):
        """
        Compute the Ky Fan norm of arbitrary B. Rank of Ky Fan norm is set at init.
        """
        singular_values = jnp.linalg.svdvals(B)
        return jnp.sum(singular_values[: self.rank])

    def _evaluate(self, X: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        noise_term = self.noise * jax.random.normal(key, shape=())
        return self._kyfan(X - self.A) + noise_term
