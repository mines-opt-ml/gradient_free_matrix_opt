"""Synthetic matrix-regression objective."""

import jax
import jax.numpy as jnp
from jax import random

from obj_func.obj_func_class import MatrixObjectiveFunction


class MatrixRegression(MatrixObjectiveFunction):
    def __init__(
        self,
        target_rank: int,
        m: int,
        n: int,
        ill_conditioned: bool = False,
        seed: int = 0,
        noise: float = 0.0,
    ):
        assert m >= n >= target_rank, "Need m >= n >= target_rank."
        super().__init__(rank=target_rank, noise=noise)

        key = random.PRNGKey(seed)
        k1, k2, k3 = random.split(key, 3)

        A = random.normal(k1, shape=(m, n))
        B = random.normal(k2, shape=(n, n))

        self.P, _ = jnp.linalg.qr(A)
        self.Q, _ = jnp.linalg.qr(B)

        synthetic_sing_vals = jnp.zeros(n)

        if ill_conditioned:
            non_zero_sing_vals = jnp.array(
                [10 * 2 ** (-j) for j in range(target_rank)]
            )
        else:
            non_zero_sing_vals = random.uniform(k3, shape=(target_rank,))

        synthetic_sing_vals = synthetic_sing_vals.at[:target_rank].set(
            non_zero_sing_vals
        )
        self.Sigma = jnp.diag(synthetic_sing_vals)

    def _evaluate(self, X: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        projected_matrix = self.P.T @ X @ self.Q
        value = jnp.sum((projected_matrix - self.Sigma) ** 2)
        noise_term = self.noise * random.normal(key, shape=())
        return value + noise_term
