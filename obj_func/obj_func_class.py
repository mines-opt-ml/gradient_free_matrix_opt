"""Shared interface for matrix objective functions."""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp


class MatrixObjectiveFunction(ABC):
    """Base class that provides query counting and batched evaluation."""

    def __init__(self, rank: int, noise: float = 0.0):
        self.rank = rank
        self.noise = noise
        self.number_of_queries_made = 0
        self._evaluate_vectorized = jax.vmap(self._evaluate, in_axes=(0, 0))
        self.is_convex = False  # default, can be overwritten by subclasses

    @abstractmethod
    def _evaluate(self, X: jnp.ndarray, key: jax.Array = None) -> jnp.ndarray:
        """Evaluate the objective without modifying the query counter."""

    def __call__(self, X: jnp.ndarray, key: jax.Array = None) -> jnp.ndarray:
        self.number_of_queries_made += 1
        return self._evaluate(X, key)

    def call_vectorized(
        self,
        X_arr: jnp.ndarray,
        key: jax.Array = None,
    ) -> jnp.ndarray:
        number_of_inputs = X_arr.shape[0]
        keys = jax.random.split(key, number_of_inputs)
        self.number_of_queries_made += number_of_inputs
        return self._evaluate_vectorized(X_arr, keys)

    def get_number_of_queries(self) -> int:
        return self.number_of_queries_made

    def reset_number_of_queries(self) -> None:
        self.number_of_queries_made = 0

    def gradient(self,  X: jnp.ndarray, key: jax.Array = None) -> jnp.ndarray:
        """Compute the gradient of the objective function at X, for testing purposes,
        only valid if objective is differentiable of course, and if noise is zero.
        Set `key=None` to have zero noise. """
        # This can be overwritten by subclasses if they have a more efficient way to compute the gradient.
        return jax.grad(self._evaluate)(X, key)
