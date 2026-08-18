import jax.numpy as jnp


def apply_sensing_operator(
    A: jnp.ndarray,  # shape: (num_samples, m, n)
    X: jnp.ndarray,  # shape: (m, n)
) -> jnp.ndarray:  # shape: (num_samples,)
    return jnp.sum(A * X, axis=(1, 2))
