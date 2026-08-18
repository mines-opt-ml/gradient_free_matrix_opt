import jax.numpy as jnp


def adjoint_sensing_operator(
    y: jnp.ndarray,  # shape: (num_samples,)
    A: jnp.ndarray,  # shape: (num_samples, m, n)
    **kwargs,
) -> jnp.ndarray:  # shape: (m, n)
    return jnp.sum(y[:, None, None] * A, axis=0)
