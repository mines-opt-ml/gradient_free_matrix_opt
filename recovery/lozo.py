import jax.numpy as jnp


def lozo(
    y: jnp.ndarray,  # shape: (num_samples,)
    A: jnp.ndarray,  # shape: (num_samples, m, n)
    **kwargs,
) -> jnp.ndarray:  # shape: (m, n)
    """
    This is a hack to get LOZO (see "Enhancing Zeroth-Order Fine-Tuning for Language Models with Low-rank Structures" by Chen et al, ICLR 2025) working wihtin our framework.
    Ensure that whenever you call this method, the sampling-scheme is set appropriately.
    """
    return jnp.sum(y[:, None, None] * A, axis=0)
