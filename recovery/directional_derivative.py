import jax
import jax.numpy as jnp


# exact directional derivatives of obj_func at X along each direction in Z.
# this is the oracle every recovery method is charged one query per direction for.
def directional_derivatives(
    obj_func,
    X: jax.Array,  # shape: (m, n)
    Z: jax.Array,  # shape: (num_directions, m, n)
    key: jax.Array,
    approximate: bool = False,
    **kwargs,
) -> jax.Array:  # shape: (num_directions,)
    if approximate:
        return directional_derivatives_approximate(obj_func, X, Z, key, **kwargs)
    def f(X_):
        return obj_func(X_, key)

    def directional_derivative_single(Z_i):
        _, deriv = jax.jvp(f, (X,), (Z_i,))
        return deriv

    return jax.vmap(directional_derivative_single)(Z)

def directional_derivatives_approximate(
    obj_func,
    X: jax.Array,  # shape: (m, n)
    Z: jax.Array,  # shape: (num_directions, m, n)
    key: jax.Array,
    h: float = None,
    centered: bool = False,
) -> jax.Array:  # shape: (num_directions,)
    def f(X_):
        return obj_func(X_, key)

    if h is None:
        eps = jnp.finfo(X.dtype).eps
        # Assumes f(X) is O(1)
        # could be improved, e.g., see Numerical Recipes 3rd edition, section 5.7
        h = jnp.sqrt(eps) 
        if centered:
            h = jnp.pow(eps, 1/3)  # centered difference is more accurate, so can use a larger h
    if centered:
        def directional_derivative_single(Z_i):
            return (f(X + h * Z_i) - f(X - h * Z_i)) / (2 * h)
    else:
        # precache f(X) to save on computation:
        f_X = f(X)
        def directional_derivative_single(Z_i):
            return (f(X + h * Z_i) - f_X) / h

    return jax.vmap(directional_derivative_single)(Z)
