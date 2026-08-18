import jax.numpy as jnp
import jax.scipy as jsp


def apply_A(
    v: jnp.ndarray,  # (mn, ) flattened gradient approximation
    A: jnp.ndarray,  # (numsample, matrixsize, matrixsize)
) -> jnp.ndarray:
    dim = A.shape

    return jnp.tensordot(A, v.reshape(dim[2], dim[1]).T, axes=([1, 2], [0, 1])).reshape(-1)


def apply_At(
    v: jnp.ndarray,  # (numsamples, ) flattened gradient approximation
    A: jnp.ndarray,  # (numsample, matrixsize, matrixsize)
) -> jnp.ndarray:
    dim = A.shape

    return jnp.tensordot(v, A, axes=(0, 0)).T.reshape(-1)


def apply_AAt(
    v: jnp.ndarray,  # (numsamples, ) flattened gradient approximation
    A: jnp.ndarray,  # (numsample, matrixsize, matrixsize)
) -> jnp.ndarray:
    return apply_A(apply_At(v, A), A)


def pseudo_inverse_CG(
    y: jnp.ndarray,  # (numsamples, ) flattened gradient approximation
    A: jnp.ndarray,  # (numsample, matrixsize, matrixsize)
    **kwargs,
) -> jnp.ndarray:
    iters = kwargs["iters"]
    atol = 1e-8
    dim = A.shape
    zero_matrix = jnp.zeros((dim[1], dim[2]), dtype=y.dtype)

    if not bool(jnp.isfinite(y).all()):
        return zero_matrix

    if bool(jnp.linalg.norm(y) <= atol):
        return zero_matrix

    def Normal(
        y: jnp.ndarray,  # (numsamples, ) flattened gradient approximation
    ) -> jnp.ndarray:
        return apply_AAt(y, A)

    sol, _ = jsp.sparse.linalg.cg(
        Normal, y, tol=1e-7, atol=atol, maxiter=iters
    )  # hard coding tolerance for now. Unlikely to ever be reached.

    if not bool(jnp.isfinite(sol).all()):
        return zero_matrix

    recovered_matrix = apply_At(sol, A).reshape(dim[2], dim[1]).T
    if not bool(jnp.isfinite(recovered_matrix).all()):
        return zero_matrix

    return recovered_matrix
