import jax.numpy as jnp
import numpy as np

# from matrix_zoro.RandomLinearAlgebra.R_SVD import rand_reduced_SVD
from recovery.adjoint_sensing_operator import adjoint_sensing_operator
from recovery.sensing_operator import apply_sensing_operator

EPSILON = 1e-10  # will be added to denominators for stability


def thresholding_step(
    X: jnp.ndarray,  # shape (m,n)
    target_rank: int,
):
    """
    subroutine that does the singular value thresholding step.
    """
    print(f"Nan and inf status {jnp.isnan(X).any() | jnp.isinf(X).any()}")

    U, S, VH = jnp.linalg.svd(X)  # jax.scipy.linalg.svd(X, lapack_driver="gesvd")  #
    if jnp.isnan(U).any:
        np.linalg.svd(np.asarray(X))

    ur = U[:, :target_rank]
    vr = VH.T[:, :target_rank]
    sr = S[:target_rank]
    X_out = jnp.array(ur @ jnp.diag(sr) @ vr.T)
    P = jnp.array(ur @ ur.T)
    print(f"U is currently {jnp.linalg.norm(U)}")
    return X_out, P, ur, vr


def iterative_hard_thresholding(
    y: jnp.ndarray,  # shape (num_samples, )
    A: jnp.ndarray,  # shape (num_samples, m, n)
    X: jnp.ndarray | None = None,  # the | works in Python 3.10+
    **kwargs,
) -> jnp.ndarray:  # shape (m, n)
    target_rank = kwargs["target_rank"]
    iters = kwargs["iters"]
    if X is None:
        X = adjoint_sensing_operator(y, A)
    for i in range(iters):
        if i == 0:
            X, P, _, _ = thresholding_step(X, target_rank)

        gradient = adjoint_sensing_operator(apply_sensing_operator(A, X) - y, A)
        projected_gradient = P @ gradient
        denominator = jnp.linalg.norm(apply_sensing_operator(A, projected_gradient)) ** 2 + EPSILON
        step_size = (jnp.linalg.matrix_norm(projected_gradient) ** 2) / denominator

        Yk = X - step_size * gradient
        X, P, _, _ = thresholding_step(Yk, target_rank)

    return X


def spectral_iterative_hard_thresholding(
    y: jnp.ndarray,  # shape (num_samples, )
    A: jnp.ndarray,  # shape (num_samples, m, n)
    **kwargs,
) -> jnp.ndarray:  # shape (m, n)
    target_rank = kwargs["target_rank"]
    iters = kwargs["iters"]
    X = adjoint_sensing_operator(y, A)

    for i in range(iters):
        if i == 0:
            X, P, _, _ = thresholding_step(X, target_rank)

        gradient = adjoint_sensing_operator(apply_sensing_operator(A, X) - y, A)
        projected_gradient = P @ gradient
        denominator = jnp.linalg.norm(apply_sensing_operator(A, projected_gradient)) ** 2 + EPSILON
        step_size = (jnp.linalg.matrix_norm(projected_gradient) ** 2) / denominator
        print(f"Step size is {step_size}")

        Yk = X - step_size * gradient
        print(f"Max value in Yk is {jnp.abs(Yk).max()} and min value is {jnp.abs(Yk).min()}")
        X, P, ur, vr = thresholding_step(Yk, target_rank)
        print(f"X is currently {jnp.linalg.norm(X)}")

    # return the projected recovered matrix
    print(f"Denominator is {denominator}")
    return ur @ vr.T
