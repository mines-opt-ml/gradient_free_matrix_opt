import jax.numpy as jnp
from jax import grad

from recovery.adjoint_sensing_operator import adjoint_sensing_operator
from recovery.sensing_operator import apply_sensing_operator

# from matrix_zoro.RandomLinearAlgebra.R_SVD import rand_reduced_SVD


def BM_linesearch(
    y: jnp.ndarray,  # shape: (num_samples,)
    A: jnp.ndarray,  # shape: (num_samples, m, n)
    L: jnp.ndarray,  # shape: (m, r)
    R: jnp.ndarray,  # shape: (n, r)
    gradL: jnp.ndarray,
    gradR: jnp.ndarray,
    eta: float = 1,
    rho: float = 0.5,
    c: float = 0.5,
) -> float:  # "ideal" step size
    loss = _BMLossFunc(y, A, L, R)
    gradLnormsq = jnp.linalg.norm(gradL, ord="fro") ** 2
    gradRnormsq = jnp.linalg.norm(gradR, ord="fro") ** 2
    while _BMLossFunc(y, A, L - eta * gradL, R - eta * gradR) > loss - c * eta * (gradLnormsq + gradRnormsq) and eta >= 1e-10:
        eta = rho * eta
        # print(eta)
    return eta


# initialize B-M factorization - (L0, R0) = SVD_r(A*(y))
# This function was formerly called _BM_init.
def spectral_initialization(  # ToDO: Test this as a standalone recovery method.
    y: jnp.ndarray,  # shape: (num_samples,)
    A: jnp.ndarray,  # shape: (num_samples, m, n)
    **kwargs,
) -> tuple:  # tuple with L0(m, r) and R0(n, r), the initial BM factorization
    target_rank = kwargs["target_rank"]

    U, S, VH = jnp.linalg.svd(adjoint_sensing_operator(y, A), full_matrices=True)

    # truncate
    Ur = U[:, :target_rank]
    Vr = VH.T[:, :target_rank]
    Sr = S[:target_rank]

    L0 = Ur @ jnp.diag(jnp.sqrt(Sr))
    R0 = Vr @ jnp.diag(jnp.sqrt(Sr))
    return L0, R0


# loss function for "projected" gradient descent on the BM factorization
# loss = ||A(LR^T)-y||_2^2 + 1/8 * ||L^T L - R^T R||_F^2
def _BMLossFunc(
    y: jnp.ndarray,  # shape: (num_samples,)
    A: jnp.ndarray,  # shape: (num_samples, m, n)
    L: jnp.ndarray,  # shape: (m, r)
    R: jnp.ndarray,  # shape: (n, r)
) -> jnp.ndarray:  # loss
    ft = jnp.linalg.norm(apply_sensing_operator(A, L @ R.T) - y, ord=2) ** 2
    st = (1 / 8) * jnp.linalg.norm(L.T @ L - R.T @ R, ord="fro") ** 2
    return ft + st


# bmGD
def burer_monteiro_gradient_descent(
    y: jnp.ndarray,  # shape: (num_samples,)
    A: jnp.ndarray,  # shape: (num_samples, m, n)
    **kwargs,
) -> jnp.ndarray:  # shape (m, n)
    target_rank = kwargs["target_rank"]
    iters = kwargs["iters"]

    # get initial estimate
    L, R = spectral_initialization(y, A, target_rank=target_rank)

    # get gradient of loss wrt left and right
    dll = grad(_BMLossFunc, argnums=2)
    dlr = grad(_BMLossFunc, argnums=3)
    # iterate
    for i in range(iters):
        # update left
        gradL = dll(y, A, L, R)
        gradR = dlr(y, A, L, R)
        if jnp.isnan(gradL).any() or jnp.isnan(gradR).any():
            break

        # update right
        eta = BM_linesearch(y, A, L, R, gradL, gradR, eta=0.1, rho=0.5, c=0.1)
        L = L - eta * gradL
        R = R - eta * gradR

    return L @ R.T
