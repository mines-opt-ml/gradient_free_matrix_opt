import jax.numpy as jnp

# A is num_samples * m*n  where X is m*n


def argMinVT(A, U, y):
    AIkU = jnp.einsum("smn,mr->srn", A, U).reshape(A.shape[0], U.shape[1] * A.shape[2])
    VTvec = jnp.linalg.lstsq(AIkU, y)
    return VTvec[0].reshape(U.shape[1], A.shape[2])


def argMinU(A, VT, y):
    AVkI = jnp.einsum("smn,rn->smr", A, VT).reshape(A.shape[0], A.shape[1] * VT.shape[0])
    Uvec = jnp.linalg.lstsq(AVkI, y)
    return Uvec[0].reshape(A.shape[1], VT.shape[0])



def alternating_projections(
    y: jnp.ndarray,  # shape: (num_samples,)
    A: jnp.ndarray,  # shape: (num_samples, m, n)
    **kwargs,
) -> jnp.ndarray:  # shape: (m, n)
    
    target_rank = kwargs["target_rank"]
    iters = kwargs["iters"]

    # initialization from Jain et al
    absum = jnp.sum(y[:, None, None] * A, axis=0)
    U, S, VH = jnp.linalg.svd(absum, full_matrices=True)
    U = U[:, :target_rank]
    for i in range(iters):
        VT = argMinVT(A, U, y)
        U = argMinU(A, VT, y)

    return U @ VT

