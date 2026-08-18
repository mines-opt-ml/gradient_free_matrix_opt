import jax
import jax.numpy as jnp

from recovery.alternating_projections import alternating_projections


def test_fr():
    samples = 300
    matrix_dim = 30
    rank = 3

    tol = 0.001

    recovery_iters = 200
    recovery_algorithm = alternating_projections

    seed = 8566
    key = jax.random.key(seed)
    key, l_key, r_key = jax.random.split(key, 3)

    L = jax.random.normal(l_key, (matrix_dim, rank))
    R = jax.random.normal(r_key, (matrix_dim, rank))

    target_matrix = L @ R.T

    print("testing on full rank matrix")
    key, fr_key = jax.random.split(key, 2)
    sensing_matrix_fr = jax.random.normal(fr_key, (samples, matrix_dim, matrix_dim)) / jnp.sqrt(samples)
    y_fr = jnp.einsum("kij,ij->k", sensing_matrix_fr, target_matrix)

    key, reco_key = jax.random.split(key, 2)
    recovered_matrix = recovery_algorithm(
        y_fr, sensing_matrix_fr, key=reco_key, target_rank=rank, iters=recovery_iters
    )

    assert y_fr.shape == (samples,)
    assert recovered_matrix.shape == (matrix_dim, matrix_dim)
    assert jnp.allclose(recovered_matrix, target_matrix, atol=tol)


def test_r1():
    samples = 300
    matrix_dim = 30
    rank = 3

    tol = 0.001

    recovery_iters = 200
    recovery_algorithm = alternating_projections

    seed = 8566
    key = jax.random.key(seed)
    key, l_key, r_key = jax.random.split(key, 3)

    L = jax.random.normal(l_key, (matrix_dim, rank))
    R = jax.random.normal(r_key, (matrix_dim, rank))

    target_matrix = L @ R.T

    key, sl_key, sr_key = jax.random.split(key, 3)
    sl = jax.random.normal(sl_key, (samples, matrix_dim, 1))
    sr = jax.random.normal(sr_key, (samples, 1, matrix_dim))
    sensing_matrix_r1 = sl @ sr / jnp.sqrt(samples)

    y_r1 = jnp.einsum("kij,ij->k", sensing_matrix_r1, target_matrix)

    key, reco_key = jax.random.split(key, 2)
    recovered_matrix = recovery_algorithm(
        y_r1, sensing_matrix_r1, key=reco_key, target_rank=rank, iters=recovery_iters
    )
    assert y_r1.shape == (samples,)
    assert recovered_matrix.shape == (matrix_dim, matrix_dim)
    assert jnp.allclose(recovered_matrix, target_matrix, atol=tol)
