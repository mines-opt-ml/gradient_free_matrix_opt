import jax
import jax.numpy as jnp

from recovery.iterative_hard_thresholding import (
    spectral_iterative_hard_thresholding,
)


def test_spectral_iht_recovers_a_rank_one_spectral_direction():
    matrix_dimension = 3
    scale = 0.5
    sensing_operator = scale * jnp.eye(matrix_dimension**2).reshape(
        matrix_dimension**2,
        matrix_dimension,
        matrix_dimension,
    )
    target_matrix = jnp.diag(jnp.array([1.0, 0.0, 0.0]))
    measurements = jnp.einsum(
        "kij,ij->k",
        sensing_operator,
        target_matrix,
    )

    recovered_matrix = spectral_iterative_hard_thresholding(
        measurements,
        sensing_operator,
        target_rank=1,
        iters=1,
        key=jax.random.key(2),
    )

    assert recovered_matrix.shape == target_matrix.shape
    assert jnp.allclose(recovered_matrix, target_matrix, atol=1e-5)



