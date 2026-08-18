import jax
import jax.numpy as jnp

from recovery import pseudoinverse as pseudoinverse_module
from recovery.pseudoinverse import pseudo_inverse_CG


def test_pseudoinverse_cg_recovers_from_identity_measurements():
    matrix_dimension = 3
    sensing_operator = jnp.eye(matrix_dimension**2).reshape(
        matrix_dimension**2,
        matrix_dimension,
        matrix_dimension,
    )
    target_matrix = jnp.arange(
        1,
        matrix_dimension**2 + 1,
        dtype=jnp.float32,
    ).reshape(matrix_dimension, matrix_dimension)
    measurements = jnp.einsum(
        "kij,ij->k",
        sensing_operator,
        target_matrix,
    )

    recovered_matrix = pseudo_inverse_CG(
        measurements,
        sensing_operator,
        iters=10,
        key=jax.random.key(1),
    )

    assert recovered_matrix.shape == target_matrix.shape
    assert jnp.allclose(recovered_matrix, target_matrix, atol=1e-5)


