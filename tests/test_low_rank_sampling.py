import jax
import jax.numpy as jnp

from base_optimizer import BaseMatrixOptimizer
from obj_func.singular_value_sum import SingularValueSum
from recovery.adjoint_sensing_operator import adjoint_sensing_operator


def test_low_rank_sampling():
    """
    Simple test to verify the low-rank sampling in optimizer.py is working as expected.
    """
    init_X_seed = 8566
    init_X_key = jax.random.key(init_X_seed)

    n = 31
    m = 30
    rank = 3
    samps_per_iter = 300
    sampling_scheme = 10

    X0 = jax.random.normal(init_X_key, shape=(n, m))

    svs_obj_func = SingularValueSum(rank=rank, noise=0.1)

    optimizer = BaseMatrixOptimizer(
        X0,
        108,
        svs_obj_func,
        samps_per_iter,
        0.1,
        rank,
        0,
        1000,
        100000,
        True,
        False,
        sampling_scheme,
        "gaussian",
        adjoint_sensing_operator,
        {},
    )

    optimizer.run_optimizer()

    sampling_operator = optimizer.Z

    for i in range(samps_per_iter):
        """
        Check each matrix contained in the sampling operator is indeed low rank
        """
        sampling_matrix = sampling_operator[i, :, :]
        _, S, _ = jnp.linalg.svd(sampling_matrix)
        residual_singular_values = jnp.sum(S[sampling_scheme:])
        assert residual_singular_values <= 1e-6, f"residual sum is {residual_singular_values}"

    assert optimizer.X.shape == X0.shape
    assert svs_obj_func(optimizer.X, optimizer.get_rand_key()) <= svs_obj_func(X0, optimizer.get_rand_key())
