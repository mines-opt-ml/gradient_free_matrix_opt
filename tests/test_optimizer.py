import jax

from base_optimizer import BaseMatrixOptimizer
from obj_func.matrix_regression import MatrixRegression
from obj_func.singular_value_sum import SingularValueSum
from recovery.adjoint_sensing_operator import adjoint_sensing_operator


# tests that the output is the correct state, and that the loss function is less for the optimized version than the initial matrix
def test_optimizer_svs():
    init_X_seed = 8566
    init_X_key = jax.random.key(init_X_seed)

    n = 30
    m = 30
    rank = 3
    samps_per_iter = 300

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
        100000,
        100000,
        True,
        False,
        "full-rank",
        "gaussian",
        adjoint_sensing_operator,
        {},
    )

    optimizer.run_optimizer()

    assert optimizer.X.shape == X0.shape
    assert svs_obj_func(optimizer.X, optimizer.get_rand_key()) <= svs_obj_func(X0, optimizer.get_rand_key())


def test_optimizer_mr():
    init_X_seed = 8566
    init_X_key = jax.random.key(init_X_seed)

    n = 30
    m = 30
    rank = 3
    samps_per_iter = 300

    X0 = jax.random.normal(init_X_key, shape=(n, m))

    mr_obj_func = MatrixRegression(
        target_rank=rank,
        m=30,
        n=30,
        ill_conditioned=True,
        noise=0.1,
    )

    optimizer = BaseMatrixOptimizer(
        X0,
        108,
        mr_obj_func,
        samps_per_iter,
        0.1,
        rank,
        0,
        100000,
        100000,
        True,
        False,
        "full-rank",
        "gaussian",
        adjoint_sensing_operator,
        {},
    )

    optimizer.run_optimizer()

    assert optimizer.X.shape == X0.shape
    assert mr_obj_func(optimizer.X, optimizer.get_rand_key()) <= mr_obj_func(X0, optimizer.get_rand_key())
