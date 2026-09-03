# Gradient Free Optimization for Matrix Functions

## Installation instructions

This package requires Python 3.12. You can use the package manager of your choice;
instructions for `uv` and Conda are provided below. Run the commands from the
repository root.

### Using uv

Create the virtual environment and install ZOOM with the dependencies declared
in `pyproject.toml`:

```bash
uv sync
```

Run commands inside the environment with `uv run`, for example:

```bash
uv run pytest
```

You can also activate the generated `.venv` manually if desired.

### Using Conda

Create and activate a Python 3.12 environment:

```bash
conda create --name GFOPT python=3.12 pip
conda activate GFOPT
```

Then use the environment's Python installation to install ZOOM and the
dependencies declared in `pyproject.toml`:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

The Conda command uses an editable installation, so local code changes are
available without reinstalling the project.

## Recovery algorithm parameters

All recovery algorithms are selected with `recovery_algorithm` and configured
with the `recovery_params` dictionary passed to `BaseMatrixOptimizer`. The
optimizer supplies a fresh JAX `key` and passes its fixed `target_rank`
automatically on every recovery call. Neither `key` nor `target_rank` should
be placed in `recovery_params`.

For every directional-derivative recovery method, the optimizer also passes
the measurement vector `y` and sensing matrices `A` as the first two
arguments. These are generated internally and are not part of
`recovery_params`.

| Recovery algorithm | Required `recovery_params` | Description |
| --- | --- | --- |
| `adjoint_sensing_operator` | `{}` | Has no method-specific parameters. |
| `alternating_projections` | `iters` | `iters` is the number of alternating least-squares iterations. The recovered rank is the optimizer's `target_rank`. |
| `burer_monteiro_gradient_descent` | `iters` | `iters` is the number of factor-gradient iterations. The factorization rank is the optimizer's `target_rank`. The method uses its internal line search for the factor step size. |
| `iterative_hard_thresholding` | `iters` | `iters` is the number of hard-thresholding iterations. The projection rank is the optimizer's `target_rank`. |
| `spectral_iterative_hard_thresholding` | `iters` | Uses the optimizer's `target_rank` and the same iterations as IHT but returns the final spectral direction `U_r @ V_r.T` without the singular values. |
| `pseudoinverse` (`pseudo_inverse_CG`) | `iters` | `iters` is the maximum number of conjugate-gradient iterations. The solver tolerance is currently fixed at `1e-7`. |
| `lozo` | `{}` | Has no method-specific parameters. Requires a low-rank (integer) `sampling_scheme` on the optimizer. |

Example recovery dictionaries:

```python
adjoint_params = {}
alternating_projections_params = {"iters": 5}
burer_monteiro_params = {"iters": 5}
iht_params = {"iters": 5}
spectral_iht_params = {"iters": 5}
pseudoinverse_params = {"iters": 5}
lozo_params = {}
```

## Objective function parameters

Both objectives inherit from `MatrixObjectiveFunction`, provide the same
`objective(X, key)` call interface, and can be passed directly to
`BaseMatrixOptimizer`. The optimizer supplies the JAX key during evaluation.

### `MatrixRegression`

```python
MatrixRegression(
    target_rank,
    m,
    n,
    ill_conditioned=False,
    seed=0,
    noise=0.0,
)
```

| Parameter | Description |
| --- | --- |
| `target_rank` | Rank of the generated target matrix. Must satisfy `m >= n >= target_rank`. |
| `m` | Number of matrix rows. |
| `n` | Number of matrix columns. |
| `ill_conditioned` | When `True`, uses geometrically decreasing nonzero singular values. |
| `seed` | Random seed used to generate the fixed regression problem. |
| `noise` | Standard deviation multiplier for additive Gaussian objective noise. |

### `SingularValueSum`

```python
SingularValueSum(rank, noise=0.0)
```

| Parameter | Description |
| --- | --- |
| `rank` | Number of leading singular values included in the objective. |
| `noise` | Standard deviation multiplier for additive Gaussian objective noise. |

The current singular-value objective is
`0.5 * sum(singular_values[:rank]) ** 2 + noise`.

## Tuned hyperparameters

`hyperparameter_tuning/run_tuning.py` tunes `samps_per_iter` for every
registered recovery method (plus the sampling rank for `lozo`) on fixed
30-by-30, rank-three problems, selecting by last-iterate loss at the query
budget. All other parameters are fixed; `target_rank=3` is a problem
constraint and is not tuned. The complete machine-readable configurations and
search metadata are written to
`hyperparameter_tuning/best_hyperparameters.json`, which the final experiment
in `experiments/final_experiments/` reads.
