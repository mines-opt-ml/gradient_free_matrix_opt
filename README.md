# ZOOM
Zeroth order optimization for Matrices

## Installation instructions

ZOOM requires Python 3.12. You can use the package manager of your choice;
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
conda create --name ZOOM python=3.12 pip
conda activate ZOOM
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
| `rangefinder` | `{}` | Uses the optimizer's `target_rank`, `sampling_radius`, objective function, and generated JAX key. It does not use `samps_per_iter`. |

Example recovery dictionaries:

```python
adjoint_params = {}
alternating_projections_params = {"iters": 5}
burer_monteiro_params = {"iters": 5}
iht_params = {"iters": 5}
spectral_iht_params = {"iters": 5}
pseudoinverse_params = {"iters": 5}
rangefinder_params = {}
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

The configurations below are the winners from a single-pass sequential
coordinate search on fixed 30-by-30, rank-three problems. Every candidate was
run for exactly 1,000 optimizer steps, and the score was the minimum objective
value observed during those steps. The data, initialization, and optimizer
seeds were 2025, 2026, and 2027, respectively. `target_rank=3` was fixed as a
problem constraint and was not tuned.

Directional-derivative methods do not use `sampling_radius`. Rangefinder does
not use `samps_per_iter`, `sampling_scheme`, or `sampling_dist`; dashes mark
those inapplicable settings below. The complete machine-readable
configurations and search metadata are in
`hyperparameter_tuning/best_hyperparameters.json`.

| Recovery algorithm | Objective | `iters` | `samps_per_iter` | `step_size` | `sampling_radius` | `linesearch` | `sampling_scheme` | `sampling_dist` | Minimum objective |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |
| `adjoint_sensing_operator` | Matrix regression | — | 64 | 0.03 | — | `False` | `rank_one` | `gaussian` | 1.011278226e-14 |
| `adjoint_sensing_operator` | Singular-value sum | — | 256 | 0.1 | — | `False` | `full-rank` | `rademacher` | 5.723925857e-21 |
| `alternating_projections` | Matrix regression | 3 | 256 | 0.1 | — | `False` | `full-rank` | `gaussian` | 8.586539852e-15 |
| `alternating_projections` | Singular-value sum | 5 | 256 | 0.3 | — | `False` | `full-rank` | `rademacher` | 0.0 |
| `burer_monteiro_gradient_descent` | Matrix regression | 10 | 256 | 0.1 | — | `False` | `full-rank` | `gaussian` | 6.554761074e-15 |
| `burer_monteiro_gradient_descent` | Singular-value sum | 5 | 256 | 0.1 | — | `True` | `full-rank` | `gaussian` | 0.0 |
| `iterative_hard_thresholding` | Matrix regression | 2 | 256 | 0.1 | — | `False` | `full-rank` | `rademacher` | 6.130549610e-15 |
| `iterative_hard_thresholding` | Singular-value sum | 2 | 256 | 0.3 | — | `False` | `full-rank` | `rademacher` | 0.0 |
| `pseudoinverse` | Matrix regression | 10 | 256 | 0.3 | — | `False` | `rank_one` | `gaussian` | 3.716763208e-15 |
| `pseudoinverse` | Singular-value sum | 10 | 256 | 0.3 | — | `False` | `rank_one` | `gaussian` | 4.810729564e-26 |
| `rangefinder` | Matrix regression | — | — | 0.03 | 1e-5 | `False` | — | — | 7.731069673e-10 |
| `rangefinder` | Singular-value sum | — | — | 0.3 | 1e-5 | `True` | — | — | 2.396905341e-9 |
| `spectral_iterative_hard_thresholding` | Matrix regression | 2 | 256 | 0.003 | — | `True` | `full-rank` | `gaussian` | 8.971979696e-6 |
| `spectral_iterative_hard_thresholding` | Singular-value sum | 2 | 256 | 0.003 | — | `True` | `full-rank` | `gaussian` | 5.256619716e-6 |
