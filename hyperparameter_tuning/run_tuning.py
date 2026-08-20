"""Tune samps_per_iter for every recovery method registered by ZOOM.

For each method and objective, every candidate value of ``samps_per_iter`` is
run to the full query budget and the value with the lowest last-iterate loss
is selected. For lozo, which requires low-rank sampling, the sampling rank is
tuned jointly with samps_per_iter over a small grid. All other parameters are
fixed at the defaults below.

Run from the repository root with:

    python hyperparameter_tuning/run_tuning.py

The only output file is ``best_hyperparameters.json`` in this directory.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

# Allow the script to run directly without installing ZOOM as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from base_optimizer import BaseMatrixOptimizer
from obj_func.ky_fan_regression import KyFanRegression
from obj_func.singular_value_sum import SingularValueSum
from recovery.registry import MATRIX_RECOVERY_REGISTRY

OUTPUT_FILE = Path(__file__).with_name("best_hyperparameters.json")

# The query budget is the real stopping condition; ITERATIONS is only a cap so
# a configuration that somehow spends no queries cannot loop forever. It must
# stay above MAX_QUERIES / min(SAMPLE_CANDIDATES), or the cheapest candidate
# would be stopped by the cap instead of the budget and its last-iterate loss
# would not be comparable to the others.
ITERATIONS = 10_000
MAX_QUERIES = 50_000
MATRIX_SIZE = 30
OBJECTIVE_RANK = 3
DATA_SEED = 2025
INITIALIZATION_SEED = 2026
OPTIMIZER_SEED = 2027

# Tuned for every method.
SAMPLE_CANDIDATES = [8, 32, 64, 128, 256, 512]
# Tuned for lozo only, which requires low-rank sampling instead of the
# full-rank scheme every other method uses.
SAMPLING_RANK_CANDIDATES = [4, 8, 16]

# Fixed for every configuration, never tuned.
BASE_DEFAULTS: dict[str, Any] = {
    "step_size": 0.03,
    # Problem constraint: fixed across every configuration.
    "target_rank": OBJECTIVE_RANK,
    "linesearch": True,
    "sampling_scheme": "full-rank",
    "sampling_dist": "gaussian",
}


# The grid searched for each method, as {parameter: candidate values}. Every
# combination is evaluated and the one with the lowest last-iterate loss wins.
def search_space(method: str) -> dict[str, list[Any]]:
    space: dict[str, list[Any]] = {"samps_per_iter": SAMPLE_CANDIDATES}
    if method == "lozo":
        space["sampling_scheme"] = SAMPLING_RANK_CANDIDATES
    return space


# Fixed recovery-side parameters, never tuned. Only parameters consumed by each
# implementation are included.
RECOVERY_DEFAULTS: dict[str, dict[str, Any]] = {
    "alternating_projections": {"iters": 20},
    "iterative_hard_thresholding": {"iters": 20},
    "burer_monteiro_gradient_descent": {"iters": 20},
    "IHT_SpecGD": {"iters": 20},
    "adjoint_sensing_operator": {},
    "pseudoinverse": {"iters": 20},
    "lozo": {},
}

OBJECTIVE_NAMES = ("ky_fan_regression", "singular_value_sum")


def make_initial_matrix() -> jnp.ndarray:
    """Create the same deterministic rank-three initialization for every run."""
    key = jax.random.key(INITIALIZATION_SEED)
    left_key, right_key = jax.random.split(key)
    left = jax.random.normal(left_key, (MATRIX_SIZE, OBJECTIVE_RANK))
    right = jax.random.normal(right_key, (MATRIX_SIZE, OBJECTIVE_RANK))
    return (left @ right.T) / MATRIX_SIZE


def make_objective(name: str):
    """Create a fresh copy of one of the two provided objective functions."""
    if name == "ky_fan_regression":
        return KyFanRegression(
            rank=OBJECTIVE_RANK,
            m=MATRIX_SIZE,
            n=MATRIX_SIZE,
            seed=DATA_SEED,
            noise=0.0,
        )
    if name == "singular_value_sum":
        return SingularValueSum(rank=OBJECTIVE_RANK, noise=0.0)
    raise ValueError(f"Unknown objective: {name}")


def evaluate(
    method: str,
    objective_name: str,
    base_parameters: dict[str, Any],
    recovery_parameters: dict[str, Any],
) -> tuple[float, int]:
    """Return the last-iterate loss and iteration count at MAX_QUERIES."""
    optimizer = BaseMatrixOptimizer(
        X0=make_initial_matrix(),
        rand_seed=OPTIMIZER_SEED,
        obj_func=make_objective(objective_name),
        samps_per_iter=base_parameters["samps_per_iter"],
        step_size=base_parameters["step_size"],
        target_rank=base_parameters["target_rank"],
        target_obj_func_val=-math.inf,
        max_queries=MAX_QUERIES,
        max_time=sys.maxsize,
        report=False,
        linesearch=base_parameters["linesearch"],
        sampling_scheme=base_parameters["sampling_scheme"],
        sampling_dist=base_parameters["sampling_dist"],
        recovery_algorithm=MATRIX_RECOVERY_REGISTRY[method],
        recovery_params=recovery_parameters,
    )

    for iteration in range(1, ITERATIONS + 1):
        # Exceptions deliberately propagate: a problem in the provided
        # optimizer, objective, or recovery implementation stops the study.
        optimizer.step()
        if optimizer.reached_query_budget():
            break
    else:
        raise RuntimeError(
            f"Hit the {ITERATIONS}-iteration cap after only {optimizer.number_of_queries_made} "
            f"of {optimizer.max_queries} queries; raise ITERATIONS"
        )

    final_value = float(optimizer.function_values[-1])
    if not math.isfinite(final_value):
        raise FloatingPointError(f"{method} on {objective_name} produced a non-finite final objective value")
    return final_value, iteration


def save_results(results: dict[str, Any]) -> None:
    """Checkpoint the best configurations found so far."""
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, sort_keys=True)
        file.write("\n")


def tune(method: str, objective_name: str) -> dict[str, Any]:
    """Grid-search the method's space and keep the lowest last-iterate loss."""
    recovery_parameters = RECOVERY_DEFAULTS[method].copy()
    space = search_space(method)
    parameter_names = list(space)

    print(f"\n{method} / {objective_name}", flush=True)

    candidate_results = []
    for combination in itertools.product(*(space[name] for name in parameter_names)):
        overrides = dict(zip(parameter_names, combination))
        trial_base = {**BASE_DEFAULTS, **overrides}
        final_value, iterations = evaluate(method, objective_name, trial_base, recovery_parameters)
        candidate_results.append((final_value, list(combination)))
        settings = ", ".join(f"{name}={value}" for name, value in overrides.items())
        print(f"  {settings}: final={final_value:.10g} after {iterations} iterations", flush=True)

    final_value, selected = min(candidate_results)
    selected_overrides = dict(zip(parameter_names, selected))
    print(f"  selected {', '.join(f'{name}={value}' for name, value in selected_overrides.items())}", flush=True)

    return {
        "final_objective_value": final_value,
        "base_optimizer_parameters": {**BASE_DEFAULTS, **selected_overrides},
        "recovery_parameters": recovery_parameters,
    }


def main() -> None:
    missing = set(MATRIX_RECOVERY_REGISTRY) - set(RECOVERY_DEFAULTS)
    if missing:
        raise RuntimeError(f"Registry methods missing from tuning setup: {missing}")

    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open(encoding="utf-8") as file:
            results: dict[str, Any] = json.load(file)
        results["experiment"]["status"] = "running"
    else:
        results = {
            "experiment": {
                "status": "running",
                "search_strategy": "grid search selected by last-iterate loss",
                "max_queries": MAX_QUERIES,
                "matrix_shape": [MATRIX_SIZE, MATRIX_SIZE],
                "objective_rank": OBJECTIVE_RANK,
                "data_seed": DATA_SEED,
                "initialization_seed": INITIALIZATION_SEED,
                "optimizer_seed": OPTIMIZER_SEED,
                "objective_noise": 0.0,
                "candidate_values": {
                    "samps_per_iter": SAMPLE_CANDIDATES,
                    "sampling_scheme (lozo only)": SAMPLING_RANK_CANDIDATES,
                },
            },
            "best_parameters": {},
        }

    for method in MATRIX_RECOVERY_REGISTRY:
        results["best_parameters"].setdefault(method, {})
        for objective_name in OBJECTIVE_NAMES:
            if objective_name in results["best_parameters"][method]:
                print(f"Skipping completed {method} / {objective_name}")
                continue
            results["best_parameters"][method][objective_name] = tune(method, objective_name)
            save_results(results)

    results["experiment"]["status"] = "complete"
    save_results(results)
    print(f"\nSaved best configurations to {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
