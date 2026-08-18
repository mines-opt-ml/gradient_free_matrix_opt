"""Tune every recovery method registered by ZOOM.

Run from the repository root with:

    python hyperparameter_tuning/run_tuning.py

The only output file is ``best_hyperparameters.json`` in this directory.
"""

from __future__ import annotations

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

ITERATIONS = 5000
MAX_QUERIES = 50_000
MATRIX_SIZE = 30
OBJECTIVE_RANK = 3
DATA_SEED = 2025
INITIALIZATION_SEED = 2026
OPTIMIZER_SEED = 2027


# Every numeric parameter has five candidate values. Parameters whose entire
# valid domain has two choices are tested at both choices.
CANDIDATES: dict[str, list[Any]] = {
    "samps_per_iter": [8, 32, 64, 128, 256, 512],
    "step_size": [0.3],
    "iters": [10, 15, 20],
    "linesearch": [True],
    "sampling_scheme": ["full-rank"],
    "sampling_dist": ["gaussian"],
}


# These defaults are only the starting point for coordinate search.
BASE_DEFAULTS: dict[str, Any] = {
    "samps_per_iter": 64,
    "step_size": 0.03,
    # Problem constraint: fixed across every configuration, never tuned.
    "target_rank": OBJECTIVE_RANK,
    "linesearch": True,
    "sampling_scheme": "full-rank",
    "sampling_dist": "gaussian",
}

RECOVERY_DEFAULTS: dict[str, dict[str, Any]] = {
    "alternating_projections": {"iters": 10},
    "iterative_hard_thresholding": {"iters": 10},
    "burer_monteiro_gradient_descent": {"iters": 10},
    "adjoint_sensing_operator": {},
    "pseudoinverse": {"iters": 10},
    "rangefinder": {},
    "spectral_iterative_hard_thresholding": {"iters": 10},
}


# Only parameters consumed by each implementation are included. Rangefinder
# creates its own samples, so it does not use samps_per_iter, sampling_scheme,
# or sampling_dist. Rank-aware methods receive the fixed target_rank.
SEARCH_PARAMETERS: dict[str, list[str]] = {
    "alternating_projections": [
        "iters",
        "samps_per_iter",
    ],
    "iterative_hard_thresholding": [
        "iters",
        "samps_per_iter",
    ],
    "burer_monteiro_gradient_descent": [
        "iters",
        "samps_per_iter",
    ],
    "adjoint_sensing_operator": [
        "samps_per_iter",
    ],
    "pseudoinverse": [
        "samps_per_iter",
    ],
    "rangefinder": [],
    "spectral_iterative_hard_thresholding": [
        "iters",
        "samps_per_iter",
    ],
}

RECOVERY_PARAMETER_NAMES = {"iters"}
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
    """Return the minimum value attained before reaching MAX_QUERIES."""
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

    best_value = math.inf
    iteration_of_best = 0

    for iteration in range(1, ITERATIONS + 1):
        # Exceptions deliberately propagate: a problem in the provided
        # optimizer, objective, or recovery implementation stops the study.
        optimizer.step()
        value = float(optimizer.function_values[-1])
        if optimizer.reached_query_budget():
            print(f"reached query budget at iteration {iteration}")
            break
        if math.isfinite(value) and value < best_value:
            best_value = value
            iteration_of_best = iteration

    if not math.isfinite(best_value):
        raise FloatingPointError(f"{method} on {objective_name} produced no finite objective value")

    return best_value, iteration_of_best


def save_results(results: dict[str, Any]) -> None:
    """Checkpoint the best configurations found so far."""
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, sort_keys=True)
        file.write("\n")


def tune(method: str, objective_name: str) -> dict[str, Any]:
    """Run one readable, single-pass coordinate search."""
    base_parameters = BASE_DEFAULTS.copy()
    recovery_parameters = RECOVERY_DEFAULTS[method].copy()
    best_value = math.inf
    iteration_of_best = 0

    print(f"\n{method} / {objective_name}", flush=True)

    for parameter in SEARCH_PARAMETERS[method]:
        candidate_results = []

        for candidate in CANDIDATES[parameter]:
            trial_base = base_parameters.copy()
            trial_recovery = recovery_parameters.copy()
            destination = trial_recovery if parameter in RECOVERY_PARAMETER_NAMES else trial_base
            destination[parameter] = candidate

            value, best_iteration = evaluate(
                method,
                objective_name,
                trial_base,
                trial_recovery,
            )
            candidate_results.append((value, best_iteration, candidate))
            print(
                f"  {parameter}={candidate!r}: minimum={value:.10g} at iteration {best_iteration}",
                flush=True,
            )

        best_value, iteration_of_best, selected = min(candidate_results)
        destination = recovery_parameters if parameter in RECOVERY_PARAMETER_NAMES else base_parameters
        destination[parameter] = selected
        print(f"  selected {parameter}={selected!r}", flush=True)

    return {
        "minimum_objective_value": best_value,
        "iteration_of_minimum": iteration_of_best,
        "base_optimizer_parameters": base_parameters,
        "recovery_parameters": recovery_parameters,
    }


def main() -> None:
    missing = set(MATRIX_RECOVERY_REGISTRY) - set(SEARCH_PARAMETERS)
    if missing:
        raise RuntimeError(f"Registry methods missing from tuning setup: {missing}")

    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open(encoding="utf-8") as file:
            results: dict[str, Any] = json.load(file)
        results["experiment"]["status"] = "running"
        results["experiment"].pop("stopped_reason", None)
    else:
        results = {
            "experiment": {
                "status": "running",
                "search_strategy": "single-pass sequential coordinate search",
                "iterations_per_configuration": ITERATIONS,
                "matrix_shape": [MATRIX_SIZE, MATRIX_SIZE],
                "objective_rank": OBJECTIVE_RANK,
                "data_seed": DATA_SEED,
                "initialization_seed": INITIALIZATION_SEED,
                "optimizer_seed": OPTIMIZER_SEED,
                "objective_noise": 0.0,
                "candidate_values": CANDIDATES,
                "search_parameters_by_method": SEARCH_PARAMETERS,
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
