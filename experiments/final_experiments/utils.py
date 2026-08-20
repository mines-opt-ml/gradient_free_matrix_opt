"""Shared setup for the final 30-by-30 ZOOM query experiment.

The experiment script reads its tuned configurations from
``hyperparameter_tuning/best_hyperparameters.json`` and checkpoints completed
runs to CSV, so an interrupted invocation can be resumed by rerunning it.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp

from base_optimizer import BaseMatrixOptimizer
from obj_func.ky_fan_regression import KyFanRegression
from obj_func.singular_value_sum import SingularValueSum
from recovery.registry import MATRIX_RECOVERY_REGISTRY

EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_DIR / "data"
BEST_PARAMETERS_FILE = PROJECT_ROOT / "hyperparameter_tuning" / "best_hyperparameters.json"
QUERY_RESULTS_FILE = DATA_DIR / "objective_vs_queries.csv"

# The query budget is the real stopping condition; ITERATIONS is only a cap so a
# method that somehow spends no queries cannot loop forever. It must stay above
# MAX_QUERIES divided by the smallest tuned samps_per_iter, or a cheap
# configuration would stop early and be compared against runs that got the full
# budget.
MAX_QUERIES = 100_000
ITERATIONS = 20_000
# Each method/objective pair is run this many times, varying only the
# optimizer seed: the problem data and the starting point stay fixed.
NUM_RUNS = 10
OBJECTIVE_NAMES = ("ky_fan_regression", "singular_value_sum")

CSV_FIELDS = (
    "experiment",
    "algorithm",
    "objective",
    "iteration",
    "queries",
    "samps_per_iter",
    "objective_value",
    "minimum_objective_value",
    "iteration_of_minimum",
    "random_seed",
    "status",
    "parameter_configuration",
    "elapsed_seconds",
    "error_message",
)


def run_seeds(tuning_results: dict[str, Any]) -> list[int]:
    """The optimizer seeds for the NUM_RUNS independent runs of each pair."""
    base_seed = tuning_results["experiment"]["optimizer_seed"]
    return [base_seed + run_index for run_index in range(NUM_RUNS)]


def load_tuning_results() -> dict[str, Any]:
    """Load and validate the completed tuning output."""
    with BEST_PARAMETERS_FILE.open(encoding="utf-8") as file:
        tuning_results = json.load(file)

    experiment = tuning_results["experiment"]
    if experiment["status"] != "complete":
        raise RuntimeError("Hyperparameter tuning results are not complete")
    if experiment["matrix_shape"] != [30, 30]:
        raise RuntimeError("Expected hyperparameters tuned on 30-by-30 matrices")
    if experiment["objective_rank"] != 3:
        raise RuntimeError("Expected hyperparameters with fixed target rank 3")

    configured_methods = set(tuning_results["best_parameters"])
    registered_methods = set(MATRIX_RECOVERY_REGISTRY)
    if configured_methods != registered_methods:
        raise RuntimeError(
            "Tuned and registered recovery methods differ: "
            f"tuned={sorted(configured_methods)}, "
            f"registered={sorted(registered_methods)}"
        )

    for method in configured_methods:
        configured_objectives = set(tuning_results["best_parameters"][method])
        if configured_objectives != set(OBJECTIVE_NAMES):
            raise RuntimeError(f"{method} is tuned for {sorted(configured_objectives)}, expected {sorted(OBJECTIVE_NAMES)}")

    return tuning_results


def make_initial_matrix(matrix_size: int, initialization_seed: int) -> jnp.ndarray:
    """Draw the full-rank Gaussian starting point shared by every run."""
    return jax.random.normal(jax.random.key(initialization_seed), (matrix_size, matrix_size))


def make_objective(
    objective_name: str,
    matrix_size: int,
    target_rank: int,
    data_seed: int,
    noise: float,
):
    """Create a fresh objective for one independent run."""
    if objective_name == "ky_fan_regression":
        return KyFanRegression(
            rank=target_rank,
            m=matrix_size,
            n=matrix_size,
            seed=data_seed,
            noise=noise,
        )
    if objective_name == "singular_value_sum":
        return SingularValueSum(rank=target_rank, noise=noise)
    raise ValueError(f"Unknown objective: {objective_name}")


def build_optimizer(
    tuning_results: dict[str, Any],
    method: str,
    objective_name: str,
    optimizer_seed: int,
    max_queries: int = MAX_QUERIES,
) -> tuple[BaseMatrixOptimizer, str]:
    """Construct an optimizer from one saved winning configuration."""
    experiment = tuning_results["experiment"]
    winner = tuning_results["best_parameters"][method][objective_name]
    base_parameters = winner["base_optimizer_parameters"].copy()
    recovery_parameters = winner["recovery_parameters"].copy()

    target_rank = base_parameters["target_rank"]
    matrix_size = experiment["matrix_shape"][0]
    objective = make_objective(
        objective_name,
        matrix_size,
        target_rank,
        experiment["data_seed"],
        experiment["objective_noise"],
    )

    optimizer = BaseMatrixOptimizer(
        X0=make_initial_matrix(matrix_size, experiment["initialization_seed"]),
        rand_seed=optimizer_seed,
        obj_func=objective,
        samps_per_iter=base_parameters["samps_per_iter"],
        step_size=base_parameters["step_size"],
        target_rank=target_rank,
        target_obj_func_val=-math.inf,
        max_queries=max_queries,
        max_time=sys.maxsize,
        report=True,
        linesearch=base_parameters["linesearch"],
        sampling_scheme=base_parameters["sampling_scheme"],
        sampling_dist=base_parameters["sampling_dist"],
        recovery_algorithm=MATRIX_RECOVERY_REGISTRY[method],
        recovery_params=recovery_parameters,
    )

    # ITERATIONS is deliberately absent: it is a safety cap, not a parameter of
    # the experiment. The optimizer seed is also absent: it is the per-run
    # variable, recorded in the CSV's random_seed column instead. Runs are
    # resumed by matching against this string.
    exact_configuration = {
        "max_queries": max_queries,
        "matrix_shape": experiment["matrix_shape"],
        "data_seed": experiment["data_seed"],
        "initialization_seed": experiment["initialization_seed"],
        "objective_noise": experiment["objective_noise"],
        "base_optimizer_parameters": base_parameters,
        "recovery_parameters": recovery_parameters,
    }
    configuration_json = json.dumps(
        exact_configuration,
        sort_keys=True,
        separators=(",", ":"),
    )
    return optimizer, configuration_json


def run_optimizer(
    optimizer: BaseMatrixOptimizer,
    keep_trace: bool,
    iterations: int = ITERATIONS,
    require_full_budget: bool = True,
) -> dict[str, Any]:
    """Step until the query budget is exhausted, capped at ``iterations`` steps."""
    trace = []
    if keep_trace:
        trace.append(
            {
                "iteration": 0,
                "queries": optimizer.query_vec[0],
                "objective_value": optimizer.function_values[0],
            }
        )

    minimum_value = math.inf
    iteration_of_minimum = 0
    completed_iterations = 0
    start_time = time.perf_counter()

    for iteration in range(1, iterations + 1):
        optimizer.step()
        optimizer.report_progress(False)
        objective_value = float(optimizer.function_values[-1])
        if not math.isfinite(objective_value):
            raise FloatingPointError(f"Non-finite objective at optimizer iteration {iteration}")

        completed_iterations = iteration
        if objective_value < minimum_value:
            minimum_value = objective_value
            iteration_of_minimum = iteration

        if keep_trace:
            trace.append(
                {
                    "iteration": iteration,
                    "queries": int(optimizer.query_vec[-1]),
                    "objective_value": objective_value,
                }
            )

        if optimizer.reached_query_budget():
            break
    else:
        # Falling out of the loop means the cap stopped us, not the budget, so
        # this run is not comparable to the others. Fail rather than record it.
        if require_full_budget:
            raise RuntimeError(
                f"Hit the {iterations}-iteration cap after only {optimizer.number_of_queries_made} "
                f"of {optimizer.max_queries} queries; raise ITERATIONS"
            )

    elapsed_seconds = time.perf_counter() - start_time
    return {
        "trace": trace,
        "completed_iterations": completed_iterations,
        "final_queries": int(optimizer.query_vec[-1]),
        "final_objective_value": float(optimizer.function_values[-1]),
        "minimum_objective_value": minimum_value,
        "iteration_of_minimum": iteration_of_minimum,
        "elapsed_seconds": elapsed_seconds,
    }


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append one fully completed run and flush it to persistent storage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())


def result_row(
    experiment_name: str,
    method: str,
    objective_name: str,
    sample_count: int,
    optimizer_seed: int,
    configuration_json: str,
    result: dict[str, Any],
    iteration: int,
    queries: int,
    objective_value: float,
) -> dict[str, Any]:
    """Record one successful trace point."""
    return {
        "experiment": experiment_name,
        "algorithm": method,
        "objective": objective_name,
        "iteration": iteration,
        "queries": queries,
        "samps_per_iter": sample_count,
        "objective_value": objective_value,
        "minimum_objective_value": result["minimum_objective_value"],
        "iteration_of_minimum": result["iteration_of_minimum"],
        "random_seed": optimizer_seed,
        "status": "success",
        "parameter_configuration": configuration_json,
        "elapsed_seconds": result["elapsed_seconds"],
        "error_message": "",
    }


def failure_row(
    experiment_name: str,
    method: str,
    objective_name: str,
    sample_count: int,
    optimizer_seed: int,
    configuration_json: str,
    error: Exception,
) -> dict[str, Any]:
    """Record a failed run."""
    return {
        "experiment": experiment_name,
        "algorithm": method,
        "objective": objective_name,
        "iteration": "",
        "queries": "",
        "samps_per_iter": sample_count,
        "objective_value": "",
        "minimum_objective_value": "",
        "iteration_of_minimum": "",
        "random_seed": optimizer_seed,
        "status": "failed",
        "parameter_configuration": configuration_json,
        "elapsed_seconds": "",
        "error_message": f"{type(error).__name__}: {error}",
    }


def smoke_test(tuning_results: dict[str, Any]) -> None:
    """Exercise every saved method/objective configuration without writing data."""
    first_seed = run_seeds(tuning_results)[0]
    for method in MATRIX_RECOVERY_REGISTRY:
        for objective_name in OBJECTIVE_NAMES:
            optimizer, _ = build_optimizer(tuning_results, method, objective_name, first_seed)
            run_optimizer(optimizer, keep_trace=False, iterations=1, require_full_budget=False)
            print(f"Smoke test passed: {method} / {objective_name}")
