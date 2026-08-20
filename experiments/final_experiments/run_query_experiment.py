"""Final experiment: objective value versus cumulative query count.

Runs every recovery method on every objective at its tuned configuration,
repeating each pair NUM_RUNS times with different optimizer seeds, and records
the full trace of each run. Completed runs already present in the CSV are
skipped, so an interrupted invocation can be resumed by rerunning it.

Run from anywhere with:

    uv run python experiments/final_experiments/run_query_experiment.py
"""

from __future__ import annotations

import argparse
import csv
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

from utils import (
    MATRIX_RECOVERY_REGISTRY,
    OBJECTIVE_NAMES,
    QUERY_RESULTS_FILE,
    append_csv,
    build_optimizer,
    failure_row,
    load_tuning_results,
    result_row,
    run_optimizer,
    run_seeds,
    smoke_test,
)

EXPERIMENT_NAME = "objective_vs_queries"


def completed_runs() -> set[tuple[str, str, str, str]]:
    """Find runs already present in the CSV, keyed by their exact configuration.

    Every trace row for a run is appended in a single write once the run
    finishes, so one success row is enough to mark the run as done. Including
    the configuration means results carried over from a different setup are
    rerun rather than silently accepted.
    """
    if not QUERY_RESULTS_FILE.exists():
        return set()
    completed = set()
    with QUERY_RESULTS_FILE.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["status"] == "success":
                completed.add((row["algorithm"], row["objective"], row["random_seed"], row["parameter_configuration"]))
    return completed


def run_query_experiment(tuning_results: dict[str, Any]) -> None:
    completed = completed_runs()
    seeds = run_seeds(tuning_results)

    for method in MATRIX_RECOVERY_REGISTRY:
        for objective_name in OBJECTIVE_NAMES:
            for optimizer_seed in seeds:
                optimizer, configuration_json = build_optimizer(tuning_results, method, objective_name, optimizer_seed)
                if (method, objective_name, str(optimizer_seed), configuration_json) in completed:
                    print(f"Skipping completed run: {method} / {objective_name} / seed {optimizer_seed}")
                    continue

                print(f"Query run: {method} / {objective_name} / seed {optimizer_seed}", flush=True)
                sample_count = optimizer.samps_per_iter

                try:
                    result = run_optimizer(optimizer, keep_trace=True)
                except Exception as error:
                    append_csv(
                        QUERY_RESULTS_FILE,
                        [
                            failure_row(
                                EXPERIMENT_NAME,
                                method,
                                objective_name,
                                sample_count,
                                optimizer_seed,
                                configuration_json,
                                error,
                            )
                        ],
                    )
                    raise

                append_csv(
                    QUERY_RESULTS_FILE,
                    [
                        result_row(
                            EXPERIMENT_NAME,
                            method,
                            objective_name,
                            sample_count,
                            optimizer_seed,
                            configuration_json,
                            result,
                            iteration=point["iteration"],
                            queries=point["queries"],
                            objective_value=point["objective_value"],
                        )
                        for point in result["trace"]
                    ],
                )
                print(
                    f"  minimum={result['minimum_objective_value']:.10g}, "
                    f"iterations={result['completed_iterations']}, "
                    f"queries={result['final_queries']}, "
                    f"seconds={result['elapsed_seconds']:.1f}",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one unsaved step for every configuration and exit.",
    )
    args = parser.parse_args()

    tuning_results = load_tuning_results()
    if args.smoke_test:
        smoke_test(tuning_results)
        return

    run_query_experiment(tuning_results)
    print(f"Query data: {QUERY_RESULTS_FILE}")


if __name__ == "__main__":
    main()
