"""Experiment 2: samples per iteration versus minimum objective value.

Sweeps SAMPLE_COUNTS for every recovery method and objective, recording one
summary row per run. Completed rows already present in the CSV are skipped, so
an interrupted invocation can be resumed by rerunning it.

Run from anywhere with:

    uv run python experiments/final_experiments/run_sample_experiment.py
"""

from __future__ import annotations

import argparse
import csv
import time
from typing import Any

from utils import (
    MATRIX_RECOVERY_REGISTRY,
    OBJECTIVE_NAMES,
    RANGEFINDER,
    SAMPLE_COUNTS,
    SAMPLE_RESULTS_FILE,
    FailOnRecoveryOutput,
    NumericalRecoveryError,
    append_csv,
    build_optimizer,
    failure_row,
    load_tuning_results,
    replace_csv,
    result_row,
    run_optimizer,
    smoke_test,
    summarize_partial_run,
)

EXPERIMENT_NAME = "samples_vs_minimum"


def completed_runs() -> set[tuple[str, str, str, str]]:
    """Find complete or inapplicable rows already saved, keyed by configuration.

    Including the configuration means results carried over from a different
    query budget are rerun rather than silently accepted.
    """
    if not SAMPLE_RESULTS_FILE.exists():
        return set()
    completed = set()
    with SAMPLE_RESULTS_FILE.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["status"] in {"success", "failed", "not_applicable"}:
                completed.add(
                    (row["algorithm"], row["objective"], row["samps_per_iter"], row["parameter_configuration"])
                )
    return completed


def prune_obsolete_sample_counts() -> None:
    """Remove rows outside the currently configured sample-count study."""
    if not SAMPLE_RESULTS_FILE.exists():
        return

    with SAMPLE_RESULTS_FILE.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    configured_counts = {str(value) for value in SAMPLE_COUNTS}
    retained_rows = [row for row in rows if not row["samps_per_iter"] or row["samps_per_iter"] in configured_counts]
    removed_count = len(rows) - len(retained_rows)
    if removed_count:
        replace_csv(SAMPLE_RESULTS_FILE, retained_rows)
        print(f"Removed {removed_count} obsolete sample-count rows.", flush=True)


def not_applicable_row(
    method: str,
    objective_name: str,
    optimizer_seed: int,
    configuration_json: str,
) -> dict[str, Any]:
    """Record that a method does not take a samps_per_iter setting."""
    return {
        "experiment": EXPERIMENT_NAME,
        "algorithm": method,
        "objective": objective_name,
        "iteration": "",
        "queries": "",
        "samps_per_iter": "",
        "objective_value": "",
        "minimum_objective_value": "",
        "iteration_of_minimum": "",
        "random_seed": optimizer_seed,
        "status": "not_applicable",
        "parameter_configuration": configuration_json,
        "elapsed_seconds": "",
        "error_message": f"{RANGEFINDER} does not use samps_per_iter",
    }


def run_sample_experiment(tuning_results: dict[str, Any]) -> None:
    completed = completed_runs()
    optimizer_seed = tuning_results["experiment"]["optimizer_seed"]

    for method in MATRIX_RECOVERY_REGISTRY:
        for objective_name in OBJECTIVE_NAMES:
            if method == RANGEFINDER:
                _, configuration_json = build_optimizer(tuning_results, method, objective_name)
                if (method, objective_name, "", configuration_json) not in completed:
                    append_csv(
                        SAMPLE_RESULTS_FILE,
                        [not_applicable_row(method, objective_name, optimizer_seed, configuration_json)],
                    )
                    print(f"Sample run: {method} / {objective_name}: not applicable", flush=True)
                continue

            for sample_count in SAMPLE_COUNTS:
                optimizer, configuration_json = build_optimizer(
                    tuning_results,
                    method,
                    objective_name,
                    samps_per_iter=sample_count,
                )
                if (method, objective_name, str(sample_count), configuration_json) in completed:
                    print(f"Skipping completed sample run: {method} / {objective_name} / {sample_count}")
                    continue

                print(f"Sample run: {method} / {objective_name} / {sample_count}", flush=True)
                run_started = time.perf_counter()
                try:
                    result = run_optimizer(optimizer, keep_trace=False)
                except Exception as error:
                    append_csv(
                        SAMPLE_RESULTS_FILE,
                        [
                            failure_row(
                                EXPERIMENT_NAME,
                                method,
                                objective_name,
                                sample_count,
                                optimizer_seed,
                                configuration_json,
                                error,
                                partial_result=summarize_partial_run(optimizer),
                                elapsed_seconds=time.perf_counter() - run_started,
                            )
                        ],
                    )
                    # A numerical blowup is a result for this sweep, not a bug;
                    # anything else means the harness itself is broken.
                    if not isinstance(error, (FloatingPointError, NumericalRecoveryError)):
                        raise
                    print(f"  failed numerical run: {error}", flush=True)
                    continue

                append_csv(
                    SAMPLE_RESULTS_FILE,
                    [
                        result_row(
                            EXPERIMENT_NAME,
                            method,
                            objective_name,
                            sample_count,
                            optimizer_seed,
                            configuration_json,
                            result,
                            iteration=result["completed_iterations"],
                            queries=result["final_queries"],
                            objective_value=result["final_objective_value"],
                        )
                    ],
                )
                print(
                    f"  minimum={result['minimum_objective_value']:.10g}, seconds={result['elapsed_seconds']:.1f}",
                    flush=True,
                )

    prune_obsolete_sample_counts()


def refresh_failed_minima(tuning_results: dict[str, Any]) -> None:
    """Reproduce failed sample runs and retain their finite partial results."""
    if not SAMPLE_RESULTS_FILE.exists():
        raise FileNotFoundError("Run the sample experiment before refreshing it")

    with SAMPLE_RESULTS_FILE.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    failed_row_indexes = [
        index for index, row in enumerate(rows) if row["experiment"] == EXPERIMENT_NAME and row["status"] == "failed"
    ]
    optimizer_seed = tuning_results["experiment"]["optimizer_seed"]

    for row_index in failed_row_indexes:
        previous_row = rows[row_index]
        method = previous_row["algorithm"]
        objective_name = previous_row["objective"]
        sample_count = int(previous_row["samps_per_iter"])
        print(f"Refreshing failed sample run: {method} / {objective_name} / {sample_count}", flush=True)

        optimizer, configuration_json = build_optimizer(
            tuning_results,
            method,
            objective_name,
            samps_per_iter=sample_count,
        )
        run_started = time.perf_counter()
        try:
            result = run_optimizer(optimizer, keep_trace=False)
        except (FloatingPointError, NumericalRecoveryError) as error:
            partial_result = summarize_partial_run(optimizer)
            rows[row_index] = failure_row(
                EXPERIMENT_NAME,
                method,
                objective_name,
                sample_count,
                optimizer_seed,
                configuration_json,
                error,
                partial_result=partial_result,
                elapsed_seconds=time.perf_counter() - run_started,
            )
            if partial_result is None:
                print("  failed before completing a finite iteration", flush=True)
            else:
                print(
                    f"  failed after {partial_result['completed_iterations']} iterations; "
                    f"minimum before failure={partial_result['minimum_objective_value']:.10g}",
                    flush=True,
                )
            continue

        rows[row_index] = result_row(
            EXPERIMENT_NAME,
            method,
            objective_name,
            sample_count,
            optimizer_seed,
            configuration_json,
            result,
            iteration=result["completed_iterations"],
            queries=result["final_queries"],
            objective_value=result["final_objective_value"],
        )
        print(f"  completed successfully; minimum={result['minimum_objective_value']:.10g}", flush=True)

    replace_csv(SAMPLE_RESULTS_FILE, rows)
    print(f"Refreshed {len(failed_row_indexes)} failed rows.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one unsaved step for every configuration and exit.",
    )
    parser.add_argument(
        "--refresh-failed-minima",
        action="store_true",
        help="Rerun existing failed configurations, save the minimum reached before failure, and exit.",
    )
    args = parser.parse_args()

    tuning_results = load_tuning_results()
    if args.smoke_test:
        smoke_test(tuning_results)
        return

    if args.refresh_failed_minima:
        refresh_failed_minima(tuning_results)
    else:
        run_sample_experiment(tuning_results)
    print(f"Sample data: {SAMPLE_RESULTS_FILE}")


if __name__ == "__main__":
    main()
