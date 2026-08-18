import copy
import sys
import time
from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp

from obj_func.obj_func_class import MatrixObjectiveFunction
from recovery.directional_derivative import directional_derivatives


class BaseMatrixOptimizer:
    def __init__(
        self,
        X0: jnp.ndarray,  # shape: (m, n)
        rand_seed: int,
        obj_func: MatrixObjectiveFunction,
        samps_per_iter: int,
        step_size: float,
        target_rank: int,
        target_obj_func_val: float,
        max_queries: int,
        max_time: int,
        report: bool,
        linesearch: bool,
        sampling_scheme: Literal["full-rank"] | int,  # if int then we do low-rank sampling
        sampling_dist: Literal["gaussian", "rademacher"],
        # recovery parameters
        recovery_algorithm: Callable,
        recovery_params: dict[str, Any],  # depends on the recovery algorithm
    ):

        # set starting parameters for optimizer class
        self.X = copy.copy(X0)
        self.dim0, self.dim1 = X0.shape
        self.obj_func = obj_func

        self.key = jax.random.key(rand_seed)
        self.step_size = step_size
        self.samps_per_iter = samps_per_iter
        self.target_rank = target_rank
        self.target_obj_func_val = target_obj_func_val
        self.max_queries = max_queries
        self.max_time = max_time
        self.report = report

        self.recovery_algorithm = recovery_algorithm
        self.recovery_params = recovery_params.copy()

        self.linesearch = linesearch
        self.sampling_scheme = sampling_scheme
        self.sampling_dist = sampling_dist
        self.function_values = [float(self.obj_func(self.X, self.get_rand_key()))]
        self.number_of_queries_made = 0
        self.query_vec = [0]

        self.Z = self.get_sampling_matrix()  # essential to do this here for rank-aware methods.

        self.time_vec = [0.0]

    # get a new random key, need to do this every time when computing a random number
    def get_rand_key(self) -> jax.Array:
        key, subKey = jax.random.split(self.key)
        # assign new key to optimizer
        self.key = key

        # return key for use in rand function
        return subKey

    # find z
    def get_sampling_matrix(self) -> jnp.ndarray:
        if isinstance(self.sampling_scheme, int):
            sampling_rank = self.sampling_scheme
            assert sampling_rank <= min(self.dim0, self.dim1), "sampling rank must be less than min(m,n)"
            # random normalized z matrix construced from an outer product
            if self.sampling_dist == "gaussian":
                vec1 = jax.random.normal(self.get_rand_key(), shape=(self.samps_per_iter, self.dim0, sampling_rank))
                vec2 = jax.random.normal(self.get_rand_key(), shape=(self.samps_per_iter, sampling_rank, self.dim1))
            elif self.sampling_dist == "rademacher":
                vec1 = jax.random.rademacher(self.get_rand_key(), shape=(self.samps_per_iter, self.dim0, sampling_rank))
                vec2 = jax.random.rademacher(self.get_rand_key(), shape=(self.samps_per_iter, sampling_rank, self.dim1))

            return vec1 @ vec2 / (sampling_rank * jnp.sqrt(self.samps_per_iter))
        else:
            # random normalized z matrix
            if self.sampling_dist == "gaussian":
                return jax.random.normal(self.get_rand_key(), shape=(self.samps_per_iter, self.dim0, self.dim1)) / jnp.sqrt(
                    self.samps_per_iter
                )
            elif self.sampling_dist == "rademacher":
                return jax.random.rademacher(self.get_rand_key(), shape=(self.samps_per_iter, self.dim0, self.dim1)) / jnp.sqrt(
                    self.samps_per_iter
                )

    # calculate directional derivatives of the objective function at X in the directions of Z
    def directional_derivative(self) -> jnp.ndarray:
        y = directional_derivatives(self.obj_func, self.X, self.Z, self.get_rand_key())
        self.number_of_queries_made += self.samps_per_iter
        return y

    # runs the recovery algorithm to get an estimate of the gradient of the objective function at X
    def recover_gradient(self, y: jnp.ndarray) -> jnp.ndarray:
        self.recovery_params["key"] = self.get_rand_key()
        self.recovery_params["target_rank"] = self.target_rank
        gradient = self.recovery_algorithm(y, self.Z, **self.recovery_params)

        return gradient

    # line search method for step size
    def backtrackinglinesearch(self, f_X, gradient_estimate, alpha=1, rho=0.75, c=0.001):
        current_value = f_X
        while (
            self.obj_func(self.X - alpha * gradient_estimate, self.get_rand_key())
            > current_value - c * alpha * jnp.trace(jnp.transpose(gradient_estimate) @ gradient_estimate)
            and alpha > 1e-10
        ):
            alpha = rho * alpha
            # self.number_of_queries_made += 1 # We don't count function evaluations for now.
        # self.number_of_queries_made += 1
        return alpha

    # performs one step
    def step(self):
        start_time = time.time()

        if getattr(self.recovery_algorithm, "__name__", None) in ("pseudo_inverse_CG", "adjoint_sensing_operator", "lozo"):
            # get the sampling matrix
            self.Z = self.get_sampling_matrix()

        # calculate directional derivatives
        y = self.directional_derivative()

        # get gradient estimate
        f_X = self.obj_func(self.X, self.get_rand_key())
        gradient_estimate = self.recover_gradient(y)

        # linesearch
        if self.linesearch:
            self.step_size = self.backtrackinglinesearch(f_X, gradient_estimate)

        # take step
        self.X = self.X - self.step_size * gradient_estimate
        f_X = self.obj_func(self.X, self.get_rand_key())
        # self.number_of_queries_made += 1 #Don't count function evals for now.
        end_time = time.time()

        cumulative_time = self.time_vec[-1] + end_time - start_time
        self.time_vec.append(cumulative_time)
        self.function_values.append(float(f_X))
        self.query_vec.append(self.number_of_queries_made)

    # check ending conditions
    def reached_query_budget(self) -> bool:
        return self.number_of_queries_made >= self.max_queries

    def reached_time_limit(self) -> bool:
        return self.time_vec[-1] >= self.max_time

    def reached_target_of_value(self) -> bool:
        return self.function_values[-1] <= self.target_obj_func_val

    def report_progress(self, final: bool):
        if self.report == True and final == False:
            string_to_print = (
                f"Current function value: {self.function_values[-1]}, number of queries made: {self.number_of_queries_made}"
            )
            sys.stdout.write(string_to_print + "\n")
            sys.stdout.flush()

        if final == True:
            string_to_print = (
                f"Final function value: {self.function_values[-1]}, number of queries made: {self.number_of_queries_made}"
            )
            sys.stdout.write(string_to_print + "\n")
            sys.stdout.flush()

    # assumes optimizer is already initialized
    def run_optimizer(self):
        finished = False
        while not finished:
            self.step()

            if self.reached_query_budget() or self.reached_time_limit() or self.reached_target_of_value():
                finished = True

            self.report_progress(finished)
