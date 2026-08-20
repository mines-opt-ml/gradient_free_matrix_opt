import sys
from pathlib import Path

import matplotlib.pyplot as plt
from jax import random

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from base_optimizer import BaseMatrixOptimizer
from obj_func.ky_fan_regression import KyFanRegression
from recovery.registry import MATRIX_RECOVERY_REGISTRY

## Parameters
step_size = 0.5
max_queries = 5e5
target_rank = 3
lozo_rank = 8
num_samples_per_iter = 250
max_time = 12000
# matrix shape
m = 30
n = 30

## Construct recovery params
recovery_params = {"r": target_rank, "iters": 50}

# Random key for JAX
seed = 2028
key = random.PRNGKey(seed)
k1, k2 = random.split(key, 2)

# Define parts of the objective function
obj_func = KyFanRegression(target_rank, m, n, seed, 0.0)


X = random.normal(k1, shape=(m, n))
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
for recovery_algorithm in MATRIX_RECOVERY_REGISTRY:
    print(f"Running optimization with recovery algorithm: {recovery_algorithm}")

    if recovery_algorithm == "lozo":
        sampling_scheme = lozo_rank
    else:
        sampling_scheme = "full-rank"

    optimizer = BaseMatrixOptimizer(
        X,
        rand_seed=0,
        samps_per_iter=num_samples_per_iter,
        step_size=step_size,
        target_rank=target_rank,
        target_obj_func_val=1e-10,
        max_queries=max_queries,
        max_time=max_time,
        report=True,
        obj_func=obj_func,
        recovery_algorithm=MATRIX_RECOVERY_REGISTRY[recovery_algorithm],
        recovery_params=recovery_params,
        linesearch=True,
        sampling_scheme=sampling_scheme,
        sampling_dist="gaussian",
    )

    # Run the optimization
    optimizer.run_optimizer()

    ax2.semilogy(optimizer.time_vec, optimizer.function_values, linewidth=2, label=recovery_algorithm)
    ax1.semilogy(optimizer.query_vec, optimizer.function_values, linewidth=2, label=recovery_algorithm)

ax1.set_title("Queries vs Objective Function Value")
ax2.set_title("Time vs Objective Function Value")
ax1.set_xlabel("Number of Queries Made")
ax1.set_ylabel("Objective Function Value")
ax2.set_xlabel("Time Elapsed")
ax2.set_ylabel("Objective Function Value")
ax1.legend()
ax2.legend()
fig1.savefig("experiments/ky_fan_test/kyfan_queries.png")
fig2.savefig("experiments/ky_fan_test/kyfan_time.png")
