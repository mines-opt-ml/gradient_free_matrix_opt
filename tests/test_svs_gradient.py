#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


# You could run `pip install -e .` if you use pip but a bit tricky with anaconda (?), so do this instead.
from pathlib import Path
import sys
# Get the current file's parent directory and add it to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from recovery.directional_derivative import directional_derivatives
from obj_func.singular_value_sum import SingularValueSum

if __name__ == "__main__":

    m, n = 5, 6
    r    = 5 # SVD with full rank is just 1/2||X||_F^2, so gradient is just X
    init_X_key = jax.random.key(12345)
    X0 = jax.random.normal(init_X_key, shape=(m, n))
    f = SingularValueSum(r, noise=0)

    print('X is\n', X0)
    print('Gradient is\n', f.gradient(X0))
    print('Singular values of gradient are\n', jnp.linalg.svdvals( f.gradient(X0) ))


    # DOUBLE-CHECK USING DIRECTIONAL DERIVATIVES
    Z = jnp.eye( m*n ).reshape(m*n, m, n) # shape (num_directions, m, n)
    # -- All 3 versions of directioal derivatives give the same answer --
    y = directional_derivatives( f, X0, Z, key=None )  # exact jax directional derivatives via jax.jvp
    # y = directional_derivatives( f, X0, Z, key=None, approximate=True )
    # y = directional_derivatives( f, X0, Z, key=None, approximate=True, centered =True )
    Y = y.reshape(m, n)
    print('Gradient, estimated via directional derivatives is\n', Y)