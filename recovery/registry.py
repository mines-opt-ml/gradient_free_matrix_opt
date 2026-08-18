from recovery.adjoint_sensing_operator import adjoint_sensing_operator
from recovery.alternating_projections import alternating_projections
from recovery.burer_monteiro_gradient_descent import burer_monteiro_gradient_descent
from recovery.iterative_hard_thresholding import iterative_hard_thresholding, spectral_iterative_hard_thresholding
from recovery.lozo import lozo
from recovery.pseudoinverse import pseudo_inverse_CG

MATRIX_RECOVERY_REGISTRY = {
    "alternating_projections": alternating_projections,
    "iterative_hard_thresholding": iterative_hard_thresholding,
    "burer_monteiro_gradient_descent": burer_monteiro_gradient_descent,
    "IHT_SpecGD": spectral_iterative_hard_thresholding,
    "adjoint_sensing_operator": adjoint_sensing_operator,
    "pseudoinverse": pseudo_inverse_CG,
    "lozo": lozo,
}
