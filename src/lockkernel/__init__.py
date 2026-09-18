"""Locking kernel universality: thresholds and exponents.

The package is organised around one object, the locking kernel W(u), and one
identity, the exact parametric solution of the locking self consistency in
`lockkernel.parametric`.  The `lockkernel.measured` module is the lab half:
it fits a MEASURED synchronization branch for the threshold and the exponent,
with error bars and refusals, and reads the kernel tail back off the exponent
through the same dictionary the theory half proves.
"""
__version__ = "1.1.0"

from . import kernels, lineshapes, measured, parametric  # noqa: F401
from .measured import (BranchFit, beta_relative_sigma,  # noqa: F401
                       fit_branch, kernel_tail_from_beta,
                       points_for_beta)
