"""Optional post-processing for a route produced by the strict v6-compatible run.

This module is deliberately separate from run_spline.py. It never changes the ACO
winner. Save the route as a NumPy array of lattice IDs or waypoint coordinates and
call the functions here only after the search has finished.
"""
from __future__ import annotations

import numpy as np
import fem_beam3d as fem
import geometry as geo


def verify_coords(coords, mat, sec, bands, weights=(1.0,), *,
                  elem_len=0.005, smooth=5e-4,
                  excitation="prescribed",
                  prescribed_disp=(1e-6, 1e-6, 1e-6),
                  prescribed_mask=(True, True, True),
                  drive_dir=(0.0, 1.0, 0.0), rayleigh=(0.0, 2e-7),
                  n_band=2000):
    """Recalculate the v6 winner on a dense UNIFORM grid, without changing it.

    This is a verification value, not the objective used to guide the v6 ACO.
    """
    coords = np.asarray(coords, dtype=float)
    nodes, elements, _ = geo.mesh_spline(coords, elem_len, smooth=smooth)
    total = 0.0
    for weight, band in zip(weights, bands):
        value = fem.evaluate_mesh_power(
            nodes, elements, mat, sec, band,
            freq_mode="uniform", n_band=int(n_band),
            drive_dir=drive_dir, rayleigh=rayleigh,
            excitation=excitation,
            prescribed_disp=prescribed_disp,
            prescribed_mask=prescribed_mask,
        )
        total += float(weight) * float(value)
    return float(total)
