"""
altobjective.py
Conventional vibration-aware routing objective, used only for the controlled
comparison requested in the revision: the same ant colony, lattice, construction
rules, geometric checks and budget, with the band-integrated input power replaced by
a resonance-avoidance criterion of the kind used in the routing literature (natural
frequencies kept away from the operating band).

    J_alt = sum_k w_k * ( N_k + exp(-d_k / (0.1 * width_k)) )

N_k      number of natural frequencies of the pipe inside band k (clamped at B, the
         prescribed translations fixed at A, i.e. the same boundary conditions as the
         harmonic analysis), computed with the same beam mesh as the search;
d_k      distance from the band edges of the nearest natural frequency outside the
         band (tie-break between routes with the same count: the further the nearest
         mode from the band, the smaller the term; it lies in (0, 1], so it never
         outranks a difference of one mode).

Usage in a run script (copy of run_spline.py for the comparison):
    OBJECTIVE = "modes_in_band"
    ...
    def objective_spline(coords):
        nodes, elems, _ = geo.mesh_spline(coords, ELEM, smooth=SMOOTH)
        if OBJECTIVE == "modes_in_band":
            import altobjective as AO
            return AO.modes_in_band(nodes, elems, mat, sec, BANDS, band_weights, PRESCRIBED_MASK)
        return fem.band_integrated_power(...)        # unchanged
The winner is then re-evaluated in input power by rerank_campaign.py, which does not
read the objective, only the routes.
"""
import numpy as np
import fem_beam3d as fem


def modes_in_band(nodes, elements, mat, sec, bands, weights=None, prescribed_mask=(True, True, True),
                  drive_node=0, fixed_node=None):
    if fixed_node is None:
        fixed_node = len(nodes) - 1
    weights = [1.0] * len(bands) if weights is None else list(weights)
    KM = fem.assemble(nodes, elements, mat, sec)
    extra = [6 * drive_node + i for i in range(3) if prescribed_mask[i]]
    J = 0.0
    for (lo, hi), w in zip(bands, weights):
        f = fem.natural_frequencies(nodes, elements, mat, sec, fixed_node, band_hz=(lo, hi), margin=1.0,
                                    extra_fixed_dofs=extra, KM=KM)
        f = np.asarray(f, float)
        inside = (f >= lo) & (f <= hi)
        n_in = int(inside.sum())
        outside = f[~inside]
        if outside.size:
            d = float(np.min(np.minimum(np.abs(outside - lo), np.abs(outside - hi))))
        else:
            d = hi - lo
        J += w * (n_in + np.exp(-d / (0.1 * (hi - lo))))
    return float(J)
