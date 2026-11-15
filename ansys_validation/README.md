# Verification against Ansys BEAM188

This folder reproduces the verification of the ACOROUTE beam model against Ansys Mechanical APDL (element BEAM188) reported in the manuscript. Both codes solve the same route, on the same mesh, with the same material, damping and boundary conditions, by the direct (full) harmonic method at the same frequencies.

## Model

| Item | Value |
|---|---|
| Geometry | reported route of the staircase case (`revision_results/stair_6_10`), 1257 mm |
| Mesh | 630 nodes, 629 two-node elements of about 2 mm on the smoothed centreline |
| Pipe | outer radius 1.575 mm, wall thickness 0.5 mm |
| Material | E = 210 GPa, nu = 0.3, rho = 7850 kg/m3 |
| Damping | Rayleigh, alpha = 0, beta = 2e-7 s |
| Node A (first) | translations prescribed, UX = UY = UZ = 1e-6 m in phase; rotations free |
| Node B (last) | clamped |
| Ansys element | BEAM188, cubic shape functions (`KEYOPT(3)=3`), consistent mass, `CTUBE` section |
| Harmonic analysis | full method, 11168 frequencies from 2 Hz to 11 kHz |
| Modal analysis | same boundary conditions with the translations of A fixed, up to 12 kHz |

The frequency list is a 2 Hz grid refined around every natural frequency below about 2.5 kHz. With stiffness-proportional damping the half-power bandwidth is 2 pi beta f^2 (1.3 Hz at 1 kHz, 0.013 Hz at 100 Hz), so a uniform grid does not resolve the low-frequency peaks.

## Files

| File | Content |
|---|---|
| `stair_6_10_harmonic.inp` | APDL input with the mesh and the frequency list embedded |
| `run_ansys_smp.bat` | runs the input in batch with shared-memory parallel processing (Windows) |
| `export_only.inp` | export of the reactions from an existing `stair_6_10_harm.rst` |
| `stair_6_10_nodes.txt` | node coordinates of the shared mesh (node 1 = A, last node = B) |
| `route_final_ids_stair_6_10.npy` | lattice indices of the reported route |
| `make_ansys_package.py` | regenerates the input, the node file and the ACOROUTE reference files |
| `compare_ansys.py` | reads the Ansys output and compares it with the ACOROUTE direct solver |
| `acoroute_stair_6_10_modes.txt` | ACOROUTE natural frequencies below 12 kHz |

## How to run

1. Run Ansys. On Windows, double-click `run_ansys_smp.bat` (it looks for Ansys 2026 R1 through `AWP_ROOT261`; edit `ANSYS_EXE` in the file for another version). Otherwise start Mechanical APDL with shared-memory parallel processing and use *File > Read Input From* with `stair_6_10_harmonic.inp`. Do not paste the commands in the command line, because `*VWRITE` only works when read from a file.
2. Ansys writes `stair_6_10_ansys_modes.txt` (natural frequencies), `stair_6_10_ansys_A.txt` and `stair_6_10_ansys_B.txt` (real and imaginary parts of the reactions at A and B) and `stair_6_10_ansys_done.txt` in its working directory. Copy them to this folder if the working directory is different.
3. Run `python compare_ansys.py`. It computes the active input power from the Ansys reactions at A, runs the ACOROUTE direct solver at the Ansys frequencies and writes `comparison_summary.txt`, `comparison_input_power.txt`, `comparison_modes.txt` and the figures `fig_ansys_input_power` and `fig_ansys_reactions_B`.

Shared-memory parallel processing is recommended. In a distributed run each process writes its own results file and the files must be combined before the export; the input tries to do this (`/AUX2`, `COMBINE,RST`), but the shared-memory run avoids the step.

To change the frequency range, the base step or the refinement, edit `F_MAX`, `DF_BASE`, `REFINE_REL` and `PTS_PER_BW` in `make_ansys_package.py` and run it again. Set `USE_COWPER = 1` in the input to impose the shear stiffness of ACOROUTE (Cowper) instead of the value computed by Ansys for the section.

## Results obtained

Ansys Mechanical APDL 2026 R1, differences of ACOROUTE relative to Ansys.

| Range [kHz] | Modes | max abs(df/f) [%] | dJ/J [%] | median level difference [dB] | 95th percentile [dB] |
|---|---|---|---|---|---|
| 0-1 | 31 | 0.017 | -0.25 | 0.04 | 0.65 |
| 1-3 | 22 | 0.052 | +0.32 | 0.15 | 0.96 |
| 3-6 | 25 | 0.082 | +0.45 | 0.17 | 0.90 |
| 6-10 | 19 | 0.100 | below 0.01 | 0.02 | 0.22 |
| 0-10 | 97 | 0.100 | +0.42 | 0.05 | 0.74 |

J is the active input power integrated over the range of each row. In the 6-10 kHz band the root mean square over frequency of the resultant reaction at B differs by +0.31 % (force) and +0.40 % (moment). The largest pointwise differences occur at the lightly damped peaks at low frequency, where the half-power bandwidth is of the order of the difference between the natural frequencies of the two models.
