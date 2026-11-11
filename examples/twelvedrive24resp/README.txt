Frame 12x24 -- 12 drives, 24 control accelerometers (+12 response-only)
======================================================================
Built 2026-11-07 by examples/testbeds/code/build_testbeds.py.  See
examples/testbeds/README.md for how and why.

Frame: NEW 24-node 2D lattice, 4 rows x 6 nodes, x and y at every node
(48 DOFs).  Same construction as the 18-node frame (row springs, risers,
0.6 diagonal braces, +/-6 % spring and +/-8 % mass variation), but with
CROSSED braces in every bay and one more row.
  row A nodes 1-6 (shakers), B 7-12, C 13-18, D 19-24.
  45 flexible modes, 120.0-678.5 Hz (none above 680 Hz), Rayleigh damping
  at 4x nominal (zeta_target 0.04), to match the 10x14 bed.

Why crossed braces: without them the corner node's x DOF is joined only to
its neighbour's x DOF, so driving every row-A DOF loses rank (the 5x/6x
pair of the 18-node frame).  Uncrossed, 12x24 is rank-deficient at every
line; crossed, it is full rank 12.
  sigma_min/sigma_max over 100-1000 Hz: median 3.1e-2, 10th pct 2.8e-3,
  min 6.4e-4 (10x14: 2.5e-2 / 2.4e-3 / 6.1e-4).
Brace stiffness was not lowered to spread the modes higher: 0.3 and 0.15
spread them to 707 and 828 Hz but cost 1-2 orders of magnitude in
conditioning.

Channels (channel_table_frame12x24.csv, in this order):
  1-24  control accels   rows B and C, x and y (nodes 7-18)
  25-36 response accels  row D, x and y (nodes 19-24), monitor only
  37-48 force (drives)   row A, x and y (nodes 1-6)

Rattlesnake setup (results/case/):
  linear     sdynpy_frame12x24_profile.xlsx            hardware 6, sdynpy_frame12x24_system.npz
  nonlinear  sdynpy_frame12x24_profile_nonlinear.xlsx  hardware 9, sdynpy_frame12x24_system_nonlinear_boundedsoften.npz
  spec       flat_spec_frame12x24.mat -- 0.001 on every control ASD,
             100-1000 Hz, no cross terms, 1 Hz lines to 2048 Hz
  law        PLACEHOLDER: control_laws.py :: match_trace_pseudoinverse
             '1e-3,0.95,-9.0,,1,0.05,1' (run 40's).  Change once the law
             list is settled.
  system ID  1.0 V rms in both profiles (the nonlinear design needs no
             backoff: 0.20 % worst-mode softening at 1 V rms).
  test profile  start-only (-18 dB, Start Environment, Start Streaming);
             step levels with the harness -- the nonlinear integrator runs
             ~8x slower than real time.

Nonlinear: bounded softening on the 34 modes above 300 Hz (11 below stay
exactly linear), alpha 0.35, 1 % shift at 0 dB, loop gain 0.064, sized on a
SYNTHESIZED 0 dB drive -- see examples/testbeds/README.md.

Not yet: frame_plant.py has no '12x24' layout (it loads only the 18-node
model), so check_control_laws_on_dev.py can't run this plant offline yet.
