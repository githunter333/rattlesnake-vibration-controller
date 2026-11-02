Frame 10x14 -- 10 drives, 14 control accelerometers (+4 response-only)
=====================================================================
Built 2026-10-31 from the same 18-node frame model as sixdrive12resp
(the linear part of sdynpy_frame6x12_system_nonlinear.npz).

Frame: 2D pin-jointed lattice, 3 rows x 6 nodes, x and y at every node.
  bottom row nodes 1-6 (shakers), middle row 7-12, top row 13-18.
  33 elastic modes, 120-900 Hz, 2-6 % damping.

Channels (channel_table_frame10x14.csv, in this order):
  1-14  control accels   middle row x and y at nodes 7-12, plus 13x, 14x
  15-18 response accels  15x, 16x, 17x, 18x (as in the 6x12 case)
  19-28 force (drives)   x and y at nodes 1, 2, 3, 6; y at 4, 5

Why these drives: full rank 10 at every line (sigma_min/sigma_max over
100-1000 Hz: median 2.5e-2, 10th pct 2.4e-3, min 6.1e-4).  Driving both 5x
and 6x loses a rank -- node 6's x DOF is joined only to node 5's x DOF -- which
is why the 6x12 case is rank 5, not 6.  frame_plant.py layout '10x14_rankdef'
keeps that pair if you want the redundant-drive case.

Rattlesnake setup:
  hardware: SDynPy, file results/case/sdynpy_frame10x14_system.npz (linear;
            all 36 DOFs, the channel table picks which are used)
  spec:     results/case/flat_spec_frame10x14.mat -- 0.001 on every control
            ASD, 100-1000 Hz, no cross terms (same as flat_spec_frame6x12.mat)
  control channels: the first 14

Offline: control_laws/tests/frame_plant.py builds the same FRF
  python control_laws/tests/check_control_laws_on_dev.py --plant 10x14
