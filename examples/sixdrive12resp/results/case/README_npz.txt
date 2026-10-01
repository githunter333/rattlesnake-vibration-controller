Plant models for the sixdrive12resp example
===========================================
Six-shaker frame, twelve response channels, eight used for control.
Everything is numpy .npz, loadable with np.load(path, allow_pickle=True).

sdynpy_frame6x12_system.npz
    The LINEAR article.  mass, stiffness, damping (36x36), transformation,
    coordinate (node/direction records), enforce_symmetry.  33 flexible modes
    between 120.0 and 899.5 Hz.  This is the plant every linear-plant result
    was measured on.

sdynpy_frame6x12_system_nonlinear_allmodes.npz
    Intermediate.  Carries nl_drive_dof_indices and the mode-shape bookkeeping
    the builder consumes.  Included because without it
    build_nonlinear_frf_system_boundedsoften.py cannot run.

sdynpy_frame6x12_system_nonlinear_boundedsoften.npz
    The NONLINEAR article in current use.  The linear matrices plus a bounded
    softening kernel g(q) = alpha + (1-alpha)*exp(-(q/qt)^2) on the 27 modes
    above 300 Hz, and a quadratic damping term c2*qd*|qd| on the same modes.
    The six modes below 300 Hz are exactly linear by construction -- their
    nonlinear term is identically zero -- which gives a linear control group
    inside the same FRF measurement.

    Extra fields: nl_target_mode_shapes, nl_soft_qts, nl_soft_wn2s,
    nl_soft_alpha (0.35), nl_soft_mask, nl_c2s, nl_q0_refs, nl_q_design_refs,
    nl_q_sysid_refs, nl_target_freqs_hz, nl_target_zetas,
    nl_drive_dof_indices, nl_design_target_shift, nl_sysid_backoff_db,
    nl_reference_drive_level_vrms.

    At 0 dB this plant softens every one of the 27 modes by 0.90-1.00% and
    raises their damping ratios by 12.4%.  Above 0 dB individual modes break
    away while the median stays orderly -- see
    results/notes/modal_shift_and_damping_results.txt.

    CAVEAT ON ZETA_RATIO.  The builder sets ZETA_RATIO = 1.10 but sizes c2
    with the sinusoidal describing function, which understates the Gaussian
    result by (3/4)*sqrt(2*pi) = 1.880.  The realised damping rise at 0 dB is
    +0.124, not the +0.100 the constant implies.  The plant is bounded and
    stable and no run is invalidated; the constant simply does not mean what
    its name says.

NOT INCLUDED.  build_nonlinear_frf_system_boundedsoften.py also reads
run40_matchtrace_refresh005_spec.nc4 for the shaped control drive used in the
self-consistent calibration.  That file is ~11 MB and stays out of the repo,
so the builder cannot be re-run from a fresh clone without it.  The built
article above is the deliverable; rebuilding is not required to use it.

These three are exempted from the **/*.npz rule by explicit negations at the
bottom of .gitignore.  Other .npz files in this directory are earlier
cubic-kernel attempts and remain ignored.
