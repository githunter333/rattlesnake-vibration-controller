"""
optimal_diagonal_control_fast.py

Speed-optimized variant of optimal_diagonal_control: same buzz baseline and
three-step budgeted scheduler (inherited unchanged from optimal_diagonal_
control), but each bin's SDP is replaced by an unconstrained Burer-Monteiro
factored solve (X = L L^H, L is N x r) whenever it's SAFE to do so --
falling back to the base class's real, coherence-capped SDP otherwise.

WHY THIS EXISTS: the SDP's cost is dominated by N (drive count), scaling
roughly ~N^4 with the PSD-cone/coherence-constraint machinery. Benchmarked
at a 12-drive/30-response scale: SDP ~105-119 ms/bin vs. the factored solve
~5.4-5.7 ms/bin (~18-20x faster), matching SDP accuracy closely (mean diff
-0.02 dB, max 0.32 dB on real system H) -- but ONLY when unconstrained.
Left unconstrained, the factored solve was measured to drive pairwise drive
coherence to EXACTLY 1.0 in ~100% of bins at a 12-drive/30-response scale --
perfectly rank-deficient drives, which would make a live H1/H2 estimator's
reference CPSD matrix singular. That's fine if nothing is re-estimating H
live, but unsafe the moment it is.

COUNTER-MEASUREMENT, 2026-09-17 (run 13, this 6-drive/8-control system): the
saved drive CPSD showed median pairwise coherence 0.407, 2.4% of pairs above
0.99 and 0.0% at exactly 1.0 -- the claim above did NOT reproduce. Eigenvalue
participation was 1.21 of 6, so the drive was concentrated but not
rank-deficient.

Run 13 is still not admissible evidence, but for ONE reason, not two: its
saved metadata reads control_python_function = pseudoinverse_control and
update_tf_during_control = 1, so profile_13 was never loaded and whatever it
measured is not this law.

RETRACTED 2026-09-17, same day, and worth keeping as a caution about this
metric. This paragraph previously also called run 13's system identification
weak, "minimum multiple coherence 0.492 against 0.75-0.88 for the twelve-run
comparison set". That comparison was invalid: 0.492 is run 13's CHANNEL-7
minimum and 0.75-0.88 is the range of the OTHER channels' minima in the other
runs. Control channel 7 sits between 0.28 and 0.54 in ALL THIRTEEN
identifications taken on this system -- it is a property of the rig, not of
any run. Scored consistently, run 13's identification was the BEST of the set
(worst per-channel minimum 0.531; run 14's is 0.411, run 05's 0.284). Every
run's identification has median coherence 0.998 and 5th percentile ~0.97.
When judging a system ID here, use the median and the 5th percentile, or the
per-channel minima compared like against like -- a bare minimum over all
channels and lines is dominated by channel 7 at the band edge and says
nothing about run quality.

Treat the 1.0-coherence claim as established at 12 drives and UNVERIFIED at 6
until run 14 lands.

PREDICTED for run 14 by offline dry run on run 01's identification (this exact
parameter string, law constructed the way Rattlesnake constructs it, frozen H):
after 25 cycles and 500 refined bins, median pairwise coherence 0.501, 2.8% of
pairs above 0.99, 0.0% at exactly 1.0, participation 1.09 of 6. Close to what
run 13 showed, which is unexplained given run 13 ran a different law -- read
nothing into the resemblance until run 14 is scored.

THE SAFETY RULE (this is the important part): rather than trying to detect
Rattlesnake's "Update Transfer Function During Control" checkbox directly
(control()'s signature doesn't expose it), this watches the actual
safety-relevant property instead -- has H MOVED since the last call, by
more than frf_update_threshold in relative Frobenius norm? If it has not,
nothing is meaningfully re-estimating it right now and the fast
unconstrained path is safe. The moment it has, every solve for the rest of
that call uses the real coherence-capped SDP instead, via
super()._solve_one_bin(). This degrades gracefully: a run that starts
static and has TF-update enabled mid-test downgrades to the safe path as
soon as the plant model actually moves, rather than trusting a stale
assumption from init time. See _h_moved().

REVISED 2026-09-17, and the previous version of this paragraph was wrong
in a way worth recording. The gate used to be `not np.array_equal(H,
H_cache)` -- ANY bitwise difference -- and this docstring claimed that with
TF-update off "H [is] set once at system_id_update() and never re-passed".
That is not what Rattlesnake does: random_vibration_sys_id_data_analysis.py
:431-475 passes a transfer_function to control() EVERY cycle in both
regimes, and system_id_update() is never called during control at all
(:190, only from perform_control_prediction). With the update off the array
happens to be the same frozen sysid_frf each time, so array_equal held and
the fast path ran -- the right outcome by the wrong mechanism. With the
update ON the environment republishes an incrementally-averaged FRF every
cycle (:390-394), so H was never bit-identical, EVERY bin fell through to
the SDP, and this law was byte-for-byte the base class. It was only ever
fast in the configuration it was least needed in, and any comparison of
"fast" against "optimal diagonal" with the update on was a law against
itself. A relative-change threshold fixes that; frf_update_threshold = 0
restores the old any-change behaviour.

extra_parameters: same values as optimal_diagonal_control -- the 6th is
this subclass's own, the 7th is parsed by the base class, so ONE string
works for both laws:
    "reg,frf_update_threshold,max_bins_per_update,error_threshold_db,max_drive_coherence,bm_rank,startup_test_level_cap_db"
    bm_rank - factorization rank r for the fast solve (default 4; the SDP's
              own constrained solutions rarely need rank >3-4 for 99% of
              their energy, and the unconstrained solve needs even less)
    startup_test_level_cap_db - ceiling on the FIRST control command only,
              in dB re the specification trace (default -9.0). Handled
              entirely in the base class; see its docstring, including the
              caveat that this law is open loop so the ceiling does not bind
              after cycle 1.
max_drive_coherence still applies to the SDP fallback path exactly as in
the base class; it has no effect on the fast path (which is, by design,
unconstrained).
"""

import os
import importlib.util
import numpy as np
import scipy.optimize as _scipy_optimize  # module import, not `from ... import minimize` --
                                            # a bare top-level `minimize` name would itself get
                                            # picked up as a spurious "control law" candidate by
                                            # Rattlesnake's loose function-detection heuristic
                                            # (>=12 parameters, which minimize's signature has)

# Rattlesnake loads "Control Python Script" files standalone via
# importlib.util.spec_from_file_location (see components/utilities.py),
# not as part of the control_laws package -- a `from .optimal_diagonal_
# control import ...` relative import fails there with "attempted
# relative import with no known parent package". Load the sibling file
# the same way Rattlesnake itself loads control scripts instead.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_base_path = os.path.join(_this_dir, "optimal_diagonal_control.py")
_spec = importlib.util.spec_from_file_location("optimal_diagonal_control_base", _base_path)
_base_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base_module)
optimal_diagonal_control = _base_module.optimal_diagonal_control


# 10/ln(10), squared: converts a natural-log ratio to dB and back.
_C_DB = 10.0/np.log(10.0)
_C_DB2 = _C_DB**2
# Floor under yhat and y inside the dB objective, so a bin passing through
# zero response gives a large finite penalty rather than an overflow.
_DB_FLOOR = 1e-300
# Used only if the singular-value spectrum cannot be measured at all.
_FALLBACK_SPREAD = 0.0152


class optimal_diagonal_control_fast(optimal_diagonal_control):
    def __init__(self, *args, **kwargs):
        # This subclass's own state is set up BEFORE super().__init__(),
        # not after.  The base constructor calls _initialize() whenever it
        # is handed a non-None transfer_function, and _initialize() runs
        # _refine_batch() -> _solve_one_bin(), which is overridden here and
        # reads self.bm_rank / self.n_fast_solves / self._h_changed_this_
        # call.  With those assignments after the super() call the law
        # crashed with AttributeError: 'optimal_diagonal_control_fast'
        # object has no attribute 'n_fast_solves' (found 2026-09-17 by a
        # harness that constructs the law with the FRF in hand).  Rattle-
        # snake itself builds control laws before system ID, with
        # transfer_function=None, so the crash never fired in the app -- it
        # was latent, and would have fired the moment a law was constructed
        # with an FRF already available.
        self.bm_rank = 4
        extra_parameters = kwargs.get('extra_parameters', args[3] if len(args) > 3 else '')
        if extra_parameters:
            try:
                parts = [p.strip() for p in str(extra_parameters).split(',') if p.strip() != '']
                if len(parts) >= 6:
                    self.bm_rank = int(float(parts[5]))
            except ValueError:
                pass

        self.n_fast_solves = 0
        self.n_safe_solves = 0
        self._h_changed_this_call = False

        print(f"[optimal_diagonal_control_fast] bm_rank={self.bm_rank}", flush=True)

        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    def _refine_batch(self, transfer_function):
        # Same H_changed check the base class's _refine_batch computes
        # internally (before self.H_cache gets mutated by any bin solves
        # this call) -- duplicated here (not read back out of the base
        # class, which doesn't expose it) so _solve_one_bin knows, this
        # call, whether it's safe to use the fast unconstrained path.
        H_clean = np.nan_to_num(transfer_function, nan=0.0, posinf=0.0, neginf=0.0)
        self._h_changed_this_call = self._h_moved(H_clean)
        super()._refine_batch(transfer_function)

    def _h_moved(self, H_clean):
        """Has the FRF moved enough to make the unconstrained fast path
        unsafe?

        CHANGED 2026-09-17.  This used to be `not np.array_equal(H_clean,
        self.H_cache)` -- ANY difference at all, down to the last bit.  That
        reads as the conservative choice and is in fact a silent failure:
        with "Update Transfer Function During Control" on, the environment
        republishes an incrementally-averaged FRF every single cycle
        (random_vibration_sys_id_data_analysis.py:390-394), so H is never
        bit-identical, every bin falls through to the SDP, and
        optimal_diagonal_control_fast is byte-for-byte the base class.  The
        law was only ever fast with FRF update OFF -- the configuration it
        was least needed in -- and a run comparing "fast" against "optimal
        diagonal" with the update on was comparing a law against itself.

        The threshold is frf_update_threshold, the SAME relative-change
        metric the base class already uses to decide which refined bins are
        stale enough to re-solve (optimal_diagonal_control.py's `drifted`
        selection).  So the fast path survives ordinary averaging jitter and
        a genuinely moving plant still demotes every bin to the coherence-
        capped SDP, which is what the gate exists for.  Setting
        frf_update_threshold to 0 restores the old any-change behaviour.
        """
        if self.H_cache is None:
            return True
        # IN-BAND ONLY.  Measured over the whole array this metric is dominated
        # by the out-of-band bins, where there is no excitation, the live
        # estimate is noise, and the relative change per cycle is enormous.
        # Run 28 measured it: median 4.37 -- a 437% change every cycle, max
        # 179 -- while the base class's own drift metric, which looks only at
        # refined (in-band) bins, read 0.17 on the same data.  A factor of 25
        # between two metrics of the same quantity.  That is why runs 21 and 28
        # demoted on 99% of cycles: the gate was responding to out-of-band
        # noise, never to the plant.  Same mistake as measuring the
        # singular-value spread over all lines -- see _effective_drive_rcond.
        Hc, Hn = self.H_cache, H_clean
        if (Hc.shape[0] == self.y_diag_target.shape[0]
                and Hn.shape[0] == self.y_diag_target.shape[0]):
            in_band = self.y_diag_target.max(axis=1) > 0
            if in_band.any():
                Hc, Hn = Hc[in_band], Hn[in_band]
        baseline = np.linalg.norm(Hc.reshape(-1))
        if not np.isfinite(baseline) or baseline <= 0:
            return True
        delta = np.linalg.norm((Hn - Hc).reshape(-1))
        if not np.isfinite(delta):
            return True
        rel = delta/baseline
        moved = rel > self.frf_update_threshold

        # Record what the FRF actually does, whether or not it trips the gate.
        # The demotion threshold should be set from the movement this rig
        # really produces, and that cannot be recovered afterwards -- only the
        # FINAL FRF is saved, so a run file shows cumulative drift (about 28%
        # between the system ID and end of control on run 21) and says nothing
        # about the per-cycle movement the gate tests.
        self._h_move_log = getattr(self, '_h_move_log', [])
        self._h_move_log.append(float(rel))
        n = len(self._h_move_log)
        if moved or n <= 5 or n % 25 == 0:
            a = np.asarray(self._h_move_log)
            print(f"[optimal_diagonal_control_fast] FRF moved {rel:.4f} "
                  f"(threshold {self.frf_update_threshold:g}, "
                  f"{'SDP' if moved else 'fast'} path) -- over {n} calls: "
                  f"median {np.median(a):.4f}, 90th {np.percentile(a, 90):.4f}, "
                  f"max {a.max():.4f}, demoted {100*np.mean(a > self.frf_update_threshold):.0f}%; "
                  # PER-BIN counters, not per-call.  The demotion rate above
                  # counts CALLS; these count the solves that actually took
                  # each path, which is the thing "was the fast law used?"
                  # really asks.  Confirming run 29 needed three indirect
                  # arguments (metadata, the gate log, and a drive-coherence
                  # signature above the SDP's 0.95 cap) only because these
                  # counters existed but were never printed.
                  f"solves: {self.n_fast_solves} fast / {self.n_safe_solves} SDP "
                  f"({100*self.n_fast_solves/max(1, self.n_fast_solves+self.n_safe_solves):.0f}% fast)",
                  flush=True)
        return moved

    # ------------------------------------------------------------------
    # Unconstrained Burer-Monteiro factored solve: X = L L^H, L is N x r.
    # Same objective as the base class's SDP minus the coherence-cap
    # constraint: || diag(H X H^H) - y ||^2 + reg * ||X||_F^2.
    # Gradient (Wirtinger, verified against finite differences): for
    # g(X) = objective as a function of X directly, grad_L = 2*(grad_X g)@L.
    # ------------------------------------------------------------------
    @staticmethod
    def _pack(L):
        return np.concatenate([L.real.ravel(), L.imag.ravel()])

    @staticmethod
    def _unpack(v, N, r):
        n = N * r
        return v[:n].reshape(N, r) + 1j * v[n:].reshape(N, r)

    def _bm_obj_and_grad(self, v, H, y_target, N, r):
        """Objective and gradient for the factored solve.

        error_domain 'db' (the default since 2026-09-17) minimizes
            sum_m [10 log10(yhat_m / y_m)]^2
        directly.  'linear' restores the original
            sum_m (yhat_m - y_m)^2.

        WHY THIS CHANGED.  The linear objective caps what abandoning a channel
        can cost at that channel's own target squared -- drive its response to
        zero and the penalty stops growing -- so on a plant whose rows have
        unequal gain the solver sells the hardest channels to buy the easy
        ones, and is correct to, given what it was asked to minimize.  On the
        6-drive/8-control frame with a flat -30 dB specification and row gains
        spanning 3.8 to 10.3 dB, that cost 13X+ and 14X+ about 5 dB each: run
        14 converged to a self-predicted 9.47 and 8.28 dB rms on those two
        channels against a reachability floor of 4.3 and 3.9, while beating
        that floor on 8X+.  Pooled, 4.65 dB against a 2.38 dB floor.  Measured
        offline on run 14's own FRF, the dB objective takes the pooled figure
        to 3.06 dB and 13X+ to 5.52.

        The dB objective is NOT convex, which is why it lives here and not in
        the base class's SDP; the base class reaches the same objective by
        iteratively reweighted least squares instead, keeping every subproblem
        convex and its coherence-cap constraint valid.  The two agree on what
        they are minimizing but will not agree to the last decimal.
        """
        L = self._unpack(v, N, r)
        B = H @ L
        Ydiag = np.sum(np.abs(B) ** 2, axis=1)
        X = L @ L.conj().T
        f_reg = self.reg * np.sum(np.abs(X) ** 2)
        g_reg = 2 * self.reg * X

        if self.error_domain == 'db':
            # f = sum (C ln(yhat/y))^2, C = 10/ln10.  df/dyhat = 2 C^2
            # ln(yhat/y) / yhat.  Guarded: a bin can pass through yhat = 0.
            Yc = np.maximum(Ydiag, _DB_FLOOR)
            yc = np.maximum(y_target, _DB_FLOOR)
            logr = np.log(Yc / yc)
            f_fit = _C_DB2 * np.sum(logr ** 2)
            dfdy = 2.0 * _C_DB2 * logr / Yc
            dfdy = np.where(y_target > 0, dfdy, 0.0)
        else:
            resid = Ydiag - y_target
            f_fit = np.sum(resid ** 2)
            dfdy = 2.0 * resid

        f = f_fit + f_reg
        gX = (H.conj().T @ (dfdy[:, None] * H)) + g_reg
        gL = 2 * (gX @ L)
        grad = np.concatenate([gL.real.ravel(), gL.imag.ravel()])
        return f, grad

    def _buzz_seeded_L(self, H, y_target, r, rcond=1e-12):
        """Rank-r warm start: eigendecompose the pinv/buzz diagonal-only
        closed-form solution for this bin, keep the top-r eigenpairs."""
        N = H.shape[1]
        Hpinv = np.linalg.pinv(H, rcond=rcond)
        Xbuzz = Hpinv @ np.diag(y_target).astype(complex) @ Hpinv.conj().T
        Xbuzz = (Xbuzz + Xbuzz.conj().T) / 2
        w, V = np.linalg.eigh(Xbuzz)
        order = np.argsort(-w)[:r]
        w_top = np.clip(w[order], 0, None)
        return (V[:, order] * np.sqrt(w_top)[None, :]).astype(complex)

    def _bm_solve(self, H, y_target):
        M, N = H.shape
        r = min(self.bm_rank, N)
        L0 = self._buzz_seeded_L(H, y_target, r)
        v0 = self._pack(L0)
        # Tolerances matter a lot here: on REAL (not synthetic-random) FRF
        # data, ~70% of bins hit a 200-iteration cap at tight tolerances
        # (ftol=1e-14, gtol=1e-10) without visibly improving dB accuracy --
        # checked directly (bm_accuracy tuning script): loosening to
        # maxiter=50/ftol=1e-9/gtol=1e-7 cuts time ~5.7x (18.7ms -> 3.3ms
        # mean on real bins) for only +0.4 dB mean / +2.7 dB worst-case
        # versus the SDP's own accuracy on the same bins -- a real,
        # verified speedup (~3.6x vs the SDP mean on real data), much more
        # modest than synthetic-random-H benchmarks suggested but honest.
        res = _scipy_optimize.minimize(self._bm_obj_and_grad, v0, args=(H, y_target, N, r), jac=True,
                                        method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-9, 'gtol': 1e-7})
        L = self._unpack(res.x, N, r)
        return L @ L.conj().T

    # ------------------------------------------------------------------
    def _effective_drive_rcond(self, H):
        """The subspace threshold actually used, as a MULTIPLE of this plant's
        own singular-value spread.

        drive_rcond > 0 is read as a multiple of the smallest STRUCTURAL
        singular-value ratio; drive_rcond < 0 is an absolute ratio, |value|,
        for reproducing an old run or pinning a value by hand.

        WHY A MULTIPLE.  An absolute ratio has to be re-tuned for every test
        article, because it is a threshold on a quantity that is a property of
        the structure.  The rig sweep (runs 22-25) put the optimum at 3.0e-2 on
        this frame, and this frame's smallest structural direction sits at
        sigma_5/sigma_1 = 0.0152 -- so the tuned optimum is 2.0x that, which is
        the default here.  On a differently conditioned article 2.0x tracks the
        plant automatically where 3.0e-2 would not.

        FINDING THE STRUCTURAL FLOOR.  This 8-response/6-drive frame is rank 5:
        its measured sigma_k/sigma_1 medians run 1.000, 0.319, 0.084, 0.032,
        0.015, 0.0017, and that last value is the identification noise floor,
        not a direction the plant has.  Scaling off sigma_min would therefore
        track how good the system ID was rather than how the structure is
        conditioned.  The floor is found instead by the largest gap in the
        spectrum -- here 8.9x between sigma_5 and sigma_6, against 2.1-3.8x
        everywhere else -- and the reference is the smallest direction above
        that gap.

        Computed once per FRF from the median over in-band bins, so a single
        noisy line cannot move it.
        """
        if self.drive_rcond < 0:
            return -self.drive_rcond          # absolute override
        if self.drive_rcond == 0:
            return 0.0
        # Only cache once the FULL band is in hand.  H_cache is populated by
        # _initialize before _refine_batch runs, so the control loop always
        # has it -- but _solve_one_bin called directly (a harness, a unit
        # test) would otherwise compute the reference from the ONE bin it was
        # handed and cache that answer forever.  Found 2026-09-18 when an FRF
        # sensitivity test reported 10.34 dB for a configuration the control
        # loop scores at 5.95.
        if self.H_cache is None:
            return float(np.clip(self.drive_rcond*_FALLBACK_SPREAD, 1e-4, 0.5))
        key = self.H_cache.shape
        if getattr(self, '_rcond_cache', None) is not None and self._rcond_key == key:
            return self._rcond_cache
        Hb = self.H_cache
        if Hb.ndim == 2:
            Hb = Hb[np.newaxis]
        # IN-BAND ONLY.  Out-of-band lines are not controlled and are
        # conditioned quite differently; including them moved the reference
        # from 0.0152 to 0.1435 on this frame, a factor of 9, which is the
        # difference between the tuned 3.0e-2 and a useless 0.29.
        if Hb.shape[0] == self.y_diag_target.shape[0]:
            in_band = self.y_diag_target.max(axis=1) > 0
            if in_band.any():
                Hb = Hb[in_band]
        try:
            sv = np.linalg.svd(np.nan_to_num(Hb), compute_uv=False)
        except np.linalg.LinAlgError:
            return self.drive_rcond*_FALLBACK_SPREAD
        ratios = sv/np.maximum(sv[:, :1], 1e-300)
        med = np.median(ratios, axis=0)                      # (N,)
        med = med[np.isfinite(med) & (med > 0)]
        if med.size < 2:
            ref = _FALLBACK_SPREAD
        else:
            gaps = med[:-1]/np.maximum(med[1:], 1e-300)
            ref = float(med[int(np.argmax(gaps))])           # last direction above the gap
        eff = float(np.clip(self.drive_rcond*ref, 1e-4, 0.5))
        self._rcond_cache = eff; self._rcond_key = key
        print(f"[optimal_diagonal_control_fast] drive_rcond {self.drive_rcond:g} x "
              f"plant spread {ref:.4f} -> effective {eff:.4g}", flush=True)
        return eff

    def _restricted_basis(self, H):
        """Right-singular directions of H carrying sigma >= drive_rcond*sigma_max,
        as an (N, k) matrix -- or None for no restriction.

        WHY THE FACTORED PATH NEEDS THIS AND THE SDP PATH DOES NOT.  The dB
        objective removes the ceiling on what missing a channel costs, which
        is the point, but it also means the solver will spend unlimited drive
        to chase a channel the plant barely reaches.  Unconstrained on run
        14's FRF it did exactly that: pooled error 1.93 dB, better than the
        rcond-1e-3 reachability floor itself, bought with +10.4 dB more drive
        on average, +30.2 dB on the worst bin, at eigenvalue participation
        1.02 -- a nearly rank-one drive pushed down a near-singular direction.
        The base class's SDP does not need this because its coherence-cap
        constraint already restrains it (+2.6 dB); the factored path is
        unconstrained by design and has nothing else holding it.

        reg is NOT the right lever and was measured not to be: it penalizes
        every direction equally, so raising it from 1e-6 to 1.0 gave back the
        accuracy (2.31 dB) while still leaving +7.8 dB mean drive and a +25.5
        dB worst bin.  Restricting the SUBSPACE is what the reachability floor
        itself does (restrict_rcond), and it targets the actual problem.

        Measured on run 14, pooled / mean drive / worst bin, against the
        as-shipped linear objective at 4.68 dB and 0 dB drive:

            unrestricted   1.93 dB   +10.4 dB   +30.2 dB
            rcond 1e-3     2.44 dB    +5.2 dB   +30.2 dB
            rcond 1e-2     3.83 dB   -12.2 dB    +8.7 dB   <- default
            rcond 5e-2     4.35 dB   -20.7 dB    +8.7 dB

        The 1e-2 default is the one point on that curve that is better than
        the law as it stands on BOTH axes -- 0.85 dB more accurate for 12 dB
        LESS drive -- so it is safe to adopt without knowing the rig's
        headroom.  Set drive_rcond to 0 to disable the restriction.
        """
        # APPLIES IN BOTH ERROR DOMAINS since 2026-09-18.  It was briefly
        # gated to 'db' only, to keep error_domain=0 reproducing the
        # pre-2026-09-17 law exactly.  Run 22 is what that cost: the factored
        # path in its DEFAULT linear mode then had nothing constraining it at
        # all -- no coherence cap, which it cannot express, and no subspace
        # restriction -- and it commanded a drive whose response depends on
        # inter-drive cancellation the rig cannot physically realize.
        #
        # THE MEASUREMENT THAT SETTLES IT.  Perturbing the commanded drive by
        # 1%, which is less than the difference between what run 22 commanded
        # and what its saved drive CPSD actually shows:
        #
        #     configuration                      rms    level  | +1% drive
        #     factored, unconstrained           4.64   -0.42   | 11.85  +16.69
        #     factored + drive_rcond 1e-2       3.73   +1.94   |  4.68   +3.80
        #     SDP + coherence cap 0.95          4.16   -0.50   |  7.23   +7.92
        #     SDP, cap off                      4.09   -0.49   |  7.98   +9.09
        #
        # Run 22 measured +14.29 dB level against -0.39 predicted from the
        # commanded drive; a 1% realization error predicts +16.69.  The
        # unconstrained solve is not wrong, it is UNREALIZABLE -- its accuracy
        # lives in cancellation between drives that survives only if the drive
        # is reproduced to a precision no rig delivers.  drive_rcond bounds
        # that sensitivity directly, and does it better than the coherence cap,
        # which only halves it.
        #
        # Set drive_rcond to 0 to disable and recover the run-22 behaviour.
        rc = self._effective_drive_rcond(H)
        if not (rc > 0):
            return None
        try:
            _, sv, Vh = np.linalg.svd(H, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        if sv.size == 0 or not np.isfinite(sv[0]) or sv[0] <= 0:
            return None
        k = int(np.sum(sv >= rc*sv[0]))
        if k < 1:
            k = 1
        if k >= H.shape[1]:
            return None          # nothing excluded, skip the change of basis
        return Vh[:k].conj().T

    def _solve_one_bin(self, H, y_target, X_warm=None):
        # X_warm is the base class's IRLS warm start; the factored solve here
        # reaches the dB objective directly and has no use for it, but the
        # signature has to match so the SDP fallback below still gets it.
        if not self._h_changed_this_call:
            # H is unchanged since the last call -- nothing is re-
            # estimating it live right now, so the fast unconstrained
            # solve is safe even though it drives coherence toward 1.0.
            self.n_fast_solves += 1
            try:
                # Solve in the well-conditioned drive subspace and map back,
                # so the dB objective cannot buy accuracy with drive down a
                # near-singular direction -- see _restricted_basis.  The
                # coherence cap does not apply on this path either way, so
                # nothing is lost by working in a rotated basis here.
                Vk = self._restricted_basis(H)
                if Vk is None:
                    return self._bm_solve(H, y_target)
                Xk = self._bm_solve(H @ Vk, y_target)
                return Vk @ Xk @ Vk.conj().T
            except Exception as e:
                print(f"[optimal_diagonal_control_fast] BM solve raised {type(e).__name__}: {e} "
                      f"-- falling back to SDP for this bin", flush=True)
        self.n_safe_solves += 1
        return super()._solve_one_bin(H, y_target, X_warm=X_warm)

    # ------------------------------------------------------------------
    # Explicit pass-through overrides -- required, not decorative.
    # Rattlesnake's control-law loader (components/random_vibration_sys_id_
    # environment.py) only recognizes a class as a valid class-style control
    # law if 'system_id_update' and 'control' are in the class's OWN
    # __dict__, not just inherited from a base class. Without these, this
    # class is invisible to the "Control Python Function" dropdown even
    # though it fully implements both via inheritance.
    def system_id_update(self, *args, **kwargs):
        return super().system_id_update(*args, **kwargs)

    def control(self, *args, **kwargs):
        return super().control(*args, **kwargs)
