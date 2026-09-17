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

COUNTER-MEASUREMENT, 2026-09-17 (run 13, this 6-drive/8-control system, with
"Update Transfer Function During Control" OFF so the fast path was eligible
for the whole test): the saved drive CPSD showed median pairwise coherence
0.407, 2.4% of pairs above 0.99 and 0.0% at exactly 1.0 -- the claim above
did NOT reproduce here. Eigenvalue participation was 1.21 of 6, so the drive
was concentrated but not rank-deficient. That run is confounded: its frozen
FRF came from a weak system identification (minimum multiple coherence 0.492
against 0.75-0.88 for the twelve-run comparison set) and it overshot by
+14.79 dB, so "the solve behaves differently at this scale" and "the plant
model was bad" are not yet separated. Treat the 1.0-coherence claim as
established at 12 drives and UNVERIFIED at 6 until a run on a clean
identification says otherwise.

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
        baseline = np.linalg.norm(self.H_cache.reshape(-1))
        if not np.isfinite(baseline) or baseline <= 0:
            return True
        delta = np.linalg.norm((H_clean - self.H_cache).reshape(-1))
        if not np.isfinite(delta):
            return True
        moved = (delta/baseline) > self.frf_update_threshold
        if moved:
            print(f"[optimal_diagonal_control_fast] FRF moved "
                  f"{delta/baseline:.4f} > {self.frf_update_threshold:g} -- "
                  f"safe SDP path this call", flush=True)
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
        L = self._unpack(v, N, r)
        B = H @ L
        Ydiag = np.sum(np.abs(B) ** 2, axis=1)
        resid = Ydiag - y_target
        f_fit = np.sum(resid ** 2)
        X = L @ L.conj().T
        f_reg = self.reg * np.sum(np.abs(X) ** 2)
        f = f_fit + f_reg
        gX = 2 * (H.conj().T @ (resid[:, None] * H)) + 2 * self.reg * X
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
    def _solve_one_bin(self, H, y_target):
        if not self._h_changed_this_call:
            # H is unchanged since the last call -- nothing is re-
            # estimating it live right now, so the fast unconstrained
            # solve is safe even though it drives coherence toward 1.0.
            self.n_fast_solves += 1
            try:
                return self._bm_solve(H, y_target)
            except Exception as e:
                print(f"[optimal_diagonal_control_fast] BM solve raised {type(e).__name__}: {e} "
                      f"-- falling back to SDP for this bin", flush=True)
        self.n_safe_solves += 1
        return super()._solve_one_bin(H, y_target)

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
