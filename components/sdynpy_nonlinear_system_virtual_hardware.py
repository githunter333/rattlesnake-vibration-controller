"""
Nonlinear variant of the synthetic SDynPy "hardware": same linear
mass/damping/stiffness integration as sdynpy_system_virtual_hardware.py,
plus an amplitude-dependent softening-stiffness + quadratic-damping term
on one OR MORE target modes (see examples/sixdrive12resp/code/
build_nonlinear_frf_system.py for the single-mode case and
build_nonlinear_frf_system_allmodes.py for the all-modes case -- how the
nonlinear coefficients are calibrated and stored in the system .npz).

This is a NEW file that subclasses SDynPySystemAcquisition without
modifying it -- the existing linear hardware class, and every live run
that depends on it, is untouched. Only read() is overridden: scipy's
signal.lsim (exact for LTI systems only) is replaced with a fixed-step
RK4 integrator, since the nonlinear restoring force makes the system
state-dependent in a way lsim can't handle.

Modal form for EACH target mode's mass-normalized coordinate q_i (mn=1):
    q_i'' + 2*zeta0_i*wn_i*q_i' + wn_i^2*q_i + k3_i*q_i^3 + c2_i*q_i'*|q_i'| = Qn_i(t)
projected onto the physical state via each mode's own mass-normalized
eigenvector phi_i (phi_i^T M phi_i = 1), summed across all target modes:
    x'' = [linear part, same A matrix as the base class]
          - sum_i phi_i*(k3_i*q_i^3 + c2_i*q_i'*|q_i'|)
    q_i = phi_i^T M x_disp,  q_i' = phi_i^T M x_vel
which is why M is needed once at setup (to build the fixed projection
matrix Phi^T M) even though the base class discards M/C/K after building
its state-space matrices. Vectorized across modes throughout (a single
mode is just the n_modes=1 case, no special-casing needed).

System file format: accepts EITHER the single-mode keys (nl_target_mode_
shape (ndof,), nl_k3/nl_c2 scalars -- build_nonlinear_frf_system.py) OR
the all-modes keys (nl_target_mode_shapes (ndof,n_modes), nl_k3s/nl_c2s
(n_modes,) arrays -- build_nonlinear_frf_system_allmodes.py). Both are
normalized internally to the (ndof,n_modes)/(n_modes,) all-modes shape.

RATTLESNAKE_NONLINEARITY_STRENGTH (env var, default 1.0): multiplies
(k3, c2) together for EVERY target mode uniformly. strength=0 recovers
the pure linear baseline exactly (the RK4 integrator then just reproduces
what lsim would have given, modulo integration scheme -- verified in the
build/validation scripts).
"""

from .sdynpy_system_virtual_hardware import SDynPySystemAcquisition, SDynPySystemOutput
from .utilities import Channel
import numpy as np
from typing import List
import multiprocessing as mp
import time
import os


class SDynPyNonlinearSystemAcquisition(SDynPySystemAcquisition):
    def __init__(self, system_file: str, queue: mp.queues.Queue):
        super().__init__(system_file, queue)
        d = self.sdynpy_system_data

        if 'nl_target_mode_shapes' in d and 'nl_k3s' in d and 'nl_c2s' in d:
            self._has_nonlinearity = True
            self.nl_phis = np.atleast_2d(d['nl_target_mode_shapes'])       # (ndof, n_modes)
            self.nl_k3_base = np.atleast_1d(d['nl_k3s']).astype(float)     # (n_modes,)
            self.nl_c2_base = np.atleast_1d(d['nl_c2s']).astype(float)     # (n_modes,)
        elif 'nl_target_mode_shape' in d and 'nl_k3' in d and 'nl_c2' in d:
            self._has_nonlinearity = True
            self.nl_phis = np.asarray(d['nl_target_mode_shape'])[:, None]  # (ndof, 1)
            self.nl_k3_base = np.atleast_1d(float(d['nl_k3']))
            self.nl_c2_base = np.atleast_1d(float(d['nl_c2']))
        else:
            self._has_nonlinearity = False
            print("[SDynPyNonlinearSystemAcquisition] WARNING: system file has no nl_* keys -- "
                  "behaving as a pure linear system (use build_nonlinear_frf_system.py or "
                  "build_nonlinear_frf_system_allmodes.py's output).", flush=True)
            self.nl_phis = None
            self.nl_k3_base = None
            self.nl_c2_base = None

        strength_env = os.environ.get('RATTLESNAKE_NONLINEARITY_STRENGTH')
        self.nl_strength = float(strength_env) if strength_env not in (None, '') else 1.0

        # Set up once M is available (create_response_channels loads it locally
        # and discards it -- redo that one step here to build the fixed
        # Phi^T M projection matrix, and to know which output channels
        # are acceleration-type and therefore need the nonlinear correction).
        self._nl_w_modal = None          # (n_modes, ndof) = Phi^T M, fixed
        self._nl_response_coeff = None   # (n_response_channels, n_modes) = -(phi_response_row . phi_i), per channel/mode
        self._nl_accel_mask = None       # (n_response_channels,) bool, True where channel_type is acceleration

        n_modes = self.nl_phis.shape[1] if self._has_nonlinearity else 0
        print(f"[SDynPyNonlinearSystemAcquisition] nonlinearity_strength={self.nl_strength:g} "
              f"n_modes={n_modes} "
              f"(k3 range=[{self.nl_k3_base.min():.3g},{self.nl_k3_base.max():.3g}], "
              f"c2 range=[{self.nl_c2_base.min():.3g},{self.nl_c2_base.max():.3g}])"
              if self._has_nonlinearity else
              f"[SDynPyNonlinearSystemAcquisition] nonlinearity_strength={self.nl_strength:g} n_modes=0",
              flush=True)

    def create_response_channels(self, channel_data: List[Channel]):
        super().create_response_channels(channel_data)
        if not self._has_nonlinearity:
            return
        M = self.sdynpy_system_data['mass']
        self._nl_w_modal = self.nl_phis.T @ M   # (n_modes, ndof): q = self._nl_w_modal @ x_disp

        # response-channel-ordered (not full channel_data-ordered) coefficient
        # and accel-type mask, matching self.phi_response's row order
        coeffs = []
        accel_mask = []
        response_idx = 0
        for i, channel in enumerate(channel_data):
            if not self.response_channels[i]:
                continue
            coeffs.append(-(self.phi_response[response_idx] @ self.nl_phis))  # (n_modes,)
            accel_mask.append(channel.channel_type.lower() in ['accel', 'acceleration', 'acc'])
            response_idx += 1
        self._nl_response_coeff = np.array(coeffs)  # (n_response_channels, n_modes)
        self._nl_accel_mask = np.array(accel_mask, dtype=bool)

    def read(self):
        """Same force-buffering prologue/epilogue as the base class's
        read(); only the integration step (lsim -> RK4) and the output
        computation (add the nonlinear acceleration correction on
        acceleration-type response channels) differ."""
        self._check_frf_switch()
        start_time = time.time()
        while self.force_buffer.shape[0] < self.times.size:
            try:
                forces = self.queue.get(timeout=self.frame_time)
            except mp.queues.Empty:
                forces = np.zeros((self.force_buffer.shape[-1], self.times.size))
            self.force_buffer = np.concatenate((self.force_buffer, forces.T), axis=0)

        this_force = self.force_buffer[:self.times.size]
        self.force_buffer = self.force_buffer[self.times.size:]

        active = self._has_nonlinearity and self.nl_strength != 0.0
        if active:
            k3s = self.nl_k3_base * self.nl_strength    # (n_modes,)
            c2s = self.nl_c2_base * self.nl_strength    # (n_modes,)

        A, B, C_out, D_out = self.system.A, self.system.B, self.system.C, self.system.D
        ndof = A.shape[0] // 2
        w_modal = self._nl_w_modal    # (n_modes, ndof)
        phis = self.nl_phis           # (ndof, n_modes)

        def nl_terms_fn(x_state):
            """Returns (n_modes,) nonlinear generalized-force term per mode."""
            if not active:
                return None
            x_disp = x_state[:ndof]
            x_vel = x_state[ndof:]
            q = w_modal @ x_disp     # (n_modes,)
            qd = w_modal @ x_vel     # (n_modes,)
            return k3s * q ** 3 + c2s * qd * np.abs(qd)   # (n_modes,)

        def deriv(x_state, u):
            xdot = A @ x_state + B @ u
            if active:
                nl_terms = nl_terms_fn(x_state)
                xdot[ndof:] -= phis @ nl_terms   # (ndof,)
            return xdot

        # RK4 substeps between each reported sample -- a single step per
        # sample (dt ~ 1/sample_rate) is NOT fine enough to resolve this
        # system's higher modes (up to ~900 Hz) against a fixed-step
        # integrator; verified (debug_rk4_vs_lsim.py prototype) that a
        # single-substep RK4 disagreed with the linear system's exact lsim
        # baseline by ~13% (relative, on acceleration outputs specifically,
        # where the C matrix's omega^2-scale coefficients amplify small
        # state errors) at strength=0. RK4 is 4th-order (error ~ 1/N_SUB^4),
        # so N_SUB=4 should already be ~0.13/4^4 =~ 0.05% -- far better than
        # N_SUB=20's ~1e-6 (verified accurate but too slow for real time at
        # this frame size) while staying well within a real-time budget.
        N_SUB = 4
        n_steps = self.times.size
        x_out_states = np.empty((n_steps, A.shape[0]))
        x = self.state.copy()
        dt_outer = self.times[1] - self.times[0] if n_steps > 1 else 0.0
        dt = dt_outer / N_SUB
        for i in range(n_steps):
            x_out_states[i] = x
            if i == n_steps - 1:
                break
            u0_full = this_force[i]
            u1_full = this_force[i + 1]
            for s in range(N_SUB):
                u0 = u0_full + (s / N_SUB) * (u1_full - u0_full)
                u1 = u0_full + ((s + 1) / N_SUB) * (u1_full - u0_full)
                um = u0_full + ((s + 0.5) / N_SUB) * (u1_full - u0_full)
                k1 = deriv(x, u0)
                k2 = deriv(x + 0.5 * dt * k1, um)
                k3_ = deriv(x + 0.5 * dt * k2, um)
                k4 = deriv(x + dt * k3_, u1)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3_ + k4)

        self.state[:] = x_out_states[-1]

        # linear part of the output, exactly as lsim would have given it
        sys_out = np.einsum('on,tn->to', C_out, x_out_states) + np.einsum('oi,ti->to', D_out, this_force)

        # nonlinear correction on acceleration-type response channels only
        # (displacement/velocity channels are already exact -- they come
        # straight from the true nonlinear state trajectory, not from a
        # linear-acceleration formula the way the accel output row is)
        if active and self._nl_response_coeff is not None:
            nl_terms_series = np.array([nl_terms_fn(x_out_states[i]) for i in range(n_steps)])  # (n_steps, n_modes)
            correction = nl_terms_series @ self._nl_response_coeff.T  # (n_steps, n_response_channels)
            correction[:, ~self._nl_accel_mask] = 0.0
            sys_out[:, self.response_channels] += correction

        integration_time = time.time() - start_time
        remaining_time = self.frame_time - integration_time
        if remaining_time > 0.0:
            time.sleep(remaining_time)

        return sys_out.T[..., ::self.integration_oversample]
