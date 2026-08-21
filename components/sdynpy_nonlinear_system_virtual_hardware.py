"""
Nonlinear variant of the synthetic SDynPy "hardware": same linear
mass/damping/stiffness integration as sdynpy_system_virtual_hardware.py,
plus an amplitude-dependent softening-stiffness + quadratic-damping term
on one target mode (see examples/sixdrive12resp/code/
build_nonlinear_frf_system.py for how the nonlinear coefficients are
calibrated and stored in the system .npz).

This is a NEW file that subclasses SDynPySystemAcquisition without
modifying it -- the existing linear hardware class, and every live run
that depends on it, is untouched. Only read() is overridden: scipy's
signal.lsim (exact for LTI systems only) is replaced with a fixed-step
RK4 integrator, since the nonlinear restoring force makes the system
state-dependent in a way lsim can't handle.

Modal form for the target mode's mass-normalized coordinate q (mn=1):
    q'' + 2*zeta0*wn*q' + wn^2*q + k3*q^3 + c2*q'*|q'| = Qn(t)
projected onto the physical state via the mode's own mass-normalized
eigenvector phi (phi^T M phi = 1):
    x'' = [linear part, same A matrix as the base class] - phi*(k3*q^3 + c2*q'*|q'|)
    q = phi^T M x_disp,  q' = phi^T M x_vel
which is why M is needed once at setup (to build the fixed projection row
vector phi^T M) even though the base class discards M/C/K after building
its state-space matrices.

RATTLESNAKE_NONLINEARITY_STRENGTH (env var, default 1.0): multiplies
(k3, c2) together. strength=0 recovers the pure linear baseline exactly
(the RK4 integrator then just reproduces what lsim would have given,
modulo integration scheme -- verified in the build/validation script).
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
        self._has_nonlinearity = 'nl_k3' in d and 'nl_c2' in d and 'nl_target_mode_shape' in d
        if not self._has_nonlinearity:
            print("[SDynPyNonlinearSystemAcquisition] WARNING: system file has no nl_* keys -- "
                  "behaving as a pure linear system (use build_nonlinear_frf_system.py's output).",
                  flush=True)
            self.nl_phi = None
            self.nl_k3_base = 0.0
            self.nl_c2_base = 0.0
        else:
            self.nl_phi = d['nl_target_mode_shape']
            self.nl_k3_base = float(d['nl_k3'])
            self.nl_c2_base = float(d['nl_c2'])

        strength_env = os.environ.get('RATTLESNAKE_NONLINEARITY_STRENGTH')
        self.nl_strength = float(strength_env) if strength_env not in (None, '') else 1.0

        # Set up once M is available (create_response_channels loads it locally
        # and discards it -- redo that one step here to build the fixed
        # phi^T M projection row vector, and to know which output channels
        # are acceleration-type and therefore need the nonlinear correction).
        self._nl_w_modal = None          # (ndof,) = phi^T M, fixed
        self._nl_response_coeff = None   # (n_response_channels,) = -(phi_response_row . phi), per channel
        self._nl_accel_mask = None       # (n_response_channels,) bool, True where channel_type is acceleration

        print(f"[SDynPyNonlinearSystemAcquisition] nonlinearity_strength={self.nl_strength:g} "
              f"(k3={self.nl_k3_base*self.nl_strength:.3f}, c2={self.nl_c2_base*self.nl_strength:.6f})",
              flush=True)

    def create_response_channels(self, channel_data: List[Channel]):
        super().create_response_channels(channel_data)
        if not self._has_nonlinearity:
            return
        M = self.sdynpy_system_data['mass']
        self._nl_w_modal = self.nl_phi @ M   # fixed row vector: q = self._nl_w_modal @ x_disp

        # response-channel-ordered (not full channel_data-ordered) coefficient
        # and accel-type mask, matching self.phi_response's row order
        coeffs = []
        accel_mask = []
        response_idx = 0
        for i, channel in enumerate(channel_data):
            if not self.response_channels[i]:
                continue
            coeffs.append(-(self.phi_response[response_idx] @ self.nl_phi))
            accel_mask.append(channel.channel_type.lower() in ['accel', 'acceleration', 'acc'])
            response_idx += 1
        self._nl_response_coeff = np.array(coeffs)
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

        if not self._has_nonlinearity or self.nl_strength == 0.0:
            k3 = c2 = 0.0
        else:
            k3 = self.nl_k3_base * self.nl_strength
            c2 = self.nl_c2_base * self.nl_strength

        A, B, C_out, D_out = self.system.A, self.system.B, self.system.C, self.system.D
        ndof = A.shape[0] // 2
        w_modal = self._nl_w_modal
        phi = self.nl_phi

        def nl_accel(x_state):
            if phi is None or (k3 == 0.0 and c2 == 0.0):
                return 0.0, 0.0
            x_disp = x_state[:ndof]
            x_vel = x_state[ndof:]
            q = w_modal @ x_disp
            qd = w_modal @ x_vel
            nl_term = k3 * q ** 3 + c2 * qd * abs(qd)
            return nl_term, q  # nl_term used both for state accel correction and output correction

        def deriv(x_state, u):
            nl_term, _ = nl_accel(x_state)
            xdot = A @ x_state + B @ u
            xdot[ndof:] -= phi * nl_term if phi is not None else 0.0
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
        if phi is not None and self._nl_response_coeff is not None and (k3 != 0.0 or c2 != 0.0):
            nl_terms = np.array([nl_accel(x_out_states[i])[0] for i in range(n_steps)])  # (n_steps,)
            correction = np.outer(nl_terms, self._nl_response_coeff)  # (n_steps, n_response_channels)
            correction[:, ~self._nl_accel_mask] = 0.0
            sys_out[:, self.response_channels] += correction

        integration_time = time.time() - start_time
        remaining_time = self.frame_time - integration_time
        if remaining_time > 0.0:
            time.sleep(remaining_time)

        return sys_out.T[..., ::self.integration_oversample]
