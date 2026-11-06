"""Nonlinear variant of the synthetic SDynPy "hardware".

Same linear mass/damping/stiffness integration as
``sdynpy_system_virtual_hardware.py``, plus an amplitude-dependent
softening-stiffness and quadratic-damping term on one or more target modes.

This is a port of ``components/sdynpy_nonlinear_system_virtual_hardware.py``
from branch ``control-law-development-2026-09`` onto the current package layout.
It is the hardware the sixdrive12resp bounded-softening runs (runs 49-52, 55,
56) were measured on; that module did not survive the move to
``src/rattlesnake/`` and was absent from ``control-laws-on-dev``, which is why
runs 53 and 54 -- the fast law's pair on the moving plant -- were never done.
The physics below is carried over unchanged.  What changed in the port:

  * it subclasses the current ``SDynPySystemAcquisition``, whose
    ``create_response_channels`` keeps ``phi_response`` and ``ndofs`` for us
    rather than discarding them, so the modal projection is built from the same
    matrices the linear state space was built from (no duplicated assembly);
  * the force-buffer prologue and the decimated return follow the current base
    class exactly (it retains the last force sample between frames, and returns
    ``[..., :-1:integration_oversample]``);
  * ``_check_frf_switch()`` is dropped -- it belonged to the FRF-switch study
    and has no counterpart here.

Modal form for EACH target mode's mass-normalized coordinate q_i (mn=1):

    q_i'' + 2 zeta0_i wn_i q_i' + wn_i^2 q_i + k3_i q_i^3
            + c2_i q_i' |q_i'| = Qn_i(t)

projected onto the physical state through each mode's own mass-normalized
eigenvector phi_i (phi_i^T M phi_i = 1) and summed over target modes:

    x'' = [linear part, the same A matrix as the base class]
          - sum_i phi_i (k3_i q_i^3 + c2_i q_i' |q_i'|)
    q_i = phi_i^T M x_disp,   q_i' = phi_i^T M x_vel

which is why M is needed once at setup even though the base class discards
M/C/K after building its state matrices.  Vectorized across modes throughout.

BOUNDED SOFTENING (2026-09-24) takes precedence when the system file carries
``nl_soft_qts``.  The restoring force saturates onto a residual linear
stiffness ``alpha wn^2`` instead of reversing sign, so there is no finite
potential well and the plant cannot escape at any amplitude or strength -- the
failure that ended the cubic runs 42/44/46/48.

System file format: accepts the single-mode keys (``nl_target_mode_shape``,
scalar ``nl_k3``/``nl_c2``), the all-modes keys (``nl_target_mode_shapes``,
``nl_k3s``/``nl_c2s``) or the bounded-softening keys (``nl_soft_qts``,
``nl_soft_wn2s``, ``nl_soft_alpha``).  All are normalized internally to the
(ndof, n_modes) / (n_modes,) all-modes shape.

RATTLESNAKE_NONLINEARITY_STRENGTH (environment, default 1.0) multiplies
(k3, c2) uniformly for every target mode.  0 recovers the linear baseline
exactly, except for the integration scheme.
"""

import multiprocessing as mp
import os
import time
from typing import List

import numpy as np

from rattlesnake.hardware.hardware_utilities import Channel
from rattlesnake.hardware.sdynpy_system_virtual_hardware import (
    SDynPySystemAcquisition,
    SDynPySystemMetadata,
    SDynPySystemOutput,
)

#: RK4 substeps between reported samples.  A single step per sample does NOT
#: resolve this system's higher modes (to ~900 Hz): a single-substep RK4 was
#: measured ~13 % off the linear system's exact lsim baseline at strength 0, on
#: acceleration outputs, where C's omega^2-scale coefficients amplify small
#: state errors.  RK4 is 4th order, so 4 substeps is ~0.13/4^4 = ~0.05 %, far
#: better than 20 substeps' ~1e-6 (accurate but too slow for real time here).
N_SUB = 4


class SDynPyNonlinearSystemMetadata(SDynPySystemMetadata):
    """Identical to the linear metadata; the hardware type comes from the file."""


class SDynPyNonlinearSystemOutput(SDynPySystemOutput):
    """Unchanged from the linear output stage."""


class SDynPyNonlinearSystemAcquisition(SDynPySystemAcquisition):
    """The linear SDynPy integrator with a modal nonlinear restoring force."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nl_ready = False
        self._has_nonlinearity = False
        self._nl_soft = False
        self.nl_phis = None
        self.nl_k3_base = None
        self.nl_c2_base = None
        self.nl_soft_qt_base = None
        self.nl_soft_wn2 = None
        self.nl_soft_alpha = None
        self._nl_w_modal = None
        self._nl_response_coeff = None
        self._nl_accel_mask = None
        strength = os.environ.get("RATTLESNAKE_NONLINEARITY_STRENGTH")
        self.nl_strength = float(strength) if strength not in (None, "") else 1.0

    def _read_nonlinear_keys(self):
        """Normalize whichever nl_* key set the system file carries."""
        d = self.sdynpy_system_data
        if "nl_target_mode_shapes" in d and "nl_soft_qts" in d:
            self._has_nonlinearity = True
            self._nl_soft = True
            self.nl_phis = np.atleast_2d(d["nl_target_mode_shapes"])
            self.nl_soft_qt_base = np.atleast_1d(d["nl_soft_qts"]).astype(float)
            self.nl_soft_wn2 = np.atleast_1d(d["nl_soft_wn2s"]).astype(float)
            self.nl_soft_alpha = float(d["nl_soft_alpha"])
            self.nl_c2_base = (
                np.atleast_1d(d["nl_c2s"]).astype(float)
                if "nl_c2s" in d
                else np.zeros_like(self.nl_soft_qt_base)
            )
        elif "nl_target_mode_shapes" in d and "nl_k3s" in d and "nl_c2s" in d:
            self._has_nonlinearity = True
            self.nl_phis = np.atleast_2d(d["nl_target_mode_shapes"])
            self.nl_k3_base = np.atleast_1d(d["nl_k3s"]).astype(float)
            self.nl_c2_base = np.atleast_1d(d["nl_c2s"]).astype(float)
        elif "nl_target_mode_shape" in d and "nl_k3" in d and "nl_c2" in d:
            self._has_nonlinearity = True
            self.nl_phis = np.asarray(d["nl_target_mode_shape"])[:, None]
            self.nl_k3_base = np.atleast_1d(float(d["nl_k3"]))
            self.nl_c2_base = np.atleast_1d(float(d["nl_c2"]))
        else:
            self._has_nonlinearity = False
            print(
                "[SDynPyNonlinearSystemAcquisition] WARNING: system file has no "
                "nl_* keys -- behaving as a pure linear system.",
                flush=True,
            )
            return

        n_modes = self.nl_phis.shape[1]
        tag = (
            f"[SDynPyNonlinearSystemAcquisition] "
            f"nonlinearity_strength={self.nl_strength:g} n_modes={n_modes}"
        )
        if self._nl_soft:
            if self.nl_strength < 0.0:
                raise ValueError(
                    "RATTLESNAKE_NONLINEARITY_STRENGTH must be >= 0 for a "
                    f"bounded-softening system file (got {self.nl_strength:g}); "
                    "strength scales qt by 1/sqrt(strength) and a negative "
                    "value has no meaning."
                )
            finite = np.isfinite(self.nl_soft_qt_base)
            qf = self.nl_soft_qt_base[finite]
            print(
                f"{tag} BOUNDED SOFTENING alpha={self.nl_soft_alpha:.3f} "
                f"({int(finite.sum())}/{n_modes} modes softened, "
                f"qt range=[{qf.min():.3g},{qf.max():.3g}], "
                f"c2 range=[{self.nl_c2_base.min():.3g},"
                f"{self.nl_c2_base.max():.3g}]) -- restoring force is bounded "
                "below by alpha*wn^2*q, escape is impossible",
                flush=True,
            )
        else:
            print(
                f"{tag} CUBIC (k3 range=[{self.nl_k3_base.min():.3g},"
                f"{self.nl_k3_base.max():.3g}], c2 range="
                f"[{self.nl_c2_base.min():.3g},{self.nl_c2_base.max():.3g}])",
                flush=True,
            )

    def create_response_channels(self, channel_data: List[Channel]):
        super().create_response_channels(channel_data)
        self._read_nonlinear_keys()
        self._nl_ready = True
        if not self._has_nonlinearity:
            return

        # q = (Phi^T M) x_disp.  phi_response and ndofs are retained by the base
        # class, so this uses the same matrices the state space was built from.
        M = self.sdynpy_system_data["mass"]
        self._nl_w_modal = self.nl_phis.T @ M

        # Per response channel (in phi_response row order): the coefficient that
        # maps the modal nonlinear force onto that channel's acceleration, and
        # whether the channel is acceleration-type at all.
        coeffs = []
        accel_mask = []
        response_index = 0
        for i, channel in enumerate(channel_data):
            if not self.response_channels[i]:
                continue
            coeffs.append(-(self.phi_response[response_index] @ self.nl_phis))
            accel_mask.append(
                str(channel.channel_type).lower() in ["accel", "acceleration", "acc"]
            )
            response_index += 1
        self._nl_response_coeff = np.array(coeffs)
        self._nl_accel_mask = np.array(accel_mask, dtype=bool)

    def read(self):
        """The base class's read(), with lsim replaced by RK4 and the nonlinear
        acceleration correction added on acceleration-type response channels."""
        start_time = time.time()
        while self.force_buffer.shape[0] < self.times.size:
            try:
                forces = self.queue.get(timeout=self.frame_time)
            except mp.queues.Empty:
                print("Warning! SDynPy integrator ran out of samples!")
                forces = np.zeros((self.force_buffer.shape[-1], self.times.size))
            self.force_buffer = np.concatenate((self.force_buffer, forces.T), axis=0)

        this_force = self.force_buffer[: self.times.size]
        # Keep the last force sample: it is the first sample of the next frame.
        self.force_buffer = self.force_buffer[self.times.size - 1 :]

        active = self._has_nonlinearity and self.nl_strength != 0.0
        soft = active and self._nl_soft
        k3s = c2s = qts = soft_wn2 = soft_a = None
        if active:
            c2s = self.nl_c2_base * self.nl_strength
            if soft:
                # strength scales qt as qt/sqrt(strength), reproducing
                # k3_eff = strength*k3 in the small-amplitude limit while
                # leaving alpha -- and therefore boundedness -- untouched.
                qts = self.nl_soft_qt_base / np.sqrt(self.nl_strength)
                soft_wn2 = self.nl_soft_wn2
                soft_a = self.nl_soft_alpha
            else:
                k3s = self.nl_k3_base * self.nl_strength

        A, B = self.system.A, self.system.B
        C_out, D_out = self.system.C, self.system.D
        ndof = A.shape[0] // 2
        w_modal = self._nl_w_modal
        phis = self.nl_phis

        def nl_terms_fn(x_state):
            """(n_modes,) nonlinear generalized-force term per mode."""
            q = w_modal @ x_state[:ndof]
            qd = w_modal @ x_state[ndof:]
            if soft:
                # Departure from linear: wn^2 q (1-alpha) (exp(-(q/qt)^2) - 1).
                # expm1 is exact as q -> 0 and saturates at -1, so it cannot
                # overflow at any amplitude.  Unsoftened modes carry qt = inf,
                # giving expm1(0) = 0 exactly: they stay bit-identically linear.
                return (
                    soft_wn2 * q * (1.0 - soft_a) * np.expm1(-((q / qts) ** 2))
                    + c2s * qd * np.abs(qd)
                )
            return k3s * q**3 + c2s * qd * np.abs(qd)

        def deriv(x_state, u):
            xdot = A @ x_state + B @ u
            if active:
                xdot[ndof:] -= phis @ nl_terms_fn(x_state)
            return xdot

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
                um = u0_full + ((s + 0.5) / N_SUB) * (u1_full - u0_full)
                u1 = u0_full + ((s + 1.0) / N_SUB) * (u1_full - u0_full)
                k1 = deriv(x, u0)
                k2 = deriv(x + 0.5 * dt * k1, um)
                k3_ = deriv(x + 0.5 * dt * k2, um)
                k4 = deriv(x + dt * k3_, u1)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3_ + k4)

        self.state[:] = x_out_states[-1]

        # The linear part of the output, exactly as lsim would have given it.
        sys_out = np.einsum("on,tn->to", C_out, x_out_states) + np.einsum(
            "oi,ti->to", D_out, this_force
        )

        # The nonlinear correction applies only to acceleration-type response
        # channels: displacement and velocity channels already come straight
        # from the true nonlinear state trajectory, while the acceleration row
        # is a linear formula in the state.  C_state has one row per channel,
        # so response_channels indexes sys_out's columns directly.
        if active and self._nl_response_coeff is not None:
            nl_series = np.array(
                [nl_terms_fn(x_out_states[i]) for i in range(n_steps)]
            )
            correction = nl_series @ self._nl_response_coeff.T
            correction[:, ~self._nl_accel_mask] = 0.0
            sys_out[:, self.response_channels] += correction

        integration_time = time.time() - start_time
        remaining_time = self.frame_time - integration_time
        if remaining_time > 0.0 and self.sleep:
            time.sleep(remaining_time)

        # The last sample is the initial state of the next frame.
        return sys_out.T[..., : -1 : self.integration_oversample]
