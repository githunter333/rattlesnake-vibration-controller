"""Rattlesnake Type-1 (generator) adapter for forcefinder's SPR random control law.

forcefinder's ``RandomControlSourcePathReceiver`` is a spec-first Type-2 "Class" control law
(``__init__`` takes the specification first, plus ``system_id_update`` / ``control``). Rattlesnake's
current loader mis-detects a plain class as Type-3 ("Interactive") and crashes on ``get_ui_class``
(the same issue as the project's ``band_average`` class). This module wraps it in a Type-1 generator, which
Rattlesnake drives without the interactive-UI machinery: create once, ``next()``, then ``send()`` the
per-frame data tuple. The generator instantiates the SPR law and runs ``system_id_update`` once on the
first frame (front-loading the L-curve inverse), then calls ``control`` each frame.

Set the inverse technique and options in the Rattlesnake control-parameters box. Note forcefinder uses
``name = value`` syntax (with an equals sign), one per line, e.g.:

    df = 1.0
    ISE_technique = auto_tikhonov_by_l_curve
    use_buzz = False
"""
import importlib.util
import os

import sys
import types

import numpy as np

# Rattlesnake's controller processes run on PyQt6. Importing forcefinder loaded PyQt5 as well --
# its package __init__ imports a PyQt5 GUI module (transient_quality_evaluation.transient_gui_plot),
# and sdynpy's geometry module imports pyvistaqt -> qtpy, which picks PyQt5 first -- and two Qt
# bindings in one process crashed a controller worker (SIGBUS in garbage collection, 2026-09-27;
# the run recorded nothing). The GUI module is replaced by a placeholder here (the control law
# never plots), and qtpy is told to use the binding Rattlesnake already has.
os.environ.setdefault("QT_API", "pyqt6")
_GUI = "forcefinder.transient_quality_evaluation.transient_gui_plot"
if _GUI not in sys.modules:
    _stub = types.ModuleType(_GUI)
    _stub.SpectrogramGUI = _stub.LevelErrorGUI = None
    sys.modules[_GUI] = _stub

from forcefinder.rattlesnake_control_laws.spr_random_control_law import RandomControlSourcePathReceiver


def spr_random_generator():
    """Type-1 generator wrapping ``RandomControlSourcePathReceiver`` (front-loads the ISE inverse)."""
    law = None
    data = yield                                       # next() lands here; first send() supplies data
    while True:
        (specification, warning_levels, abort_levels, transfer_function,
         noise_response_cpsd, noise_reference_cpsd, sysid_response_cpsd,
         sysid_reference_cpsd, multiple_coherence, frames, total_frames,
         extra_parameters, last_response_cpsd, last_output_cpsd) = data
        if law is None:                                # build the SPR law + run the inverse once
            law = RandomControlSourcePathReceiver(
                specification, warning_levels, abort_levels, extra_parameters, transfer_function,
                noise_response_cpsd, noise_reference_cpsd, sysid_response_cpsd, sysid_reference_cpsd,
                multiple_coherence, frames, total_frames, last_response_cpsd, last_output_cpsd)
            law.system_id_update(transfer_function, noise_response_cpsd, noise_reference_cpsd,
                                 sysid_response_cpsd, sysid_reference_cpsd, multiple_coherence,
                                 frames, total_frames)
        drive = law.control(transfer_function, multiple_coherence, frames, total_frames,
                            last_response_cpsd, last_output_cpsd)
        data = yield drive


def spr_random_study_generator():
    """``spr_random_generator`` from a stale or noisy design FRF -- a study aid.

    The design-FRF lines of the band-average study wrapper (``design_frf_file:``,
    ``design_noise_level:``, ``design_noise_seed:``, ``design_buzz_file:``, colon syntax) are stripped
    from the parameters before forcefinder reads them, and the transfer function
    is replaced by that design on every frame, with the wrapper's own noise model; a
    ``design_buzz_file`` replaces the system identification's response CPSD (forcefinder's
    ``use_buzz`` cross terms) the same way.
    """
    spec = importlib.util.spec_from_file_location(
        "band_average_study_wrapper",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "band_average_study_wrapper.py"))
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    inner = spr_random_generator()
    next(inner)
    design = None
    data = yield
    while True:
        data = list(data)
        study, data[11] = wrapper._split(data[11])
        if design is None:
            design = wrapper._design(data[3], study)
        data[3] = design
        if "design_buzz_file" in study:                # the healthy article's buzz test, identified before it changed
            data[6] = np.load(study["design_buzz_file"])["cpsd"]
        drive = inner.send(tuple(data))
        data = yield drive
