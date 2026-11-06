"""Run a sixdrive12resp profile unattended, saving spectral data at each level.

The bounded-softening runs were GUI-driven: an operator stepped the test level
and pressed "Save Spectral Data" at each one, which is why make_moving_plant_
figures.py scores one file per level (``<base>_spec<level>.nc4``, reading
``response_cpsd_real`` / ``drive_cpsd_real``) rather than a streaming file.

This drives the same sequence with no GUI, and it is PACED BY DATA, not by the
clock.  The nonlinear integrator runs at about an eighth of real time, so a
wall-clock schedule written in data-seconds fires roughly eight times too early
-- a 45 s-a-level profile ramped through all five levels inside the first ~34 s
of data and saved five bit-identical files.  The level is therefore stepped, and
the spectral data saved, only once the controller's OWN frame count has advanced
by `--settle-frames` since the last change.  Both go through the same calls the
GUI buttons use:

    send_environment_command(env, RandomVibrationCommands.ADJUST_TEST_LEVEL, level)
    send_environment_command(env, RandomVibrationCommands.SAVE_CONTROL_DATA, path)

The frame count arrives on the GUI update queue as the first element of each
CONTROL_UPDATE payload, so progress is read from the controller rather than
guessed.

Adapted from the rattlesnake-band-average study's run_headless.py, which is the
proven driver for this package layout.  Its gotchas still apply and are kept:
the __main__ guard (macOS spawns), a raised controller timeout for a real-time
system ID, float coercion on event data, RandomVibrationInstructions injected
into Start Environment, a raised file-descriptor limit, and draining the GUI
update queue (Rattlesnake's own trimmer dies on macOS at Queue.qsize(), so
nothing else empties it and control updates accumulate in memory).

    python run_headless_levels.py profile.xlsx out_base [wait_s] [--levels -18,...]

Writes out_base_spec<level>.nc4 per level plus out_base_stream.nc4.
"""

import os
import queue
import resource
import sys
import threading
import time
import traceback

_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _hard), _hard))


def log(*a):
    print(*a, flush=True)


def watch(q, stop, env_name, state):
    """Drain the GUI update queue, recording control progress as it goes.

    Draining is not optional: Rattlesnake's own trimmer dies on macOS at its
    first Queue.qsize(), so nothing else empties this queue and every control
    update -- the full response CPSD each cycle -- would accumulate in memory.
    """
    while not stop.is_set():
        try:
            item = q.get(timeout=0.25)
        except queue.Empty:
            continue
        try:
            name, (command, payload) = item
            if name == env_name and getattr(command, "name", "") == "CONTROL_UPDATE":
                state["updates"] += 1
                if isinstance(payload, (tuple, list)) and payload:
                    state["frames"] = int(payload[0])
        except Exception:
            pass


def level_tag(level):
    """Match the historical file naming: -18 -> 'm18', 0 -> '0'."""
    lv = int(round(float(level)))
    return f"m{abs(lv)}" if lv < 0 else f"{lv}"


def run(xlsx, out_base, levels, settle_frames, per_level_cap, overall_cap):
    import openpyxl
    from rattlesnake.engine import RattlesnakeController
    from rattlesnake.load_utilities import load_metadata_from_workbook
    from rattlesnake.process.streaming import StreamType, StreamMetadata
    from rattlesnake.environment.random_vibration_sys_id_environment import (
        RandomVibrationCommands,
        RandomVibrationInstructions,
    )

    stream_nc4 = out_base + "_stream.nc4"
    if os.path.exists(stream_nc4):
        os.remove(stream_nc4)

    workbook = openpyxl.load_workbook(xlsx)
    hardware_metadata, environment_metadata_list, profile_event_list = (
        load_metadata_from_workbook(workbook)
    )
    env = environment_metadata_list[0]
    env_name = env.environment_name
    sysid_metadata = env.sysid_metadata

    last_level = 0.0
    for ev in profile_event_list:
        name = getattr(ev.command, "name", "")
        if isinstance(ev.data, (int, str)) and not isinstance(ev.data, bool):
            try:
                ev.data = float(ev.data)
            except (TypeError, ValueError):
                pass
        if name == "ADJUST_TEST_LEVEL" and isinstance(ev.data, float):
            last_level = ev.data
        elif name == "START_ENVIRONMENT":
            ev.data = RandomVibrationInstructions(ev.environment_name, last_level)
    log(
        f"loaded {os.path.basename(xlsx)}; env={env_name!r}; "
        f"{len(profile_event_list)} profile events; hardware "
        f"type {getattr(hardware_metadata.hardware_type, 'name', '?')}"
    )
    log(f"levels {levels}; settle = {settle_frames} frames of data a level; "
        f"caps {per_level_cap:g} s a level, {overall_cap:g} s overall")

    rat = RattlesnakeController(threaded=False, timeout=600)
    stop_watch = threading.Event()
    state = {"updates": 0, "frames": 0}
    threading.Thread(
        target=watch,
        args=(rat.queue_container.gui_update_queue, stop_watch, env_name, state),
        daemon=True,
    ).start()

    saved = []

    try:
        rat.initialize_hardware(hardware_metadata)
        log("hardware initialized")
        rat.initialize_environments(environment_metadata_list)
        log("environments initialized")
        rat.initialize_system_id(sysid_metadata, env_name)
        log("system id initialized")
        log("running system id (real-time)...")
        rat.run_system_id(sysid_metadata, env_name)
        log("system id done")

        stream = StreamMetadata(StreamType.PROFILE_INSTRUCTION, stream_nc4)
        rat.set_stream_metadata(stream)
        rat.start_acquisition(stream)
        log("acquisition started")
        rat.clear_blocking()
        t0 = time.time()
        rat.start_profile(profile_event_list)
        log("profile started (start-only events; levels stepped from here)")

        def wait_for(target_frames, label, cap):
            """Block until the controller's frame count reaches target_frames."""
            t_level = time.time()
            last_log = 0.0
            while state["frames"] < target_frames:
                if time.time() - t_level > cap:
                    log(f"  [{label}] CAP {cap:g} s reached at frame "
                        f"{state['frames']} of {target_frames} -- moving on")
                    return False
                if time.time() - t0 > overall_cap:
                    log(f"  [{label}] overall cap reached -- stopping")
                    return False
                if time.time() - last_log > 20:
                    rate = state["frames"] / max(time.time() - t0, 1e-9)
                    log(f"  [{label}] frame {state['frames']}/{target_frames}  "
                        f"updates={state['updates']}  {rate:.2f} frames/s wall  "
                        f"t={time.time()-t0:5.0f}s")
                    last_log = time.time()
                time.sleep(1.0)
            return True

        for i, level in enumerate(levels):
            if i > 0:
                rat.send_environment_command(
                    env_name, RandomVibrationCommands.ADJUST_TEST_LEVEL, float(level)
                )
                log(f"  [level] -> {level:+g} dB at frame {state['frames']}")
            target = state["frames"] + settle_frames
            ok = wait_for(target, f"{level:+g} dB", per_level_cap)
            path = f"{out_base}_spec{level_tag(level)}.nc4"
            try:
                if os.path.exists(path):
                    os.remove(path)
                rat.send_environment_command(
                    env_name, RandomVibrationCommands.SAVE_CONTROL_DATA, path
                )
                time.sleep(3.0)      # let the write land before the next level
                saved.append((level, path, state["frames"]))
                log(f"  [save] {level:+g} dB at frame {state['frames']} -> "
                    f"{os.path.basename(path)}")
            except Exception:
                log(f"  [save] FAILED at {level:+g} dB:\n" + traceback.format_exc())
            if not ok and time.time() - t0 > overall_cap:
                break
    except Exception:
        log("ERROR:\n" + traceback.format_exc())
    finally:
        log("shutting down...")
        stop_watch.set()
        try:
            rat.shutdown()
        except Exception:
            log("shutdown error:\n" + traceback.format_exc())
        log(f"saved {len(saved)} spectral files (frame count at each save):")
        for level, path, frame in saved:
            ok = os.path.exists(path)
            sz = os.path.getsize(path) / 1e3 if ok else 0
            log(f"   {level:+g} dB  frame {frame:5d}  {'OK ' if ok else 'MISSING'}  "
                f"{sz:8.1f} kB  {os.path.basename(path)}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=")[0]: a.split("=", 1)[-1] for a in sys.argv[1:] if a.startswith("--")}
    xlsx, out_base = args[0], args[1]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from prepare_bsoft_profiles import schedule

    _, levels, _ = schedule()
    if "--levels" in opts:
        levels = [float(v) for v in opts["--levels"].split(",")]
    #: 60 frames at frames_in_cpsd = 20 is three full CPSD refreshes a level
    settle = int(opts.get("--settle-frames", 60))
    per_level = float(opts.get("--level-cap", 900))
    overall = float(opts.get("--overall-cap", 5400))
    run(xlsx, out_base, levels, settle, per_level, overall)
