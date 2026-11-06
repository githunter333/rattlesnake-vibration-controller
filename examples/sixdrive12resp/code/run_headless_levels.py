"""Run a sixdrive12resp profile unattended, saving spectral data at each level.

The bounded-softening runs were GUI-driven: an operator stepped the test level
and pressed "Save Spectral Data" at each one, which is why make_moving_plant_
figures.py scores one file per level (``<base>_spec<level>.nc4``, reading
``response_cpsd_real`` / ``drive_cpsd_real``) rather than a streaming file.

This drives the same sequence with no GUI.  The level schedule comes from the
profile's Test Profile sheet (prepare_bsoft_profiles.py writes it); the only
thing done here that a profile event cannot express is the per-level save, which
goes through the same controller call the GUI button uses:

    send_environment_command(env, RandomVibrationCommands.SAVE_CONTROL_DATA, path)

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


def drain(q, stop):
    while not stop.is_set():
        try:
            q.get(timeout=0.25)
        except queue.Empty:
            pass


def level_tag(level):
    """Match the historical file naming: -18 -> 'm18', 0 -> '0'."""
    lv = int(round(float(level)))
    return f"m{abs(lv)}" if lv < 0 else f"{lv}"


def run(xlsx, out_base, wait_seconds, save_plan):
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
    log("save plan: " + ", ".join(f"{t:g}s -> {lv:+g} dB" for t, lv in save_plan))

    rat = RattlesnakeController(threaded=False, timeout=600)
    stop_drain = threading.Event()
    threading.Thread(
        target=drain, args=(rat.queue_container.gui_update_queue, stop_drain), daemon=True
    ).start()

    saved = []

    def saver(t0):
        """Issue SAVE_CONTROL_DATA at each planned time, as the GUI button does."""
        for t_at, level in sorted(save_plan):
            delay = t_at - (time.time() - t0)
            if delay > 0:
                time.sleep(delay)
            path = f"{out_base}_spec{level_tag(level)}.nc4"
            try:
                if os.path.exists(path):
                    os.remove(path)
                rat.send_environment_command(
                    env_name, RandomVibrationCommands.SAVE_CONTROL_DATA, path
                )
                saved.append((level, path))
                log(f"  [save] t={time.time()-t0:5.0f}s  {level:+g} dB -> "
                    f"{os.path.basename(path)}")
            except Exception:
                log(f"  [save] FAILED at {level:+g} dB:\n" + traceback.format_exc())

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
        log("profile started")
        threading.Thread(target=saver, args=(t0,), daemon=True).start()

        while time.time() - t0 < wait_seconds:
            time.sleep(10)
            sz = os.path.getsize(stream_nc4) / 1e6 if os.path.exists(stream_nc4) else 0
            log(f"  t={time.time()-t0:5.0f}s  stream={sz:7.1f} MB  saved={len(saved)}")
    except Exception:
        log("ERROR:\n" + traceback.format_exc())
    finally:
        log("shutting down...")
        stop_drain.set()
        try:
            rat.shutdown()
        except Exception:
            log("shutdown error:\n" + traceback.format_exc())
        log(f"saved {len(saved)} spectral files:")
        for level, path in saved:
            ok = os.path.exists(path)
            sz = os.path.getsize(path) / 1e3 if ok else 0
            log(f"   {level:+g} dB  {'OK ' if ok else 'MISSING'}  {sz:8.1f} kB  {path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=")[0]: a.split("=", 1)[-1] for a in sys.argv[1:] if a.startswith("--")}
    xlsx = args[0]
    out_base = args[1]
    wait = float(args[2]) if len(args) > 2 else 330.0
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from prepare_bsoft_profiles import schedule

    _, saves, _ = schedule()
    if "--saves" in opts:
        saves = [
            (float(p.split(":")[0]), float(p.split(":")[1]))
            for p in opts["--saves"].split(",")
        ]
    run(xlsx, out_base, wait, saves)
