"""Make the bounded-softening profiles loadable and runnable unattended.

The sixdrive12resp bounded-softening runs (49-52, 55, 56) were driven from the
GUI: an operator stepped the test level and pressed "Save Spectral Data" at each
one.  That is why profiles 42-45 carry an EMPTY Test Profile sheet, and why runs
53 and 54 -- the fast law's pair on the moving plant -- were never done.

Two things have to change before they run unattended:

  hardware type   The profiles name type 7, which was the SDynPy nonlinear
                  integrator on branch control-law-development-2026-09.  In the
                  current enum 7 is SDYNPY_FRF and the nonlinear integrator is
                  9 (restored in sdynpy_nonlinear_system_virtual_hardware.py).
                  Loading one of these profiles unchanged would silently select
                  the WRONG hardware, so this is not cosmetic.

  test profile    A level schedule, so the run steps itself.

Nothing else in the profile is touched: the law, its parameters, the spec file
and all spectral settings stay exactly as they were saved.

    python prepare_bsoft_profiles.py            # report only
    python prepare_bsoft_profiles.py --write    # apply
"""

import argparse
import os

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES = os.path.normpath(os.path.join(HERE, "..", "results", "case", "profiles"))

#: profile stem -> the run number it was prepared for
TARGETS = {
    "profile_42_optdiag_bsoft_trimoff": 51,
    "profile_43_optdiag_bsoft_trimon": 52,
    "profile_44_optdiagfast_bsoft_trimoff": 53,
    "profile_45_optdiagfast_bsoft_trimon": 54,
}

OLD_NONLINEAR_TYPE = 7      # what these files were written against
NEW_NONLINEAR_TYPE = 9      # HardwareType.SDYNPY_NONLINEAR here

#: the levels the historical optimal_diagonal moving-plant run covered
LEVELS = (-18, -12, -6, -3, 0)
STEP = 45.0                 # seconds per level before full level
FULL_HOLD = 90.0            # seconds at 0 dB
START = 1.0                 # Start Environment


def schedule():
    """(events, save_times) -- events for the Test Profile sheet, and the time at
    which each level's spectral data should be saved (end of its dwell)."""
    events = [("0.0", "ENV", "Adjust Test Level", str(LEVELS[0]))]
    saves = []
    t = START
    events.append((f"{t:g}", "ENV", "Start Environment", None))
    events.append((f"{t:g}", "Global", "Start Streaming", None))
    for i, level in enumerate(LEVELS):
        dwell = FULL_HOLD if i == len(LEVELS) - 1 else STEP
        saves.append((round(t + dwell - 2.0, 3), level))
        t += dwell
        if i < len(LEVELS) - 1:
            events.append((f"{t:g}", "ENV", "Adjust Test Level", str(LEVELS[i + 1])))
    events.append((f"{t:g}", "Global", "Stop Streaming", None))
    events.append((f"{t + 4:g}", "Global", "Stop Hardware", None))
    return events, saves, t + 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="apply the changes")
    a = ap.parse_args()

    events, saves, end = schedule()
    print(f"schedule: levels {list(LEVELS)}, {STEP:g} s a step, {FULL_HOLD:g} s at full "
          f"level, ends {end:g} s")
    print("  saves at: " + ", ".join(f"{t:g}s ({lv:+d} dB)" for t, lv in saves))
    print()

    for stem, run in sorted(TARGETS.items(), key=lambda kv: kv[1]):
        path = os.path.join(PROFILES, stem + ".xlsx")
        if not os.path.exists(path):
            print(f"  run {run}: MISSING {stem}.xlsx")
            continue
        book = openpyxl.load_workbook(path)
        hardware = book["Hardware"]
        env_sheet = next(s for s in book.sheetnames if "Random" in s)
        hw_type = hardware.cell(1, 2).value
        n_events = book["Test Profile"].max_row - 1

        note = []
        if hw_type == OLD_NONLINEAR_TYPE:
            note.append(f"hardware {OLD_NONLINEAR_TYPE} -> {NEW_NONLINEAR_TYPE}")
        elif hw_type == NEW_NONLINEAR_TYPE:
            note.append("hardware already 9")
        else:
            note.append(f"hardware is {hw_type}, NOT the nonlinear type -- skipped")
        note.append(f"{n_events} events -> {len(events)}")
        print(f"  run {run}: {stem}")
        print(f"      {'; '.join(note)}  (env sheet {env_sheet!r})")

        if not a.write or hw_type not in (OLD_NONLINEAR_TYPE, NEW_NONLINEAR_TYPE):
            continue

        hardware.cell(1, 2, NEW_NONLINEAR_TYPE)
        sheet = book["Test Profile"]
        for row in range(sheet.max_row, 1, -1):
            for col in range(1, 5):
                sheet.cell(row, col, None)
        for i, (t, env, op, data) in enumerate(events, start=2):
            sheet.cell(i, 1, t)
            sheet.cell(i, 2, env_sheet if env == "ENV" else "Global")
            sheet.cell(i, 3, op)
            sheet.cell(i, 4, data)
        book.save(path)
        print("      written")

    if not a.write:
        print("\n(report only -- pass --write to apply)")


if __name__ == "__main__":
    main()
