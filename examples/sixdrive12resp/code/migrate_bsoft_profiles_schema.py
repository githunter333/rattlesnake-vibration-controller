"""Relay the bounded-softening profiles' environment sheet onto the current schema.

These profiles were saved against the environment worksheet layout on branch
control-law-development-2026-09.  The current layout inserted six rows, so
loading one unchanged fails (`percent_lines_out = float(cell(11,2))` reads
'N', the old "Allow Automatic Aborts").  Everything below each insertion is
off by one or more rows, which would otherwise be read silently as the wrong
field.

This maps OLD ROW LABEL -> NEW ROW, so nothing depends on either layout's
numbering.  Only the environment sheet is rewritten; Channel Table, Hardware
and Test Profile are untouched.

SIX FIELDS ARE NEW and have no counterpart in the old profile.  Their values
are a choice, not a recovery, and they are recorded here because they affect
the run:

  Percent Lines Out            10     abort/warning threshold; the old schema
                                      had no such field
  Sigma Clipping                5     effectively no clipping, matching a
                                      schema that could not clip at all.  A
                                      lower value would impose clipping the
                                      historical runs did not have.
  System ID Samples per Frame  4096   the old schema had one frame size for
                                      both control and system ID; this is it
  System ID Low Bandwidth         0   full band
  System ID High Bandwidth     2048   sample_rate / 2
  System ID Ramp Time           0.5   short ramp on the system-ID signal; the
                                      old sheet carried Ramp Fraction % = 0
                                      and no ramp time at all

    python migrate_bsoft_profiles_schema.py [--write]
"""

import argparse
import os
import shutil

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES = os.path.normpath(os.path.join(HERE, "..", "results", "case", "profiles"))

STEMS = [
    "profile_42_optdiag_bsoft_trimoff",
    "profile_43_optdiag_bsoft_trimon",
    "profile_44_optdiagfast_bsoft_trimoff",
    "profile_45_optdiagfast_bsoft_trimon",
]

#: the current layout, in order: (label, source) where source is either
#: ("old", <old label>) to carry a value across, or ("new", value)
CURRENT = [
    ("Control Type", ("old", "Control Type")),
    ("Samples Per Frame:", ("old", "Samples Per Frame:")),
    ("Test Level Ramp Time:", ("old", "Test Level Ramp Time:")),
    ("COLA Window:", ("old", "COLA Window:")),
    ("COLA Overlap %:", ("old", "COLA Overlap %:")),
    ("COLA Window Exponent:", ("old", "COLA Window Exponent:")),
    ("Update System ID During Control:", ("old", "Update System ID During Control:")),
    ("Frames in CPSD:", ("old", "Frames in CPSD:")),
    ("CPSD Window:", ("old", "CPSD Window:")),
    ("CPSD Overlap %:", ("old", "CPSD Overlap %:")),
    ("Percent Lines Out", ("new", 10)),
    ("Allow Automatic Aborts", ("old", "Allow Automatic Aborts")),
    ("Control Python Script:", ("old", "Control Python Script:")),
    ("Control Python Function:", ("old", "Control Python Function:")),
    ("Control Parameters:", ("old", "Control Parameters:")),
    ("Control Channels (1-based):", ("old", "Control Channels (1-based):")),
    ("Sigma Clipping", ("new", 5)),
    ("System ID Samples per Frame", ("new", 4096)),
    ("System ID Averaging:", ("old", "System ID Averaging:")),
    ("Noise Averages:", ("old", "Noise Averages:")),
    ("System ID Averages:", ("old", "System ID Averages:")),
    ("Exponential Averaging Coefficient:", ("old", "Exponential Averaging Coefficient:")),
    ("System ID Estimator:", ("old", "System ID Estimator:")),
    ("System ID Level (V RMS):", ("old", "System ID Level (V RMS):")),
    ("System ID Low Bandwidth", ("new", 0)),
    ("System ID High Bandwidth", ("new", 2048)),
    ("System ID Ramp Time", ("new", 0.5)),
    ("System ID Signal Type:", ("old", "System ID Signal Type:")),
    ("System ID Window:", ("old", "System ID Window:")),
    ("System ID Overlap %:", ("old", "System ID Overlap %:")),
    ("System ID Burst On %:", ("old", "System ID Burst On %:")),
    ("System ID Burst Pretrigger %:", ("old", "System ID Burst Pretrigger %:")),
    ("System ID Ramp Fraction %:", ("old", "System ID Ramp Fraction %:")),
    ("Specification File:", ("old", "Specification File:")),
    ("Response Transformation Matrix:", ("old", "Response Transformation Matrix:")),
    ("Output Transformation Matrix:", ("old", "Output Transformation Matrix:")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    for stem in STEMS:
        path = os.path.join(PROFILES, stem + ".xlsx")
        book = openpyxl.load_workbook(path)
        sheet_name = next(s for s in book.sheetnames if "Random" in s)
        sheet = book[sheet_name]

        old = {}
        for i in range(1, sheet.max_row + 1):
            label = sheet.cell(i, 1).value
            if label is not None:
                old[str(label).strip()] = sheet.cell(i, 2).value

        if "Percent Lines Out" in old:
            print(f"  {stem}: already migrated, skipped")
            continue

        rows, missing = [], []
        for label, (kind, src) in CURRENT:
            if kind == "old":
                key = str(src).strip()
                if key not in old:
                    missing.append(key)
                    rows.append((label, None))
                else:
                    rows.append((label, old[key]))
            else:
                rows.append((label, src))

        print(f"  {stem}: {len(old)} old rows -> {len(rows)} current rows"
              + (f"   MISSING: {missing}" if missing else ""))
        if missing:
            print("      refusing to write: an old label was not found")
            continue
        if not a.write:
            continue

        shutil.copy(path, path + ".preschema.bak")
        for i in range(sheet.max_row, 0, -1):
            for c in (1, 2):
                sheet.cell(i, c, None)
        for i, (label, value) in enumerate(rows, start=1):
            sheet.cell(i, 1, label)
            sheet.cell(i, 2, value)
        book.save(path)
        print("      written (old kept as .preschema.bak)")

    if not a.write:
        print("\n(report only -- pass --write to apply)")


if __name__ == "__main__":
    main()
