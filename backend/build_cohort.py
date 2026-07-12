#!/usr/bin/env python3
"""
Build an empirical cohort-percentile model from YOUR OWN library of swings.

Percentiles in Swing Optimizer Pro are normally *estimated* from published
research benchmarks. This tool replaces that with real, level-grouped percentiles
computed from the .mot/.trc files on your machine: a "72nd percentile" then means
"better than 72% of the <level> swings in your library".

Two steps:

  1) Generate a manifest template listing your swing files:

        python build_cohort.py init --dir ~/swings --out cohort_manifest.csv

     Then open cohort_manifest.csv and fill in, for each row:
        level        one of: youth | high_school | college | professional
        height_in    athlete height in inches   (or set height_m)
        weight_lb    athlete weight in pounds    (or set weight_kg)
     (bat_oz / bat_in are optional; sensible defaults are used.)

  2) Build the cohort model consumed by the analyzer:

        python build_cohort.py build --manifest cohort_manifest.csv \
               --out cohort_percentiles.json

Place cohort_percentiles.json next to analyzer.py (the default --out) and the
backend will use it automatically. Set COHORT_MODEL_PATH to override the location.
"""
import argparse
import csv
import glob
import json
import os
import sys

from analyzer import RefinedHittingOptimizer, SWINGAI_LABELS

MANIFEST_COLUMNS = [
    'mot_file', 'trc_file', 'level',
    'height_in', 'weight_lb', 'height_m', 'weight_kg',
    'bat_oz', 'bat_in', 'athlete', 'notes',
]
VALID_LEVELS = {'youth', 'high_school', 'college', 'professional'}


def _num(row, key):
    v = (row.get(key) or '').strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def cmd_init(args):
    mots = sorted(glob.glob(os.path.join(os.path.expanduser(args.dir), '**', '*.mot'), recursive=True))
    if not mots:
        print(f"No .mot files found under {args.dir}", file=sys.stderr)
        return 1
    rows = []
    for mot in mots:
        stem = os.path.splitext(mot)[0]
        trc = stem + '.trc'
        rows.append({
            'mot_file': mot,
            'trc_file': trc if os.path.exists(trc) else '',
            'level': '', 'height_in': '', 'weight_lb': '',
            'height_m': '', 'weight_kg': '', 'bat_oz': '', 'bat_in': '',
            'athlete': '', 'notes': '',
        })
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out} with {len(rows)} swing(s).")
    print("Now fill in the 'level', 'height_in' (or height_m), and 'weight_lb' (or weight_kg) columns,")
    print("then run:  python build_cohort.py build --manifest", args.out)
    return 0


def cmd_build(args):
    with open(args.manifest, newline='') as f:
        rows = list(csv.DictReader(f))

    # level -> dim_key -> [values]
    buckets = {}
    processed = 0
    skipped = []
    for i, row in enumerate(rows, start=2):  # 2 = first data row (after header)
        mot = (row.get('mot_file') or '').strip()
        level = (row.get('level') or '').strip().lower()
        if not mot:
            continue
        if level not in VALID_LEVELS:
            skipped.append((mot, f"level '{level}' is blank/invalid"))
            continue

        # Demographics: prefer metric, else convert imperial.
        h_m = _num(row, 'height_m') or (_num(row, 'height_in') and _num(row, 'height_in') * 0.0254)
        w_kg = _num(row, 'weight_kg') or (_num(row, 'weight_lb') and _num(row, 'weight_lb') * 0.453592)
        if not h_m or not w_kg:
            skipped.append((mot, "missing height/weight"))
            continue

        bat_kg = (_num(row, 'bat_oz') or 31.0) * 0.0283495
        bat_m = (_num(row, 'bat_in') or 34.0) * 0.0254
        trc = (row.get('trc_file') or '').strip()

        try:
            opt = RefinedHittingOptimizer(body_mass_kg=w_kg, body_height_m=h_m,
                                          skill_level=level, bat_mass_kg=bat_kg, bat_length_m=bat_m)
            kin = opt.load_mot_file(mot)
            if kin is None or len(kin) == 0:
                skipped.append((mot, "empty/invalid .mot"))
                continue
            trc_data = opt.load_trc_file(trc) if trc and os.path.exists(trc) else None
            diag = opt.comprehensive_diagnosis(kin, os.path.basename(mot), trc_data=trc_data)
            dims = diag.get('swingai_report', {}).get('dimensions', {})
            for key, d in dims.items():
                val = d.get('value')
                if isinstance(val, (int, float)):
                    buckets.setdefault(level, {}).setdefault(key, []).append(round(float(val), 3))
            processed += 1
        except Exception as e:
            skipped.append((mot, f"error: {e}"))

    # Assemble compact model: sorted value list + n per (level, dim).
    model = {'_meta': {'source': 'user_library', 'files_processed': processed}}
    for level, dims in buckets.items():
        model[level] = {}
        for key, vals in dims.items():
            model[level][key] = {'n': len(vals), 'values': sorted(vals)}

    with open(args.out, 'w') as f:
        json.dump(model, f, indent=1)

    print(f"Processed {processed} swing(s) -> {args.out}")
    for level in sorted(k for k in model if k != '_meta'):
        dims = model[level]
        sample_n = max((d['n'] for d in dims.values()), default=0)
        print(f"  {level:<14} n={sample_n:<4} ({len(dims)} dimensions)")
    if skipped:
        print(f"\nSkipped {len(skipped)} file(s):")
        for mot, why in skipped[:20]:
            print(f"  - {os.path.basename(mot)}: {why}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")
    return 0


def main():
    p = argparse.ArgumentParser(description="Build a level-grouped cohort-percentile model from your swing library.")
    sub = p.add_subparsers(dest='cmd', required=True)

    pi = sub.add_parser('init', help='Scan a folder and write a manifest template to fill in.')
    pi.add_argument('--dir', required=True, help='Folder containing your .mot/.trc files (searched recursively).')
    pi.add_argument('--out', default='cohort_manifest.csv', help='Manifest CSV to write.')
    pi.set_defaults(func=cmd_init)

    pb = sub.add_parser('build', help='Run every listed swing and write the cohort-percentile model.')
    pb.add_argument('--manifest', required=True, help='Filled-in manifest CSV.')
    pb.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'cohort_percentiles.json'),
                    help='Output JSON model (default: alongside analyzer.py).')
    pb.set_defaults(func=cmd_build)

    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == '__main__':
    main()
