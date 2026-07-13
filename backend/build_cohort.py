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
        height_cm    athlete height in centimetres  (imperial height_in also accepted)
        weight_kg    athlete weight in kilograms    (imperial weight_lb also accepted)
     (bat_oz / bat_in are optional; sensible defaults are used.)

  2) Build the cohort model consumed by the analyzer:

        python build_cohort.py build --manifest cohort_manifest.csv \
               --out cohort_percentiles.json

Place cohort_percentiles.json next to analyzer.py (the default --out) and the
backend will use it automatically. Set COHORT_MODEL_PATH to override the location.

Automation (no manual manifest): keep an athletes.json that maps each folder to
its athlete's level + demographics, then just run:

        python build_cohort.py auto            # reads athletes.json, rebuilds

Any new .mot/.trc dropped into a listed folder is picked up on the next run,
so this command is what a file-watcher (e.g. a macOS launchd agent) should call.
athletes.json format:

    {
      "cohort_out": "cohort_percentiles.json",
      "athletes": [
        {"dir": "~/Downloads/kike_swing_data_monocular", "athlete": "kike",
         "level": "high_school", "height_cm": 173, "weight_kg": 68}
      ]
    }
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
    'height_cm', 'weight_kg', 'height_in', 'weight_lb', 'height_m',
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
            'level': '', 'height_cm': '', 'weight_kg': '', 'height_in': '',
            'weight_lb': '', 'height_m': '', 'bat_oz': '', 'bat_in': '',
            'athlete': '', 'notes': '',
        })
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out} with {len(rows)} swing(s).")
    print("Now fill in the 'level', 'height_cm' (or height_in), and 'weight_kg' (or weight_lb) columns,")
    print("then run:  python build_cohort.py build --manifest", args.out)
    return 0


def _run_swing(buckets, mot, trc, level, h_m, w_kg, bat_kg, bat_m):
    """Analyze one swing and fold its dimension values into buckets[level][dim].
    Returns None on success or an error string to report as skipped."""
    try:
        opt = RefinedHittingOptimizer(body_mass_kg=w_kg, body_height_m=h_m,
                                      skill_level=level, bat_mass_kg=bat_kg, bat_length_m=bat_m)
        kin = opt.load_mot_file(mot)
        if kin is None or len(kin) == 0:
            return "empty/invalid .mot"
        trc_data = opt.load_trc_file(trc) if trc and os.path.exists(trc) else None
        diag = opt.comprehensive_diagnosis(kin, os.path.basename(mot), trc_data=trc_data)
        dims = diag.get('swingai_report', {}).get('dimensions', {})
        for key, d in dims.items():
            val = d.get('value')
            if isinstance(val, (int, float)):
                buckets.setdefault(level, {}).setdefault(key, []).append(round(float(val), 3))
        return None
    except Exception as e:
        return f"error: {e}"


def _write_model(buckets, processed, out, skipped):
    """Serialize buckets to the compact cohort model and print a summary."""
    model = {'_meta': {'source': 'user_library', 'files_processed': processed}}
    for level, dims in buckets.items():
        model[level] = {}
        for key, vals in dims.items():
            model[level][key] = {'n': len(vals), 'values': sorted(vals)}
    with open(out, 'w') as f:
        json.dump(model, f, indent=1)

    print(f"Processed {processed} swing(s) -> {out}")
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


def _pair_trc(root):
    """Index every .trc under root by lowercased filename stem (for cross-subfolder pairing)."""
    idx = {}
    for trc in glob.glob(os.path.join(root, '**', '*.trc'), recursive=True):
        idx.setdefault(os.path.splitext(os.path.basename(trc))[0].lower(), trc)
    return idx


def cmd_build(args):
    with open(args.manifest, newline='') as f:
        rows = list(csv.DictReader(f))

    buckets, processed, skipped = {}, 0, []
    for row in rows:
        mot = (row.get('mot_file') or '').strip()
        level = (row.get('level') or '').strip().lower()
        if not mot:
            continue
        if level not in VALID_LEVELS:
            skipped.append((mot, f"level '{level}' is blank/invalid"))
            continue
        # Demographics: metric preferred (cm/kg); imperial accepted as a fallback.
        h_m = (_num(row, 'height_cm') and _num(row, 'height_cm') / 100.0) \
            or _num(row, 'height_m') or (_num(row, 'height_in') and _num(row, 'height_in') * 0.0254)
        w_kg = _num(row, 'weight_kg') or (_num(row, 'weight_lb') and _num(row, 'weight_lb') * 0.453592)
        if not h_m or not w_kg:
            skipped.append((mot, "missing height/weight"))
            continue
        bat_kg = (_num(row, 'bat_oz') or 31.0) * 0.0283495
        bat_m = (_num(row, 'bat_in') or 34.0) * 0.0254
        err = _run_swing(buckets, mot, (row.get('trc_file') or '').strip(), level, h_m, w_kg, bat_kg, bat_m)
        if err:
            skipped.append((mot, err))
        else:
            processed += 1

    _write_model(buckets, processed, args.out, skipped)
    return 0


def _discover(cfg, config_path):
    """Scan cfg['discover_roots'] for athlete folders (any immediate subfolder
    containing .mot files) not already in the config, and register each as a
    stub entry that needs demographics. Returns the list of newly-added names.
    Stubs are held OUT of the cohort until level/height/weight are filled in."""
    roots = cfg.get('discover_roots') or []
    if not roots:
        return []
    known = set()
    for a in cfg.get('athletes', []):
        if a.get('dir'):
            known.add(os.path.realpath(os.path.expanduser(a['dir'])))
    added = []
    for root in roots:
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            child = os.path.join(root, name)
            if not os.path.isdir(child):
                continue
            rp = os.path.realpath(child)
            if rp in known:
                continue
            if not glob.glob(os.path.join(child, '**', '*.mot'), recursive=True):
                continue  # not a swing folder
            cfg.setdefault('athletes', []).append({
                'dir': child, 'athlete': name,
                'level': '', 'height_cm': None, 'weight_kg': None,
                'needs_demographics': True,
            })
            known.add(rp)
            added.append(name)
    if added:
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2)
    return added


def cmd_auto(args):
    """Rebuild the cohort with no manual manifest, driven by an athletes config
    that maps each folder to its athlete demographics. Ideal for automation:
    drop new .mot/.trc files into a known folder and re-run this command.

    If the config has a `discover_roots` list, new athlete folders under those
    roots are auto-registered (held out of the cohort until you add their
    level/height/weight)."""
    with open(args.config) as f:
        cfg = json.load(f)

    newly = _discover(cfg, args.config)
    if newly:
        print(f"Registered {len(newly)} new athlete folder(s): {', '.join(newly)}")
        print("  -> add each one's level + height/weight in the config to include them in the cohort.")

    out = args.out or cfg.get('cohort_out') or os.path.join(os.path.dirname(__file__), 'cohort_percentiles.json')
    if not os.path.isabs(out):  # resolve relative to the config file, not the CWD
        out = os.path.join(os.path.dirname(os.path.abspath(args.config)), out)
    dirs = [os.path.expanduser(a['dir']) for a in cfg.get('athletes', []) if a.get('dir')]

    # Cheap change-detection so a file-watcher can poll frequently without cost:
    # skip the rebuild unless some .mot/.trc (or the config) is newer than the model.
    if getattr(args, 'if_changed', False) and os.path.exists(out):
        newest = 0.0
        for d in dirs:
            for ext in ('mot', 'trc'):
                for f in glob.glob(os.path.join(d, '**', '*.' + ext), recursive=True):
                    try:
                        newest = max(newest, os.path.getmtime(f))
                    except OSError:
                        pass
        try:
            newest = max(newest, os.path.getmtime(args.config))
        except OSError:
            pass
        if newest <= os.path.getmtime(out):
            print("cohort up to date — nothing changed, skipping rebuild")
            return 0

    buckets, processed, skipped = {}, 0, []
    for a in cfg.get('athletes', []):
        d = os.path.expanduser(a['dir'])
        level = str(a.get('level', '')).strip().lower()
        if level not in VALID_LEVELS:
            skipped.append((d, f"level '{level}' invalid"))
            continue
        h_m = (a.get('height_cm') and float(a['height_cm']) / 100.0) \
            or a.get('height_m') or (a.get('height_in') and float(a['height_in']) * 0.0254)
        w_kg = a.get('weight_kg') or (a.get('weight_lb') and float(a['weight_lb']) * 0.453592)
        if not h_m or not w_kg:
            skipped.append((d, "missing height/weight"))
            continue
        bat_kg = float(a.get('bat_oz', 31.0)) * 0.0283495
        bat_m = float(a.get('bat_in', 34.0)) * 0.0254
        trc_idx = _pair_trc(d)
        for mot in sorted(glob.glob(os.path.join(d, '**', '*.mot'), recursive=True)):
            trc = trc_idx.get(os.path.splitext(os.path.basename(mot))[0].lower(), '')
            err = _run_swing(buckets, mot, trc, level, h_m, w_kg, bat_kg, bat_m)
            if err:
                skipped.append((mot, err))
            else:
                processed += 1

    _write_model(buckets, processed, out, skipped)
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

    pa = sub.add_parser('auto', help='Rebuild from an athletes config (no manual manifest). Good for automation.')
    pa.add_argument('--config', default=os.path.join(os.path.dirname(__file__), 'athletes.json'),
                    help='JSON mapping each athlete folder to level + demographics (default: athletes.json).')
    pa.add_argument('--out', default=None, help='Output JSON model (default: config cohort_out or cohort_percentiles.json).')
    pa.add_argument('--if-changed', dest='if_changed', action='store_true',
                    help='Skip the rebuild unless a .mot/.trc file (or the config) is newer than the existing model. For frequent polling by a watcher.')
    pa.set_defaults(func=cmd_auto)

    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == '__main__':
    main()
