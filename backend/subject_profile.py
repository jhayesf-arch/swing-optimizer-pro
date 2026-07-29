#!/usr/bin/env python3
"""
Subject profile — analyze EVERY swing for one athlete and aggregate them.

A single swing is one noisy sample. This module runs all of an athlete's swings
through the engine and produces:

  * one report per swing (so any individual trial can be inspected), and
  * an averaged view with the same shape as a single-swing report, so the
    frontend can render it through the existing code path.

Two things make the average trustworthy rather than naive:

  * **Outlier exclusion** — a single bad capture (dropped marker, mistracked
    frame) can drag a raw mean badly. Values are screened with a MAD-based
    modified z-score and excluded from the average; every exclusion is reported,
    and the swing itself stays selectable.
  * **Consistency** — the spread of each metric across swings is a coaching
    metric in its own right. A hitter whose separation ranges 25-50 degrees has
    a repeatability problem, not a range problem; the mean alone hides that.

Usage:

    python subject_profile.py --athlete jett
    python subject_profile.py --athlete jett --json profile.json
    python subject_profile.py --dir ~/Downloads/jett_swing_data \
        --level college --height-cm 193 --weight-kg 84
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from typing import Dict, List, Optional, Set

from analyzer import (
    DRILL_LIBRARY,
    RefinedHittingOptimizer,
    SWINGAI_LABELS,
    SWINGAI_PHASES,
    SWINGAI_WEIGHTS,
    load_cohort_model,
)

# Outlier screening only kicks in once there are enough swings for the spread to
# mean anything; below this, every swing is kept.
MIN_N_FOR_OUTLIERS = 5
OUTLIER_Z = 3.5  # modified z-score threshold (Iglewicz & Hoaglin)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _pair_trc(root: str) -> Dict[str, str]:
    """Index every .trc under root by lowercased filename stem, so .mot files in
    OpenSimData/Kinematics/ pair with markers in MarkerData/ (OpenCap layout)."""
    idx: Dict[str, str] = {}
    for trc in glob.glob(os.path.join(root, "**", "*.trc"), recursive=True):
        idx.setdefault(os.path.splitext(os.path.basename(trc))[0].lower(), trc)
    return idx


def _outlier_indices(values: List[float]) -> Set[int]:
    """MAD-based modified z-score. Robust for the small samples we deal with —
    unlike a mean/std rule, one extreme value can't hide itself by inflating
    the very spread used to detect it."""
    if len(values) < MIN_N_FOR_OUTLIERS:
        return set()
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values])
    if mad == 0:
        return set()
    return {i for i, v in enumerate(values) if abs(0.6745 * (v - med) / mad) > OUTLIER_Z}


def _round_smart(x: float) -> float:
    """Round to a sensible precision for the magnitude (stride ratios need more
    decimals than degrees)."""
    a = abs(x)
    if a < 1:
        return round(x, 3)
    if a < 10:
        return round(x, 2)
    return round(x, 1)


def _consistency(values: List[float]) -> Optional[float]:
    """0-100 repeatability score from the coefficient of variation.
    100 = identical every swing; lower = more scattered. None when undefined."""
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    if mean == 0:
        return None
    cv = statistics.pstdev(values) / abs(mean)
    return round(max(0.0, 100.0 * (1.0 - min(cv, 1.0))), 1)


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------
def analyze_subject(
    mot_dir: str,
    level: str,
    height_m: float,
    weight_kg: float,
    bat_kg: float = 0.88,
    bat_m: float = 0.864,
    athlete: Optional[str] = None,
) -> Dict:
    """Run every swing in mot_dir and return per-swing reports plus an averaged view."""
    mot_dir = os.path.expanduser(mot_dir)
    trc_idx = _pair_trc(mot_dir)
    mots = sorted(glob.glob(os.path.join(mot_dir, "**", "*.mot"), recursive=True))

    swings: List[Dict] = []
    errors: List[Dict] = []

    for mot in mots:
        stem = os.path.splitext(os.path.basename(mot))[0]
        trc = trc_idx.get(stem.lower())
        try:
            opt = RefinedHittingOptimizer(
                body_mass_kg=weight_kg, body_height_m=height_m,
                skill_level=level, bat_mass_kg=bat_kg, bat_length_m=bat_m,
            )
            kin = opt.load_mot_file(mot)
            if kin is None or len(kin) == 0:
                errors.append({"file": os.path.basename(mot), "error": "empty/invalid .mot"})
                continue
            trc_data = opt.load_trc_file(trc) if trc and os.path.exists(trc) else None
            diag = opt.comprehensive_diagnosis(kin, os.path.basename(mot), trc_data=trc_data)
            rep = diag.get("swingai_report", {})
            swings.append({
                "index": len(swings) + 1,
                "name": stem,
                "mot_file": mot,
                "has_markers": trc_data is not None,
                "swing_score": rep.get("swing_score", 0.0),
                "overall_percentile": rep.get("overall_percentile"),
                "percentile_basis": rep.get("percentile_basis"),
                "efficiency_score": diag.get("efficiency_score", 0),
                "dimensions": rep.get("dimensions", {}),
                "lead_leg_block": rep.get("lead_leg_block", {}),
                "prescriptions": rep.get("prescriptions", []),
                "metrics": diag.get("metrics", {}),
            })
        except Exception as e:  # one bad file shouldn't sink the whole profile
            errors.append({"file": os.path.basename(mot), "error": str(e)})

    if not swings:
        return {
            "athlete": athlete or os.path.basename(mot_dir.rstrip("/")),
            "level": level, "n_swings": 0, "swings": [], "errors": errors,
        }

    agg = aggregate_swings(swings, level, height_m, weight_kg, bat_kg, bat_m)
    return {
        "athlete": athlete or os.path.basename(mot_dir.rstrip("/")),
        "level": level,
        "n_swings": len(swings),
        "height_m": height_m,
        "weight_kg": weight_kg,
        "swings": swings,
        "errors": errors,
        **agg,
    }


def aggregate_swings(
    swings: List[Dict],
    level: str,
    height_m: float,
    weight_kg: float,
    bat_kg: float = 0.88,
    bat_m: float = 0.864,
) -> Dict:
    """Given per-swing reports, compute cross-swing statistics and an averaged
    view shaped like a single-swing report. Shared by the folder-based profile
    and the multi-file upload endpoint."""
    # ---- per-dimension statistics across swings ---------------------------
    def _measured(s, key):
        """A dimension counts only when the capture actually measured it."""
        d = s["dimensions"].get(key)
        return (d is not None and d.get("available") is not False
                and isinstance(d.get("value"), (int, float)))

    dim_keys = [k for k in SWINGAI_WEIGHTS if any(_measured(s, k) for s in swings)]
    unavailable_keys = [k for k in SWINGAI_WEIGHTS
                        if k not in dim_keys and any(k in s["dimensions"] for s in swings)]
    stats: Dict[str, Dict] = {}
    for key in dim_keys:
        idxs = [i for i, s in enumerate(swings) if _measured(s, key)]
        vals = [float(swings[i]["dimensions"][key]["value"]) for i in idxs]
        outliers = _outlier_indices(vals)
        kept = [v for i, v in enumerate(vals) if i not in outliers]
        if not kept:  # every value flagged (degenerate) — fall back to all
            kept, outliers = vals, set()
        stats[key] = {
            "mean": statistics.fmean(kept),
            "median": statistics.median(kept),
            "std": statistics.pstdev(kept) if len(kept) > 1 else 0.0,
            "min": min(kept),
            "max": max(kept),
            "n_used": len(kept),
            "n_excluded": len(outliers),
            "consistency": _consistency(kept),
            "values": [_round_smart(v) for v in vals],
            # swing numbers (1-based) whose value was excluded from the average
            "excluded_swings": sorted(swings[idxs[i]]["index"] for i in outliers),
        }

    # ---- averaged view, shaped exactly like a single-swing report ----------
    rater = RefinedHittingOptimizer(
        body_mass_kg=weight_kg, body_height_m=height_m, skill_level=level,
        bat_mass_kg=bat_kg, bat_length_m=bat_m,
    )
    avg_dims: Dict[str, Dict] = {}
    # Dimensions no swing could measure stay in the report, flagged, so the
    # athlete sees "not measured" rather than a silently missing row.
    for key in unavailable_keys:
        src = next((s["dimensions"][key] for s in swings if key in s["dimensions"]), {})
        avg_dims[key] = {
            "label": src.get("label", SWINGAI_LABELS.get(key, key)),
            "unit": src.get("unit", ""), "description": src.get("description", ""),
            "value": None, "stars": 0, "badge": "unavailable", "available": False,
            "unavailable_reason": src.get("unavailable_reason", "not measurable in these captures"),
        }
    for key in dim_keys:
        st = stats[key]
        excluded_pos = {i for i, s in enumerate(swings)
                        if _measured(s, key) and s["index"] in st["excluded_swings"]}
        members = [s["dimensions"][key] for i, s in enumerate(swings)
                   if _measured(s, key) and i not in excluded_pos]
        template = members[0]

        star_vals = [d.get("stars", 3) for d in members]
        pct_vals = [d["percentile"] for d in members if isinstance(d.get("percentile"), (int, float))]
        n_vals = [d["percentile_n"] for d in members if isinstance(d.get("percentile_n"), (int, float))]
        avg_stars = int(round(statistics.fmean(star_vals))) if star_vals else 3

        entry = {
            "label": template.get("label", SWINGAI_LABELS.get(key, key)),
            "unit": template.get("unit", ""),
            "description": template.get("description", ""),
            "value": _round_smart(st["mean"]),
            "stars": avg_stars,
            "badge": rater._rating_to_badge(avg_stars),
            # spread across swings — the coaching signal a single swing can't show
            "consistency": st["consistency"],
            "range": [_round_smart(st["min"]), _round_smart(st["max"])],
            "n_swings": st["n_used"],
        }
        if pct_vals:
            entry["percentile"] = int(round(statistics.fmean(pct_vals)))
        if n_vals:
            entry["percentile_n"] = int(round(statistics.fmean(n_vals)))
        avg_dims[key] = entry

    # Prescriptions for the average, ranked by bat-speed impact (same rule as
    # a single swing: dimension weight x star deficit).
    prescriptions = []
    for key, weight in SWINGAI_WEIGHTS.items():
        d = avg_dims.get(key)
        if not d or d.get("available") is False or d["stars"] >= 4:
            continue
        drill = DRILL_LIBRARY.get(key)
        if not drill:
            continue
        prescriptions.append({
            "key": key, "label": d["label"], "stars": d["stars"],
            "percentile": d.get("percentile", 50),
            "impact": round(weight * (5 - d["stars"]), 4),
            "cue": drill["cue"], "drill": drill["drill"], "why": drill["why"],
        })
    prescriptions.sort(key=lambda p: p["impact"], reverse=True)
    for i, p in enumerate(prescriptions):
        p["priority"] = i + 1

    phases = {}
    for phase_key, meta in SWINGAI_PHASES.items():
        pdims = [{**avg_dims[d], "key": d} for d in meta["dimensions"] if d in avg_dims]
        if not pdims:
            continue
        rated = [d for d in pdims if d.get("available") is not False]
        avg_stars = sum(d["stars"] for d in rated) / max(1, len(rated))
        phases[phase_key] = {
            "label": meta["label"], "icon": meta["icon"],
            "avg_stars": round(avg_stars, 1),
            "badge": rater._rating_to_badge(round(avg_stars)),
            "dimensions": pdims,
        }

    scores = [s["swing_score"] for s in swings]
    pcts = [s["overall_percentile"] for s in swings if isinstance(s.get("overall_percentile"), (int, float))]
    weighted_consistency = [
        (SWINGAI_WEIGHTS[k], stats[k]["consistency"])
        for k in dim_keys if stats[k]["consistency"] is not None
    ]
    overall_consistency = (
        round(sum(w * c for w, c in weighted_consistency) / sum(w for w, _ in weighted_consistency), 1)
        if weighted_consistency else None
    )

    # Averaged scalar metrics for the hero stats.
    avg_metrics: Dict[str, float] = {}
    for mkey in (swings[0].get("metrics") or {}):
        mvals = [s["metrics"][mkey] for s in swings
                 if isinstance(s.get("metrics", {}).get(mkey), (int, float))
                 and not isinstance(s["metrics"][mkey], bool)]
        if mvals:
            avg_metrics[mkey] = _round_smart(statistics.fmean(mvals))

    average_report = {
        "swing_score": round(statistics.fmean(scores), 1),
        "overall_percentile": int(round(statistics.fmean(pcts))) if pcts else None,
        "percentile_basis": swings[0].get("percentile_basis"),
        "skill_level": level,
        "phases": phases,
        "dimensions": avg_dims,
        "prescriptions": prescriptions,
        "lead_leg_block": swings[0].get("lead_leg_block", {}),
        # aggregate-only fields
        "is_average": True,
        "n_swings": len(swings),
        "overall_consistency": overall_consistency,
        "swing_score_range": [round(min(scores), 1), round(max(scores), 1)],
    }

    return {
        "average": average_report,
        "average_metrics": avg_metrics,
        "stats": stats,
    }


def analyze_from_config(config_path: str, athlete_name: str) -> Dict:
    """Look an athlete up in athletes.json and profile them."""
    with open(config_path) as f:
        cfg = json.load(f)
    for a in cfg.get("athletes", []):
        if str(a.get("athlete", "")).lower() != athlete_name.lower():
            continue
        h_m = (a.get("height_cm") and float(a["height_cm"]) / 100.0) \
            or a.get("height_m") or (a.get("height_in") and float(a["height_in"]) * 0.0254)
        w_kg = a.get("weight_kg") or (a.get("weight_lb") and float(a["weight_lb"]) * 0.453592)
        level = str(a.get("level", "")).strip().lower()
        if not h_m or not w_kg or not level:
            raise SystemExit(
                f"Athlete '{athlete_name}' is missing level/height/weight in {config_path}."
            )
        return analyze_subject(
            a["dir"], level, h_m, w_kg,
            bat_kg=float(a.get("bat_oz", 31.0)) * 0.0283495,
            bat_m=float(a.get("bat_in", 34.0)) * 0.0254,
            athlete=a.get("athlete"),
        )
    known = ", ".join(str(a.get("athlete")) for a in cfg.get("athletes", []))
    raise SystemExit(f"Athlete '{athlete_name}' not found in {config_path}. Known: {known}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_summary(p: Dict) -> None:
    if not p["n_swings"]:
        print(f"No swings analyzed for {p['athlete']}.")
        for e in p.get("errors", []):
            print(f"  - {e['file']}: {e['error']}")
        return

    avg = p["average"]
    print(f"\n{p['athlete']}  ({p['level']})  —  {p['n_swings']} swings")
    print(f"  Swing Score : {avg['swing_score']}  (range {avg['swing_score_range'][0]}–{avg['swing_score_range'][1]})")
    if avg.get("overall_percentile") is not None:
        print(f"  Percentile  : {avg['overall_percentile']}th  [{avg.get('percentile_basis')}]")
    if avg.get("overall_consistency") is not None:
        print(f"  Consistency : {avg['overall_consistency']}/100  (100 = identical every swing)")

    print(f"\n  {'dimension':<32} {'avg':>9} {'range':>17} {'consist':>8} {'★':>6}")
    print("  " + "-" * 78)
    for key, d in avg["dimensions"].items():
        if d.get("available") is False:
            print(f"  {d['label'][:32]:<32} {'not measured':>9} {'—':>17} {'—':>8} {'—':>6}")
            continue
        rng = f"{d['range'][0]} – {d['range'][1]}"
        con = "—" if d.get("consistency") is None else f"{d['consistency']:.0f}"
        print(f"  {d['label'][:32]:<32} {d['value']:>9} {rng:>17} {con:>8} {'★' * d['stars']:>6}")

    missing = [d for d in avg["dimensions"].values() if d.get("available") is False]
    if missing:
        print("\n  Not measured in these captures:")
        for d in missing:
            print(f"    - {d['label']}: {d.get('unavailable_reason', 'unavailable')}")

    flagged = {k: s for k, s in p["stats"].items() if s["excluded_swings"]}
    if flagged:
        print("\n  Excluded from the average as outliers:")
        for k, s in flagged.items():
            print(f"    - {SWINGAI_LABELS.get(k, k)}: swing(s) {s['excluded_swings']}")

    least = sorted(
        [(k, d) for k, d in avg["dimensions"].items() if d.get("consistency") is not None],
        key=lambda kv: kv[1]["consistency"],
    )[:3]
    if least:
        print("\n  Least repeatable (biggest swing-to-swing scatter):")
        for k, d in least:
            print(f"    - {d['label']}: {d['consistency']:.0f}/100  (range {d['range'][0]}–{d['range'][1]} {d['unit']})")

    if p.get("errors"):
        print(f"\n  {len(p['errors'])} file(s) failed:")
        for e in p["errors"][:5]:
            print(f"    - {e['file']}: {e['error']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze every swing for one athlete and aggregate.")
    ap.add_argument("--athlete", help="Athlete name from athletes.json")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "athletes.json"))
    ap.add_argument("--dir", help="Swing folder (instead of --athlete)")
    ap.add_argument("--level", default="high_school", help="youth | high_school | college | professional")
    ap.add_argument("--height-cm", type=float)
    ap.add_argument("--weight-kg", type=float)
    ap.add_argument("--bat-oz", type=float, default=31.0)
    ap.add_argument("--bat-in", type=float, default=34.0)
    ap.add_argument("--json", help="Write the full profile to this JSON file")
    args = ap.parse_args()

    load_cohort_model()  # blend percentiles against the user's library when present

    if args.athlete:
        profile = analyze_from_config(args.config, args.athlete)
    elif args.dir:
        if not args.height_cm or not args.weight_kg:
            raise SystemExit("--dir requires --height-cm and --weight-kg")
        profile = analyze_subject(
            args.dir, args.level.lower(), args.height_cm / 100.0, args.weight_kg,
            bat_kg=args.bat_oz * 0.0283495, bat_m=args.bat_in * 0.0254,
        )
    else:
        raise SystemExit("Pass --athlete <name> or --dir <folder>")

    _print_summary(profile)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(profile, f, indent=1)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
