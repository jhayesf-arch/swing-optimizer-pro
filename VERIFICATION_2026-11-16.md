# Calculation verification, 2026-11-16

Independent audit of every calculation in `backend/`: kinematics, ground force,
inverse dynamics, muscle estimation, scoring. Method: read each calculation,
state the correct formula, run a synthetic input with a known answer (or real
capture 1001 where a known answer exists), compare. Test scripts live outside
the repo. No repo code was changed by the audit.

Verdicts: CORRECT (verified numerically), BUG (wrong, with evidence),
QUESTIONABLE (defensible but misleading or fragile), UNVERIFIED (no source to
check against).

## 1. Verified correct

| Calculation | Evidence |
|---|---|
| Pelvis angular velocity (unwrap → Butterworth 15 Hz → Savitzky-Golay derivative) | A·sin(ωt): 188.46 vs 188.50 deg/s at 60/120/240 Hz |
| Sequence timing (sub-frame parabola, ms = frames·dt·1000) | 16.5 ms measured vs 17 expected; sign positive = pelvis first |
| Hand speed from .trc (15 Hz Butterworth, np.gradient, ×2.23694) | 10.00 m/s → 10.014 m/s, 22.40 mph |
| Max separation, x-factor stretch, `_surges` instability detector | synthetic profiles reproduce expected values |
| `_lead_foot_touchdown`, stride plant from pelvis ω sign change | plant frame 77 = expected; t = 1.15 for load ending 1.20 |
| `_crop_to_swing`, `_swing_window`, `_hand_peak_time` | swing kept with a 50 cm one-frame glitch; peak 5.283 for true 5.30 |
| IK trunk-flip detection (2500 deg/s) | flagged at the injected flips |
| Ground force F = m·a + m·g (`grf_estimation.py`) | static body: 794.610 N vertical, 0 horizontal; a = +2 m/s² → 956.59 vs 956.61 N |
| de Leva mass fractions | sum 1.000, every value matches Table 4 |
| Bottom-up Newton-Euler leg chain | three hand statics cases match to 0.01 N·m |
| OpenSim ID pelvis force axes | translation axes 1 0 0 / 0 1 0 / 0 0 1; rotated static poses all give (0, 794.3, 0) N |
| No-load ID on quiet standing | pelvis_ty_force +794.0 N (0.999 BW) |
| Model-consistent loads | force residual 0.01 N (raw estimate: 289 N) |
| External loads XML | calcn_r/l, ground frame, identifiers match; model ground = .trc ground (heel within 2 cm) |
| Fast muscle solver vs true QP (SLSQP) | max activation diff 9e-4; constraint violation 1.7e-4 N·m |
| Moment arm sign convention | equals −dL/dq (FD, 3e-4 m); hip flexors +, extensors −; OpenSim SO with physiology off gives the same activations (soleus 0.607 vs 0.602) |
| `_ramp`, `_rate_dimension`, both orientations | verified at, between and beyond every threshold, 64 tables |
| Swing Score weights, coverage, evidence keys | sum 1.0; 15/15 keys resolve; plausibility limits |
| Cohort/percentile data provenance | user's own .mot/.trc only; no Driveline/OBP data |

Also settled: the earlier disagreement between the fast solver and OpenSim's
StaticOptimization on calf/shin muscles. OpenSim's Ipopt hit `max_iterations`
(200) on many frames and stopped at non-optimal points (higher cost at every
frame, Σa² 20.4 vs 18.2). The fast solver is right.

## 2. Bugs

### Change the Swing Score users see

**Status (2026-11-17): B1–B5 fixed on branch `claude/identify-swing-optimizer-repo-V0Ixe`; regression tests in `backend/test_phase_report.py`.**

**B1. Torso-to-pelvis ratio uses the relative lumbar rate.**
`analyzer.py:1501` `peak_shoulder_w = max|lumbar_omega|` (twist relative to
pelvis); `:1540` ratio = that / pelvis. Benchmarks (Escamilla 1.1–1.6; penalty
below 0.8; swing-score evidence key weight 0.12) are for the absolute torso
rate. Synthetic: pelvis 314, lumbar 157, absolute torso 463 deg/s → code 0.50,
correct 1.47. `peak_shoulder_omega_rad_s` and `peak_torso_omega_3d_deg_s` have
the same error. Fix: use `thorax_abs_omega` (already computed at `:1401`).

**B2. "Pelvis Direction at Contact" is measured at foot plant.**
`analyzer.py:3121` indexes `pelvis_angle[stride['plant_frame']]`, 100–200 ms
before contact, when hips are still closed (Taguchi: −28 to −33° at plant).
Also absolute capture-frame yaw, so it depends on camera placement; and when
stride is missing the code substitutes 45° (`:3126`) and still scores it.
`upper_torso_direction_at_contact` (`:3139`) is pelvis_dev / ratio, not a
measurement. Combined weight 0.11. Fix: use `rotation['contact_frame']` and
`pelvis_rotation_at_contact_deg`; mark both unavailable without stride; set
upper-torso direction to weight 0 until a thorax angle at contact exists.

**B3. Missing knee data scores as a zero lead-leg block.**
`analyzer.py:3154` `llb_val = llb.get(..., 0.0)` with no `available` flag;
`swing_score.py:112` and `mechanics_score.py:181` skip only on
`available is False`. A good college swing drops 98.8 → 88.8 and a front-leg
drill is prescribed for data that doesn't exist. Fix:
`'available': bool(llb)` plus `unavailable_reason`.

**B4. Lumbar joint-limit repair bypassed.**
`:1171–1198` repairs lumbar_rotation, but `_om1d` (`:1389`, used at `:1401`)
and `_omega_1d` (`:1577`) re-read the raw column. Sequence timing, torso
speed, `torso_peak_at_edge`, instability flag and the chart all see the
unrepaired spike (synthetic −90→+85 flip: 807 vs 2975 deg/s). Fix: route
`lumbar_rotation` through the repaired array in both helpers.

**B5. Percentiles inflated ~40 points.**
`analyzer.py:268` `PERCENTILE_ANCHORS = [5,25,50,75,92]` map the 5-star
threshold to the 92nd percentile, but for pelvis_speed, torso_speed,
stride_length, lead_leg_block and torso_coil the 5-star threshold *is* the
published level mean. A college hitter at 678 deg/s is told "92nd percentile";
1 SD below (truly ~16th) is told 50th. `overall_percentile` and coach text
inherit it. Fix: normal CDF from the published mean/SD for mean-anchored
dimensions, or re-anchor those tables so 3 stars = mean.

### Other

**B6. Bat inertia added without the parallel-axis shift.**
`opensim_id.py:401–429` adds m_bat to hand_r mass and m·L²/12 to Ixx/Izz but
leaves `mass_center` unchanged. The bat's centre is ~L/2 ≈ 0.43 m beyond the
hand, so the correct combined inertia about the new centre is ≈ 0.22 kg·m²,
not 0.055. Bat load on elbow/shoulder understated ~4×; whole bat on hand_r
(elbow_flex_r 229 vs _l 54 N·m on 1001). Fix: compute combined mass centre
and inertia; split 50/50 between hands or document the one-hand assumption.

**B7. TRC and MOT frame indices mixed.** `analyzer.py:1861–1864,1896` index
the .trc with a .mot frame index and .mot sample rate. 120 Hz trc + 60 Hz
mot → stride 0.000 m, touchdown None. Fix: map by time.

**B8. TRC `Units` header ignored.** `load_trc_file` and `grf_estimation.py`
assume metres; a millimetre file gives 22,401 mph hand speed and 163 kN GRF.
`load_trc_file` also returns None if line 1 contains "time". Fix: parse the
Units line, scale by 0.001 for mm; search the header from line 2.

**B9. Static optimization: non-converged frames kept; withheld result leaks.**
`static_optimization.py:137` keeps frames where Ipopt reported "could not find
a solution" (≥20 of 73 on 1001) with no flag. `muscle_forces.py:181–184`
writes `seconds`, `residual_moment_check`, `reserves` before the withheld
return, contradicting the docstring. Fix: raise `optimizer_max_iterations`
to ≥1000 and parse the log; move the writes after the gate.

**B10 (low impact).** `grf_estimation.py:56` a missing marker becomes (0,0,0)
silently (CoM 0.996 → 0.566 m); `_calculate_trc_sequence_timing:2279` has the
sign of `proper_sequence` inverted (unused).

## 3. Questionable

- **Contact frame has four definitions**: .trc deceleration (rotation
  metrics), peak pelvis ω (stride, lead-leg block, skeleton window),
  pelvis-peak-based `time_to_contact_s`, and plant (B2). Lead-leg block
  measured to peak pelvis ω reads 15° where contact gives 30°. Compute once,
  pass `rotation['contact_frame']` everywhere.
- **Lead-leg block window** ends at peak pelvis ω; Escamilla's 59°/32° runs to
  bat contact, so skilled hitters under-rate.
- **`upper_torso_rotation_range`** = pelvis range + separation (cap 120), an
  arithmetic composite with weight 0.025 and a research tier it doesn't earn.
  Recommend weight 0.
- **Torso speed** is a 3-D magnitude (axial + bending) compared against
  Escamilla's axial value; skews high.
- **3-D pelvis ω** sums squared Euler rates; true magnitude differs by up to
  −3.5% at 10° list. Small.
- **Torso coil at plant** uses `abs()`: shoulders *open* 14° at plant scores
  5 stars.
- **Stride description text** says "elite ~75–90% of height" beside a table
  giving 5 stars at 47–48%.
- **Hand speed**: wrist-marker speed vs Blast's handle sensor; two conflicting
  Blast ranges hard-coded. Unverified.
- **Thresholds without a level-stratified source**: pro torso_speed 897
  (midpoint of two studies), pelvis/torso rotation-range tables (Fortenbaugh
  2011 is a GRF paper), direction-at-contact tables, youth/HS separation
  (scaled from a pro-only study).
- **Mechanics pelvis-drive** scores on timing alone when speed is implausible
  and still displays the implausible value; `sep == 0.0` is skipped but 0.1
  scores 0.
- **Per-foot weight split** (`bottom_up_id.py:157`): sigmoid on ankle-marker
  height; a 4 cm marker offset on 1001 gives 28–41% left while standing on
  both feet. This split is reused by the model-consistent loads and is the
  main reason the pelvis residual *moment* (128 → 340 N·m standing, 2959 on
  the swing) stays far above the Hicks limit and muscle loads are withheld.
  Add a dead-band or use heel/toe-to-floor distance. Report the moment
  residual in `opensim_id_status`.
- **`check_required_grf` vertical_min −0.25 BW** lets the ground "pull" 0.2 BW.
  Suggest −0.05.
- **GRF filtering**: Butterworth 10 Hz then a 0.10 s Savitzky-Golay window
  (8 Hz at 0.50); standing noise ±12% BW, so "peak 1.82 BW" on 1001 is not
  separable from noise.
- **Segment CoM fractions** in `grf_estimation.py` differ from de Leva (trunk
  0.50 vs 0.569 from midHip; head collapsed onto neck); static CoM ~3 cm low.
  `bottom_up_id.SEG` is Winter's table labelled de Leva.
- **Muscle max isometric forces** are the generic model values, not scaled
  with the athlete (only lengths are scaled). Docstring claims otherwise.
- **Hicks residual limit** uses body weight rather than peak net external
  force (stricter than the paper); fine, should be stated. `share_of_id_moment`
  is peak/peak, so a joint whose reserves carry 48× a tiny moment passes.
- **Handedness** in `dynamics_compare` accepts only 'right'/'left'; 'R'/'L'
  silently fall back to Left/Right labels. `ours_from_metrics` drops genuine
  zeros.
- **Entered-mass vs model-mass mismatch** is a comparison-table note only;
  should be a status flag computed in `_run_id`.

## 4. Suggested order of fixes

1. B1, B2, B3, B4, B5 (score-changing), then re-score the demo and the
   reference swings and note what moved.
2. B7, B8 (robustness), B6 (bat), B9 (muscle gating).
3. Unify the contact frame; move `upper_torso_rotation_range` and
   `upper_torso_direction_at_contact` to weight 0; fix the stride text.
4. Per-foot split dead-band and moment-residual reporting, then re-test
   whether muscle loads can pass on video-only captures.
