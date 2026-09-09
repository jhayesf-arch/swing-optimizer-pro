# Metric Formulas

For each metric: the inputs it reads, the equation it computes, and a
one-sentence plain-English meaning. Evidence tiers and citations are in
[`METRICS_RESEARCH.md`](METRICS_RESEARCH.md); this document is the math.

---

## Notation and shared machinery

**Inputs come from two files.**

- `.mot` — OpenSim joint-coordinate time series. Every column here is a joint
  angle in degrees. Referred to as `mot.<column>`.
- `.trc` — 3-D marker positions in metres. Referred to as `trc.<marker>`.

**Signal conditioning (applied everywhere unless stated).**

- Angles are unwrapped, then low-pass filtered with a **zero-lag 4th-order
  Butterworth** at 15 Hz:
  `y = filtfilt(b, a, x)`.
- Derivatives use **Savitzky-Golay** on the filtered signal
  (`window ≈ 100 ms, polyorder = 3`):
  `ω = SG(θ, deriv=1)`, `α = SG(θ, deriv=2)`.
- Sampling `dt = mean(diff(time))`, `fs = 1/dt` (60 Hz on OpenCap output).

**Segment inertia** (de Leva 1996 percentages of body height and mass; `M` is
body mass, `H` is body height):

```
mass  = M · mass_pct
length = H · length_pct
I     = mass · (length · rg_pct)²
```

| Segment | mass_pct | length_pct | rg_pct |
|---|---|---|---|
| forearm    | 0.016  | 0.146 | 0.303 |
| upper_arm  | 0.028  | 0.186 | 0.322 |
| trunk      | 0.497  | 0.288 | 0.496 |
| thigh      | 0.100  | 0.245 | 0.323 |
| shank      | 0.0465 | 0.246 | 0.302 |

**Bat inertia (thin rod about the handle):**
`I_bat = ⅓ · m_bat · L_bat²`.

---

## 1 · Rotational kinetics

Uses `mot.pelvis_rotation` for the pelvis angle and
`mot.pelvis_rotation + mot.lumbar_rotation` for the absolute thorax angle. The
sum is intentional — the raw `lumbar_rotation` is *relative* to the pelvis, and
adding the pelvis contribution gives the thorax's motion in the ground frame,
which is what the literature reports and what sequences peak in the correct order.

### `peak_hip_torque_Nm`
- **Inputs:** `mot.pelvis_rotation`.
- **Formula:** `τ_hip = I_trunk · α_pelvis`, take `max(|τ_hip|)` over the swing.
- **Meaning:** Peak twist force the pelvis is producing to rotate the whole trunk.

### `peak_shoulder_torque_Nm`
- **Inputs:** `mot.pelvis_rotation`, `mot.lumbar_rotation`.
- **Formula:** `I_sh = I_trunk + 2·(I_upperarm + I_forearm) + I_bat`;
  `τ_sh = I_sh · α_thorax`; `max(|τ_sh|)`.
- **Meaning:** Peak twist force behind the upper torso and arms rotating together.

### `peak_hip_power_W` · `peak_shoulder_power_W`
- **Formula:** `P = τ · ω`, evaluated at each frame; take `max(|P|)`.
- **Meaning:** Instantaneous rotational power delivered at that joint's peak
  effort.

### `hip_inertia_kg_m2` · `shoulder_inertia_kg_m2`
- **Formula:** the two `I` expressions above.
- **Meaning:** How much rotational "weight" the pelvis and shoulder axes each
  have to spin.

### `inertia_ratio`
- **Formula:** `I_sh / I_hip`.
- **Meaning:** How much heavier the upper system (trunk + arms + bat) is to
  rotate than the pelvis alone.

### `hip_power_per_kg` · `shoulder_power_per_kg`
- **Formula:** `peak_power / M`.
- **Meaning:** Peak power expressed per kg of body mass, so bigger athletes
  aren't automatically ranked higher.

---

## 2 · Separation and sequence

### `max_separation_deg` (X-Factor)
- **Inputs (preferred):** `.trc` markers
  `r_shoulder`, `l_shoulder`, `r.ASIS`, `l.ASIS`, `r.PSIS`, `l.PSIS`, `C7`.
- **Inputs (fallback):** `mot.lumbar_rotation` — used when `.trc` absent.
- **Formula (marker path):** build the pelvis frame from ASIS + PSIS (ISB
  convention: ML from `l.ASIS − r.ASIS`, forward from ASIS-midpoint minus
  PSIS-midpoint, superior from ML × forward). Build the thorax ML axis from
  `l_shoulder − r_shoulder`, referenced against `C7` to remove scapular slide.
  Project the thorax ML into the pelvis's transverse plane and take:

  ```
  sep(t) = atan2(thorax_ml · pelvis_fwd, thorax_ml · pelvis_ml) (degrees)
  sep(t) = sep(t) − median(sep[0 : 0.25 s])
  max_separation_deg = max(|sep(t)|) over swing window
  ```

- **Meaning:** How far your shoulders open past your hips at the point of
  maximum coil — the loaded "elastic" in the swing.

### `x_factor_stretch_deg`
- **Inputs:** `sep(t)` as above, plus the transition frame (last pelvis-omega
  sign reversal before contact).
- **Formula:** `max(|sep|) between transition and contact − |sep at transition|`.
- **Meaning:** Extra separation you *gain* after starting the downswing, which
  matters more for bat speed than the static number above.

### `sequence_timing_ms`
- **Inputs:** filtered `pelvis_omega`, `thorax_abs_omega`.
- **Formula:** find the frame index of each peak with a sub-frame quadratic fit
  (~1 ms resolution instead of 17 ms), then:
  `Δt_ms = (peak_thorax_frac − peak_pelvis_frac) · dt · 1000`.
- **Meaning:** How long after your hips peak your torso peaks — the first link
  of the kinetic chain.

### `torso_arm_sequence_gap_ms`
- **Formula:** same sub-frame method, `(peak_arm_frac − peak_thorax_frac) · dt · 1000`.
- **Meaning:** How long after your torso peaks your lead arm peaks — the second
  link of the chain.

### `proper_sequence`
- **Formula:** `pelvis_peak ≤ thorax_peak ≤ arm_peak` with ±1-frame tolerance.
- **Meaning:** True when everything fires in the right order (proximal to
  distal); false if the order is broken.

### `torso_to_pelvis_rot_ratio`
- **Formula:** `peak_thorax_ω / peak_pelvis_ω`. Capped at 2.0× to reject IK
  artifacts (empirical p90 = 1.47 on clean data).
- **Meaning:** How much faster your torso spins than your hips at peak; ~1.4–1.8
  in elite hitters.

---

## 3 · Pelvis / lower half

### `peak_pelvis_omega_3d_deg_s`
- **Inputs:** `mot.pelvis_rotation`, `mot.pelvis_tilt`, `mot.pelvis_list`.
- **Formula:** each angle filtered and differentiated to `ω`, then
  `ω_3d(t) = √(ω_rot² + ω_tilt² + ω_list²)`; `max(ω_3d)` × 180/π.
- **Meaning:** Peak speed of your pelvis's total 3-D rotation.

### `time_to_peak_pelvis_ms`
- **Formula:** `(peak_pelvis_frac − swing_start_frame) · dt · 1000`.
- **Meaning:** How long from swing start to the pelvis hitting top speed.

### `time_to_peak_pelvis_from_plant_ms`
- **Formula:** `(peak_pelvis_frac − plant_frame) · dt · 1000`. Sign preserved:
  negative = pelvis peaked before foot planted.
- **Meaning:** How long after your front foot lands your pelvis reaches top
  speed; elite is 50–100 ms.

### `pelvis_rotation_at_contact_deg`
- **Inputs:** `mot.pelvis_rotation`, plus the contact frame.
- **Formula:** `|pelvis_rotation[contact_frame]|` in degrees (absolute angle,
  filtered).
- **Meaning:** How open your hips are at the moment you hit the ball.

### `pelvis_rotation_excursion_deg`
- **Formula:** `|pelvis_angle[contact] − pelvis_angle[transition]|` × 180/π.
- **Meaning:** How much total rotation your hips swept from top of backswing
  through contact.

### `pelvis_decel_rate_deg_s2`
- **Inputs:** `ω_pelvis_3d(t)` over the 200 ms after its peak.
- **Formula:** `min(dω/dt)` over that window, absolute value, converted to °/s².
- **Meaning:** How sharply your hips brake after peak speed — sharp braking is
  what transfers energy up to the torso.

### `peak_lead_hip_ir_torque_Nm`
- **Inputs:** `mot.hip_rotation_l` (RH hitter) or `mot.hip_rotation_r` (LH).
  Returns `0.0` when handedness is unknown.
- **Formula:** `τ = I_thigh · α_hip_rot`; `max(|τ|)`.
- **Meaning:** Peak inward-twist force from your front hip — the actuator
  behind pelvis rotation.

### `pelvis_tilt_range_deg` · `pelvis_list_range_deg`
- **Inputs:** `mot.pelvis_tilt`, `mot.pelvis_list`.
- **Formula:** filtered signal, `max − min` over the swing window.
- **Meaning:** Range of forward/back pelvic tilt (tilt) and side-to-side tilt
  (list) through the swing.

### `pelvis_tz_range_m`
- **Inputs:** `mot.pelvis_tz`.
- **Formula:** `max − min` over the swing window.
- **Meaning:** How far your pelvis drifts laterally through the swing.

---

## 4 · Stride and plant

### `plant_frame` / `plant_method`
- **Inputs:** `.trc` foot markers when present, else `pelvis_tz`.
- **Formula:** front-foot vertical velocity crossing zero (marker path), or the
  first frame after swing start where lateral CoM stops moving (fallback).
- **Meaning:** The frame index at which your front foot lands.

### `stride_length_m`
- **Inputs (marker):** `.trc.r_ankle`, `l_ankle` at contact.
- **Formula:** `‖ankle_lead − ankle_trail‖` in the horizontal plane, at contact.
- **Meaning:** How wide your base is at contact, measured ankle-to-ankle in
  metres.

### `stride_ratio`
- **Formula:** `stride_length_m / body_height_m`.
- **Meaning:** Stride length as a fraction of your height, so it's fair across
  body sizes.

### `stride_efficiency_pct`
- **Inputs:** `pelvis_tx(t)`, `plant_frame`.
- **Formula:** `100 · (forward_CoM_displacement / stride_length)`.
- **Meaning:** Fraction of your stride that turned into actual forward momentum
  toward the pitch (our own definition, tier C).

---

## 5 · Lower-body kinematics

For `side ∈ {r, l}` and `joint ∈ {hip_flexion, knee_angle, ankle_angle}`:

### `peak_<joint>_<side>_deg`
- **Inputs:** `mot.<joint>_<side>`.
- **Formula:** filtered signal, `max(|θ|)`.
- **Meaning:** Peak angle that joint reaches through the swing.

### `hip_flexion_asymmetry_deg` · `knee_flexion_asymmetry_deg`
- **Formula:** `|peak_r − peak_l|`.
- **Meaning:** Left-vs-right imbalance at that joint.

---

## 6 · Lower-body kinetics (Newton-Euler)

For each side, using thigh inertia for the knee and shank for the ankle:

### `peak_knee_torque_<side>_Nm`
- **Inputs:** `mot.knee_angle_<side>`.
- **Formula:** `α_knee = SG(θ_knee, deriv=2)`; `τ = I_thigh · α`; `max(|τ|)`.
- **Meaning:** Peak twist force at the knee during the swing.

### `peak_ankle_torque_<side>_Nm`
- **Inputs:** `mot.ankle_angle_<side>`.
- **Formula:** `τ = I_shank · α_ankle`; `max(|τ|)`.
- **Meaning:** Peak twist force at the ankle during the swing.

### `peak_knee_power_<side>_W`
- **Formula:** `P = τ_knee · ω_knee`, `max(|P|)`.
- **Meaning:** Peak power the knee is delivering (torque × how fast it's
  turning).

> All lower-body torques carry systematic error without a force plate; use for
> ranking swings, not as an absolute joint load.

---

## 7 · Energy transfer

### `pelvis_ke_J`
- **Formula:** `½ · I_hip · ω_pelvis²` at the pelvis peak.
- **Meaning:** Rotational kinetic energy in the pelvis at its peak speed.

### `torso_ke_J`
- **Formula:** `½ · I_trunk · ω_thorax²` at the thorax peak. `ω_thorax` capped
  at 2.0× `ω_pelvis` to reject IK artifacts.
- **Meaning:** Rotational kinetic energy in the torso.

### `arm_ke_J`
- **Formula:** `2 · ½ · (I_upperarm + I_forearm) · ω_arm²` — both arms.
- **Meaning:** Rotational kinetic energy in the arms.

### `forearm_ke_J`
- **Formula:** `2 · ½ · I_forearm · ω_elbow²`.
- **Meaning:** Rotational KE contributed by the forearm segments alone.

### `bat_ke_J`
- **Formula:** `½ · m_bat · v_wrist² + ½ · I_bat · ω_arm²` (translational +
  rotational). Wrist speed from `.trc` when present, else estimated from arm ω ×
  forearm length.
- **Meaning:** Kinetic energy the bat is carrying.

### `total_energy_transfer_J`
- **Formula:** `pelvis_ke + torso_ke + arm_ke + forearm_ke + bat_ke`.
- **Meaning:** Total rotational KE flowing through the whole chain.

### `torso_to_arm_transfer_ratio` · `pelvis_to_torso_transfer_ratio`
- **Formula:** ratio of the two peak KEs above (`torso/arm`, `pelvis/torso`).
- **Meaning:** How well energy handed off from one segment to the next.

### `energy_transfer_proxy_pct`
- **Formula:** `100 · (arm_ke + forearm_ke + bat_ke) / total_energy_transfer_J`.
- **Meaning:** Share of total chain energy that reached the distal segments
  (our own ratio, tier C — not a validated efficiency measure).

---

## 8 · Pelvis forces (linear inverse dynamics)

Segment mass = `M · trunk_mass_pct = 0.497 · M`.

### `peak_pelvis_force_ap_N` · `peak_pelvis_force_vert_N` · `peak_pelvis_force_lat_N`
- **Inputs:** `mot.pelvis_tx`, `mot.pelvis_ty`, `mot.pelvis_tz`.
- **Formula:** each translation filtered, differentiated twice to acceleration,
  then `F = m · a` component-wise; `max(|F|)`.
- **Meaning:** Peak fore-aft, vertical, and lateral force acting on the pelvis
  segment.

### `peak_pelvis_force_resultant_N`
- **Formula:** `max(√(F_ap² + F_vert² + F_lat²))`.
- **Meaning:** Peak 3-D magnitude of the pelvis-segment force.

> These are kinematics-derived *proxies*, not measured joint reaction forces
> (no ground-reaction data). Compare between swings, not as absolute load.

---

## 9 · Weight shift

### `lateral_sway_range_m`
- **Inputs:** `mot.pelvis_tz` over the swing window.
- **Formula:** `max − min`.
- **Meaning:** Total side-to-side pelvis drift during the swing.

### `lateral_sway_at_plant_m`
- **Formula:** `pelvis_tz[plant] − median(pelvis_tz[0:swing_start])`.
- **Meaning:** How far your pelvis has slid sideways by the time your front
  foot lands.

### `weight_shift_timing_pct`
- **Formula:** frame index of `argmax(|pelvis_tz|)` expressed as % of swing
  duration.
- **Meaning:** How far into the swing your peak lateral shift happens
  (0% = start, 100% = contact).

---

## 10 · Arms

For `side ∈ {r, l}`, using `mot.arm_flex_<side>`, `mot.elbow_flex_<side>`,
`mot.pro_sup_<side>`:

### `peak_arm_flex_l_deg` · `peak_elbow_flex_l_deg`
- **Formula:** filtered signal, `max(|θ|)`.
- **Meaning:** Peak lead-shoulder flexion and lead-elbow flexion.

### `arm_flex_asymmetry_deg`
- **Formula:** `|peak_arm_flex_r − peak_arm_flex_l|`.
- **Meaning:** Difference between how far each shoulder flexes.

### `peak_prosup_<side>_deg`
- **Formula:** `max(|θ_prosup|)`.
- **Meaning:** Peak forearm pronation/supination — the rolling of the forearm
  as the barrel comes through.

---

## 11 · Hand speed and contact

### `max_hand_speed_mph` (measured — requires `.trc`)
- **Inputs:** `.trc.r_wrist_ulna`, `l_wrist_ulna` (radius/study variants also
  accepted).
- **Formula:** each XYZ filtered, `v = √(ẋ² + ẏ² + ż²)`; `max(v)`, take the
  faster wrist, convert m/s → mph (× 2.23694).
- **Meaning:** Peak speed your hands reach through the zone; the single best
  predictor of bat/ball outcome.

### `estimated_hand_speed_mph` (fallback — no `.trc`)
- **Formula:** `ω_arm × (0.204 · H)` (forearm + hand ≈ 20.4% of body height),
  converted to mph.
- **Meaning:** An estimate of hand speed when we don't have marker data.

### `hand_speed_peak_time_s`
- **Formula:** `time[argmax(v_wrist)]`.
- **Meaning:** The instant hand speed peaks — precedes ball contact by a few
  frames because the wrists top out while the barrel is still whipping past.

### `hand_contact_time_s` / `contact_detection_method`
- **Formula (preferred):** with `.trc` — sharpest deceleration of hand speed
  after its peak: `contact = time[argmin(dv/dt over the 150 ms after peak)]`,
  method `trc_hand_deceleration`.
- **Formula (fallback):** without `.trc` — peak of
  `|thorax_ω + arm_ω + elbow_ω|`, method `peak_pelvis_omega_fallback` (least
  accurate).
- **Meaning:** The time when the bat meets the ball, from the "force feedback"
  spike the ball puts on your hands.

### `rotational_acceleration_deg_s2`
- **Inputs:** `mot.arm_flex_r`.
- **Formula:** `α_arm = SG(θ_arm, deriv=2)`; `max(|α_arm|)` × 180/π.
- **Meaning:** Peak angular acceleration of the arm system — how quickly you're
  ramping the bat into the zone.

### `time_to_contact_s`
- **Formula:** `(peak_pelvis_frame − swing_start_frame) · dt`.
- **Meaning:** Rough duration of your swing from start to contact.

### `pelvis_torso_contribution_pct`
- **Formula:** `100 · ω_pelvis / (ω_pelvis + ω_arm + ε)`.
- **Meaning:** Share of the peak angular velocity coming from the pelvis vs.
  the arms (our own ratio, tier C — parallels a proprietary industry metric
  but is not identical).

---

## 12 · Composite score

### `swing_composite_score_v1`
- **Formula:** starts at 100, subtracts penalties across the findings ruleset
  (bad sequence −20, low separation −15, high lateral sway −10, etc.), floored
  at 0. Full weightings live in `comprehensive_diagnosis()` in `analyzer.py`.
- **Meaning:** Our own overall grade out of 100. Not a literature-validated
  score — versioned so changes are transparent.

---

## Events used across many metrics

**Swing start** — walk backwards from the pelvis-ω peak until the first frame
that drops below `max(20°/s, 12% of peak)` after having been above it. Hard
600 ms bound in case the trial starts mid-swing.

**Contact frame** — see §11. Ranked from best to worst:
`trc_hand_deceleration` > `peak_hand_speed` > `peak_pelvis_omega_fallback`.

**Transition frame** — the last sign change in `pelvis_ω` before contact;
approximates the top of the backswing.

---

## Reliability guards (how "unreliable" gets flagged)

Same file, `metric_evidence.py`:

- **Clipped separation:** raw `lumbar_rotation` pinned at ±90° for ≥ 3 frames →
  `max_separation`, `x_factor_stretch` unreliable (monocular capture failure).
- **Anatomical bound:** any separation > 70° → same metrics unreliable
  (shoulder marker line inflated by scapular slide).
- **No markers:** any "at contact" metric → unreliable, ±100 ms timing
  uncertainty.
- **Slow trial:** peak pelvis ω < 300°/s → sequence metrics unreliable
  (rehearsal-speed swing).
- **No hand speed:** `max_hand_speed_mph` unreliable when `.trc` absent.
