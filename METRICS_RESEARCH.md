# Metric Research Basis

Technical companion to [`METRICS.md`](METRICS.md). That document explains the metrics
to athletes; this one records, for each metric, **where it comes from and how far the
evidence actually goes** — so that any claim made about this system can be traced to a
source, and the metrics that are our own invention are not mistaken for established
science.

## Evidence ratings

| Rating | Meaning |
|---|---|
| **A** | Directly established in peer-reviewed literature, with published reference values for the movement. |
| **B** | Mechanically sound and literature-supported in principle, but our specific operational definition (window, normalization, event anchor) is our own choice. |
| **C** | Our own composite or proxy. Internally consistent and useful for tracking change, but **not** a validated construct. Do not present as a research-backed measurement. |

**A "C" is not a defect.** Composite scores are how a product turns numbers into
guidance. The requirement is that they are labeled honestly.

---

## Known limitations that apply system-wide

1. **No measured ground reaction force.** All kinetics here are derived from segment
   kinematics alone. Without force plates the ground-contact term is missing from the
   force balance, so joint torques and pelvis forces carry systematic error that grows
   toward the ground. Lower-extremity values are the most affected. Treat every kinetic
   number as valid for *comparison between swings*, not as absolute joint load.
2. **Markerless input.** OpenCap keypoint error (~10–20 mm vs. marker-based) propagates
   into every derivative. Acceleration-derived quantities — torque, power, deceleration
   rate — are the noisiest, since each differentiation amplifies it.
3. **Capture rate.** At 60 Hz one frame is ~17 ms, which is the same order as the
   sequence gaps being measured. Sub-frame parabolic peak interpolation is used to push
   timing resolution to ~1 ms, but genuine uncertainty remains; `sequence_indeterminate`
   flags swings where the gap is within one frame.
4. **Reference values are mostly adult.** Published golf and baseball norms come from
   college/professional adults. The stated ICP is high-school-and-below, so percentile
   comparisons for youth athletes are extrapolation until a youth cohort is collected.

---

## Rotational kinetics

| Metric | Units | Rating | Basis |
|---|---|---|---|
| `peak_hip_torque_Nm` | N·m | B | Rigid-body τ = I·α. Nesbit & Serrano (2005) for golf-swing magnitudes. Limitation 1 applies. |
| `peak_shoulder_torque_Nm` | N·m | B | As above; trunk + arms + bat inertia about the shoulder axis. |
| `peak_hip_power_W` | W | B | P = τ·ω. MacWilliams et al. (1998); Anderson et al. (2006). |
| `peak_shoulder_power_W` | W | B | As above. |
| `hip_inertia_kg_m2` | kg·m² | **A** | de Leva (1996) adjusted segment inertia parameters — the standard source. |
| `shoulder_inertia_kg_m2` | kg·m² | **A** | de Leva (1996), summed over trunk, both arms, and bat. |
| `inertia_ratio` | — | C | Derived convenience ratio; no independent literature meaning. |
| `hip_power_per_kg` | W/kg | **A** | Body-mass normalization per Winter, *Biomechanics and Motor Control of Human Movement*. |
| `shoulder_power_per_kg` | W/kg | **A** | As above. |

## Separation and sequence

| Metric | Units | Rating | Basis |
|---|---|---|---|
| `max_separation_deg` (X-Factor) | deg | **A** | Cheetham et al. (2001); Myers et al. (2008) linked separation to ball velocity. Measured pre-peak-pelvis, which is our window choice. |
| `x_factor_stretch_deg` | deg | **A** | Cheetham et al. (2008): separation *gained after transition* predicts speed better than static X-Factor. Computed as `max_separation − |separation at swing start|`. |
| `sequence_timing_ms` | ms | **A** | Pelvis→torso peak lag. Kwon et al. (2013); Putnam (1993) for the underlying proximal-to-distal principle. Elite gaps ~30–70 ms. |
| `torso_arm_sequence_gap_ms` | ms | **A** | Torso→lead-arm lag, the second chain link. Kwon et al. (2013). Same sub-frame peaks as above, so the two gaps sum to pelvis→arm. |
| `proper_sequence` | bool | **A** | Proximal-to-distal ordering. Putnam (1993); Kwon et al. (2013). ±1 frame tolerance. |
| `pelvis_decel_rate_deg_s2` | deg/s² | B | Putnam (1993): proximal deceleration is the mechanism of distal acceleration. The principle is established; the 200 ms post-peak window and steepest-slope definition are ours. |

## Pelvis

| Metric | Units | Rating | Basis |
|---|---|---|---|
| `peak_pelvis_omega_3d_deg_s` | deg/s | **A** | The most-cited single power correlate in rotational sport. Fortenbaugh (2011); Cheetham et al. (2008). |
| `time_to_peak_pelvis_from_plant_ms` | ms | **A** | Welch et al. (1995): elite hitters peak ~50–100 ms after front-foot plant. Negative values mean the pelvis peaked before plant — spinning open early — and the sign is preserved because that pattern is diagnostic. |
| `time_to_peak_pelvis_ms` | ms | B | Same quantity measured from swing start instead of plant. Available when plant detection fails; less comparable to literature. |
| `pelvis_rotation_at_contact_deg` | deg | **A** | Welch et al. (1995); Fortenbaugh et al. (2011): ~40–75° open in professionals. Single-frame read, so it carries no differentiation noise — the most robust metric in this group on markerless data. Contact is proxied by peak pelvis angular velocity. |
| `pelvis_tilt_range_deg` | deg | B | Anterior/posterior tilt range. Fortenbaugh (2011) reports tilt; the range-over-swing form is ours. |
| `pelvis_list_range_deg` | deg | B | Pelvic obliquity. McNally et al. (2014) relate obliquity to weight shift. |
| `pelvis_tz_range_m` | m | B | Lateral pelvis translation. Welch et al. (1995). |
| `pelvis_torso_contribution_pct` | % | **C** | Pelvis share of pelvis + arm angular velocity at peak. Conceptually parallel to Blast Motion's "Body Rotation", but **their formula is proprietary and unpublished — this is our approximation and the two should not be expected to agree.** Renamed away from their term deliberately. |

## Stride and plant

| Metric | Units | Rating | Basis |
|---|---|---|---|
| `stride_length_m` | m | **A** | Welch et al. (1995); Fortenbaugh (2011). Ankle-to-ankle horizontal distance at contact. |
| `stride_ratio` | — | **A** | Height-normalized stride, standard in the same sources. |
| `stride_efficiency_pct` | % | **C** | Fraction of stride converted to forward CoM motion. Not a named metric in the literature; our own definition. |
| `plant_frame` / `plant_method` | frame / label | B | Event detection methodology per Bruening & Ridge (2012). |

## Lower-body kinematics and kinetics

| Metric | Units | Rating | Basis |
|---|---|---|---|
| `peak_hip_flexion_r/l_deg` | deg | **A** | Standard joint-angle reporting. Fortenbaugh (2011); Welch et al. (1995). |
| `peak_knee_flexion_r/l_deg` | deg | **A** | As above. Lead-knee extension through contact is a documented bat-speed correlate. |
| `peak_ankle_dorsiflexion_r/l_deg` | deg | **A** | As above. |
| `hip_flexion_asymmetry_deg` | deg | B | Krysak et al. (2017) associate >15% asymmetry with injury risk; our peak-difference form is a simplification. |
| `knee_flexion_asymmetry_deg` | deg | B | As above. |
| `peak_knee_torque_r/l_Nm` | N·m | B | τ = I·α with de Leva thigh inertia. Chu et al. (2010); MacWilliams et al. (1998). Limitation 1 applies strongly. |
| `peak_ankle_torque_r/l_Nm` | N·m | B | As above with shank inertia. Most GRF-sensitive metric in the system — interpret with caution. |
| `peak_knee_power_r/l_W` | W | B | P = τ·ω. Fortenbaugh (2011). |
| `peak_lead_hip_ir_torque_Nm` | N·m | B | MacWilliams et al. (1998) identify lead-hip internal rotation as the drive behind pelvis rotation. Returns 0 when handedness is unknown rather than guessing the leg, since the trail hip is a different action. |

## Energy transfer

| Metric | Units | Rating | Basis |
|---|---|---|---|
| `pelvis_ke_J`, `torso_ke_J`, `arm_ke_J`, `forearm_ke_J` | J | B | KE = ½Iω² with de Leva inertias. `forearm_ke_J` is named for the segment carrying the energy, not the elbow joint, which has no inertia of its own. |
| `total_energy_transfer_J` | J | B | Sum of segment KE including the bat. Putnam (1993) for segment energy transfer. |
| `torso_to_arm_transfer_ratio` | — | B | Peak-KE ratio. The rigorous form follows joint-power flow (Robertson & Winter 1980), not peak energies; ours is the simplification. |
| `pelvis_to_torso_transfer_ratio` | — | B | As above. |
| `torso_to_pelvis_rot_ratio` | — | **A** | Kwon et al. (2013); ~1.4–1.8 in elite rotational athletes. Capped at 2.0× pelvis omega to reject IK artifacts. |
| `energy_transfer_proxy_pct` | % | **C** | Distal share of total chain energy. **Deliberately not called "efficiency": the literature has no agreed formula for kinetic-chain efficiency.** A true one would follow joint-power flow (Robertson & Winter 1980). Our own defined ratio. |

## Pelvis forces (linear inverse dynamics)

| Metric | Units | Rating | Basis |
|---|---|---|---|
| `peak_pelvis_force_ap_N` | N | **C** | F = m·a on the pelvis segment. **A kinematics-derived proxy, not a joint reaction force** — no measured GRF, so the ground-contact term is absent (Limitation 1). Robertson et al., *Research Methods in Biomechanics*, on why the distinction matters. |
| `peak_pelvis_force_vert_N` | N | **C** | As above. |
| `peak_pelvis_force_lat_N` | N | **C** | As above. |
| `peak_pelvis_force_resultant_N` | N | **C** | 3-D magnitude of the above. |

## Weight shift

| Metric | Units | Rating | Basis |
|---|---|---|---|
| `lateral_sway_range_m` | m | B | McNally et al. (2014). |
| `lateral_sway_at_plant_m` | m | B | As above. |
| `weight_shift_timing_pct` | % of swing | B | MacWilliams et al. (1998) establish weight-transfer timing; the literature reports it in ms from plant, we report % of swing. |

## Arms

| Metric | Units | Rating | Basis |
|---|---|---|---|
| `peak_arm_flex_l_deg` | deg | **A** | Escamilla et al. (2009), baseball hitting joint angles. |
| `peak_elbow_flex_l_deg` | deg | **A** | As above. |
| `arm_flex_asymmetry_deg` | deg | B | Krysak et al. (2017) asymmetry concept. |
| `peak_prosup_r/l_deg` | deg | **A** | Welch et al. (1995). |

## Bat / output

| Metric | Units | Rating | Basis |
|---|---|---|---|
| `max_hand_speed_mph` | mph | **A** | Direct measurement from TRC wrist markers. Fortenbaugh (2011). Preferred over the estimate below whenever markers exist. |
| `estimated_hand_speed_mph` | mph | B | Fallback: ω × lever arm when markers are unavailable. Mechanically sound; should be validated against bat-sensor ground truth before being quoted as measured. |
| `rotational_acceleration_deg_s2` | deg/s² | B | Peak angular acceleration. Nesbit (2005). |
| `time_to_contact_s` | s | B | Swing start to peak pelvis omega (contact proxy). Conceptually aligned with Blast Motion's metric of the same name, which is proprietary; not validated against a measured contact event. |
| `swing_composite_score_v1` | 0–100 | **C** | **Our composite grade, not a measurement.** No literature supports a single scalar "swing score". Versioned so scores stay comparable across releases; the weighting is in `comprehensive_diagnosis()`. |

---

## References

- Anderson, B. C., Wright, I. C., & Stefanyshyn, D. J. (2006). Segmental sequencing of kinetic energy in the golf swing. *The Engineering of Sport 6*.
- Bruening, D. A., & Ridge, S. T. (2012). Automated event detection algorithms in pathological gait. *Gait & Posture*, 35(3).
- Cheetham, P. J., Martin, P. E., Mottram, R. E., & St. Laurent, B. F. (2001). The importance of stretching the "X-Factor" in the downswing of golf. *Optimising Performance in Golf*.
- Cheetham, P. J., et al. (2008). The energy flow of the golf swing. *Science and Golf V*.
- Chu, Y., Sell, T. C., & Lephart, S. M. (2010). The relationship between biomechanical variables and driving performance during the golf swing. *Journal of Sports Sciences*, 28(11).
- de Leva, P. (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters. *Journal of Biomechanics*, 29(9), 1223–1230.
- Escamilla, R. F., et al. (2009). A comparison of age level on baseball hitting kinematics. *Journal of Applied Biomechanics*, 25(3).
- Fortenbaugh, D. (2011). *The Biomechanics of the Baseball Swing*. Doctoral dissertation, University of Miami.
- Krysak, S., et al. (2017). Kinetic asymmetry and injury risk in rotational athletes. *Sports Biomechanics*.
- Kwon, Y.-H., Han, K. H., Como, C., Lee, S., & Singhal, K. (2013). Validity of the X-Factor computation methods and relationship between the X-Factor parameters and clubhead velocity. *Sports Biomechanics*, 12(3).
- MacKenzie, S. J. (2012). Club position relative to the golfer's swing plane meaningfully affects swing dynamics. *Sports Biomechanics*, 11(2).
- MacWilliams, B. A., Choi, T., Perezous, M. K., Chao, E. Y., & McFarland, E. G. (1998). Characteristic ground-reaction forces in baseball pitching. *American Journal of Sports Medicine*, 26(1).
- McNally, M. P., Borstad, J. D., Oñate, J. A., & Chaudhari, A. M. (2014). Stride leg ground reaction forces in baseball pitching. *Journal of Applied Biomechanics*, 31(4).
- Myers, J., et al. (2008). The role of upper torso and pelvis rotation in driving performance during the golf swing. *Journal of Sports Sciences*, 26(2).
- Nesbit, S. M., & Serrano, M. (2005). Work and power analysis of the golf swing. *Journal of Sports Science and Medicine*, 4(4).
- Putnam, C. A. (1993). Sequential motions of body segments in striking and throwing skills. *Journal of Biomechanics*, 26(Suppl 1).
- Robertson, D. G. E., & Winter, D. A. (1980). Mechanical energy generation, absorption and transfer amongst segments during walking. *Journal of Biomechanics*, 13(10).
- Robertson, D. G. E., Caldwell, G. E., Hamill, J., Kamen, G., & Whittlesey, S. N. *Research Methods in Biomechanics* (2nd ed.).
- Taguchi, Y., et al. (2023). Kinematic sequence in Division I collegiate batters. *Fukushima Journal of Medical Science*.
- Uhlrich, S. D., et al. (2023). OpenCap: Human movement dynamics from smartphone videos. *PLOS Computational Biology*, 19(10).
- Welch, C. M., Banks, S. A., Cook, F. F., & Draovitch, P. (1995). Hitting a baseball: A biomechanical description. *JOSPT*, 22(5).
- Winter, D. A. *Biomechanics and Motor Control of Human Movement* (4th ed.).

---

## Validation status

None of the kinetic outputs have yet been validated against an independent
ground truth. The planned first check is OpenSim Inverse Dynamics run on the
same OpenCap trial, comparing knee and ankle moments (target: r > 0.85, RMSE
< 15 N·m). Until that is done, no accuracy claim should be made for any
metric rated **B** or **C** above.
