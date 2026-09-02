"""Per-metric evidence rating and per-capture reliability.

Two independent questions are answered here, and keeping them apart matters:

  1. EVIDENCE  — does published research back this metric at all?
  2. RELIABLE  — can THIS capture support it?

They are genuinely independent. Hip-shoulder separation is among the best
established metrics in the rotational-sport literature (evidence A) and is
simultaneously worthless on a monocular capture, where the lumbar coordinate
parks against its joint stop for half a second. Reporting only the citation
would imply a confidence the data does not carry; reporting only the data
quality would hide that some numbers are ours rather than the field's.

Evidence tiers
  A  Established in peer-reviewed work, with published reference values.
  B  Mechanically sound and literature-supported in principle, but our specific
     operational definition — window, normalization, event anchor — is our own.
  C  Our own composite or proxy. Useful for tracking change over time; not a
     validated construct, and not to be presented as one.
"""

A, B, C = 'A', 'B', 'C'

# metric -> (tier, citation, note)
METRIC_EVIDENCE = {
    # ── Pelvis / lower half ─────────────────────────────────────────────────
    'peak_pelvis_omega_3d_deg_s': (
        A, 'Fortenbaugh 2011; Cheetham et al. 2008',
        'Most-cited single power correlate in rotational sport. '
        'Reference: 500-700 deg/s competitive HS/college, 700-900 professional.'),
    'time_to_peak_pelvis_from_plant_ms': (
        A, 'Welch et al. 1995',
        'Elite hitters peak ~50-100 ms after front-foot plant. Negative means the '
        'pelvis peaked before the foot landed — spinning open early.'),
    'time_to_peak_pelvis_ms': (
        B, 'Welch et al. 1995 (adapted)',
        'Same quantity measured from swing start rather than plant. Used when '
        'plant detection fails; less comparable to published values.'),
    'pelvis_rotation_at_contact_deg': (
        A, 'Welch et al. 1995; Fortenbaugh 2011',
        'Published range 40-75 deg in professionals. Absolute angle, so it is '
        'expressed in the capture ground frame and shifts with setup orientation.'),
    'pelvis_rotation_excursion_deg': (
        B, 'Welch et al. 1995 (adapted)',
        'Rotation swept from the top to contact. Invariant to setup orientation, '
        'so it compares across sessions where the absolute angle does not.'),
    'pelvis_decel_rate_deg_s2': (
        B, 'Putnam 1993',
        'Proximal deceleration is the established mechanism of distal '
        'acceleration; the 200 ms window and steepest-slope definition are ours.'),
    'peak_lead_hip_ir_torque_Nm': (
        B, 'MacWilliams et al. 1998',
        'Lead-hip internal rotation drives pelvis rotation. Returns 0 when '
        'handedness is unknown rather than guessing the leg.'),
    'pelvis_tilt_range_deg': (B, 'Fortenbaugh 2011', 'Range-over-swing form is ours.'),
    'pelvis_list_range_deg': (B, 'McNally et al. 2014', 'Obliquity as a weight-shift proxy.'),
    'pelvis_tz_range_m': (B, 'Welch et al. 1995', ''),

    # ── Separation and sequence ─────────────────────────────────────────────
    'max_separation_deg': (
        A, 'Cheetham et al. 2001; Myers et al. 2008',
        'X-Factor. Strongly linked to ball velocity. Requires a trustworthy '
        'thorax-vs-pelvis signal — see the capture caveats.'),
    'x_factor_stretch_deg': (
        A, 'Cheetham et al. 2008',
        'Separation gained after transition predicts speed better than static '
        'X-Factor. Same signal dependency as above.'),
    'sequence_timing_ms': (
        A, 'Kwon et al. 2013; Putnam 1993',
        'Pelvis-to-torso peak lag. Elite gaps ~30-70 ms.'),
    'torso_arm_sequence_gap_ms': (A, 'Kwon et al. 2013', 'Second link of the chain.'),
    'proper_sequence': (A, 'Putnam 1993; Kwon et al. 2013', 'Proximal-to-distal ordering.'),
    'torso_to_pelvis_rot_ratio': (A, 'Kwon et al. 2013', 'Elite ~1.4-1.8.'),

    # ── Stride ──────────────────────────────────────────────────────────────
    'stride_length_m': (A, 'Welch et al. 1995; Fortenbaugh 2011', ''),
    'stride_ratio': (A, 'Welch et al. 1995; Fortenbaugh 2011', 'Height-normalized.'),
    'stride_efficiency_pct': (
        C, 'no published basis',
        'Fraction of stride converted to forward CoM motion. Our own definition; '
        'not a named metric in the literature.'),

    # ── Lower-body kinematics / kinetics ────────────────────────────────────
    'peak_hip_flexion_r_deg': (A, 'Fortenbaugh 2011; Welch et al. 1995', ''),
    'peak_hip_flexion_l_deg': (A, 'Fortenbaugh 2011; Welch et al. 1995', ''),
    'peak_knee_flexion_r_deg': (A, 'Fortenbaugh 2011; Chu et al. 2010', ''),
    'peak_knee_flexion_l_deg': (A, 'Fortenbaugh 2011; Chu et al. 2010', ''),
    'peak_ankle_dorsiflexion_r_deg': (A, 'Fortenbaugh 2011', ''),
    'peak_ankle_dorsiflexion_l_deg': (A, 'Fortenbaugh 2011', ''),
    'hip_flexion_asymmetry_deg': (B, 'Krysak et al. 2017', 'Peak-difference form is ours.'),
    'knee_flexion_asymmetry_deg': (B, 'Krysak et al. 2017', 'Peak-difference form is ours.'),
    'peak_knee_torque_r_Nm': (B, 'Chu et al. 2010; MacWilliams et al. 1998', 'No measured GRF.'),
    'peak_knee_torque_l_Nm': (B, 'Chu et al. 2010; MacWilliams et al. 1998', 'No measured GRF.'),
    'peak_ankle_torque_r_Nm': (B, 'Winter, Biomechanics', 'Most GRF-sensitive metric here.'),
    'peak_ankle_torque_l_Nm': (B, 'Winter, Biomechanics', 'Most GRF-sensitive metric here.'),
    'peak_knee_power_r_W': (B, 'Fortenbaugh 2011', ''),
    'peak_knee_power_l_W': (B, 'Fortenbaugh 2011', ''),

    # ── Rotational kinetics ─────────────────────────────────────────────────
    'peak_hip_torque_Nm': (B, 'Nesbit & Serrano 2005', 'No measured GRF.'),
    'peak_shoulder_torque_Nm': (B, 'Nesbit & Serrano 2005', 'No measured GRF.'),
    'peak_hip_power_W': (B, 'MacWilliams et al. 1998; Anderson et al. 2006', ''),
    'peak_shoulder_power_W': (B, 'MacWilliams et al. 1998; Anderson et al. 2006', ''),
    'hip_inertia_kg_m2': (A, 'de Leva 1996', 'Standard segment inertia parameters.'),
    'shoulder_inertia_kg_m2': (A, 'de Leva 1996', 'Standard segment inertia parameters.'),
    'hip_power_per_kg': (A, 'Winter, Biomechanics', 'Body-mass normalization.'),
    'shoulder_power_per_kg': (A, 'Winter, Biomechanics', 'Body-mass normalization.'),
    'inertia_ratio': (C, 'derived', 'Convenience ratio; no independent meaning.'),

    # ── Energy ──────────────────────────────────────────────────────────────
    'pelvis_ke_J': (B, 'Putnam 1993', 'KE = 1/2 I w^2 with de Leva inertias.'),
    'torso_ke_J': (B, 'Putnam 1993', ''),
    'arm_ke_J': (B, 'Putnam 1993', ''),
    'forearm_ke_J': (B, 'Putnam 1993', ''),
    'total_energy_transfer_J': (B, 'Putnam 1993', ''),
    'torso_to_arm_transfer_ratio': (
        B, 'Robertson & Winter 1980 (simplified)',
        'The rigorous form follows joint-power flow, not peak-energy ratios.'),
    'pelvis_to_torso_transfer_ratio': (
        B, 'Robertson & Winter 1980 (simplified)',
        'The rigorous form follows joint-power flow, not peak-energy ratios.'),
    'energy_transfer_proxy_pct': (
        C, 'no agreed formula exists',
        'Distal share of chain energy. The literature has no standard definition '
        'of kinetic-chain efficiency; this is our own ratio.'),

    # ── Pelvis forces ───────────────────────────────────────────────────────
    'peak_pelvis_force_ap_N': (
        C, 'Robertson et al., Research Methods in Biomechanics',
        'Kinematics-derived proxy, not a joint reaction force: no measured GRF, '
        'so the ground-contact term is absent. Compare between swings, not as load.'),
    'peak_pelvis_force_vert_N': (C, 'as above', 'Kinematics-derived proxy.'),
    'peak_pelvis_force_lat_N': (C, 'as above', 'Kinematics-derived proxy.'),
    'peak_pelvis_force_resultant_N': (C, 'as above', 'Kinematics-derived proxy.'),

    # ── Weight shift ────────────────────────────────────────────────────────
    'lateral_sway_range_m': (B, 'McNally et al. 2014', ''),
    'lateral_sway_at_plant_m': (B, 'McNally et al. 2014', ''),
    'weight_shift_timing_pct': (
        B, 'MacWilliams et al. 1998',
        'Literature reports ms from plant; we report % of swing.'),

    # ── Arms / output ───────────────────────────────────────────────────────
    'peak_arm_flex_l_deg': (A, 'Escamilla et al. 2009', ''),
    'peak_elbow_flex_l_deg': (A, 'Escamilla et al. 2009', ''),
    'arm_flex_asymmetry_deg': (B, 'Krysak et al. 2017', ''),
    'peak_prosup_r_deg': (A, 'Welch et al. 1995', ''),
    'peak_prosup_l_deg': (A, 'Welch et al. 1995', ''),
    'max_hand_speed_mph': (
        A, 'Fortenbaugh 2011',
        'Direct measurement from wrist markers. Requires a .trc.'),
    'estimated_hand_speed_mph': (
        B, 'rigid-body estimate',
        'Angular velocity x lever arm, used when markers are absent. Not '
        'validated against a bat sensor.'),
    'rotational_acceleration_deg_s2': (B, 'Nesbit & Serrano 2005', ''),
    'time_to_contact_s': (
        B, 'Blast Motion (proprietary), conceptually',
        'Their formula is unpublished; ours is an approximation, not a match.'),
    'pelvis_torso_contribution_pct': (
        C, 'Blast Motion (proprietary), conceptually',
        'Their "Body Rotation" formula is unpublished. Our own approximation; '
        'the two should not be expected to agree numerically.'),
    'swing_composite_score_v1': (
        C, 'no published basis',
        'Our composite grade, not a measurement. No literature supports a single '
        'scalar swing score. Versioned so it stays comparable across releases.'),
}


def _capture_flags(diagnosis: dict) -> dict:
    """Read the capture conditions that decide which metrics this trial supports."""
    rot = diagnosis.get('_rotation_context') or {}
    metrics = diagnosis.get('metrics') or {}
    if not isinstance(metrics, dict):
        metrics = getattr(metrics, '__dict__', {}) or {}

    contact_method = metrics.get('contact_detection_method', 'none')
    return {
        'separation_clipped': bool(rot.get('separation_signal_saturated')),
        'separation_source': rot.get('separation_source', 'model_lumbar_rotation'),
        'has_markers': bool(metrics.get('max_hand_speed_mph', 0) > 0),
        'contact_measured': contact_method in ('trc_hand_deceleration',),
        'contact_inferred': contact_method in (
            'peak_hand_speed', 'hand_deceleration_impact', 'peak_pelvis_omega_fallback'),
        'peak_pelvis_dgs': float(metrics.get('peak_pelvis_omega_3d_deg_s', 0.0) or 0.0),
        'max_separation_deg': float(metrics.get('max_separation_deg', 0.0) or 0.0),
        'separation_raw_peak_deg': float(rot.get('separation_raw_peak_deg', 0.0) or 0.0),
    }


def build_metric_evidence(diagnosis: dict) -> dict:
    """Return per-metric evidence tier plus whether THIS capture supports it.

    Every metric keeps its published tier regardless of the capture. The capture
    only sets `reliable` and attaches a caveat, so a good citation can never mask
    data that cannot support the number.
    """
    f = _capture_flags(diagnosis)
    out = {}

    for name, (tier, citation, note) in METRIC_EVIDENCE.items():
        reliable, caveat = True, ''

        # Separation-derived metrics inherit the thorax-vs-pelvis signal quality.
        if name in ('max_separation_deg', 'x_factor_stretch_deg'):
            # Judge on the larger of the windowed value and the raw signal peak: a
            # windowed number can look sane while the signal behind it is not.
            sep_val = max(float(f.get('max_separation_deg') or 0.0),
                          float(f.get('separation_raw_peak_deg') or 0.0))
            if f['separation_clipped']:
                reliable = False
                caveat = ('Thorax-vs-pelvis signal is clipped at its joint limit on '
                          'this capture, so no separation value can be trusted. '
                          'Common on monocular captures.')
            elif sep_val > 70.0:
                # Anatomical plausibility, checked regardless of source. Live humans
                # reach roughly 50-70 deg of thorax-vs-pelvis axial rotation; elite
                # X-Factor is 40-60. A larger number is not a big coil, it is a
                # broken measurement — and it slipped through when only the clip and
                # the source were checked, because marker-derived values are not
                # clipped and so were being passed as sound.
                reliable = False
                caveat = (f'{sep_val:.0f} deg exceeds the ~70 deg anatomical limit for '
                          'thorax-vs-pelvis rotation, so this is a measurement '
                          'artifact rather than a coil. The shoulder-marker line '
                          'carries scapular motion on top of true thoracic rotation, '
                          'which inflates it; a dedicated thorax segment '
                          '(sternum + C7 + T-spine) is needed to separate the two.')
            elif f['separation_source'] == 'model_lumbar_rotation':
                caveat = ('Taken from the model coordinate. Supply a .trc to compute '
                          'this from markers, which avoids the joint-limit problem.')

        # "At contact" metrics are only as good as the contact event.
        if name in ('pelvis_rotation_at_contact_deg', 'pelvis_rotation_excursion_deg',
                    'time_to_contact_s'):
            if not f['has_markers']:
                reliable = False
                caveat = ('Contact was inferred from joint angles; carries roughly '
                          '+/-100 ms of timing uncertainty. Supply a .trc for a '
                          'measured contact event.')
            elif f['contact_measured']:
                caveat = ('Contact taken from hand deceleration. Meaningful only on '
                          'swings that actually strike a ball — on dry swings there '
                          'is no impact to detect.')

        # Measured vs estimated hand speed.
        if name == 'max_hand_speed_mph' and not f['has_markers']:
            reliable = False
            caveat = 'No marker data; estimated_hand_speed_mph was used instead.'

        # Sequencing needs enough movement to resolve at 60 Hz.
        if name in ('sequence_timing_ms', 'torso_arm_sequence_gap_ms', 'proper_sequence'):
            if 0 < f['peak_pelvis_dgs'] < 300:
                reliable = False
                caveat = (f'Peak pelvis speed {f["peak_pelvis_dgs"]:.0f} deg/s is well '
                          'below competitive range; this may not be a full-intent '
                          'swing, and sequence timing is not interpretable if not.')

        # Every kinetic output lacks measured ground reaction force.
        if name.endswith(('_Nm', '_W', '_N')) or name.endswith('_ke_J'):
            if not caveat:
                caveat = 'No measured ground reaction force; valid for comparison between swings, not as absolute load.'

        out[name] = {
            'tier': tier,
            'citation': citation,
            'note': note,
            'reliable': reliable,
            'caveat': caveat,
        }

    return out


TIER_LABELS = {
    A: 'Research-backed',
    B: 'Literature-based, our definition',
    C: 'Our own measure',
}
