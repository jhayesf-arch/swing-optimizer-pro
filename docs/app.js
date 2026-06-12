const DEMO_DIAGNOSIS = {
    swing_score: 74,
    efficiency_score: 82,
    metrics: {
        estimated_hand_speed_mph: 24.2, max_hand_speed_mph: 24.2,
        max_separation_deg: 44.1, peak_hip_power_W: 3820, hip_power_per_kg: 14.2,
        sequence_timing_ms: 48, kinetic_chain_efficiency_pct: 52.3,
        torso_to_pelvis_rot_ratio: 1.31, total_energy_transfer_J: 412,
        stride_efficiency_pct: 94, stride_ratio: 0.74, proper_sequence: true,
        pelvis_ke_J: 98.4, torso_ke_J: 74.1, arm_ke_J: 61.2, bat_ke_J: 38.7,
        time_to_contact_s: 0.155, rotational_acceleration_deg_s2: 8400,
        body_rotation_ratio_pct: 44.2,
    },
    findings: [
        'Optimal Proximal-to-Distal Kinetic Chain demonstrated.',
        'Elite X-Factor Separation Stretch (44.1°).',
        'Elite Lower-Half Power Generation.',
        'Elite Distal Energy Amplification (52.3%).',
        'Excellent Torso-to-Pelvis Velocity Amplification (1.31).',
        'Good Total Kinetic Chain Energy (412 J, 15.3 J/kg).',
        'Efficient Stride Mechanics & Center of Mass Control.',
    ],
    recommendations: [
        'Good foundation. To push toward elite: focus on violent hip deceleration at front foot plant to whip stored energy into the torso and arms.',
        'Focus on lead-leg bracing at contact and sequential deceleration of the pelvis to whip maximum energy into the hands.',
    ],
    swingai_report: {
        skill_level: 'college',
        swing_score: 74,
        phases: {
            balance_load: { label: 'Balance & Load', icon: '⚖️', avg_stars: 3.7, badge: 'satisfactory', dimensions: [
                { key: 'negative_move', label: 'Negative Move', value: '0.06', unit: 'm', stars: 4, badge: 'satisfactory', description: 'Good backward weight shift before stride.' },
                { key: 'pelvis_load', label: 'Pelvis Load', value: '88', unit: 'J', stars: 4, badge: 'satisfactory', description: 'Strong pelvis kinetic energy at load.' },
                { key: 'upper_torso_load', label: 'Upper Torso Load', value: '52', unit: 'J', stars: 3, badge: 'satisfactory', description: 'Adequate upper torso coil.' },
            ]},
            stride: { label: 'Stride', icon: '👣', avg_stars: 4.0, badge: 'satisfactory', dimensions: [
                { key: 'stride_length', label: 'Stride Length', value: '0.74', unit: 'x Ht', stars: 4, badge: 'satisfactory', description: 'Good stride length relative to height.' },
                { key: 'forward_move', label: 'Forward Move', value: '94', unit: '%', stars: 4, badge: 'satisfactory', description: 'Efficient forward momentum.' },
            ]},
            power_move: { label: 'Power Move', icon: '💥', avg_stars: 4.3, badge: 'excellent', dimensions: [
                { key: 'max_hip_shoulder_separation', label: 'Max Hip-Shoulder Separation', value: '44.1', unit: '°', stars: 5, badge: 'excellent', description: 'Elite X-Factor separation.' },
                { key: 'pelvis_rotation_range', label: 'Pelvis Total Rotation Range', value: '68', unit: '°', stars: 4, badge: 'satisfactory', description: 'Good pelvis rotation range.' },
                { key: 'upper_torso_rotation_range', label: 'Upper Torso Rotation Range', value: '92', unit: '°', stars: 4, badge: 'satisfactory', description: 'Good shoulder rotation range.' },
            ]},
            contact: { label: 'Contact & Follow-Through', icon: '🎯', avg_stars: 3.5, badge: 'satisfactory', dimensions: [
                { key: 'pelvis_direction_at_contact', label: 'Pelvis Direction at Contact', value: '18', unit: '°', stars: 4, badge: 'satisfactory', description: 'Hips well open at contact.' },
                { key: 'upper_torso_direction_at_contact', label: 'Upper Torso Direction at Contact', value: '28', unit: '°', stars: 3, badge: 'satisfactory', description: 'Shoulders moderately open.' },
                { key: 'kinetic_chain_efficiency', label: 'Kinetic Chain Efficiency', value: '52.3', unit: '%', stars: 5, badge: 'excellent', description: 'Elite distal energy transfer.' },
                { key: 'sequence_quality', label: 'Sequence Quality', value: '48', unit: 'ms', stars: 4, badge: 'satisfactory', description: 'Good proximal-to-distal sequence timing.' },
                { key: 'hand_speed', label: 'Hand / Bat Speed', value: '24.2', unit: 'mph', stars: 4, badge: 'satisfactory', description: 'Above average hand speed for college level (Blast benchmark: 21-25 mph).' },
                { key: 'follow_through_quality', label: 'Follow-Through Quality', value: '42', unit: '°', stars: 3, badge: 'satisfactory', description: 'Adequate follow-through arc.' },
            ]},
        },
    },
    grf_estimation: {},
};

// Representative proximal-to-distal kinematic sequence for the demo swing.
// (Real analyses receive this from the backend as diagnosis.kinematic_sequence.)
DEMO_DIAGNOSIS.kinematic_sequence = (function () {
    const peaks = { 'Pelvis': [118, 705], 'Torso': [140, 930], 'Lead Arm': [158, 1120], 'Hands/Bat': [173, 1325] };
    const widths = { 'Pelvis': 50, 'Torso': 44, 'Lead Arm': 37, 'Hands/Bat': 31 };
    const time_ms = [], series = { 'Pelvis': [], 'Torso': [], 'Lead Arm': [], 'Hands/Bat': [] };
    for (let t = 0; t <= 215; t += 3) {
        time_ms.push(t);
        for (const k in peaks) {
            const [pt, pv] = peaks[k], w = widths[k];
            series[k].push(Math.round(pv * Math.exp(-((t - pt) ** 2) / (2 * w * w))));
        }
    }
    const peakObj = {};
    for (const k in peaks) peakObj[k] = { t_ms: peaks[k][0], value: peaks[k][1] };
    return { time_ms, series, peaks: peakObj, contact_ms: 173, units: 'deg/s' };
})();

document.addEventListener('DOMContentLoaded', () => {
    // -----------------------------------------
    // Elements
    // -----------------------------------------
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const trcInput = document.getElementById('trc-input');
    const trcLabel = document.getElementById('trc-label');

    const uploadSection = document.getElementById('upload-section');
    const resultsSection = document.getElementById('results-section');
    const btnBack = document.getElementById('btn-back');
    const loadingOverlay = document.getElementById('loading-overlay');
    
    const errorToast = document.getElementById('error-toast');
    const errorMessage = document.getElementById('error-message');
    const closeToast = document.getElementById('close-toast');
    
    const API_BASE = window.location.hostname.includes('github.io')
        ? 'https://swing-optimizer-pro.onrender.com'
        : '';

    // -----------------------------------------
    // State
    // -----------------------------------------
    let pendingUploadFile = null;
    let selectedSkillLevel = 'high_school';
    let lastAnalysis = null;

    // -----------------------------------------
    // Init
    // -----------------------------------------
    checkBackendHealth();
    initSkillPills();

    // -----------------------------------------
    // Skill Level Pills
    // -----------------------------------------
    function initSkillPills() {
        document.querySelectorAll('.skill-pill').forEach(pill => {
            pill.addEventListener('click', () => {
                document.querySelectorAll('.skill-pill').forEach(p => p.classList.remove('active'));
                pill.classList.add('active');
                selectedSkillLevel = pill.dataset.level;
            });
        });
    }

    // -----------------------------------------
    // Event Listeners
    // -----------------------------------------
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            promptDemographicsForUpload(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            promptDemographicsForUpload(e.target.files[0]);
        }
    });

    trcInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            trcLabel.textContent = `✓ ${e.target.files[0].name}`;
        }
    });

    document.getElementById('load-demo').addEventListener('click', () => {
        renderDashboard(DEMO_DIAGNOSIS, 'demo_swing.mot');
    });
    
    document.getElementById('btn-cancel-demo').addEventListener('click', hideDemoModal);
    document.getElementById('btn-run-physics').addEventListener('click', () => {
        hideDemoModal();
        if (pendingUploadFile) {
            handleUpload(pendingUploadFile);
        }
    });

    btnBack.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        fileInput.value = '';
    });
    
    closeToast.addEventListener('click', hideError);

    // Advanced toggle
    document.getElementById('toggle-advanced').addEventListener('click', () => {
        const panels = document.getElementById('advanced-panels');
        const icon = document.getElementById('toggle-icon');
        const isHidden = panels.classList.contains('hidden');
        panels.classList.toggle('hidden');
        icon.classList.toggle('open', isHidden);
    });

    // Export the current analysis as a clean, print-ready PDF report.
    document.getElementById('btn-report')?.addEventListener('click', exportReport);

    function exportReport() {
        if (!lastAnalysis) return;
        const { diagnosis, filename } = lastAnalysis;
        const rep = diagnosis.swingai_report || {};
        const m = diagnosis.metrics || {};
        const skillLabels = { youth: 'Youth', high_school: 'High School', college: 'College', professional: 'Professional' };
        const skill = skillLabels[rep.skill_level || selectedSkillLevel] || (rep.skill_level || '—');
        const val = id => { const e = document.getElementById(id); return e ? e.value : ''; };
        const num = (x, d) => (x || x === 0) ? Number(x).toFixed(d) : '—';

        const score = (diagnosis.swing_score != null) ? Math.round(diagnosis.swing_score) : '—';
        const eff = (diagnosis.efficiency_score != null) ? diagnosis.efficiency_score : '—';
        const hand = num(m.estimated_hand_speed_mph, 1);
        const pelvis = m.peak_pelvis_omega_3d_deg_s ? Math.round(m.peak_pelvis_omega_3d_deg_s) : '—';
        const date = new Date().toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' });
        const stars = n => '★'.repeat(n) + '☆'.repeat(Math.max(0, 5 - n));
        const badgeColor = b => b === 'excellent' ? '#0a7d4d' : b === 'off-target' ? '#c02626' : '#b6791b';
        const esc = s => String(s == null ? '' : s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

        let dimsHtml = '';
        ['balance_load', 'stride', 'power_move', 'contact'].forEach(k => {
            const ph = rep.phases && rep.phases[k];
            if (!ph) return;
            dimsHtml += `<tr class="phase-row"><td colspan="4">${esc(ph.label)}</td></tr>`;
            (ph.dimensions || []).forEach(d => {
                dimsHtml += `<tr><td>${esc(d.label)}</td>`
                    + `<td style="text-align:right;white-space:nowrap;">${esc(d.value)} ${esc(d.unit || '')}</td>`
                    + `<td style="color:#e0a800;letter-spacing:1px;white-space:nowrap;">${stars(d.stars || 0)}</td>`
                    + `<td style="color:${badgeColor(d.badge)};text-transform:capitalize;">${esc((d.badge || '').replace('-', ' '))}</td></tr>`;
            });
        });
        const list = arr => (arr || []).map(x => `<li>${esc(x)}</li>`).join('') || '<li>—</li>';

        const html = `
          <div class="pr-page">
            <div class="pr-head">
              <div>
                <div class="pr-brand">Swing Optimizer <span>Pro</span></div>
                <div class="pr-sub">Baseball Swing Biomechanics Report</div>
              </div>
              <div class="pr-meta"><div>${date}</div><div>${esc(filename || '')}</div></div>
            </div>
            <div class="pr-athlete">
              <span><b>Skill:</b> ${esc(skill)}</span>
              <span><b>Height:</b> ${esc(val('height-ft'))}'${esc(val('height-in'))}"</span>
              <span><b>Weight:</b> ${esc(val('weight-lbs'))} lb</span>
              <span><b>Age:</b> ${esc(val('athlete-age'))}</span>
              <span><b>Bat:</b> ${esc(val('bat-weight-oz'))} oz / ${esc(val('bat-length-in'))}"</span>
            </div>
            <div class="pr-scores">
              <div class="pr-score-main"><div class="pr-score-num">${score}</div><div class="pr-score-lbl">Swing Score / 100</div></div>
              <div class="pr-stat"><div>${hand}</div><span>Hand Speed (mph)</span></div>
              <div class="pr-stat"><div>${eff}</div><span>Efficiency Score</span></div>
              <div class="pr-stat"><div>${pelvis}</div><span>Pelvis Vel (°/s)</span></div>
            </div>
            <h3 class="pr-h3">12-Dimension Breakdown</h3>
            <table class="pr-table"><thead><tr><th>Dimension</th><th style="text-align:right;">Value</th><th>Rating</th><th>Status</th></tr></thead><tbody>${dimsHtml}</tbody></table>
            <div class="pr-cols">
              <div><h3 class="pr-h3">Mechanical Findings</h3><ul>${list(diagnosis.findings)}</ul></div>
              <div><h3 class="pr-h3">Prescriptions</h3><ul>${list(diagnosis.recommendations)}</ul></div>
            </div>
            <div class="pr-foot">Generated by Swing Optimizer Pro · For performance training use only — not a medical diagnosis.</div>
          </div>`;

        const host = document.getElementById('print-report');
        host.innerHTML = html;
        const prevTitle = document.title;
        document.title = 'SwingReport_' + String(filename || 'swing').replace(/\.[^.]+$/, '');
        window.addEventListener('afterprint', () => { document.title = prevTitle; }, { once: true });
        window.print();
    }

    // -----------------------------------------
    // API Calls
    // -----------------------------------------
    async function checkBackendHealth() {
        try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 15000);
            const res = await fetch(`${API_BASE}/api/health`, { signal: controller.signal });
            clearTimeout(timeout);
            if (!res.ok) throw new Error('not ok');
        } catch (err) {
            showError("Backend is waking up — this may take up to 60 seconds on first load. Please try your upload again shortly.");
        }
    }

    async function handleUpload(file) {
        if (!file.name.endsWith('.mot')) {
            showError("Please upload a .mot file");
            return;
        }
        
        showLoading();
        
        const formData = new FormData();
        formData.append('file', file);

        const trcFile = trcInput.files[0];
        if (trcFile) formData.append('trc_file', trcFile);

        const demo = getDemographics();
        formData.append('height_m', demo.height_m);
        formData.append('weight_kg', demo.weight_kg);
        formData.append('skill_level', selectedSkillLevel);
        formData.append('bat_mass_kg', demo.bat_mass_kg);
        formData.append('bat_length_m', demo.bat_length_m);
        
        const doUpload = () => fetch(`${API_BASE}/api/analyze/upload`, { method: 'POST', body: formData });
        try {
            let response;
            try {
                response = await doUpload();
            } catch (_) {
                // Backend may have been sleeping — wait and retry once
                await new Promise(r => setTimeout(r, 5000));
                response = await doUpload();
            }
            let data;
            try { data = await response.json(); }
            catch (_) { throw new Error(`Server error (HTTP ${response.status})`); }

            if (data.success) {
                renderDashboard(data.data, data.filename);
            } else {
                showError(data.error || "Analysis failed");
                hideLoading();
            }
        } catch (err) {
            showError("Network error — backend may still be waking up. Please try again in a moment.");
            console.error(err);
            hideLoading();
        }
    }
    
    // -----------------------------------------
    // Rendering
    // -----------------------------------------
    
    function renderDashboard(diagnosis, filename) {
        hideLoading();
        uploadSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        // Remember the latest analysis so the report can be exported to PDF.
        lastAnalysis = { diagnosis, filename };

        // Re-trigger animations
        document.querySelectorAll('.anim-slide-up').forEach(el => {
            el.style.animation = 'none';
            el.offsetHeight;
            el.style.animation = null; 
        });
        
        // Smoothly count a number up to its target (respects reduced-motion).
        function animateCount(el, target, decimals) {
            if (!el) return;
            target = Number(target) || 0;
            decimals = decimals || 0;
            const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
            if (reduce) { el.textContent = target.toFixed(decimals); return; }
            const duration = 900, start = performance.now();
            (function tick(now) {
                const t = Math.min((now - start) / duration, 1);
                const eased = 1 - Math.pow(1 - t, 3); // easeOutCubic
                el.textContent = (target * eased).toFixed(decimals);
                if (t < 1) requestAnimationFrame(tick);
                else el.textContent = target.toFixed(decimals);
            })(start);
        }

        // ---- SWING SCORE HERO ----
        document.getElementById('filename-display').textContent = filename;
        const swingScore = diagnosis.swing_score || 0;
        const effScore = diagnosis.efficiency_score || 0;
        const handSpeed = diagnosis.metrics.estimated_hand_speed_mph || 0;

        // Score ring animation
        const ringFill = document.getElementById('ring-fill');
        const circumference = 2 * Math.PI * 52;
        const dashVal = (swingScore / 100) * circumference;
        ringFill.style.strokeDasharray = `${dashVal} ${circumference}`;
        ringFill.classList.remove('clr-green', 'clr-yellow', 'clr-red');
        if (swingScore >= 70) ringFill.classList.add('clr-green');
        else if (swingScore >= 45) ringFill.classList.add('clr-yellow');
        else ringFill.classList.add('clr-red');

        animateCount(document.getElementById('swing-score-number'), swingScore, 0);
        animateCount(document.getElementById('exit-velo-number'), handSpeed, 1);
        animateCount(document.getElementById('efficiency-number'), effScore, 0);

        // Skill badge
        const skillLabels = {
            youth: 'Youth', high_school: 'High School',
            college: 'College', professional: 'Professional'
        };
        const skillLevel = diagnosis.swingai_report?.skill_level || selectedSkillLevel;
        document.getElementById('skill-badge-display').textContent = skillLabels[skillLevel] || skillLevel;

        // Hide data-warning (no longer needed without exit velo guess)
        document.getElementById('data-warning').classList.add('hidden');

        // ---- PELVIS ANGULAR VELOCITY ----
        const pelvisOmega = diagnosis.metrics?.peak_pelvis_omega_3d_deg_s || 0;
        const pelvisEl = document.getElementById('pelvis-omega-value');
        if (pelvisOmega > 0) animateCount(pelvisEl, pelvisOmega, 0); else pelvisEl.textContent = '—';

        // ---- KINEMATIC SEQUENCE ----
        // Prefer the backend's computed sequence; otherwise derive it client-side
        // from the per-frame skeleton joints already in the response.
        renderKinematicSequence(diagnosis.kinematic_sequence || computeSequenceFromSkeleton(diagnosis.skeleton_frames));

        // ---- SWINGAI 4-PHASE CARDS ----
        if (diagnosis.swingai_report) {
            renderSwingAIReport(diagnosis.swingai_report);
            setTimeout(() => init3DSkeleton(diagnosis.skeleton_frames), 100);
            attachSkeletonClickHandlers();
        }

        // ---- FINDINGS & RECOMMENDATIONS ----
        const findingsList = document.getElementById('findings-list');
        const recList = document.getElementById('recommendation-list');
        
        findingsList.innerHTML = '';
        diagnosis.findings.forEach(f => {
            const li = document.createElement('li');
            li.textContent = f;
            findingsList.appendChild(li);
        });
        
        recList.innerHTML = '';
        diagnosis.recommendations.forEach(r => {
            const li = document.createElement('li');
            li.textContent = r;
            recList.appendChild(li);
        });

        // ---- ADVANCED PHYSICS ----
        const m = diagnosis.metrics;
        let handSpeedHtml = '';
        if (m.max_hand_speed_mph > 0) {
            handSpeedHtml = createMetric('Max Hand Speed', m.max_hand_speed_mph.toFixed(1), 'mph');
        }
        
        document.getElementById('rotational-metrics').innerHTML = `
            ${createMetric('Max Separation', (m.max_separation_deg || 0).toFixed(1), '°')}
            ${handSpeedHtml}
            ${createMetric('Peak Hip Power', (m.peak_hip_power_W || 0).toFixed(0), 'W')}
            ${createMetric('Rel. Hip Power', (m.hip_power_per_kg || 0).toFixed(1), 'W/kg')}
            ${createMetric('Sequence Timing', (m.sequence_timing_ms || 0).toFixed(0), 'ms')}
            ${createMetric('Chain Efficiency', (m.kinetic_chain_efficiency_pct || 0).toFixed(1), '%')}
            ${createMetric('Torso/Pelvis Ratio', (m.torso_to_pelvis_rot_ratio || 0).toFixed(2), '')}
            ${createMetric('Total Chain KE', (m.total_energy_transfer_J || 0).toFixed(0), 'J')}
            ${m.time_to_contact_s > 0 ? createMetric('Time to Contact', (m.time_to_contact_s * 1000).toFixed(0), 'ms') : ''}
            ${m.rotational_acceleration_deg_s2 > 0 ? createMetric('Rotational Accel', (m.rotational_acceleration_deg_s2 / 1000).toFixed(1), 'k°/s²') : ''}
            ${m.body_rotation_ratio_pct > 0 ? createMetric('Body Rotation Ratio', (m.body_rotation_ratio_pct || 0).toFixed(1), '%') : ''}
        `;

        const grf = diagnosis.grf_estimation || {};
        document.getElementById('stride-metrics').innerHTML = `
            ${createMetric('Stride Efficiency', (m.stride_efficiency_pct || 0).toFixed(0), '%')}
            ${createMetric('Stride Ratio', (m.stride_ratio || 0).toFixed(2), 'x Ht')}
            ${createMetric('Proper Sequence', m.proper_sequence ? 'YES ✅' : 'NO ❌', '', true)}
            ${createMetric('Pelvis KE', (m.pelvis_ke_J || 0).toFixed(1), 'J')}
            ${createMetric('Torso KE', (m.torso_ke_J || 0).toFixed(1), 'J')}
            ${createMetric('Arm KE', (m.arm_ke_J || 0).toFixed(1), 'J')}
            ${createMetric('Bat KE', (m.bat_ke_J || 0).toFixed(1), 'J')}
            ${grf.peak_grf_vert_BW ? createMetric('Peak GRF Vert', (grf.peak_grf_vert_BW * 100).toFixed(0), '% BW') : ''}
            ${grf.peak_grf_ap_N ? createMetric('Peak GRF AP', grf.peak_grf_ap_N.toFixed(0), 'N') : ''}
        `;

        // Reset advanced panel state
        document.getElementById('advanced-panels').classList.add('hidden');
        document.getElementById('toggle-icon').classList.remove('open');

        saveToHistory({ filename, date: new Date().toLocaleDateString(), score: swingScore, efficiency: effScore, handSpeed: handSpeed.toFixed(1), skill: skillLabels[skillLevel] || skillLevel });
        renderHistory();
    }

    // -----------------------------------------
    // Longitudinal History (localStorage)
    // -----------------------------------------
    const HISTORY_KEY = 'swingopt_history';

    function saveToHistory(entry) {
        const history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
        history.push(entry);
        if (history.length > 50) history.shift(); // cap at 50 sessions
        localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    }

    function renderHistory() {
        const history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
        const panel = document.getElementById('history-panel');
        if (history.length < 2) { panel.style.display = 'none'; return; }
        panel.style.display = '';

        // Sparkline
        const canvas = document.getElementById('history-chart');
        const ctx = canvas.getContext('2d');
        canvas.width = canvas.offsetWidth || 600;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const scores = history.map(h => h.score);
        const minS = Math.min(...scores), maxS = Math.max(...scores);
        const range = maxS - minS || 1;
        const W = canvas.width, H = canvas.height, pad = 12;
        ctx.strokeStyle = 'var(--accent, #00d4ff)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        scores.forEach((s, i) => {
            const x = pad + (i / (scores.length - 1)) * (W - 2 * pad);
            const y = H - pad - ((s - minS) / range) * (H - 2 * pad);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
        // Dots
        scores.forEach((s, i) => {
            const x = pad + (i / (scores.length - 1)) * (W - 2 * pad);
            const y = H - pad - ((s - minS) / range) * (H - 2 * pad);
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fillStyle = 'var(--accent, #00d4ff)';
            ctx.fill();
        });

        // Table (last 10)
        const recent = history.slice(-10).reverse();
        document.getElementById('history-table').innerHTML = `
            <table style="width:100%;border-collapse:collapse;">
                <thead><tr style="color:var(--text-muted,#888);text-align:left;">
                    <th style="padding:4px 8px;">Date</th>
                    <th style="padding:4px 8px;">File</th>
                    <th style="padding:4px 8px;">Level</th>
                    <th style="padding:4px 8px;">Score</th>
                    <th style="padding:4px 8px;">Efficiency</th>
                    <th style="padding:4px 8px;">Hand Speed</th>
                </tr></thead>
                <tbody>${recent.map(h => `
                    <tr style="border-top:1px solid rgba(255,255,255,0.07);">
                        <td style="padding:4px 8px;">${h.date}</td>
                        <td style="padding:4px 8px;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${h.filename}</td>
                        <td style="padding:4px 8px;">${h.skill}</td>
                        <td style="padding:4px 8px;font-weight:600;">${h.score}</td>
                        <td style="padding:4px 8px;">${h.efficiency}</td>
                        <td style="padding:4px 8px;">${h.handSpeed} mph</td>
                    </tr>`).join('')}
                </tbody>
            </table>`;

        document.getElementById('clear-history').onclick = () => {
            localStorage.removeItem(HISTORY_KEY);
            panel.style.display = 'none';
        };
    }

    // -----------------------------------------
    // Drill / Cue Tags per dimension
    // -----------------------------------------
    const DIM_CUES = {
        'negative_move':                    ['Hip hinge load drill', 'Toe tap rhythm'],
        'pelvis_load':                      ['Hip load & coil', 'Resistance band hip load'],
        'upper_torso_load':                 ['Shoulder coil drill', 'Bat behind back rotation'],
        'stride_length':                    ['Stride length marker drill', 'Soft front foot landing'],
        'forward_move':                     ['Linear momentum drill', 'Step-and-hit tee work'],
        'max_hip_shoulder_separation':      ['X-Factor stretch drill', 'Hip-shoulder separation tee'],
        'pelvis_rotation_range':            ['Hip turn drill', 'Pivot & rotate med ball'],
        'upper_torso_rotation_range':       ['Shoulder turn drill', 'Rotational med ball throw'],
        'pelvis_direction_at_contact':      ['Hip clearing drill', 'Open hips at contact cue'],
        'upper_torso_direction_at_contact': ['Shoulder square drill', 'Contact point extension'],
        'kinetic_chain_efficiency':         ['Whip drill', 'Decelerate hips — fire torso'],
        'sequence_quality':                 ['Sequence timing drill', 'Pelvis-first cue'],
        'hand_speed':                       ['Bat speed overload/underload', 'Wrist snap drill'],
        'follow_through_quality':           ['Full finish drill', 'High hands follow-through'],
    };

    // -----------------------------------------
    // Kinematic Sequence Chart (Canvas 2D)
    // -----------------------------------------
    const KINE_COLORS = { 'Pelvis': '#38bdf8', 'Torso': '#34d399', 'Lead Arm': '#fbbf24', 'Hands/Bat': '#f472b6' };
    const KINE_ORDER = ['Pelvis', 'Torso', 'Lead Arm', 'Hands/Bat'];
    let lastKineSeq = null;

    // Fallback: derive a kinematic sequence from the 3D skeleton frames
    // (per-frame joint positions) when the backend didn't supply one. Each
    // segment's rotation about the vertical axis is differentiated into deg/s.
    function computeSequenceFromSkeleton(sk) {
        if (!sk || !Array.isArray(sk.frames) || sk.frames.length < 5) return null;
        const frames = sk.frames;
        const fps = (sk.fps && sk.fps > 0) ? sk.fps : 60;
        const dt = 1 / fps;
        const segs = {
            'Pelvis': ['LHip', 'RHip'],
            'Torso': ['LShoulder', 'RShoulder'],
            'Lead Arm': ['LShoulder', 'LElbow'],
            'Hands/Bat': ['LElbow', 'LWrist'],
        };
        const horizAngle = (f, a, b) => {
            const pa = f[a], pb = f[b];
            if (!pa || !pb) return null;
            return Math.atan2(pb[2] - pa[2], pb[0] - pa[0]); // rotation about vertical (Y up)
        };
        const series = {}, peaks = {}, present = [];
        for (const k in segs) {
            const [a, b] = segs[k];
            const ang = [];
            let ok = true;
            for (const f of frames) { const v = horizAngle(f, a, b); if (v === null) { ok = false; break; } ang.push(v); }
            if (!ok) continue;
            for (let i = 1; i < ang.length; i++) {
                while (ang[i] - ang[i - 1] > Math.PI) ang[i] -= 2 * Math.PI;
                while (ang[i] - ang[i - 1] < -Math.PI) ang[i] += 2 * Math.PI;
            }
            const w = ang.map((_, i) => {
                let d;
                if (i === 0) d = (ang[1] - ang[0]) / dt;
                else if (i === ang.length - 1) d = (ang[i] - ang[i - 1]) / dt;
                else d = (ang[i + 1] - ang[i - 1]) / (2 * dt);
                return Math.abs(d * 180 / Math.PI);
            });
            const sm = w.map((_, i) => (w[Math.max(0, i - 1)] + w[i] + w[Math.min(w.length - 1, i + 1)]) / 3);
            series[k] = sm.map(x => Math.round(x));
            present.push(k);
        }
        if (!present.length) return null;
        const time_ms = frames.map((_, i) => Math.round(i * dt * 1000));
        for (const k of present) {
            let pi = 0;
            for (let i = 1; i < series[k].length; i++) if (series[k][i] > series[k][pi]) pi = i;
            peaks[k] = { t_ms: time_ms[pi], value: series[k][pi] };
        }
        const cf = sk.contact_frame;
        const contact_ms = (cf != null && time_ms[cf] != null) ? time_ms[cf]
            : (peaks['Hands/Bat'] || peaks['Lead Arm'] || peaks['Pelvis']).t_ms;
        return { time_ms, series, peaks, contact_ms, units: 'deg/s' };
    }

    function renderKinematicSequence(seq) {
        const panel = document.getElementById('kinematic-panel');
        if (!seq || !seq.series || !seq.time_ms || !seq.time_ms.length) {
            lastKineSeq = null;
            panel.style.display = 'none';
            return;
        }
        panel.style.display = '';
        lastKineSeq = seq;

        // Legend
        const present = KINE_ORDER.filter(k => Array.isArray(seq.series[k]));
        document.getElementById('kinematic-legend').innerHTML = present.map(k => {
            const pk = seq.peaks && seq.peaks[k];
            const detail = pk ? ` <span class="kine-peak">${Math.round(pk.value)}°/s @ ${Math.round(pk.t_ms)}ms</span>` : '';
            return `<span class="kine-chip"><span class="kine-dot" style="background:${KINE_COLORS[k]}"></span>${k}${detail}</span>`;
        }).join('');

        // Caption: is the sequence proper proximal-to-distal?
        const times = present.map(k => seq.peaks && seq.peaks[k] ? seq.peaks[k].t_ms : null).filter(t => t != null);
        let proper = times.length === present.length;
        for (let i = 1; i < times.length; i++) if (times[i] <= times[i - 1]) proper = false;
        const cap = document.getElementById('kinematic-caption');
        cap.innerHTML = proper
            ? '<span class="val-good" style="font-weight:700;">✓ Proper sequence</span> <span class="text-muted">— segments peak in order, building speed up the chain.</span>'
            : '<span class="val-warn" style="font-weight:700;">⚠ Sequence flag</span> <span class="text-muted">— segments do not peak strictly proximal-to-distal; energy may leak out of the chain.</span>';

        const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (reduce) { drawKine(1); return; }
        const start = performance.now(), dur = 900;
        (function step(now) {
            const p = Math.min((now - start) / dur, 1);
            drawKine(1 - Math.pow(1 - p, 3));
            if (p < 1 && lastKineSeq === seq) requestAnimationFrame(step);
        })(start);
    }

    function drawKine(progress) {
        const seq = lastKineSeq;
        const canvas = document.getElementById('kinematic-chart');
        if (!seq || !canvas) return;
        const wrap = canvas.parentElement;
        const cssW = wrap.clientWidth || 600, cssH = 300;
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        canvas.width = cssW * dpr; canvas.height = cssH * dpr;
        canvas.style.width = cssW + 'px'; canvas.style.height = cssH + 'px';
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, cssW, cssH);

        const padL = 46, padR = 14, padT = 14, padB = 32;
        const plotW = cssW - padL - padR, plotH = cssH - padT - padB;
        const present = KINE_ORDER.filter(k => Array.isArray(seq.series[k]));
        const tMax = seq.time_ms[seq.time_ms.length - 1] || 1;
        let vMax = 0;
        present.forEach(k => seq.series[k].forEach(v => { if (v > vMax) vMax = v; }));
        vMax = vMax * 1.08 || 1;
        const X = t => padL + (t / tMax) * plotW;
        const Y = v => padT + plotH - (v / vMax) * plotH;
        const css = getComputedStyle(document.documentElement);
        const muted = (css.getPropertyValue('--text-muted') || '#94a3b8').trim();

        // Gridlines + Y labels (deg/s)
        ctx.font = '10px Inter, sans-serif'; ctx.textBaseline = 'middle';
        const ySteps = 4;
        for (let i = 0; i <= ySteps; i++) {
            const v = (vMax / ySteps) * i, y = Y(v);
            ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(cssW - padR, y); ctx.stroke();
            ctx.fillStyle = muted; ctx.textAlign = 'right';
            ctx.fillText(Math.round(v), padL - 6, y);
        }
        // X labels (ms)
        ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        const xSteps = 5;
        for (let i = 0; i <= xSteps; i++) {
            const t = (tMax / xSteps) * i;
            ctx.fillStyle = muted; ctx.fillText(Math.round(t), X(t), cssH - padB + 8);
        }
        ctx.fillText('Time (ms)', padL + plotW / 2, cssH - 12);
        ctx.save(); ctx.translate(12, padT + plotH / 2); ctx.rotate(-Math.PI / 2);
        ctx.fillText('Angular velocity (°/s)', 0, 0); ctx.restore();

        // Contact line
        if (seq.contact_ms != null) {
            const cx = X(seq.contact_ms);
            ctx.strokeStyle = 'rgba(255,255,255,0.28)'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
            ctx.beginPath(); ctx.moveTo(cx, padT); ctx.lineTo(cx, padT + plotH); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = muted; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            ctx.fillText('contact', cx, padT + 1);
        }

        const tReveal = tMax * progress;
        present.forEach(k => {
            const data = seq.series[k], times = seq.time_ms;
            ctx.strokeStyle = KINE_COLORS[k]; ctx.lineWidth = 2.2;
            ctx.lineJoin = 'round'; ctx.beginPath();
            let started = false;
            for (let i = 0; i < times.length; i++) {
                if (times[i] > tReveal) break;
                const x = X(times[i]), y = Y(data[i]);
                if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
            }
            ctx.stroke();
            // Peak marker (once revealed)
            const pk = seq.peaks && seq.peaks[k];
            if (pk && pk.t_ms <= tReveal) {
                ctx.fillStyle = KINE_COLORS[k];
                ctx.beginPath(); ctx.arc(X(pk.t_ms), Y(pk.value), 3.5, 0, Math.PI * 2); ctx.fill();
                ctx.strokeStyle = 'rgba(8,11,18,0.9)'; ctx.lineWidth = 1.5; ctx.stroke();
            }
        });
    }

    // Redraw on resize (debounced) so the chart stays crisp & responsive.
    let kineResizeTimer = null;
    window.addEventListener('resize', () => {
        if (!lastKineSeq) return;
        clearTimeout(kineResizeTimer);
        kineResizeTimer = setTimeout(() => drawKine(1), 150);
    });

    // -----------------------------------------
    // SwingAI Report Renderer
    // -----------------------------------------
    const PHASE_ORDER = ['balance_load', 'stride', 'power_move', 'contact'];

    function renderSwingAIReport(report) {
        const grid = document.getElementById('phase-cards-grid');
        grid.innerHTML = '';

        PHASE_ORDER.forEach((phaseKey, i) => {
            const phase = report.phases[phaseKey];
            if (!phase) return;
            const card = buildPhaseCard(phase, i);
            grid.appendChild(card);
        });
    }

    function buildPhaseCard(phase, index) {
        const card = document.createElement('div');
        card.className = 'phase-card';
        card.style.animationDelay = `${0.05 + index * 0.08}s`;

        const phaseStars = renderStars(Math.round(phase.avg_stars));
        const badgeClass = badgeCssClass(phase.badge);
        
        card.innerHTML = `
            <div class="phase-header">
                <div class="phase-header-left">
                    <span class="phase-label">${phase.label}</span>
                </div>
                <div class="phase-avg-stars">${phaseStars}</div>
            </div>
            <div class="phase-dimensions" id="phase-dims-${index}"></div>
        `;

        const dimsContainer = card.querySelector(`#phase-dims-${index}`);
        phase.dimensions.forEach(dim => {
            dimsContainer.appendChild(buildDimTile(dim));
        });

        return card;
    }

    function buildDimTile(dim) {
        const tile = document.createElement('div');
        tile.className = 'dim-tile';
        if (dim.key) tile.dataset.dimKey = dim.key;

        const badgeClass = badgeCssClass(dim.badge);
        const pillLabel = pillText(dim.badge);
        const stars = renderStars(dim.stars);

        const cues = (dim.badge !== 'excellent' && DIM_CUES[dim.key])
            ? `<div class="dim-cues">${DIM_CUES[dim.key].map(c => `<span class="dim-cue-tag">${c}</span>`).join('')}</div>`
            : '';

        tile.innerHTML = `
            <div class="dim-badge ${badgeClass}"></div>
            <div class="dim-info">
                <div class="dim-name">${dim.label}</div>
                <div class="dim-value">${dim.value} ${dim.unit}</div>
                ${cues}
            </div>
            <div class="dim-stars">${stars}</div>
            <div class="dim-pill ${badgeClass}">${pillLabel}</div>
            <div class="dim-tooltip">${dim.description}</div>
        `;

        return tile;
    }

    function renderStars(filled) {
        filled = Math.max(1, Math.min(5, filled));
        let html = '';
        for (let i = 1; i <= 5; i++) {
            html += i <= filled
                ? '<span class="star-filled">★</span>'
                : '<span class="star-empty">★</span>';
        }
        return html;
    }

    function badgeCssClass(badge) {
        const map = {
            'excellent': 'badge-excellent',
            'satisfactory': 'badge-satisfactory',
            'off_target': 'badge-off-target',
        };
        return map[badge] || 'badge-satisfactory';
    }

    function pillText(badge) {
        const map = {
            'excellent': 'Excellent',
            'satisfactory': 'Good',
            'off_target': 'Off Target',
        };
        return map[badge] || badge;
    }

    // -----------------------------------------
    // Advanced Metric Tile
    // -----------------------------------------
    function createMetric(label, value, unit, isText = false) {
        let valClass = '';
        if (!isText) {
            const num = parseFloat(value);
            if (!isNaN(num)) {
                if (label.includes('Efficiency') && num < 70) valClass = 'val-bad';
                else if (label.includes('Timing') && num < 20) valClass = 'val-warn';
            }
        }
        if (value === 'NO') valClass = 'val-bad';
        if (value === 'YES') valClass = 'val-good';

        return `
        <div class="metric-item">
            <div class="metric-label">${label}</div>
            <div class="metric-value ${valClass}">${value}<span style="font-size:0.6em; margin-left:2px">${unit}</span></div>
        </div>
        `;
    }

    // -----------------------------------------
    // Utilities
    // -----------------------------------------
    function promptDemographicsForUpload(file) {
        if (!file.name.endsWith('.mot')) {
            showError("Please upload a .mot file");
            return;
        }
        pendingUploadFile = file;
        showDemoModal();
    }

    function showDemoModal() { document.getElementById('demo-modal').classList.remove('hidden'); }
    function hideDemoModal() { document.getElementById('demo-modal').classList.add('hidden'); }
    
    // -----------------------------------------
    // Body Skeleton Canvas (static marker positions, highlight on metric click)
    // -----------------------------------------

    // Hardcoded marker positions — mid-swing contact pose (right-handed batter, front view)
    // Pelvis open to pitcher, lead arm extended forward, back arm bent, slight squat
    // Bones to highlight per metric key
    const METRIC_HIGHLIGHTS = {
        'pelvis_load':                      { bones: [['midHip','RHip'],['midHip','LHip']], color: '#f59e0b', desc: 'Pelvis — hip rotational energy storage during load.' },
        'negative_move':                    { bones: [['midHip','RHip'],['midHip','LHip'],['RHip','RKnee'],['LHip','LKnee']], color: '#a78bfa', desc: 'Pelvis & legs — backward weight shift before stride.' },
        'upper_torso_load':                 { bones: [['Neck','RShoulder'],['Neck','LShoulder'],['Neck','midHip']], color: '#8b5cf6', desc: 'Torso & shoulders — upper body coil tension at load.' },
        'stride_length':                    { bones: [['LHip','LKnee'],['LKnee','LAnkle']], color: '#fbbf24', desc: 'Lead leg — stride length from load to foot plant.' },
        'forward_move':                     { bones: [['midHip','LHip'],['LHip','LKnee']], color: '#fbbf24', desc: 'Pelvis & lead leg — forward linear momentum into the ball.' },
        'max_hip_shoulder_separation':      { bones: [['midHip','RHip'],['midHip','LHip'],['Neck','RShoulder'],['Neck','LShoulder']], color: '#10b981', desc: 'Pelvis vs shoulders — X-Factor separation angle.' },
        'pelvis_rotation_range':            { bones: [['midHip','RHip'],['midHip','LHip']], color: '#f59e0b', desc: 'Pelvis — total axial rotation from load to contact.' },
        'upper_torso_rotation_range':       { bones: [['Neck','RShoulder'],['Neck','LShoulder'],['Neck','midHip']], color: '#8b5cf6', desc: 'Torso & shoulders — total rotation from load to contact.' },
        'pelvis_direction_at_contact':      { bones: [['midHip','RHip'],['midHip','LHip']], color: '#ef4444', desc: 'Pelvis — alignment at front foot plant. Should be square to pitcher.' },
        'upper_torso_direction_at_contact': { bones: [['Neck','RShoulder'],['Neck','LShoulder']], color: '#ef4444', desc: 'Shoulders — alignment at contact.' },
        'kinetic_chain_efficiency':         { bones: [['midHip','RHip'],['midHip','LHip'],['Neck','midHip'],['RShoulder','RElbow'],['RElbow','RWrist']], color: '#06b6d4', desc: 'Full kinetic chain — energy flow from pelvis through torso to hands.' },
        'sequence_quality':                 { bones: [['midHip','RHip'],['midHip','LHip'],['Neck','midHip'],['RShoulder','RElbow']], color: '#06b6d4', desc: 'Proximal-to-distal sequence: pelvis → torso → arm.' },
        'hand_speed':                       { bones: [['RShoulder','RElbow'],['RElbow','RWrist'],['LShoulder','LElbow'],['LElbow','LWrist']], color: '#10b981', desc: 'Hands & forearms — peak bat/hand speed at contact.' },
        'follow_through_quality':           { bones: [['midHip','RHip'],['midHip','LHip'],['Neck','midHip']], color: '#f97316', desc: 'Pelvis & torso — deceleration arc after contact.' },
    };

    // ── 3D SKELETON ─────────────────────────────────────────────────────────
    let skeletonScene = null;

    const BONE_CONNECTIONS = [
        ['Neck','RShoulder'],['Neck','LShoulder'],
        ['RShoulder','RElbow'],['RElbow','RWrist'],
        ['LShoulder','LElbow'],['LElbow','LWrist'],
        ['Neck','midHip'],
        ['midHip','RHip'],['midHip','LHip'],
        ['RHip','RKnee'],['RKnee','RAnkle'],
        ['LHip','LKnee'],['LKnee','LAnkle'],
    ];

    // Visual palette for the skeleton (kept brighter than the panel bg for contrast)
    const SKEL_BONE = 0x8b97b8;     // limb / segment colour
    const SKEL_JOINT = 0xe2e8f5;    // bright mocap-style joint markers
    const SKEL_HEAD = 0x9aa6c8;
    const SKEL_BAT = 0xc9a06a;      // wood bat
    const JOINT_NAMES = ['Neck','RShoulder','LShoulder','RElbow','LElbow','RWrist','LWrist','midHip','RHip','LHip','RKnee','LKnee','RAnkle','LAnkle'];

    // Anatomically-tapered limb radii (metres) keyed by "proximal|distal".
    const BONE_RADII = {
        'Neck|midHip':[0.05,0.066],                                   // trunk
        'Neck|RShoulder':[0.03,0.028],'Neck|LShoulder':[0.03,0.028],  // clavicle
        'RShoulder|RElbow':[0.034,0.025],'LShoulder|LElbow':[0.034,0.025],
        'RElbow|RWrist':[0.025,0.018],'LElbow|LWrist':[0.025,0.018],
        'midHip|RHip':[0.055,0.045],'midHip|LHip':[0.055,0.045],
        'RHip|RKnee':[0.058,0.04],'LHip|LKnee':[0.058,0.04],          // thigh
        'RKnee|RAnkle':[0.04,0.026],'LKnee|LAnkle':[0.04,0.026],      // shank
        'RAnkle|RToe':[0.03,0.02],'LAnkle|LToe':[0.03,0.02],          // foot
    };
    const JOINT_RADII = {
        midHip:0.05, RShoulder:0.04, LShoulder:0.04, RHip:0.044, LHip:0.044,
        RKnee:0.04, LKnee:0.04, RElbow:0.034, LElbow:0.034,
        RWrist:0.032, LWrist:0.032, RAnkle:0.036, LAnkle:0.036, Neck:0.034,
    };

    // Default reference pose: a right-handed hitter at contact (hips fired open,
    // lead leg braced, hands extended out front, bat through the zone). Metres,
    // Y up, +Z toward the pitcher. Shown when no captured marker data is present.
    const RIGHTY_SWING_POSE = {
        midHip:[0,0.92,0], Neck:[-0.05,1.46,-0.02],
        RHip:[0.13,0.93,-0.05], LHip:[-0.13,0.91,0.05],
        RKnee:[0.16,0.48,-0.16], LKnee:[-0.16,0.50,0.10],
        RAnkle:[0.22,0.07,-0.30], LAnkle:[-0.18,0.05,0.16],
        RToe:[0.24,0.02,-0.16], LToe:[-0.18,0.03,0.30],
        RShoulder:[0.18,1.44,-0.10], LShoulder:[-0.20,1.42,0.06],
        RElbow:[0.20,1.18,-0.02], LElbow:[0.05,1.22,0.14],
        RWrist:[0.31,1.16,0.18], LWrist:[0.28,1.14,0.18],
    };
    const BAT_TIP = [0.64,1.42,0.30];

    function init3DSkeleton(skeletonFrames) {
        const container = document.getElementById('skeleton-3d');
        if (!container) return;
        if (typeof THREE === 'undefined') {
            container.innerHTML = '<p class="small text-muted" style="padding:1rem;text-align:center;">3D viewer unavailable — could not load Three.js (check your connection, then reload).</p>';
            return;
        }
        container.innerHTML = '';

        const W = container.clientWidth || 260;
        const H = container.clientHeight || 460;

        let renderer;
        try {
            renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        } catch (err) {
            console.error('[skeleton] WebGL unavailable:', err);
            container.innerHTML = '<p class="small text-muted" style="padding:1rem;text-align:center;">3D viewer needs WebGL, which appears to be disabled or unsupported in this browser.</p>';
            return;
        }
        renderer.setSize(W, H);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(renderer.domElement);

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, W / H, 0.001, 5000);
        camera.position.set(0, 0, 5);

        // pivot rotates on drag; figure holds the meshes and is recentered each build
        const pivot = new THREE.Group();
        const figure = new THREE.Group();
        pivot.add(figure);
        scene.add(pivot);

        let rotX = 0.1, rotY = 0.3;
        pivot.rotation.set(rotX, rotY, 0);

        function render() { renderer.render(scene, camera); }

        // Orbit controls via mouse / touch drag — re-render on move so it updates
        // even if the rAF loop is throttled (e.g. background tab).
        let isDragging = false, prevX = 0, prevY = 0;
        function startDrag(x, y) { isDragging = true; prevX = x; prevY = y; container.style.cursor = 'grabbing'; }
        function moveDrag(x, y) {
            if (!isDragging) return;
            rotY += (x - prevX) * 0.01;
            rotX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotX + (y - prevY) * 0.01));
            prevX = x; prevY = y;
            pivot.rotation.y = rotY; pivot.rotation.x = rotX;
            render();
        }
        function endDrag() { isDragging = false; container.style.cursor = 'grab'; }
        renderer.domElement.addEventListener('mousedown', e => startDrag(e.clientX, e.clientY));
        renderer.domElement.addEventListener('touchstart', e => { startDrag(e.touches[0].clientX, e.touches[0].clientY); }, { passive: true });
        window.addEventListener('mouseup', endDrag);
        window.addEventListener('touchend', endDrag);
        window.addEventListener('mousemove', e => moveDrag(e.clientX, e.clientY));
        window.addEventListener('touchmove', e => { if (isDragging) { moveDrag(e.touches[0].clientX, e.touches[0].clientY); e.preventDefault(); } }, { passive: false });

        const contactFrame = skeletonFrames?.contact_frame || 0;
        const frames = skeletonFrames?.frames || null;

        function getPose(frameIdx) {
            if (!frames || !frames.length) return null;
            return frames[Math.min(frameIdx, frames.length - 1)];
        }

        // Detect coordinate scale once (TRC markers in mm ~hundreds/thousands → m).
        const sampleVals = frames && frames.length
            ? Object.values(frames[0]).filter(Array.isArray).flat().map(Math.abs) : [];
        const maxAbs = sampleVals.length ? Math.max(...sampleVals) : 0;
        const scale = maxAbs > 10 ? 0.001 : 1.0;

        function toV3(pose, name) {
            const p = pose && pose[name];
            return p ? new THREE.Vector3(p[0] * scale, p[1] * scale, p[2] * scale) : null;
        }

        const Y_AXIS = new THREE.Vector3(0, 1, 0);

        // A tapered, lit limb segment between two joints (rA = radius at va end).
        function addBone(va, vb, rA, rB, color, hl) {
            const dir = new THREE.Vector3().subVectors(vb, va);
            const len = dir.length();
            if (len < 1e-6) return;
            const k = hl ? 1.5 : 1;
            const geo = new THREE.CylinderGeometry(rB * k, rA * k, len, 14, 1);
            const mat = new THREE.MeshStandardMaterial({
                color, roughness: 0.55, metalness: 0.12,
                emissive: hl ? new THREE.Color(color) : 0x000000, emissiveIntensity: hl ? 0.45 : 0,
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.copy(va).add(vb).multiplyScalar(0.5);
            mesh.quaternion.setFromUnitVectors(Y_AXIS, dir.normalize());
            figure.add(mesh);
        }

        // A spherical joint marker (mocap-style).
        function addJoint(v, r, color, hl) {
            const mat = new THREE.MeshStandardMaterial({
                color, roughness: 0.35, metalness: 0.2,
                emissive: hl ? new THREE.Color(color) : 0x000000, emissiveIntensity: hl ? 0.55 : 0,
            });
            const mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 16), mat);
            mesh.position.copy(v);
            figure.add(mesh);
        }

        // Draw every segment of a pose-like map {jointName: Vector3}, with taper.
        function addBonesFor(getV, connections, hlBones, hlColor) {
            for (const [a, b] of connections) {
                const va = getV(a), vb = getV(b);
                if (!va || !vb) continue;
                const isHL = hlBones && (hlBones.has(`${a}|${b}`) || hlBones.has(`${b}|${a}`));
                const [rA, rB] = BONE_RADII[`${a}|${b}`] || [0.022, 0.018];
                addBone(va, vb, rA, rB, isHL ? new THREE.Color(hlColor) : SKEL_BONE, isHL);
            }
        }

        // Joint markers + a hand bulge at each wrist + an ellipsoid head along the spine.
        function addJointsAndExtras(getV, hlBones, hlColor) {
            for (const name of JOINT_NAMES) {
                const v = getV(name);
                if (!v) continue;
                const isHL = hlBones && [...(hlBones || [])].some(k => k.includes(name));
                addJoint(v, (JOINT_RADII[name] || 0.03) * (isHL ? 1.4 : 1), isHL ? new THREE.Color(hlColor) : SKEL_JOINT, isHL);
            }
            // Hands: a small fist just past each wrist along the forearm.
            for (const [w, e] of [['RWrist', 'RElbow'], ['LWrist', 'LElbow']]) {
                const vw = getV(w), ve = getV(e);
                if (!vw) continue;
                const reach = ve ? new THREE.Vector3().subVectors(vw, ve).normalize().multiplyScalar(0.05) : new THREE.Vector3();
                addJoint(new THREE.Vector3().addVectors(vw, reach), 0.042, SKEL_JOINT, false);
            }
            const neck = getV('Neck'), hip = getV('midHip');
            if (neck) {
                const up = hip ? new THREE.Vector3().subVectors(neck, hip).normalize() : Y_AXIS.clone();
                const head = new THREE.Mesh(new THREE.SphereGeometry(0.092, 22, 22),
                    new THREE.MeshStandardMaterial({ color: SKEL_HEAD, roughness: 0.5, metalness: 0.1 }));
                head.scale.set(0.84, 1.08, 0.84);
                head.quaternion.setFromUnitVectors(Y_AXIS, up);
                head.position.copy(neck).add(up.clone().multiplyScalar(0.14));
                figure.add(head);
            }
        }

        function buildSkeleton(pose, hlBones, hlColor) {
            while (figure.children.length) figure.remove(figure.children[0]);

            if (pose) {
                const getV = (n) => toV3(pose, n);
                addBonesFor(getV, BONE_CONNECTIONS, hlBones, hlColor);
                addJointsAndExtras(getV, hlBones, hlColor);
            }
            // Safety net: no pose, or captured data had no joints matching the
            // expected marker names → never leave the box blank, show the
            // reference right-handed swing instead.
            if (!figure.children.length) drawStaticFallback(hlBones, hlColor);
            frameCamera();
        }

        // Add a tapered wooden bat from the hands out to the barrel tip, with a knob.
        function addBat(getV) {
            const rw = getV('RWrist'), lw = getV('LWrist');
            if (!rw || !lw) return;
            const grip = new THREE.Vector3().addVectors(rw, lw).multiplyScalar(0.5);
            const tip = getV('BatTip');
            if (!tip) return;
            const axis = new THREE.Vector3().subVectors(tip, grip).normalize();
            const knob = grip.clone().add(axis.clone().multiplyScalar(-0.06));
            addBone(knob, tip, 0.016, 0.034, new THREE.Color(SKEL_BAT), false); // handle → barrel
            addJoint(knob, 0.026, SKEL_BAT, false);                              // knob
            addJoint(tip, 0.034, SKEL_BAT, false);                              // barrel end cap
        }

        function drawStaticFallback(hlBones, hlColor) {
            // Default reference figure: a right-handed swing at contact (with bat).
            const P = RIGHTY_SWING_POSE;
            const getV = (n) => (n === 'BatTip' ? new THREE.Vector3(...BAT_TIP) : (P[n] ? new THREE.Vector3(...P[n]) : null));
            const conns = BONE_CONNECTIONS.concat([['RAnkle', 'RToe'], ['LAnkle', 'LToe']]);
            addBonesFor(getV, conns, hlBones, hlColor);
            addJointsAndExtras(getV, hlBones, hlColor);
            addBat(getV);
        }

        // Recenter the figure on its bounding box and pull the camera back so the
        // whole skeleton fills the frame — robust to any units or world offset.
        function frameCamera() {
            const savedRot = pivot.rotation.clone();
            pivot.rotation.set(0, 0, 0);
            figure.position.set(0, 0, 0);
            pivot.updateMatrixWorld(true);

            const box = new THREE.Box3().setFromObject(figure);
            if (box.isEmpty()) { pivot.rotation.copy(savedRot); return; }

            const center = box.getCenter(new THREE.Vector3());
            figure.position.copy(center).multiplyScalar(-1);

            const sphere = box.getBoundingSphere(new THREE.Sphere());
            const radius = Math.max(sphere.radius, 0.1);
            const dist = (radius / Math.sin((camera.fov * Math.PI / 180) / 2)) * 1.15;
            camera.position.set(0, 0, dist);
            camera.near = Math.max(dist - radius * 2, 0.001);
            camera.far = dist + radius * 4;
            camera.updateProjectionMatrix();

            pivot.rotation.copy(savedRot);
            pivot.updateMatrixWorld(true);
            render();
        }

        const pose = getPose(contactFrame);
        // Diagnostics so an empty/blank viewer can be traced from the console.
        console.info('[skeleton] frames:', frames ? frames.length : 0,
            '| markers in frame 0:', pose ? Object.keys(pose) : '(none)',
            '| matched joints:', pose ? JOINT_NAMES.filter(n => Array.isArray(pose[n])).length : 0,
            '| scale:', scale);

        // Lighting rig (fixed to the scene, so shading shifts as the figure rotates).
        scene.add(new THREE.AmbientLight(0xb9c4dc, 0.75));
        const keyLight = new THREE.DirectionalLight(0xffffff, 1.0); keyLight.position.set(2, 4, 3); scene.add(keyLight);
        const rimLight = new THREE.DirectionalLight(0x5b8cff, 0.55); rimLight.position.set(-3, 1.5, -2.5); scene.add(rimLight);

        try {
            buildSkeleton(pose, null, '#ffffff');
        } catch (err) {
            console.error('[skeleton] build failed, showing reference pose:', err);
            try { buildSkeleton(null, null, '#ffffff'); } catch (e) { /* give up gracefully */ }
        }

        function animate() { requestAnimationFrame(animate); renderer.render(scene, camera); }
        animate();

        skeletonScene = { buildSkeleton, pose, frames, contactFrame };
    }

    function attachSkeletonClickHandlers() {
        document.querySelectorAll('.dim-tile').forEach(tile => {
            tile.style.cursor = 'pointer';
            tile.addEventListener('click', () => {
                const key = tile.dataset.dimKey;
                const hit = METRIC_HIGHLIGHTS[key];
                if (!hit) return;
                if (skeletonScene) {
                    const hlBones = new Set();
                    for (const [a,b] of hit.bones) { hlBones.add(`${a}|${b}`); hlBones.add(`${b}|${a}`); }
                    skeletonScene.buildSkeleton(skeletonScene.pose, hlBones, hit.color);
                }
                document.getElementById('skeleton-label').textContent = tile.querySelector('.dim-label')?.textContent || key;
                document.getElementById('skeleton-metric-desc').textContent = hit.desc;
                document.getElementById('skeleton-panel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            });
        });
    }

    function getDemographics() {
        const ft = parseFloat(document.getElementById('height-ft').value) || 6;
        const inc = parseFloat(document.getElementById('height-in').value) || 0;
        const lbs = parseFloat(document.getElementById('weight-lbs').value) || 180;
        const batOz = parseFloat(document.getElementById('bat-weight-oz').value) || 0;
        const batIn = parseFloat(document.getElementById('bat-length-in').value) || 0;
        return {
            height_m: ((ft * 12) + inc) * 0.0254,
            weight_kg: lbs * 0.453592,
            bat_mass_kg: batOz * 0.0283495,   // oz → kg
            bat_length_m: batIn * 0.0254       // in → m
        };
    }
    
    function showLoading() { loadingOverlay.classList.remove('hidden'); }
    function hideLoading() { loadingOverlay.classList.add('hidden'); }
    
    function showError(msg) {
        errorMessage.textContent = msg;
        errorToast.classList.add('show');
        setTimeout(hideError, 5000);
    }
    function hideError() { errorToast.classList.remove('show'); }
});
