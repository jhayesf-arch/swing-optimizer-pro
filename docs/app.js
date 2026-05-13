const DEMO_DIAGNOSIS = {
    swing_score: 74,
    efficiency_score: 82,
    metrics: {
        estimated_hand_speed_mph: 68.4, max_hand_speed_mph: 71.2,
        max_separation_deg: 44.1, peak_hip_power_W: 3820, hip_power_per_kg: 14.2,
        sequence_timing_ms: 48, kinetic_chain_efficiency_pct: 52.3,
        torso_to_pelvis_rot_ratio: 1.31, total_energy_transfer_J: 412,
        stride_efficiency_pct: 94, stride_ratio: 0.74, proper_sequence: true,
        pelvis_ke_J: 98.4, torso_ke_J: 74.1, arm_ke_J: 61.2, bat_ke_J: 38.7,
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
                { key: 'hand_speed', label: 'Hand / Bat Speed', value: '68.4', unit: 'mph', stars: 4, badge: 'satisfactory', description: 'Above average hand speed for college level.' },
                { key: 'follow_through_quality', label: 'Follow-Through Quality', value: '42', unit: '°', stars: 3, badge: 'satisfactory', description: 'Adequate follow-through arc.' },
            ]},
        },
    },
    grf_estimation: {},
};

document.addEventListener('DOMContentLoaded', () => {
    // -----------------------------------------
    // Elements
    // -----------------------------------------
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const refreshLocalBtn = document.getElementById('refresh-local');
    const localFilesList = document.getElementById('local-files-list');
    
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
    let currentLocalFilepath = null;
    let currentLocalFilename = null;
    let pendingUploadFile = null;
    let pendingLocalFilepath = null;
    let pendingLocalFilename = null;
    let selectedSkillLevel = 'high_school';

    // -----------------------------------------
    // Init
    // -----------------------------------------
    checkBackendHealth();
    fetchLocalFiles();
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
    
    refreshLocalBtn.addEventListener('click', fetchLocalFiles);

    document.getElementById('load-demo').addEventListener('click', () => {
        renderDashboard(DEMO_DIAGNOSIS, 'demo_swing.mot');
    });
    
    document.getElementById('btn-cancel-demo').addEventListener('click', hideDemoModal);
    document.getElementById('btn-run-physics').addEventListener('click', () => {
        hideDemoModal();
        if (pendingUploadFile) {
            handleUpload(pendingUploadFile);
        } else if (pendingLocalFilepath) {
            analyzeLocalFile(pendingLocalFilepath, pendingLocalFilename);
        }
    });
    
    btnBack.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        fileInput.value = '';
        currentLocalFilepath = null;
        currentLocalFilename = null;
    });
    
    closeToast.addEventListener('click', hideError);

    // Advanced toggle
    document.getElementById('toggle-advanced').addEventListener('click', () => {
        const panels = document.getElementById('advanced-panels');
        const icon = document.getElementById('toggle-icon');
        const isHidden = panels.classList.contains('hidden');
        panels.classList.toggle('hidden');
        icon.textContent = isHidden ? '▼' : '▶';
    });

    // Auto-Recalculate on Demographic Change (local files only)
    ['height-ft', 'height-in', 'weight-lbs'].forEach(id => {
        document.getElementById(id).addEventListener('change', () => {
            if (currentLocalFilepath && !resultsSection.classList.contains('hidden')) {
                analyzeLocalFile(currentLocalFilepath, currentLocalFilename);
            }
        });
    });

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

    async function fetchLocalFiles() {
        localFilesList.innerHTML = '<div class="loading-spinner"></div><p class="small text-muted mt-2">Scanning...</p>';
        try {
            const response = await fetch(`${API_BASE}/api/scan-downloads`);
            const data = await response.json();
            if (data.success) {
                renderLocalFiles(data.files);
            } else {
                throw new Error(data.error || "Failed to scan local files");
            }
        } catch (err) {
            localFilesList.innerHTML = `<p class="small text-muted">Scan failed. Is backend running?</p>`;
            console.error(err);
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
    
    async function analyzeLocalFile(filepath, filename) {
        showLoading();
        const demo = getDemographics();
        const payload = JSON.stringify({ filepath, filename, height_m: demo.height_m, weight_kg: demo.weight_kg, skill_level: selectedSkillLevel, bat_mass_kg: demo.bat_mass_kg, bat_length_m: demo.bat_length_m });
        const doLocal = () => fetch(`${API_BASE}/api/analyze/local`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: payload });
        try {
            let response;
            try {
                response = await doLocal();
            } catch (_) {
                await new Promise(r => setTimeout(r, 5000));
                response = await doLocal();
            }
            let data;
            try { data = await response.json(); }
            catch (_) { throw new Error(`Server error (HTTP ${response.status})`); }

            if (data.success) {
                currentLocalFilepath = filepath;
                currentLocalFilename = filename;
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
    function renderLocalFiles(files) {
        if (!files || files.length === 0) {
            localFilesList.innerHTML = '<p class="small text-muted">No .mot files found in ~/Downloads</p>';
            return;
        }
        
        localFilesList.innerHTML = '';
        files.forEach(f => {
            const div = document.createElement('div');
            div.className = 'file-item';
            div.innerHTML = `
                <span class="file-name">${f.filename}</span>
                <span class="file-action">▶</span>
            `;
            div.addEventListener('click', () => promptDemographicsForLocal(f.filepath, f.filename));
            localFilesList.appendChild(div);
        });
    }
    
    function renderDashboard(diagnosis, filename) {
        hideLoading();
        uploadSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        // Re-trigger animations
        document.querySelectorAll('.anim-slide-up').forEach(el => {
            el.style.animation = 'none';
            el.offsetHeight;
            el.style.animation = null; 
        });
        
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

        document.getElementById('swing-score-number').textContent = swingScore.toFixed(0);
        document.getElementById('exit-velo-number').textContent = handSpeed.toFixed(1);
        document.getElementById('efficiency-number').textContent = effScore;

        // Skill badge
        const skillLabels = {
            youth: 'Youth', high_school: 'High School',
            college: 'College', professional: 'Professional'
        };
        const skillLevel = diagnosis.swingai_report?.skill_level || selectedSkillLevel;
        document.getElementById('skill-badge-display').textContent = skillLabels[skillLevel] || skillLevel;

        // Hide data-warning (no longer needed without exit velo guess)
        document.getElementById('data-warning').classList.add('hidden');

        // ---- SWINGAI 4-PHASE CARDS ----
        if (diagnosis.swingai_report) {
            renderSwingAIReport(diagnosis.swingai_report);
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
        document.getElementById('toggle-icon').textContent = '▶';

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
                    <span class="phase-icon">${phase.icon}</span>
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
            </div>
            <div class="dim-stars">${stars}</div>
            <div class="dim-pill ${badgeClass}">${pillLabel}</div>
            <div class="dim-tooltip">${dim.description}</div>
            ${cues}
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
        pendingLocalFilepath = null;
        showDemoModal();
    }
    
    function promptDemographicsForLocal(filepath, filename) {
        pendingUploadFile = null;
        pendingLocalFilepath = filepath;
        pendingLocalFilename = filename;
        showDemoModal();
    }
    
    function showDemoModal() { document.getElementById('demo-modal').classList.remove('hidden'); }
    function hideDemoModal() { document.getElementById('demo-modal').classList.add('hidden'); }
    
    // -----------------------------------------
    // Body Skeleton Canvas (static marker positions, highlight on metric click)
    // -----------------------------------------

    // Hardcoded marker positions — mid-swing contact pose (right-handed batter, front view)
    // Pelvis open to pitcher, lead arm extended forward, back arm bent, slight squat
    const MARKERS = {
        Head:       [118, 32],
        Neck:       [114, 62],
        RShoulder:  [145, 82],   // back shoulder (right) — higher, pulled back
        LShoulder:  [82,  88],   // lead shoulder — lower, rotated forward
        RElbow:     [162, 148],  // back elbow — bent, tucked
        LElbow:     [58,  130],  // lead elbow — extended forward
        RWrist:     [150, 188],  // hands together near contact zone
        LWrist:     [138, 192],
        midHip:     [110, 200],
        RHip:       [136, 208],  // back hip — rotated open
        LHip:       [84,  208],  // lead hip — clearing
        RKnee:      [148, 300],  // back knee — bent, weight transferring
        LKnee:      [78,  292],  // lead knee — braced/extending
        RAnkle:     [152, 388],  // back foot — heel rising
        LAnkle:     [72,  385],  // lead foot — planted
        RHeel:      [142, 408],
        LHeel:      [64,  408],
    };

    const BONES = [
        ['Neck','RShoulder'],['Neck','LShoulder'],
        ['RShoulder','RElbow'],['RElbow','RWrist'],
        ['LShoulder','LElbow'],['LElbow','LWrist'],
        ['Neck','midHip'],
        ['midHip','RHip'],['midHip','LHip'],
        ['RHip','RKnee'],['RKnee','RAnkle'],['RAnkle','RHeel'],
        ['LHip','LKnee'],['LKnee','LAnkle'],['LAnkle','LHeel'],
    ];

    // Bones to highlight per metric key
    const METRIC_HIGHLIGHTS = {
        'pelvis_load':                      { bones: [['midHip','RHip'],['midHip','LHip']], color: '#f59e0b', desc: 'Pelvis — hip rotational energy storage during load.' },
        'negative_move':                    { bones: [['midHip','RHip'],['midHip','LHip'],['RHip','RKnee'],['LHip','LKnee']], color: '#a78bfa', desc: 'Pelvis & legs — backward weight shift before stride.' },
        'upper_torso_load':                 { bones: [['Neck','RShoulder'],['Neck','LShoulder'],['Neck','midHip']], color: '#8b5cf6', desc: 'Torso & shoulders — upper body coil tension at load.' },
        'stride_length':                    { bones: [['LHip','LKnee'],['LKnee','LAnkle'],['LAnkle','LHeel']], color: '#fbbf24', desc: 'Lead leg — stride length from load to foot plant.' },
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

    let activeHighlight = null;

    function drawBodyCanvas() {
        const canvas = document.getElementById('body-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const hlBones = new Set();
        let hlColor = '#ffffff';
        if (activeHighlight) {
            hlColor = activeHighlight.color;
            for (const [a, b] of activeHighlight.bones) {
                hlBones.add(`${a}|${b}`);
                hlBones.add(`${b}|${a}`);
            }
        }

        // Draw head circle
        const headHL = activeHighlight && (
            hlBones.has('Neck|RShoulder') || hlBones.has('Neck|LShoulder') || hlBones.has('Neck|midHip')
        );
        ctx.beginPath();
        ctx.arc(MARKERS.Head[0], MARKERS.Head[1], 18, 0, Math.PI * 2);
        ctx.strokeStyle = headHL ? hlColor : 'rgba(255,255,255,0.25)';
        ctx.lineWidth = headHL ? 3 : 1.5;
        ctx.stroke();

        // Draw bones
        for (const [a, b] of BONES) {
            const isHL = hlBones.has(`${a}|${b}`);
            ctx.beginPath();
            ctx.moveTo(MARKERS[a][0], MARKERS[a][1]);
            ctx.lineTo(MARKERS[b][0], MARKERS[b][1]);
            ctx.strokeStyle = isHL ? hlColor : 'rgba(255,255,255,0.25)';
            ctx.lineWidth = isHL ? 4 : 1.5;
            ctx.stroke();
        }

        // Draw joints
        for (const [name, [x, y]] of Object.entries(MARKERS)) {
            if (name === 'Head') continue;
            const isHL = activeHighlight && activeHighlight.bones.some(([a, b]) => a === name || b === name);
            ctx.beginPath();
            ctx.arc(x, y, isHL ? 5 : 3, 0, Math.PI * 2);
            ctx.fillStyle = isHL ? hlColor : 'rgba(255,255,255,0.5)';
            ctx.fill();
        }

        // Glow effect on highlighted bones
        if (activeHighlight) {
            ctx.shadowColor = hlColor;
            ctx.shadowBlur = 10;
            for (const [a, b] of activeHighlight.bones) {
                ctx.beginPath();
                ctx.moveTo(MARKERS[a][0], MARKERS[a][1]);
                ctx.lineTo(MARKERS[b][0], MARKERS[b][1]);
                ctx.strokeStyle = hlColor;
                ctx.lineWidth = 4;
                ctx.stroke();
            }
            ctx.shadowBlur = 0;
        }
    }

    // Hook metric tile clicks to canvas highlight
    function attachSkeletonClickHandlers() {
        drawBodyCanvas();
        document.querySelectorAll('.dim-tile').forEach(tile => {
            tile.style.cursor = 'pointer';
            tile.addEventListener('click', () => {
                const key = tile.dataset.dimKey;
                const hit = METRIC_HIGHLIGHTS[key];
                if (!hit) return;
                activeHighlight = hit;
                drawBodyCanvas();
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
