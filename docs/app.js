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
            ${createMetric('Max Separation', m.max_separation_deg.toFixed(1), '°')}
            ${handSpeedHtml}
            ${createMetric('Peak Hip Power', m.peak_hip_power_W.toFixed(0), 'W')}
            ${createMetric('Rel. Hip Power', m.hip_power_per_kg.toFixed(1), 'W/kg')}
            ${createMetric('Sequence Timing', m.sequence_timing_ms.toFixed(0), 'ms')}
            ${createMetric('Chain Efficiency', m.kinetic_chain_efficiency_pct.toFixed(1), '%')}
            ${createMetric('Torso/Pelvis Ratio', m.torso_to_pelvis_rot_ratio.toFixed(2), '')}
            ${createMetric('Total Chain KE', m.total_energy_transfer_J.toFixed(0), 'J')}
            ${createMetric('E_total (Bat)', m.e_total_J.toFixed(0), 'J')}
        `;
        
        document.getElementById('stride-metrics').innerHTML = `
            ${createMetric('Stride Efficiency', m.stride_efficiency_pct.toFixed(0), '%')}
            ${createMetric('Stride Ratio', m.stride_ratio.toFixed(2), 'x Ht')}
            ${createMetric('Proper Sequence', m.proper_sequence ? 'YES' : 'NO', '')}
            ${createMetric('Plant Method', m.plant_method.replace(/_/g, ' '), '', true)}
            ${createMetric('Pelvis KE', m.pelvis_ke_J.toFixed(1), 'J')}
            ${createMetric('Torso KE', m.torso_ke_J.toFixed(1), 'J')}
            ${createMetric('Arm KE', m.arm_ke_J.toFixed(1), 'J')}
            ${createMetric('Elbow KE', m.elbow_ke_J.toFixed(1), 'J')}
        `;

        // Reset advanced panel state
        document.getElementById('advanced-panels').classList.add('hidden');
        document.getElementById('toggle-icon').textContent = '▶';
    }

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

        const badgeClass = badgeCssClass(dim.badge);
        const pillLabel = pillText(dim.badge);
        const stars = renderStars(dim.stars);

        tile.innerHTML = `
            <div class="dim-badge ${badgeClass}"></div>
            <div class="dim-info">
                <div class="dim-name">${dim.label}</div>
                <div class="dim-value">${dim.value} ${dim.unit}</div>
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
