# Swing Optimizer Pro

> **Research-grade baseball hitting biomechanics analysis tool** built on OpenCap motion capture data, featuring a 12-dimension scoring framework.

---

## Overview

Swing Optimizer Pro processes OpenSim `.mot` kinematics files from [OpenCap](https://opencap.ai) and produces a comprehensive biomechanical analysis modeled after [a 4-phase / 12-dimension framework](https://winreality.com)'s 4-phase, 12-dimension evaluation framework.

### Key Features

- **12-Dimension Report** — Balance & Load, Stride, Power Move, Contact & Follow-Through phases with 1–5 star ratings and color-coded badges (Excellent / Good / Off Target)
- **Skill-Level Calibrated Thresholds** — Youth, High School, College, Professional corridors for every metric
- **Swing Score (0–100)** — Weighted aggregate of all 12 dimension ratings
- **Advanced Physics Engine** — Energy transfer ratios, kinetic chain efficiency, proximal-to-distal sequencing (Driveline-inspired), Butterworth + Savitzky-Golay signal filtering
- **Exit Velocity Prediction** — Physics-based model accounting for body mass, segmental KE transfer, and X-Factor separation
- **Local File Scan** — Automatically detects `.mot` files in `~/Downloads`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, FastAPI, NumPy, SciPy, Pandas |
| Frontend | Vanilla HTML/CSS/JS — no framework dependencies |
| Motion Data | OpenCap / OpenSim `.mot` kinematics files |

---

## Project Structure

```
hitting_optimizer/
├── backend/
│   ├── analyzer.py        # Core biomechanics engine + 4-phase report builder
│   ├── main.py            # FastAPI server + endpoints
│   └── requirements.txt
├── frontend/
│   ├── index.html         # App shell
│   ├── style.css          # Dark glassmorphism design system
│   └── app.js             # 4-phase report renderer + API client
├── jmp_converter.py       # JMP data format converter utility
├── run.sh                 # One-command launcher
└── .gitignore
```

---

## Framework Reference

This tool's analysis structure mirrors the 4-phase / 12-dimension evaluation framework:

| Phase | Dimensions |
|---|---|
| ⚖️ Balance & Load | Negative Move, Pelvis Load, Upper Torso Load |
| 👣 Stride | Stride Length, Forward Move |
| 💥 Power Move | Max Hip-Shoulder Separation, Pelvis Rotation Range, Upper Torso Rotation Range |
| 🎯 Contact & Follow-Through | Pelvis Direction at Contact, Upper Torso Direction at Contact, Kinetic Chain Efficiency, Sequence Quality |

Our physics computations map directly to each framework dimension using literature-backed thresholds stratified by skill level.

---

## License

MIT
