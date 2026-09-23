# Dynamics Comparison

The report can show lower-body joint loads from up to three independent sources
side by side:

| Column | Where it comes from | Needs |
|---|---|---|
| **Bottom-up ID (ours)** | Newton-Euler chain from the CoM-derived ground reaction force (`backend/bottom_up_id.py`) | a `.trc` |
| **OpenSim ID (same GRF)** | OpenSim Inverse Dynamics with that same GRF applied as external loads (`backend/opensim_id.py`) | OpenSim installed locally + the athlete's scaled `.osim` |
| **Reference** | Anything you upload with the trial: Cortex force-plate values, an OpenCap/OpenSimAD simulation, a lab's ID output | a `.json` or `.sto` file |

**Reading it.** Ours and OpenSim use the *same* ground force, so a gap between
them is in the moment math. A force-plate reference measures the ground force
itself, so a gap between ours and the reference points at our GRF estimate or
the per-foot split. The `Δ ours` column compares ours to the reference when one
is uploaded, otherwise to OpenSim. Green is within 15%, amber within 30%.

---

## Reference file format

### JSON (`biostrike.dynamics.v1`)

Peaks only, e.g. typed in by hand from Cortex. Every key is optional:

```json
{
  "schema": "biostrike.dynamics.v1",
  "source": "cortex",
  "label": "Cortex force plates — Emilio 1_10",
  "peaks": {
    "grf_vert_l_N": 612,
    "grf_vert_r_N": 488,
    "hip_moment_l_Nm": 240,
    "hip_moment_r_Nm": 110,
    "knee_moment_l_Nm": 130,
    "knee_moment_r_Nm": 70,
    "ankle_moment_l_Nm": 55,
    "ankle_moment_r_Nm": 35
  }
}
```

Or time series, and the peaks are computed for you:

```json
{
  "schema": "biostrike.dynamics.v1",
  "source": "cortex",
  "label": "Cortex — Emilio 1_10",
  "timeseries": {
    "time_s":  [0.000, 0.004, ...],
    "grf_l_N": [[fx, fy, fz], ...],
    "grf_r_N": [[fx, fy, fz], ...],
    "hip_moment_l_Nm": [..]
  }
}
```

- Units are SI: newtons and newton-metres.
- `l` / `r` are the athlete's **left / right**, not lead / trail. The report maps
  them to lead and trail from the handedness you enter.
- `grf_*_N` vectors must be **Y-up** so column 1 is vertical. Cortex exports are
  often Z-up; swap the columns before saving.
- Moment series can be scalars or 3-vectors. Vectors are compared by magnitude.

### OpenSim `.sto`

Upload an Inverse Dynamics output (`*_ID.sto`) directly. Hip moments are taken as
the magnitude across flexion, adduction and rotation; the ankle combines
`ankle_angle` and `subtalar`.

### Pairing files in a multi-swing upload

References pair with swings by filename. Common suffixes are ignored, so
`Emilio_1_10_cortex.json` and `Emilio_1_10_ID.sto` both pair with
`Emilio_1_10.mot`. A single reference uploaded with a single swing pairs
regardless of name.

---

## Running OpenSim ID locally (Apple Silicon M2)

OpenSim is a conda-only package, so it cannot run on the Render server. On
Render the OpenSim column stays empty and the report says why. It runs when you
start the backend yourself.

```bash
# 1. Environment. If conda cannot find an arm64 build of opensim, create the
#    env as x86_64 instead — it runs through Rosetta 2 on an M2:
#      CONDA_SUBDIR=osx-64 conda create -n biostrike python=3.11
#      conda activate biostrike && conda config --env --set subdir osx-64
conda create -n biostrike python=3.11
conda activate biostrike
conda install -c opensim-org opensim=4.5
pip install -r backend/requirements.txt

# 2. Check it imports
python -c "import opensim; print(opensim.__version__)"

# 3. Run the app locally, then open http://localhost:8000
./run.sh
```

Then, with each upload, add the athlete's scaled model. It lives in the OpenCap
session download at `OpenSimData/Model/LaiUhlrich2022_scaled.osim`. Use **+ Add
Scaled Model**, or set a default:

```bash
export OPENSIM_MODEL_PATH=~/path/to/LaiUhlrich2022_scaled.osim
```

The model carries the athlete's segment masses, so use the model from the
**same athlete's** session. The report flags it when it had to fall back to a
default model.

**Pelvis residual.** When OpenSim runs, the notes show the peak pelvis residual
force. That is the force OpenSim needs at the pelvis to balance the motion
against the ground force we applied. Under roughly 5–10% of body weight means
the applied GRF and the kinematics agree. A large residual means they don't, and
the GRF estimate on that trial shouldn't be trusted.

### Why the OpenCap dynamic simulation isn't included

`opencap-processing`'s `example_kinetics.py` (OpenSimAD) solves ground forces and
joint moments together, but its docs list macOS arm64 as unsupported, it has no
swing preset (running, walking, drop_jump, sit-to-stand and squats only), and
its authors state the results are not biomechanically validated. If you run it
elsewhere, for example on a lab PC, export its peaks to the JSON format above
and upload them as a reference.
