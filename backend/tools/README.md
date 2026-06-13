# OBP → Swing Optimizer Pro data tools

## `obp_extract.py`

Converts the **Driveline OpenBiomechanics (OBP) hitting** dataset into the
`.trc` (markers) and `.mot` (joint kinematics) files this app ingests, plus a
`manifest.csv` carrying the population strata (level, side, age, height, weight,
bat spec, exit velo).

### ⚠️ Licensing — read first
The OBP dataset is **CC BY-NC-SA 4.0** (Attribution, **Non-Commercial**, ShareAlike):
- **Non-Commercial only.** Do **not** use OBP data (or files derived from it,
  including the `.trc`/`.mot` this tool produces) to build, calibrate, validate,
  or train anything used for commercial / for-profit purposes. For commercial use,
  obtain Driveline's paid license.
- **ShareAlike.** Any redistributed derivatives must also be CC BY-NC-SA 4.0.
  Do not commit generated data into a product/commercial repository.
- **Attribution.** Credit The OpenBiomechanics Project, link the license, and note
  that changes were made. There is an additional hard exclusion for anyone affiliated
  with a professional sports organization or financial-analysis firm.

This tool is intended for **methodology development, research, and validation only.**

### Get the source data
From the OBP repo (`drivelineresearch/openbiomechanics`, `baseball_hitting/data`):
- `full_sig/landmarks.zip`     → unzip to `landmarks.csv`
- `full_sig/joint_angles.zip`  → unzip to `joint_angles.csv`
- `metadata.csv`

### Run
```bash
# a balanced validation sample (3 swings per playing level)
python3 obp_extract.py \
  --landmarks landmarks.csv --joint-angles joint_angles.csv --metadata metadata.csv \
  --out ./obp_out --per-level 3

# everything (677 swings / 98 athletes)
python3 obp_extract.py \
  --landmarks landmarks.csv --joint-angles joint_angles.csv --metadata metadata.csv \
  --out ./obp_out

# specific swings
python3 obp_extract.py ... --session-swings 6_1,21_1,84_1
```

### How it maps the data
- **Coordinates.** Driveline global (+x pitcher, +y RH box, +z up) → app frame
  (+x right, +y up, +z toward pitcher): `app(x,y,z) = (drv_y, drv_z, drv_x)`.
- **`.trc` markers** come from landmark joint centres (`lsjc`→LShoulder, `lejc`→LElbow,
  `lwjc`→LWrist, `lhjc`→LHip, `lkjc`→LKnee, `lajc`→LAnkle, mirror for right, `thorax_prox`→Neck,
  midpoint of hips→midHip, `sweet_spot`→Bat), in metres.
- **`.mot` rotations** come from Driveline's validated `joint_angles.csv` (cleaner than
  deriving angles from short joint-centre segments). Lead/rear are mapped **functionally**
  (rear/top-hand side → `_r`, lead → `_l`), so it's handedness-agnostic. **Translation**
  (`pelvis_tx/ty/tz`) is the landmark mid-hip position. Driveline event times
  (`fp_10_time`, `contact_time`) are embedded in the `.mot` header.

### Validation (12-swing sample, all 4 levels incl. a LH hitter)
Files load with `RefinedHittingOptimizer.load_mot_file` / `load_trc_file`; within-swing
pelvis-rotation range came out **106–133°** and trunk–pelvis separation **16–30°** —
physically sensible. Anatomical marker heights are correct (neck ≈1.40 m, ankle ≈0.12 m).
