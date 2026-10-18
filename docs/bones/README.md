# Bone meshes and motion

- `geometry.json` holds the bone meshes of OpenSim's LaiUhlrich2022 model (the
  model OpenCap scales to each athlete), converted from the `.vtp` files that come
  with an OpenCap session download by `biostrike/export_bones.py` in the
  biostrike-capture fork.
- `demo_motion.json` is one captured swing: each body's position and rotation in
  ground per frame, from OpenCap's `generateVisualizerJson` (opencap-core, Apache 2.0).
- The viewer places each mesh at its body's transform, scaled by the body's scale
  factors, the same way OpenCap's web viewer does (opencap-org/opencap-viewer,
  Apache 2.0). No viewer code was copied; the approach and data format were.

The meshes come from published OpenSim musculoskeletal models. Confirm the terms
of the source models before using them in a paid product.
