# Bat tracking from OpenCap's two phone videos (experimental)

Not used by the app or the Swing Score. This measures bat speed from the raw OpenCap
camera videos, so bat-speed labels can be collected for a future body-motion → bat-speed model.

## Pipeline

1. **`batcap.py`**: per swing, from the two raw `.mov` files, the synced videos, the OpenPose
   keypoint pickles, the camera calibrations and the `.trc`:
   - pairs each raw video with its calibration by evidence (OpenCap sometimes files the phones
     under swapped camera slots), and aligns raw frames to `.trc` frames
   - fits each camera's pose in the `.trc` frame (solvePnP of the 3-D markers against the 2-D
     keypoints; typically 1.5–3 px median reprojection error)
   - finds the bat in each view as the moving, wood-coloured streak leaving the hands, with a
     smoothness-constrained (Viterbi) angle track, centred on the frame of most video motion
2. **`postbat.py`**: builds the 3-D bat for a 33-inch bat anchored at the hands, picks the
   candidate that both views agree on, drops impossible frames, and reports sweet-spot speed
   (6 in from the tip). A swing counts as **measured** only if the fast part of the swing was
   genuinely tracked (`fast_window_good >= 0.75`, `good_fraction >= 0.8`, speed 40–100 mph).
3. **`overlay.py`**: draws the 3-D bat back onto both videos for visual checking.
4. **`safepkl.py`**: loads the OpenCap pickles with a loader restricted to plain data.

Bat length and sweet-spot position are constants at the top of `batcap.py`; change them for
youth bats.

## First results (Cuesta, 107 swings, 2 front phones, 120 fps)

27 of 107 swings measured (typically 60–78 mph); the rest are marked not measurable.

Why the rest fail:
- **Geometry (main cause):** near contact the bat points toward both front cameras, so the two
  views cannot fix its direction.
- **Motion blur:** at 120 fps the barrel is a faint streak in the fastest frames.
- **Busy backgrounds.**

A body-motion ANN trained on the 27 measured swings did not beat a wrist-speed baseline under
leave-one-player-out testing (peak error 19 vs 16 mph).

## Capture recommendations

- A camera to the side or behind the hitter.
- 240 fps with a fast shutter under bright light.
- High-contrast tape at the bat tip and sweet spot.
- A plain backdrop.
