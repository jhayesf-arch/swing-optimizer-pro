# Swing Optimizer Pro — Metric Guide

A plain-English explanation of every number in your swing report. No biomechanics degree required.

**About the `*`:** A metric marked with an asterisk (`*`) is **benchmarked against outside research or measured data, broken down by competition level** (youth / high school / college / pro) — so a "good" score means good *compared to other hitters at your level*. Metrics **without** an asterisk are **relative benchmarks**: there's no solid published data for them yet, so they're best for tracking *your own* progress over time rather than ranking you against others. (See the note at the bottom for the fine print.)

---

## The headline numbers (top of the report)

- **Swing Score** — Your overall grade out of 100, blended from the 12 dimensions below (the biggest hitters of the score are hand speed, hip‑shoulder separation, and sequencing).
- **Hand / Bat Speed\*** — How fast your hands are moving the bat through the zone, in mph. This is the single most reliable predictor of power and is the number pro tools (e.g. Blast) report.
- **Physics Efficiency Score** — How well the speed you generate actually transfers up the chain into the bat (vs. leaking out). Higher = less wasted effort.
- **Peak Pelvis Angular Velocity** — How fast your hips are turning at their quickest point (degrees per second). Your hips are the engine; this is the engine's RPM.

---

## The 4 phases (your 12‑dimension breakdown)

### Phase 1 — Balance & Load (getting loaded up)
- **Negative Move** — A small, controlled shift of weight *back* before you stride forward, like a hitter "gathering." Too little = no load; too much = drift.
- **Pelvis Load** — How much energy your hips store during the load, like winding up a spring.
- **Upper Torso Load** — How much your upper body coils/stores energy during the load.

### Phase 2 — Stride (moving toward the pitch)
- **Stride Length\*** — How far your front foot strides, measured relative to your height (so it's fair across body sizes).
- **Forward Move** — How efficiently your body's weight moves forward into the swing (momentum toward the ball).

### Phase 3 — Power Move (the turn)
- **Max Hip‑Shoulder Separation\*** — How much your hips lead your shoulders (the "X‑factor" stretch). The more your hips open while your shoulders stay back, the more whip you create.
- **Pelvis Total Rotation Range\*** — How far your hips rotate from load to contact.
- **Upper Torso Total Rotation Range\*** — How far your shoulders/upper body rotate from load to contact.

### Phase 4 — Contact & Follow‑Through (delivering the barrel)
- **Pelvis Direction at Contact\*** — How "open" your hips are when you hit the ball. Good hitters have their hips already cleared/open at contact.
- **Upper Torso Direction at Contact\*** — How open your shoulders are at contact (they should still lag the hips a touch).
- **Kinetic Chain Efficiency** — How much of your total energy actually ends up in the arms and bat (the payoff of good sequencing).
- **Lead‑Leg Block\*** — How much your front knee *straightens* (extends) from foot plant to contact. A firm, extending front leg "posts up" and redirects your momentum into rotation. This is one of the strongest bat‑speed correlates in the Driveline OpenBiomechanics dataset — a soft, collapsing front leg leaks energy that should whip into the barrel.
- **Sequence Quality\*** — Whether your body fires in the right *order* with the right timing (hips → torso → arms → hands), and how cleanly.
- **Hand / Bat Speed\*** — Same as the headline number: peak hand speed through the zone (mph).
- **Follow‑Through Quality** — How smoothly you decelerate after contact — abrupt stops usually mean energy leaked instead of going into the ball.

---

## Percentile ranks & Coaching Focus

Every dimension shows a **percentile** — where you rank *at your level* (youth / high school / college / pro). 50th = dead average; 80th = better than 4 of 5 hitters. The **Overall Percentile** at the top blends all dimensions. The **Coaching Focus** panel turns this into action: a body heatmap colors your weakest links (red = needs work, green = strength), and **Your Top Priorities** ranks the fixes with the biggest bat‑speed payoff first, each with a specific cue and drill.

**Two ways percentiles are computed — the report tells you which:**

- **vs research** (default) — percentiles are *estimated* from published benchmarks (Blast Motion, Fleisig, Escamilla, etc.). Good directionally, but not a measured rank against real hitters. Only Hand/Bat Speed is anchored to fully level‑stratified data.
- **vs your library** — real percentiles computed from *your own* `.mot`/`.trc` files, grouped by level. A "72nd percentile" then literally means "better than 72% of the college swings you've logged." Tiles marked **·lib** and the hero note tell you this is active.

To switch to library‑based ranking, build a cohort from the swings on your machine:

```
# 1) List your swing files (fill in level + height/weight in the CSV it writes)
python backend/build_cohort.py init --dir ~/your-swings --out cohort_manifest.csv

# 2) Build the model (drops cohort_percentiles.json next to analyzer.py)
python backend/build_cohort.py build --manifest cohort_manifest.csv
```

Each level needs at least 5 swings before its empirical percentiles are trusted; levels below that automatically fall back to the research estimate. Re‑run `build` whenever you add swings.

---

## Kinematic Sequence chart
A graph showing how fast each body part is turning, in order, through the swing. In an elite swing the peaks march left‑to‑right — **Pelvis → Torso → Lead Arm → Hands/Bat** — each one faster than the last (the "kinetic chain" / whip effect). The order is what matters most.

- **Pelvis / Torso / Lead Arm / Hands‑Bat** — Peak turning speed of each segment, and whether they fire in the correct proximal‑to‑distal order.

---

## Advanced Physics (optional deep‑dive panel)

**Speed & sequence**
- **Max Separation\*** — Same as Max Hip‑Shoulder Separation above (the X‑factor stretch).
- **Max Hand Speed\*** — Your single fastest hand‑speed reading (mph).
- **Peak Hip Power** — The raw power output of your hips at their peak (watts).
- **Rel. Hip Power** — Hip power adjusted for body weight (watts per kg), so it's fair across sizes.
- **Sequence Timing\*** — The time gap between your hips peaking and your shoulders peaking (ms); a healthy lag means good sequencing.
- **Chain Efficiency** — How much energy makes it down the chain into the bat (%).
- **Torso/Pelvis Ratio** — How much faster your torso turns than your pelvis — the "amplification" up the chain.
- **Total Chain KE** — The total energy generated through the swing (joules).
- **Time to Contact** — How quickly you get the barrel to the ball (ms) — quickness/reaction.
- **Rotational Accel** — How hard you're accelerating your rotation (how explosively you turn).
- **Body Rotation Ratio** — How much of your swing speed comes from rotating your body vs. just your arms.

**Energy & ground force**
- **Stride Efficiency** — How well your stride converts into forward energy (%).
- **Stride Ratio** — Stride length as a fraction of your height.
- **Proper Sequence** — Yes/No: did your body fire in the correct order (hips before torso before hands)?
- **Pelvis KE / Torso KE / Arm KE / Bat KE** — The energy in each body segment, so you can see the "hand‑off" building from hips to bat (joules).
- **Peak GRF Vert** — How hard you push *down* into the ground at peak, as a % of your body weight (good hitters push hard into the ground).
- **Peak GRF AP** — How hard you push *forward/back* into the ground (newtons).

---

### Note on the `*` (the fine print)
`*` metrics are anchored to external sources: **Welch 1995, Escamilla 2009, Fortenbaugh 2011, Fleisig 2013, Taguchi 2023, and Blast Motion** level benchmarks.

Two honesty caveats worth knowing:
- **Only Hand / Bat Speed is fully stratified across all four levels** (youth → high school → college → pro), via Blast Motion's published benchmarks. It's the most trustworthy comparison.
- The other `*` metrics come mostly from **college- and pro-level studies**; the youth and high‑school corridors are reasonable estimates scaled down from those (no published youth/HS data exists yet).
- Everything **without** a `*` (Negative Move, Pelvis Load, Upper Torso Load, Forward Move, Kinetic Chain Efficiency, Follow‑Through Quality, and the raw energy/power/GRF readouts) is a **relative benchmark** — great for tracking your own swing‑to‑swing progress, not for ranking against other players.
