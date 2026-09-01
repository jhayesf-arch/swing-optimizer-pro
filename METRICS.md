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
- **Stride Length\*** — How wide your base is at contact: the horizontal distance between your two ankles at ball contact, relative to your height (so it's fair across body sizes). It needs **foot markers** (a `.trc` file) — without them the report says *Not measured* rather than guessing.
- **Forward Move** — How efficiently your body's weight moves forward into the swing (momentum toward the ball).

### Phase 3 — Power Move (the turn)
- **Max Hip‑Shoulder Separation\*** — How much your hips lead your shoulders (the "X‑factor" stretch). The more your hips open while your shoulders stay back, the more whip you create.
- **Pelvis Total Rotation Range\*** — How far your hips rotate from load to contact.
- **Upper Torso Total Rotation Range\*** — How far your shoulders/upper body rotate from load to contact.

### Phase 4 — Contact & Follow‑Through (delivering the barrel)
- **Pelvis Direction at Contact\*** — How "open" your hips are when you hit the ball. Good hitters have their hips already cleared/open at contact.
- **Upper Torso Direction at Contact\*** — How open your shoulders are at contact (they should still lag the hips a touch).
- **Energy Transfer** — How much of your total energy actually ends up in the arms and bat (the payoff of good sequencing). This is our own measure, not a standard one — best for tracking your own progress rather than ranking against others.
- **Lead‑Leg Block\*** — How much your front knee *straightens* (extends) from foot plant to contact. A firm, extending front leg "posts up" and redirects your momentum into rotation. This is one of the strongest bat‑speed correlates in the Driveline OpenBiomechanics dataset — a soft, collapsing front leg leaks energy that should whip into the barrel.
- **Sequence Quality\*** — Whether your body fires in the right *order* with the right timing (hips → torso → arms → hands), and how cleanly.
- **Hand / Bat Speed\*** — Same as the headline number: peak hand speed through the zone (mph).
- **Follow‑Through Quality** — How smoothly you decelerate after contact — abrupt stops usually mean energy leaked instead of going into the ball.

---

## Percentile ranks & Coaching Focus

Every dimension shows a **percentile** — where you rank *at your level* (youth / high school / college / pro). 50th = dead average; 80th = better than 4 of 5 hitters. The **Overall Percentile** at the top blends all dimensions. The **Coaching Focus** panel turns this into action: a body heatmap colors your weakest links (red = needs work, green = strength), and **Your Top Priorities** ranks the fixes with the biggest bat‑speed payoff first, each with a specific cue and drill.

**How percentiles are computed — the report tells you which mode is active:**

- **Research‑guided (default)** — percentiles are *estimated* from published benchmarks (Blast Motion, Fleisig, Escamilla, etc.). Good directionally, but not a measured rank against real hitters. Only Hand/Bat Speed is anchored to fully level‑stratified data.
- **Blended (library + research)** — once you've built a library of *your own* swings, percentiles blend your cohort with the research benchmark using **empirical‑Bayes shrinkage**: your cohort's weight is `n / (n + 25)`, and the research benchmark carries the rest. So research still **anchors** the number (and dominates small cohorts), and your own data takes over as you log more swings — the benchmarks are never abandoned. Tiles show **·blend**; hover any bar to see the split (e.g. *"78th pct — blend of your library (68th, n=20) and research (86th); 44% library"*).

The blend constant (`COHORT_SHRINKAGE`, default 25) lives in `analyzer.py` — lower it to trust your library sooner, raise it to lean on research longer. A level still needs ≥ `COHORT_MIN_N` (5) swings before its data blends in at all.

To start blending, build a cohort from the swings on your machine:

```
# 1) List your swing files (fill in level + height/weight in the CSV it writes)
python backend/build_cohort.py init --dir ~/your-swings --out cohort_manifest.csv

# 2) Build the model (drops cohort_percentiles.json next to analyzer.py)
python backend/build_cohort.py build --manifest cohort_manifest.csv
```

Each level needs at least 5 swings before its empirical percentiles are trusted; levels below that automatically fall back to the research estimate. Re‑run `build` whenever you add swings.

---

## Analyzing several swings at once (averages &amp; consistency)

One swing is a single noisy sample. Select **multiple `.mot` files** in the uploader (or drop them together with their `.trc` markers) and the report gains a **Viewing** menu: an **Average** across all the swings, plus every individual swing.

- **The average is the headline.** It is what an athlete should be judged on — a single trial can be flattered or wrecked by one bad capture.
- **Outliers are excluded.** Each metric is screened with a MAD‑based modified z‑score, so one mistracked swing can't drag the average. Excluded trials stay fully visible in the menu.
- **Consistency is its own coaching metric.** Every tile shows a `0–100 consistent` score and the swing‑to‑swing range. A hitter averaging 38° of hip‑shoulder separation with a 25–50° range has a *repeatability* problem, not a range problem — the average alone hides that. Green ≥75, amber ≥50, red below.

A very low consistency score with a huge range (e.g. `9/100`, `50–1126 J`) usually means the **metric itself is unreliable** on that capture, not that the hitter is wildly inconsistent — treat it as a data‑quality flag.

## Ask the coach

With a server-side Anthropic key configured, a **Coach** panel appears bottom-right. Ask anything about your report in plain language, or hit **Explain** on any metric tile to ask about that number specifically.

The coach is **grounded, not generative**: it receives the numbers your physics engine already computed and may only explain those. It never calculates biomechanics itself — that boundary is what keeps every figure in the report reproducible. It's also told which metrics were *not measured* and which are *too inconsistent to trust*, so it declines to over‑interpret them rather than confidently explaining a number the engine doesn't stand behind.

An **Ask the Coach** section appears in every report with starter questions, alongside a floating button bottom‑right. Both are always visible: if no key is configured they say so plainly rather than disappearing, because a hidden feature is indistinguishable from a missing one.

To enable it, set the key on the **server** (never in the browser) and restart the backend:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Optionally set `COACH_MODEL` (defaults to `claude-sonnet-5`).

## Handedness

Which side you bat from isn't recorded in a `.mot` file, so without it the app has to **guess your lead leg** from your pose each swing — and it gets it wrong on check swings and short captures. Set it once per athlete:

```json
{ "athlete": "kike", "handedness": "right", ... }
```

A right‑handed hitter strides onto the left leg, a lefty onto the right. The profile reports how often the guess agreed with itself (e.g. *"71% agree, 4 swings used the OTHER leg"*), so you can spot the problem before you fix it. Uploads accept a `handedness` form field too.

## Capture quality

Not every trial is a swing. Some captures are check swings, warm‑up moves, or mistracked — a 5‑star pelvis rotation next to 3 mph hands is contradictory, not talent. Every swing gets a **capture quality score (0–100)** with reasons, based on peak pelvis rotation, whether the sequence order is resolvable, marker availability, unmeasurable metrics, and whether the detected swing window is physically plausible.

Trials below **55** are **held out of the averages and out of the cohort** — they still appear in the report and stay selectable, with the reason stated, so nothing is hidden. This keeps one bad capture from moving an athlete's numbers, and (more importantly) from shifting the reference distribution everyone else is ranked against. Pelvis rotation under 150°/s disqualifies a trial on its own — no competitive swing turns that slowly.

## "Not measured"

Some metrics can't be computed from every capture — stride length needs foot markers, and a capture that begins after front‑foot plant may not contain the information at all. When that happens the tile reads **Not measured** with the reason, and the dimension is **excluded from the Swing Score and from the drill prescriptions** rather than scored as a failure.

This matters: a metric the system cannot see is not the same as a metric the athlete performed badly, and reporting the second when the first is true is worse than reporting nothing.

From the command line, the same aggregation runs per athlete straight from `athletes.json`:

```
python backend/subject_profile.py --athlete jett
python backend/subject_profile.py --athlete kike --json kike_profile.json
```

### Automatic rebuilds

To avoid re‑running by hand, keep an `athletes.json` mapping each athlete's folder to their level + height (`height_cm`) + weight (`weight_kg`) — metric is preferred; `height_in`/`weight_lb` are also accepted — then rebuild with a single command (no manifest):

```
python backend/build_cohort.py auto      # reads athletes.json, regroups by level
```

On macOS you can fully automate it with a `launchd` agent that runs `build_cohort.py auto --if-changed`, triggered two ways: `WatchPaths` on `athletes.json` (so **adding/editing an athlete rebuilds immediately**) plus a `StartInterval` poll (e.g. 180s) that catches new swings for existing athletes. `--if-changed` makes the poll cheap — it exits without work unless a `.mot`/`.trc` (or the config) is newer than the model. Because the agent references only `athletes.json` and `build_cohort.py` — never individual athlete folders — **you never edit the plist again.** Use a scipy‑enabled Python so the cohort numerics match live analysis.

**Auto‑discovery.** Add a `discover_roots` list to the config (e.g. `["~/Downloads"]`) and any *new* folder under those roots that contains `.mot` files is auto‑registered as a stub entry flagged `needs_demographics`. Stubs are **held out of the cohort** until you fill in their level + height/weight (guessed body size would distort the physics) — so a new folder is captured automatically, and you only supply the demographics.

**Backup.** `athletes.json` and `cohort_percentiles.json` hold athlete demographics, so they're gitignored in this (public) repo. Keep them in a separate **private** repo instead: store the real files there and symlink them back into `backend/`, and have the launchd wrapper `git commit && git push` that private repo after each rebuild. Your config and model are then versioned and backed up, and new discovered athletes sync automatically.

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
- Everything **without** a `*` (Negative Move, Pelvis Load, Upper Torso Load, Forward Move, Energy Transfer, Follow‑Through Quality, and the raw energy/power/GRF readouts) is a **relative benchmark** — great for tracking your own swing‑to‑swing progress, not for ranking against other players.

---

### Where these numbers come from
For the research basis of every metric — the citation behind it, how strong the
evidence is, and which numbers are our own composites rather than established
science — see **[METRICS_RESEARCH.md](METRICS_RESEARCH.md)**.
