# DDS-CV-MS

## Confidence-Aware Temporal Facial Motion Analysis for Driver Drowsiness Detection

Research project exploring whether temporal facial motion from KLT tracking and optical flow can improve driver drowsiness analysis compared with static eye/mouth measurements.

### Research question

Can temporal facial motion features from KLT tracking and optical flow improve drowsiness detection compared with static eye/mouth measurements?

### Working hypothesis

Drowsiness-related behaviour can produce temporal patterns such as slow eyelid closing, prolonged eye closure, yawning, head nodding, and changes in facial motion stability. A confidence-aware tracker may reduce false alarms when lighting, head pose, or tracking quality changes.

> Current scope: establish and evaluate the temporal-tracking baseline first. Reinforcement learning is a later extension, not part of the current baseline.

---

## Current status — Sprint 4

Sprint 1–3 established the KLT tracking pipeline and a confidence/reliability analysis on controlled pilot recordings.

Sprint 4 extends the pipeline to the **YawDD** video dataset and focuses on temporal facial-motion characterization.

Current Sprint 4 work includes:

- YawDD video metadata generation
- subject-aware metadata using `subject_uid`
- a 12-video pilot spanning Normal, Talking, Yawning, and Talking&Yawning
- ROI-based KLT tracking of left eye, right eye, mouth, and head
- two-stage Haar face initialization for difficult poses
- video-time-based temporal indexing using source FPS
- tracking reliability using point coverage and forward-backward error
- 5-second temporal-window feature extraction
- full 320-video Mirror temporal processing
- subject-level temporal aggregation
- paired subject-level statistical analysis

The current pilot contains 12 videos from 12 unique subjects. The full YawDD Mirror subset contains 320 labelled recordings; the 29 Dash recordings are kept separate because their filenames do not provide the same condition labels.

### Important interpretation

YawDD is being used here to study **temporal facial motion and tracking behaviour**, especially yawning/talking effects. Yawning should not be treated as equivalent to drowsiness.

The current pilot is exploratory. It is not yet a subject-independent drowsiness classifier and does not establish generalization.

---

## Pipeline

```text
Video
  ↓
Face detection
  ↓
Facial ROIs
  ├── left eye
  ├── right eye
  ├── mouth
  └── head
  ↓
Shi–Tomasi feature detection
  ↓
Pyramidal Lucas–Kanade / KLT tracking
  ↓
Motion features
  ├── displacement
  ├── velocity / speed
  ├── vertical motion
  └── temporal statistics
  ↓
Tracking reliability
  ├── feature-point coverage
  └── forward-backward error
  ↓
Temporal facial-motion features
  ↓
Future classification / adaptive decision layer
```

---

## Repository structure

```text
DDS-CV-MS/
├── README.md
├── LICENSE
├── requirements.txt
│
├── docs/
│   ├── research_plan.md
│   ├── sprint1.md
│   ├── sprint2.md
│   ├── sprint3.md
│   └── sprint4.md
│
├── src/
│   ├── tracking/
│   │   ├── track_klt_face.py
│   │   └── track_roi_klt.py
│   │
│   ├── analysis/
│   │   ├── analyze_roi_klt.py
│   │   ├── plot_klt_results.py
│   │   ├── sprint3_robust.py
│   │   ├── analyze_yawdd_pilot.py
│   │   └── analyze_yawdd_temporal.py
│   │
│   └── datasets/
│       ├── build_yawdd_metadata.py
│       ├── select_yawdd_pilot.py
│       └── process_yawdd_pilot.py
│
├── data/       # local datasets; ignored by Git
└── outputs/    # generated results; ignored for new files
```

---

## Reproduction

Create and activate the project environment, then install dependencies:

```bash
pip install -r requirements.txt
```

### Build YawDD metadata

```bash
python src/datasets/build_yawdd_metadata.py
```

### Select the 12-video pilot

```bash
python src/datasets/select_yawdd_pilot.py
```

### Process the pilot videos

```bash
python src/datasets/process_yawdd_pilot.py
```

### Analyze pilot features

```bash
python src/analysis/analyze_yawdd_pilot.py
```

### Analyze temporal windows

```bash
python src/analysis/analyze_yawdd_temporal.py
```

### Aggregate temporal features by subject

```bash
python src/analysis/analyze_yawdd_subject.py
```

### Run subject-level paired statistics

```bash
python src/analysis/analyze_yawdd_subject_stats.py
```

Generated CSVs and figures belong under `outputs/` and should not be committed as routine generated artifacts.

The full YawDD temporal baseline currently contains 1,627 five-second windows from 320 Mirror videos and 90 subjects. Subject-level analysis is performed before inferential statistics.

---

## Dataset

The current temporal experiments use the **YawDD — Yawning Detection Dataset**.

For the labelled Mirror recordings, the current metadata contains:

- 320 Mirror videos
- 90 unique subjects
- Normal: 105
- Talking: 100
- Yawning: 102
- Talking&Yawning: 13

The Dash recordings are retained separately for future robustness experiments because their filenames do not provide equivalent condition labels.

---

## Research roadmap

1. Establish KLT tracking and reliability.
2. Characterize temporal facial motion on YawDD.
3. Scale to the full labelled Mirror subset.
4. Build subject-independent evaluation splits.
5. Compare static measurements with temporal KLT features.
6. Evaluate confidence-aware versus motion-only features.
7. Add ROI and component ablations.
8. Test robustness to illumination, pose, and tracking degradation.
9. Introduce a genuine drowsiness dataset for drowsiness classification.
10. Explore an adaptive/RL decision layer only after the non-RL baseline is established.

Detailed sprint records are in `docs/`.
