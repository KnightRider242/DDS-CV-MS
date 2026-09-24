# DDS-CV-MS

Building a system that detects drowsiness from a webcam by combining eye closure, blink duration, yawning, and head-nod motion.

# Confidence-Aware Driver Drowsiness Detection Using KLT and Optical Flow

## Research Question

Can temporal facial motion features from KLT tracking and optical flow improve drowsiness detection compared to static eye/mouth measurements?

## Hypothesis

Drowsiness produces measurable temporal motion patterns:
- slow eyelid closing
- long eye closure
- yawning mouth expansion
- downward head nodding
- reduced facial motion stability

A confidence-aware tracker can reduce false alarms when lighting, head pose, or tracking quality changes.

## Sprint 1 Goal

Build a KLT-based facial motion tracker and log motion features from webcam video.

## Sprint 2 Observations

The KLT motion tracker successfully produces measurable temporal motion signatures.

The head-nod sequence is clearly separable from the other behaviours. It produces large vertical displacement in `mean_v`, high `mean_speed`, and high `std_v`.

Blinking creates short motion spikes, especially in `mean_speed`, but the signal is not yet strongly separated because the current tracker follows the whole face instead of eye-specific regions.

The fake-drowsy sequence has low whole-face motion. This shows that whole-face KLT alone is insufficient for slow eye-closure detection. The next step is to split the face into eye, mouth, and head ROIs and compute region-specific motion features.

Tracking confidence remains high across all sessions, and forward-backward error is generally low, meaning the KLT tracking itself is stable.

## Current Features

- number of tracked points
- mean horizontal displacement
- mean vertical displacement
- motion variance
- average motion speed
- forward-backward tracking error
- tracking confidence

## Sprint 3 — ROI-Specific KLT Tracking

### Objective

Extend the whole-face KLT tracker into region-specific tracking so that motion from the eyes, mouth, and head can be analysed independently.

### ROIs

The Sprint 3 tracker uses four approximate facial regions:

- left eye
- right eye
- mouth
- head

The regions are derived from the detected face bounding box. Shi–Tomasi features are detected independently inside each ROI and tracked between consecutive frames using pyramidal Lucas–Kanade optical flow.

### Sprint 3 Features

For each ROI, the tracker records:

- number of tracked points
- mean horizontal displacement
- mean vertical displacement
- horizontal motion standard deviation
- vertical motion standard deviation
- mean motion speed
- forward-backward error
- tracking confidence

Additional combined features are:

- `eye_mean_speed`
- `eye_mean_v`
- `mouth_speed`
- `mouth_mean_v`
- `head_speed`
- `head_mean_v`
- `overall_tracking_confidence`

### Robustness Improvements

Sprint 3 also improves the tracking pipeline by:

- automatically locating the OpenCV Haar cascade
- using histogram equalisation before face detection
- waiting for a detectable face instead of requiring detection on the first webcam frame
- periodically redetecting the face to update ROI locations
- reinitialising ROI features when too few trackable points remain
- using forward-backward error to reject unreliable correspondences

### Sprint 3 Experiment

Four recordings were collected:

1. normal
2. blinking
3. fake drowsy / slow eye closure
4. head nod

Each recording was approximately 50–67 seconds long and was saved as a separate CSV in `outputs/`.

Recorded sessions:

| Condition | Frames | Duration |
|---|---:|---:|
| normal | 1324 | 66.50 s |
| blinking | 1157 | 58.10 s |
| fake drowsy | 1005 | 50.44 s |
| head nod | 1083 | 54.36 s |

### Sprint 3 Robust Analysis

A second analysis stage was added after the initial ROI analysis.

The analysis computes:

- point coverage relative to the initial feature count
- per-ROI forward-backward error thresholds
- per-ROI tracking reliability
- all-ROI reliable frames
- raw motion distributions
- motion distributions after reliability filtering

The normal-session 95th percentile forward-backward error was used as the baseline reliability threshold for each ROI:

| ROI | Normal 95th percentile FB error |
|---|---:|
| left eye | 0.04524 |
| right eye | 0.03409 |
| mouth | 0.00845 |
| head | 0.01191 |

These thresholds are analysis heuristics derived from the current normal recording; they are not trained classifier thresholds.

### Sprint 3 Results — Tracking Reliability

| Condition | All-ROI reliable frames | Mean point coverage |
|---|---:|---:|
| normal | 88.6% | 87.5% |
| blinking | 88.9% | 80.2% |
| fake drowsy | 47.9% | 76.0% |
| head nod | 45.9% | 55.4% |

The results show that tracking reliability decreases substantially during the fake-drowsy and head-nod recordings. Head nodding produces the largest reduction in feature coverage.

This demonstrates that optical-flow measurements should be interpreted together with tracking quality rather than using motion magnitude alone.

### Sprint 3 Results — Reliable-Frame Motion

Motion values below are calculated only from frames where all four ROIs satisfy their forward-backward error thresholds.

| Condition | Eye motion mean | Mouth motion mean | Head motion mean |
|---|---:|---:|---:|
| normal | 0.218 | 0.239 | 0.197 |
| blinking | 0.156 | 0.158 | 0.145 |
| fake drowsy | 0.349 | 0.428 | 0.348 |
| head nod | 0.463 | 0.535 | 0.451 |

The pilot recordings show different motion distributions:

- normal has the lowest overall reliable-frame motion
- blinking is characterised by short temporal events rather than high average motion
- the fake-drowsy recording has higher reliable-frame eye, mouth, and head motion than normal
- head nod produces the highest reliable-frame motion among the four recordings

The fake-drowsy condition is a simulated behaviour and should not be treated as ground-truth clinical or real-world drowsiness.

### Important Sprint 3 Finding — Confidence-Aware Tracking

The original `overall_tracking_confidence` metric is close to 1.0 across the recordings and therefore does not provide enough information about cumulative tracking degradation.

Sprint 3 analysis therefore uses two more informative reliability signals:

1. **Point coverage** — the fraction of the initial feature points that remain tracked.
2. **Forward-backward error** — the consistency of a feature's forward and backward optical-flow displacement.

This reveals an important difference between:

- large motion with reliable tracking
- large apparent motion caused by degraded tracking

For example, raw eye motion during head nodding has a mean of 2.404, but after restricting the analysis to reliable frames it falls to 0.463. This shows why confidence-aware filtering is important for interpreting KLT motion.

### Research Interpretation

Sprint 3 provides a pilot demonstration that ROI-specific KLT can produce region-dependent temporal motion measurements while also exposing tracking degradation under larger facial/head movements.

The results do not establish statistical significance or generalisation because only one recording was collected for each condition.

The main conclusion from Sprint 3 is:

> **Motion magnitude alone is not sufficient. Temporal facial motion should be combined with tracking reliability measures such as point coverage and forward-backward error.**

This supports the confidence-aware direction of the project.

### Research Question for Sprint 3

Does region-specific KLT produce more discriminative temporal motion features for eye closure, blinking, yawning, and head nodding than whole-face KLT?

The pilot results indicate that ROI-specific tracking provides useful region-level signals, but temporal event analysis is still required before determining how well the signals separate behaviours.

### Limitations Identified in Sprint 3

- only one recording per condition was collected
- the fake-drowsy condition is simulated
- the eye and mouth ROIs are approximate regions derived from the face bounding box
- the current overall tracking-confidence metric saturates near 1.0
- feature coverage decreases over time and is not yet continuously replenished
- `face_detected` is recorded during periodic face redetection and should not be interpreted as a frame-by-frame face-presence label
- simple motion thresholds are not sufficient to identify blink events reliably

These limitations will guide the next tracker and feature-engineering iteration.

### Status

**Sprint 3 implementation and pilot analysis complete.**

Generated analysis outputs are stored under:

`outputs/sprint3_robust/`

The next stage is **Sprint 4 — Temporal Feature Extraction**, focusing on:

- blink/event detection
- eye closing and opening speed
- event duration
- sustained low-motion windows
- head-nod velocity and duration
- mouth-opening temporal events
- confidence-weighted temporal features
- comparison of raw versus reliability-filtered motion

The goal is to move from frame-level motion measurements toward interpretable temporal biomarkers that can later be used for drowsiness classification and ablation experiments.
