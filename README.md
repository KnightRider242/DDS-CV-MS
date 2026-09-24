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

The current Sprint 3 tracker uses four approximate facial regions:

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

Record approximately 30–60 seconds for each condition:

1. normal
2. blinking
3. fake drowsy / slow eye closure
4. head nod
5. optional: yawning

Save each session as a separate CSV in `outputs/`.

### Research Question for Sprint 3

Does region-specific KLT produce more discriminative temporal motion features for eye closure, blinking, yawning, and head nodding than whole-face KLT?

### Expected Analysis

The results should be analysed rather than assumed. In particular:

- compare eye motion between normal, blinking, and fake-drowsy sessions
- compare mouth motion during yawning
- compare head motion during nodding
- examine tracking confidence and forward-backward error for each ROI
- determine whether the ROI signals provide clearer temporal separation than the Sprint 2 whole-face signals

### Status

**Sprint 3 implementation complete. Experimental data collection pending.**

After the four main recordings are complete, the next step is to generate comparable plots and quantify the ROI-specific motion distributions before moving to windowed features and classification.
