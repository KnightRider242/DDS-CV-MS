# Research Plan

## Working title

**Confidence-Aware Temporal Facial Motion Analysis for Driver Drowsiness Detection Using KLT Tracking and Optical Flow**

## Research question

Can temporal facial motion features from KLT tracking and optical flow improve drowsiness detection compared with static eye/mouth measurements?

## Hypothesis

Drowsiness-related behaviours contain temporal patterns that are not fully represented by single-frame measurements. Tracking reliability should be treated as an explicit signal so that motion caused by tracking degradation is not interpreted as behaviour.

## Core pipeline

1. Detect the face.
2. Define facial regions of interest.
3. Detect Shi–Tomasi features.
4. Track features with pyramidal Lucas–Kanade/KLT.
5. Compute displacement, speed, vertical motion, and temporal statistics.
6. Estimate reliability using feature coverage and forward-backward error.
7. Extract temporal biomarkers.
8. Evaluate a non-RL baseline before introducing an adaptive decision layer.

## Current Sprint 4 evidence

The full labelled YawDD Mirror subset has been processed at the temporal-feature level:

- 320 videos
- 90 subjects
- 1,627 five-second windows

Subject-level analysis uses a window → video → subject × condition hierarchy so that temporal windows are not treated as independent observations.

Normal/Talking comparisons across 90 paired subjects show differences in eye, mouth, and head temporal motion summaries. Normal/Yawning comparisons show that yawning-labelled recordings do not simply have higher raw motion than Normal. The binary reliable-frame percentage is close to saturation, while feature-point counts and forward-backward error provide more continuous tracking-quality variation.

These results support further testing of confidence-aware temporal features, but they do not establish a drowsiness classifier because YawDD labels yawning/talking behaviour rather than genuine drowsiness.

## Planned evaluation

- Static eye/mouth measurements vs temporal KLT features
- Motion-only vs motion + reliability
- Motion + temporal statistics vs motion + temporal statistics + reliability
- ROI ablation
- Tracking-component ablation
- Robustness to illumination, head pose, and tracking loss
- Subject-independent train/test separation
- Multiple subjects and sessions

## Feature groups for the next experiment

### A. Motion only

- eye motion
- mouth motion
- head motion

### B. Motion + temporal statistics

- temporal means
- variability
- P95 motion
- other window-level temporal descriptors

### C. Motion + temporal statistics + reliability

- feature-point coverage
- forward-backward error
- tracking-quality temporal statistics

The next comparison should use subject-independent splits and evaluate whether reliability features add predictive information beyond temporal motion features.

## Scope note

YawDD is useful for temporal facial-motion characterization, particularly yawning and talking, but yawning is not equivalent to drowsiness. A genuine drowsiness dataset is required for the final drowsiness-classification claim.

Reinforcement learning is a later extension. The perception and temporal baseline should be validated first.
