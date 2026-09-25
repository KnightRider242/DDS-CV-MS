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

## Planned evaluation

- Static eye/mouth measurements vs temporal KLT features
- Motion-only vs motion + reliability
- ROI ablation
- Tracking-component ablation
- Robustness to illumination, head pose, and tracking loss
- Subject-independent train/test separation
- Multiple subjects and sessions

## Scope note

YawDD is useful for temporal facial-motion characterization, particularly yawning and talking, but yawning is not equivalent to drowsiness. A genuine drowsiness dataset is required for the final drowsiness-classification claim.

Reinforcement learning is a later extension. The perception and temporal baseline should be validated first.
