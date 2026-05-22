# DDS-CV-MS
Building a system that detects drowsiness from a webcam by combining eye closure, blink duration, yawning, and head-nod motion, using concepts from your Computer Vision lectures instead of only a black-box CNN.

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