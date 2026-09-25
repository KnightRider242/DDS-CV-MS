# Sprint 1 — Whole-Face KLT Tracking

## Goal

Build a KLT-based facial motion tracker and log motion features from video.

## Pipeline

- Haar face detection
- Shi–Tomasi feature detection
- pyramidal Lucas–Kanade optical flow
- forward-backward tracking error
- tracking confidence
- CSV logging

## Logged features

`timestamp`, `num_tracked_points`, `mean_u`, `mean_v`, `std_u`, `std_v`, `mean_speed`, `forward_backward_error`, and `tracking_confidence`.

## Observation

Whole-face motion was useful for identifying large head movement, but it was not sufficient for slow eye-closure behaviour. This motivated ROI-based tracking in Sprint 3.
