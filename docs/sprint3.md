# Sprint 3 — ROI KLT and Tracking Reliability

## Goal

Move from whole-face tracking to region-specific temporal motion.

## ROIs

- left eye
- right eye
- mouth
- head

The tracker uses Shi–Tomasi features and pyramidal Lucas–Kanade tracking. Face detection is periodically refreshed and ROI features are reinitialized when too few points remain.

## Reliability

Sprint 3 introduced forward-backward error and point coverage as more useful reliability indicators than the existing cumulative confidence measure.

A pilot robustness analysis showed that reliability varied substantially across conditions, while raw motion magnitude alone did not provide a sufficient behavioural interpretation.

## Main conclusion

The system should distinguish:

1. **motion magnitude** — what the tracked points are doing
2. **tracking reliability** — whether the motion estimate can be trusted

This separation is the basis for the confidence-aware direction in Sprint 4.

## Limitation

The Sprint 3 controlled sequences were limited and included a simulated drowsy condition. They do not establish generalization to real-world drowsiness.
