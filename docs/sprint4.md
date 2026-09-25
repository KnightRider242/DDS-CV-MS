# Sprint 4 — Temporal YawDD Analysis

## Objective

Extend the ROI-KLT pipeline from short controlled recordings to a public video dataset and extract temporal facial-motion features.

## Dataset preparation

YawDD metadata is generated from the video filenames and directory structure.

The metadata includes:

- video_id
- video_path
- subject_id
- subject_uid
- gender
- glasses
- condition
- split_type

The `subject_uid` combines gender and subject identifier so that overlapping numeric subject IDs are not accidentally treated as the same person.

## Pilot

The first pilot contains 12 videos from 12 unique subjects:

- 3 Normal
- 3 Talking
- 3 Yawning
- 3 Talking&Yawning

The pilot is intentionally subject-aware.

## Temporal extraction

The tracker records elapsed video time using the source FPS rather than wall-clock processing time. This preserves the temporal scale of the original recording even when processing is slower than real time.

Temporal analysis currently uses 5-second windows and summarizes:

- mean motion
- median motion
- standard deviation
- 95th percentile
- maximum
- tracking reliability
- feature-point counts
- forward-backward error

## Pilot findings

The 66 pilot windows show substantial inter-subject variability.

Talking&Yawning produced the highest average motion across the three main ROI summaries in the pilot, especially for mouth motion. Yawning-only recordings did not simply produce higher overall motion than Normal recordings.

This supports treating raw motion as a temporal feature rather than a direct drowsiness label.

Tracking was generally stable in the pilot, with reliability above 98% for the condition-level summaries. Reliability nevertheless varies by recording and should remain an explicit feature.

## Next step

Scale from the 12-video pilot to the full 320-video labelled Mirror subset while preserving subject-independent evaluation. Keep the 29 Dash recordings separate for robustness analysis rather than assigning unsupported condition labels.
