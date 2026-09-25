# Sprint 4 — Temporal YawDD Analysis

## Objective

Extend the ROI-KLT pipeline from short controlled recordings to a public video dataset and extract subject-aware temporal facial-motion features.

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

The labelled Mirror subset contains 320 recordings from 90 subjects:

- Normal: 105
- Talking: 100
- Yawning: 102
- Talking&Yawning: 13

The 29 Dash recordings are kept separate because their filenames do not provide equivalent condition labels.

## Pilot

The first pilot contains 12 videos from 12 unique subjects:

- 3 Normal
- 3 Talking
- 3 Yawning
- 3 Talking&Yawning

The pilot was used to validate the temporal-processing pipeline before scaling to the full labelled Mirror subset.

## Tracking and robustness

The ROI tracker uses:

- Haar face detection
- Shi–Tomasi feature detection
- pyramidal Lucas–Kanade/KLT tracking
- forward-backward error rejection
- periodic face re-detection
- ROI point reinitialisation when point counts become too low

Face initialization uses a two-stage Haar detector: the original strict detector is attempted first, followed by a more permissive fallback for difficult poses. This was needed for a Mirror recording whose side/down-turned face was not detected by the strict detector.

The tracker also supports headless processing with `--no-display`.

Elapsed time is based on source video FPS when available rather than wall-clock processing time.

## Temporal extraction

The full Mirror subset was processed into per-video ROI-KLT CSV files.

Temporal analysis uses 5-second windows and summarizes:

- mean motion
- median motion
- standard deviation
- 95th percentile
- maximum
- vertical motion
- tracking reliability
- feature-point counts
- forward-backward error

The full temporal analysis produced:

- 1,627 temporal windows
- 320 videos
- 90 subjects

Window counts:

- Normal: 482
- Talking: 652
- Talking&Yawning: 66
- Yawning: 427

## Full-dataset temporal findings

At the window level, Talking showed higher average eye, mouth, and head motion than Normal. Talking&Yawning also showed elevated mouth and head motion summaries.

Yawning-only recordings did not simply produce higher raw motion than Normal. In the full temporal summaries, Yawning had lower mean eye, mouth, and head motion than Normal.

This is important for interpretation: YawDD is being used to characterize temporal facial behaviour, not as a direct drowsiness ground truth. Raw motion magnitude should not be treated as a direct drowsiness measure.

Tracking reliability was generally high, but varied across conditions. The binary `reliable_frame_pct` measure was close to saturation, while continuous tracking-quality measures such as point counts and forward-backward error showed more variation.

## Subject-level analysis

Because temporal windows from the same video and subject are not independent, the analysis was aggregated hierarchically:

```
window
  ↓
video
  ↓
subject × condition
```

This prevents longer recordings or subjects with more windows from dominating the subject-level analysis.

Subject coverage:

- Normal: 90 subjects
- Talking: 90 subjects
- Yawning: 87 subjects
- Talking&Yawning: 13 subjects

The subject-level analysis produces:

- `video_temporal_features.csv`
- `subject_condition_features.csv`
- `subject_condition_counts.csv`
- `subject_condition_summary.csv`
- subject-level distribution plots

## Subject-level statistical analysis

Normal was compared with each labelled condition using paired subjects and a two-sided Wilcoxon signed-rank test.

Multiple testing was controlled using Benjamini-Hochberg correction within each comparison family. Rank-biserial correlation was recorded as an effect-size measure.

### Normal vs Talking

All six primary motion summaries were statistically different after correction:

- eye motion
- mouth motion
- head motion
- eye P95
- mouth P95
- head P95

The median motion differences were positive for Talking.

The binary reliable-frame percentage did not show a significant paired difference. However, mean tracked-point count decreased and mean/95th-percentile forward-backward error increased for Talking.

This suggests that binary point-count reliability alone may be too coarse, while continuous tracking-quality signals may contain useful information.

### Normal vs Yawning

Yawning showed lower eye motion and lower eye/mouth/head P95 motion than Normal after correction. Mouth mean motion and head mean motion were not significant after correction.

This reinforces that raw motion magnitude is not a direct proxy for yawning or drowsiness.

### Normal vs Talking&Yawning

Only 13 subjects were available, so these results are treated as small-sample exploratory findings. Mouth motion and mouth P95 showed corrected differences from Normal. Several tracking-quality measures also differed.

These results should not be generalized beyond the small paired sample.

## Current interpretation

The full YawDD analysis supports three methodological points:

1. ROI-KLT can provide measurable temporal facial-motion features across a larger public video dataset.
2. Subject-level aggregation is necessary before inferential testing because temporal windows are clustered within videos and subjects.
3. Tracking quality should be represented with continuous signals such as feature-point coverage and forward-backward error rather than relying only on a near-saturated binary reliable-frame percentage.

YawDD does not establish the final drowsiness-classification claim. A genuine drowsiness dataset is required for that stage.

## Next step

Take a short break after completing the YawDD temporal baseline.

The next experiment will compare feature groups using subject-independent evaluation:

1. motion-only features
2. motion + temporal statistics
3. motion + temporal statistics + tracking reliability

The purpose is to test whether confidence-aware tracking information adds measurable predictive value beyond motion features alone.
