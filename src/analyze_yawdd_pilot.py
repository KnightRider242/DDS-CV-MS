from pathlib import Path
import pandas as pd
import numpy as np


INPUT_DIR = Path("outputs/YawDD/pilot_csv")
OUTPUT_DIR = Path("outputs/YawDD/pilot_analysis")


def percentile(series, q):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        return np.nan
    return np.percentile(series, q)


def safe_mean(series):
    series = pd.to_numeric(series, errors="coerce")
    return series.mean()


def safe_median(series):
    series = pd.to_numeric(series, errors="coerce")
    return series.median()


def analyze_file(csv_path):
    df = pd.read_csv(csv_path)

    video_id = int(csv_path.stem.replace("video_", ""))

    result = {
        "video_id": video_id,
        "frames": len(df),
    }

    # ---------------------------------------------------------
    # Video timing
    # ---------------------------------------------------------
    if "elapsed_sec" in df.columns:
        result["duration_sec"] = df["elapsed_sec"].max()

    if "frame_idx" in df.columns and "elapsed_sec" in df.columns:
        duration = df["elapsed_sec"].max()
        if duration > 0:
            result["effective_fps"] = len(df) / duration

    # ---------------------------------------------------------
    # Motion features
    # ---------------------------------------------------------
    motion_columns = {
        "eye_motion": "eye_mean_speed",
        "mouth_motion": "mouth_speed",
        "head_motion": "head_speed",
    }

    for name, column in motion_columns.items():
        if column not in df.columns:
            continue

        result[f"{name}_mean"] = safe_mean(df[column])
        result[f"{name}_median"] = safe_median(df[column])
        result[f"{name}_p95"] = percentile(df[column], 95)
        result[f"{name}_max"] = percentile(df[column], 100)

    # ---------------------------------------------------------
    # Vertical motion
    # ---------------------------------------------------------
    vertical_columns = {
        "eye_vertical": "eye_mean_v",
        "mouth_vertical": "mouth_mean_v",
        "head_vertical": "head_mean_v",
    }

    for name, column in vertical_columns.items():
        if column not in df.columns:
            continue

        result[f"{name}_mean"] = safe_mean(df[column])
        result[f"{name}_std"] = df[column].std()
        result[f"{name}_p95_abs"] = percentile(df[column].abs(), 95)

    # ---------------------------------------------------------
    # Tracking reliability
    # ---------------------------------------------------------
    confidence_columns = [
        "left_eye_confidence",
        "right_eye_confidence",
        "mouth_confidence",
        "head_confidence",
    ]

    existing_conf = [
        c for c in confidence_columns if c in df.columns
    ]

    if existing_conf:
        confidence = df[existing_conf].apply(
            pd.to_numeric, errors="coerce"
        )

        result["mean_tracking_confidence"] = confidence.mean().mean()

    # Forward-backward error
    fb_columns = [
        "left_eye_fb_error",
        "right_eye_fb_error",
        "mouth_fb_error",
        "head_fb_error",
    ]

    existing_fb = [
        c for c in fb_columns if c in df.columns
    ]

    if existing_fb:
        fb = df[existing_fb].apply(
            pd.to_numeric, errors="coerce"
        )

        result["mean_fb_error"] = fb.mean().mean()
        result["median_fb_error"] = fb.stack().median()
        result["fb_error_p95"] = fb.stack().quantile(0.95)

    # ---------------------------------------------------------
    # Point coverage
    # ---------------------------------------------------------
    point_columns = [
        "left_eye_points",
        "right_eye_points",
        "mouth_points",
        "head_points",
    ]

    existing_points = [
        c for c in point_columns if c in df.columns
    ]

    if existing_points:
        points = df[existing_points].apply(
            pd.to_numeric, errors="coerce"
        )

        result["mean_points"] = points.mean().mean()

        # A frame is considered reasonably tracked if every
        # available ROI has at least 8 points.
        reliable_frames = (points >= 8).all(axis=1)

        result["reliable_frame_pct"] = (
            reliable_frames.mean() * 100
        )

    # ---------------------------------------------------------
    # Face detection
    # ---------------------------------------------------------
    if "face_detected" in df.columns:
        result["face_detected_pct"] = (
            pd.to_numeric(
                df["face_detected"],
                errors="coerce"
            ).mean() * 100
        )

    return result


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(INPUT_DIR.glob("video_*.csv"))

    if not csv_files:
        print(f"No CSV files found in {INPUT_DIR}")
        return

    print("=" * 70)
    print("YawDD PILOT ANALYSIS")
    print("=" * 70)
    print(f"Input directory : {INPUT_DIR}")
    print(f"Videos found    : {len(csv_files)}")
    print()

    results = []

    for csv_path in csv_files:
        print(f"Analyzing {csv_path.name}")

        try:
            result = analyze_file(csv_path)
            results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")

    feature_summary = pd.DataFrame(results)

    # ---------------------------------------------------------
    # Add metadata
    # ---------------------------------------------------------
    metadata_path = Path("outputs/YawDD/metadata.csv")

    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)

        metadata_columns = [
            "video_id",
            "subject_uid",
            "gender",
            "glasses",
            "condition",
            "split_type",
        ]

        metadata_columns = [
            c for c in metadata_columns
            if c in metadata.columns
        ]

        feature_summary = feature_summary.merge(
            metadata[metadata_columns],
            on="video_id",
            how="left",
        )

    # ---------------------------------------------------------
    # Save video-level summary
    # ---------------------------------------------------------
    feature_path = OUTPUT_DIR / "feature_summary.csv"

    feature_summary.to_csv(
        feature_path,
        index=False
    )

    # ---------------------------------------------------------
    # Condition-level summary
    # ---------------------------------------------------------
    if "condition" in feature_summary.columns:

        numeric_columns = feature_summary.select_dtypes(
            include=np.number
        ).columns

        condition_summary = (
            feature_summary
            .groupby("condition")[numeric_columns]
            .agg(["mean", "median"])
        )

        condition_summary.to_csv(
            OUTPUT_DIR / "condition_summary.csv"
        )

        # Reliability-specific summary
        reliability_columns = [
            c for c in [
                "reliable_frame_pct",
                "mean_points",
                "mean_fb_error",
                "median_fb_error",
                "fb_error_p95",
                "face_detected_pct",
            ]
            if c in feature_summary.columns
        ]

        if reliability_columns:

            reliability_summary = (
                feature_summary
                .groupby("condition")[reliability_columns]
                .agg(["mean", "median"])
            )

            reliability_summary.to_csv(
                OUTPUT_DIR / "reliability_summary.csv"
            )

    # ---------------------------------------------------------
    # Print important results
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("VIDEO-LEVEL FEATURES")
    print("=" * 70)

    display_columns = [
        "video_id",
        "subject_uid",
        "condition",
        "duration_sec",
        "eye_motion_mean",
        "mouth_motion_mean",
        "head_motion_mean",
        "reliable_frame_pct",
        "mean_fb_error",
    ]

    display_columns = [
        c for c in display_columns
        if c in feature_summary.columns
    ]

    print(
        feature_summary[display_columns]
        .round(4)
        .to_string(index=False)
    )

    if "condition" in feature_summary.columns:

        print()
        print("=" * 70)
        print("CONDITION-LEVEL MOTION")
        print("=" * 70)

        motion_columns = [
            c for c in [
                "eye_motion_mean",
                "mouth_motion_mean",
                "head_motion_mean",
            ]
            if c in feature_summary.columns
        ]

        print(
            feature_summary
            .groupby("condition")[motion_columns]
            .agg(["mean", "median"])
            .round(4)
            .to_string()
        )

        print()
        print("=" * 70)
        print("CONDITION-LEVEL RELIABILITY")
        print("=" * 70)

        reliability_columns = [
            c for c in [
                "reliable_frame_pct",
                "mean_points",
                "mean_fb_error",
                "fb_error_p95",
            ]
            if c in feature_summary.columns
        ]

        print(
            feature_summary
            .groupby("condition")[reliability_columns]
            .agg(["mean", "median"])
            .round(4)
            .to_string()
        )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"Feature summary      : {feature_path}")
    print(
        f"Condition summary    : "
        f"{OUTPUT_DIR / 'condition_summary.csv'}"
    )
    print(
        f"Reliability summary  : "
        f"{OUTPUT_DIR / 'reliability_summary.csv'}"
    )


if __name__ == "__main__":
    main()