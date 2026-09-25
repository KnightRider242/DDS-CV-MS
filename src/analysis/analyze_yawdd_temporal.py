from pathlib import Path
import pandas as pd
import numpy as np

#INPUT_DIR = Path("outputs/YawDD/pilot_csv")
INPUT_DIR = Path("outputs/YawDD/full_csv")
OUTPUT_DIR = Path("outputs/YawDD/temporal_analysis")
METADATA = Path("outputs/YawDD/metadata.csv")

WINDOW_SEC = 5.0


def numeric(df, column):
    if column not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def summarize(series):
    series = series.dropna()

    if len(series) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }

    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "p95": series.quantile(0.95),
        "max": series.max(),
    }


def analyze_video(csv_path, metadata):

    df = pd.read_csv(csv_path)

    video_id = int(csv_path.stem.replace("video_", ""))

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------
    meta = metadata[
        metadata["video_id"] == video_id
    ]

    if len(meta) == 0:
        print(f"WARNING: No metadata for video {video_id}")
        return []

    meta = meta.iloc[0]

    # ---------------------------------------------------------
    # Required timing information
    # ---------------------------------------------------------
    if "elapsed_sec" not in df.columns:
        raise ValueError(
            f"{csv_path}: elapsed_sec column missing"
        )

    df["elapsed_sec"] = pd.to_numeric(
        df["elapsed_sec"],
        errors="coerce"
    )

    df = df.dropna(subset=["elapsed_sec"]).copy()

    if len(df) == 0:
        return []

    # Window index
    df["window"] = (
        df["elapsed_sec"] // WINDOW_SEC
    ).astype(int)

    rows = []

    # ---------------------------------------------------------
    # Process each temporal window
    # ---------------------------------------------------------
    for window_id, wdf in df.groupby("window"):

        start_sec = window_id * WINDOW_SEC
        end_sec = start_sec + WINDOW_SEC

        result = {
            "video_id": video_id,
            "subject_uid": meta["subject_uid"],
            "condition": meta["condition"],
            "gender": meta["gender"],
            "glasses": meta["glasses"],
            "window_id": window_id,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "frames": len(wdf),
        }

        # -----------------------------------------------------
        # Motion
        # -----------------------------------------------------
        motion_columns = {
            "eye_motion": "eye_mean_speed",
            "mouth_motion": "mouth_speed",
            "head_motion": "head_speed",
        }

        for name, column in motion_columns.items():

            values = numeric(wdf, column)

            stats = summarize(values)

            for stat_name, value in stats.items():
                result[f"{name}_{stat_name}"] = value

        # -----------------------------------------------------
        # Vertical motion
        # -----------------------------------------------------
        vertical_columns = {
            "eye_vertical": "eye_mean_v",
            "mouth_vertical": "mouth_mean_v",
            "head_vertical": "head_mean_v",
        }

        for name, column in vertical_columns.items():

            values = numeric(wdf, column)

            result[f"{name}_mean"] = values.mean()
            result[f"{name}_std"] = values.std()

            if values.dropna().empty:
                result[f"{name}_p95_abs"] = np.nan
            else:
                result[f"{name}_p95_abs"] = (
                    values.abs().quantile(0.95)
                )

        # -----------------------------------------------------
        # Tracking point counts
        # -----------------------------------------------------
        point_columns = [
            "left_eye_points",
            "right_eye_points",
            "mouth_points",
            "head_points",
        ]

        existing_points = [
            c for c in point_columns
            if c in wdf.columns
        ]

        if existing_points:

            points = wdf[existing_points].apply(
                pd.to_numeric,
                errors="coerce"
            )

            result["mean_points"] = (
                points.mean().mean()
            )

            reliable = (
                points >= 8
            ).all(axis=1)

            result["reliable_frame_pct"] = (
                reliable.mean() * 100
            )

        # -----------------------------------------------------
        # Forward-backward error
        # -----------------------------------------------------
        fb_columns = [
            "left_eye_fb_error",
            "right_eye_fb_error",
            "mouth_fb_error",
            "head_fb_error",
        ]

        existing_fb = [
            c for c in fb_columns
            if c in wdf.columns
        ]

        if existing_fb:

            fb = wdf[existing_fb].apply(
                pd.to_numeric,
                errors="coerce"
            )

            result["mean_fb_error"] = (
                fb.mean().mean()
            )

            stacked = fb.stack()

            if len(stacked) > 0:
                result["fb_error_p95"] = (
                    stacked.quantile(0.95)
                )
            else:
                result["fb_error_p95"] = np.nan

        rows.append(result)

    return rows


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_files = sorted(
        INPUT_DIR.glob("video_*.csv")
    )

    if not csv_files:
        print(
            f"No CSV files found in {INPUT_DIR}"
        )
        return

    if not METADATA.exists():
        print(
            f"Metadata file not found: {METADATA}"
        )
        return

    metadata = pd.read_csv(METADATA)

    print("=" * 70)
    print("YawDD TEMPORAL PILOT ANALYSIS")
    print("=" * 70)
    print(f"Videos  : {len(csv_files)}")
    print(f"Window  : {WINDOW_SEC:.1f} seconds")
    print()

    all_rows = []

    for csv_path in csv_files:

        print(
            f"Processing {csv_path.name}"
        )

        try:

            rows = analyze_video(
                csv_path,
                metadata
            )

            all_rows.extend(rows)

        except Exception as e:

            print(
                f"ERROR in {csv_path.name}: {e}"
            )

    temporal = pd.DataFrame(all_rows)

    if temporal.empty:
        print("No temporal data generated.")
        return

    # ---------------------------------------------------------
    # Save window-level data
    # ---------------------------------------------------------
    temporal_path = (
        OUTPUT_DIR /
        "temporal_window_features.csv"
    )

    temporal.to_csv(
        temporal_path,
        index=False
    )

    # ---------------------------------------------------------
    # Condition-level temporal summary
    # ---------------------------------------------------------
    motion_columns = [
        c for c in [
            "eye_motion_mean",
            "mouth_motion_mean",
            "head_motion_mean",
            "eye_motion_p95",
            "mouth_motion_p95",
            "head_motion_p95",
        ]
        if c in temporal.columns
    ]

    if motion_columns:

        condition_summary = (
            temporal
            .groupby("condition")[motion_columns]
            .agg(["mean", "median", "std"])
        )

        condition_summary.to_csv(
            OUTPUT_DIR /
            "temporal_condition_summary.csv"
        )

    # ---------------------------------------------------------
    # Reliability summary
    # ---------------------------------------------------------
    reliability_columns = [
        c for c in [
            "reliable_frame_pct",
            "mean_points",
            "mean_fb_error",
            "fb_error_p95",
        ]
        if c in temporal.columns
    ]

    if reliability_columns:

        reliability_summary = (
            temporal
            .groupby("condition")[reliability_columns]
            .agg(["mean", "median", "std"])
        )

        reliability_summary.to_csv(
            OUTPUT_DIR /
            "temporal_reliability_summary.csv"
        )

    # ---------------------------------------------------------
    # Print window counts
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("TEMPORAL WINDOWS")
    print("=" * 70)

    print(
        temporal
        .groupby("condition")
        .size()
        .to_string()
    )

    # ---------------------------------------------------------
    # Print motion
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("TEMPORAL MOTION SUMMARY")
    print("=" * 70)

    if motion_columns:

        print(
            temporal
            .groupby("condition")[motion_columns]
            .agg(["mean", "median"])
            .round(4)
            .to_string()
        )

    # ---------------------------------------------------------
    # Print reliability
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("TEMPORAL RELIABILITY SUMMARY")
    print("=" * 70)

    if reliability_columns:

        print(
            temporal
            .groupby("condition")[reliability_columns]
            .agg(["mean", "median"])
            .round(4)
            .to_string()
        )

    print()
    print("=" * 70)
    print("TEMPORAL ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Window features : {temporal_path}"
    )

    print(
        f"Total windows   : {len(temporal)}"
    )


if __name__ == "__main__":
    main()