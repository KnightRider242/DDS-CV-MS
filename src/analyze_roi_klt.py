import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILES = {
    "normal": "outputs/normal_roi_klt.csv",
    "blinking": "outputs/blinking_roi_klt.csv",
    "fake_drowsy": "outputs/fake_drowsy_roi_klt.csv",
    "head_nod": "outputs/head_nod_roi_klt.csv",
}

OUTPUT_DIR = "outputs/sprint3_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_columns(df, roi, keyword):
    """
    Find columns belonging to a ROI and containing a keyword.
    """
    roi = roi.lower()
    keyword = keyword.lower()

    return [
        c for c in df.columns
        if roi in c.lower() and keyword in c.lower()
    ]


def safe_mean(df, columns):
    """
    Row-wise mean across available columns.
    """
    if not columns:
        return pd.Series(np.nan, index=df.index)

    return df[columns].mean(axis=1)


def derive_features(df):
    """
    Convert ROI-specific tracker measurements into
    comparable Sprint 3 features.
    """

    out = pd.DataFrame(index=df.index)

    # --------------------------------------------------
    # Eye motion
    # --------------------------------------------------

    left_eye_speed = find_columns(df, "left_eye", "speed")
    right_eye_speed = find_columns(df, "right_eye", "speed")

    eye_speed_columns = left_eye_speed + right_eye_speed

    out["eye_speed"] = safe_mean(df, eye_speed_columns)

    left_eye_v = find_columns(df, "left_eye", "mean_v")
    right_eye_v = find_columns(df, "right_eye", "mean_v")

    eye_v_columns = left_eye_v + right_eye_v

    out["eye_vertical"] = safe_mean(df, eye_v_columns)

    # --------------------------------------------------
    # Mouth motion
    # --------------------------------------------------

    mouth_speed = find_columns(df, "mouth", "speed")
    out["mouth_speed"] = safe_mean(df, mouth_speed)

    mouth_v = find_columns(df, "mouth", "mean_v")
    out["mouth_vertical"] = safe_mean(df, mouth_v)

    # --------------------------------------------------
    # Head motion
    # --------------------------------------------------

    head_speed = find_columns(df, "head", "speed")
    out["head_speed"] = safe_mean(df, head_speed)

    head_v = find_columns(df, "head", "mean_v")
    out["head_vertical"] = safe_mean(df, head_v)

    # --------------------------------------------------
    # Tracking confidence
    # --------------------------------------------------

    confidence_columns = [
        c for c in df.columns
        if "confidence" in c.lower()
    ]

    if "overall_tracking_confidence" in df.columns:
        out["tracking_confidence"] = df["overall_tracking_confidence"]
    else:
        out["tracking_confidence"] = safe_mean(
            df,
            confidence_columns
        )

    # --------------------------------------------------
    # Forward-backward error
    # --------------------------------------------------

    fb_columns = [
        c for c in df.columns
        if "forward_backward" in c.lower()
        or "fb_error" in c.lower()
    ]

    out["forward_backward_error"] = safe_mean(
        df,
        fb_columns
    )

    return out


def load_all_data():
    all_data = []

    for condition, filepath in INPUT_FILES.items():

        if not os.path.exists(filepath):
            print(f"WARNING: missing {filepath}")
            continue

        print(f"Loading: {filepath}")

        df = pd.read_csv(filepath)

        print(f"  Frames: {len(df)}")
        print(f"  Columns: {len(df.columns)}")

        features = derive_features(df)
        features["condition"] = condition

        all_data.append(features)

    if not all_data:
        raise RuntimeError("No CSV files were found.")

    return pd.concat(all_data, ignore_index=True)


def create_summary(data):
    metrics = [
        "eye_speed",
        "eye_vertical",
        "mouth_speed",
        "mouth_vertical",
        "head_speed",
        "head_vertical",
        "tracking_confidence",
        "forward_backward_error",
    ]

    rows = []

    for condition in data["condition"].unique():

        subset = data[data["condition"] == condition]

        for metric in metrics:

            if metric not in subset.columns:
                continue

            values = subset[metric].dropna()

            if len(values) == 0:
                continue

            rows.append({
                "condition": condition,
                "metric": metric,
                "n": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),
                "p95": values.quantile(0.95),
                "max": values.max(),
            })

    summary = pd.DataFrame(rows)

    summary.to_csv(
        f"{OUTPUT_DIR}/sprint3_summary.csv",
        index=False
    )

    return summary


def plot_metric(data, metric, ylabel, filename):

    if metric not in data.columns:
        print(f"Skipping {metric}: column not available")
        return

    plt.figure(figsize=(10, 6))

    conditions = list(INPUT_FILES.keys())

    values = []
    labels = []

    for condition in conditions:

        subset = data[
            data["condition"] == condition
        ][metric].dropna()

        if len(subset) > 0:
            values.append(subset.values)
            labels.append(condition)

    if not values:
        plt.close()
        return

    plt.boxplot(
        values,
        tick_labels=labels,
        showmeans=True
    )

    plt.ylabel(ylabel)
    plt.title(f"Sprint 3: {ylabel} by Condition")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/{filename}",
        dpi=200
    )

    plt.close()


def plot_time_series(data, metric, ylabel, filename):

    if metric not in data.columns:
        return

    for condition in INPUT_FILES.keys():

        subset = data[
            data["condition"] == condition
        ][metric].dropna()

        if len(subset) == 0:
            continue

        plt.figure(figsize=(12, 5))

        plt.plot(subset.values)

        plt.xlabel("Frame")
        plt.ylabel(ylabel)
        plt.title(
            f"{condition}: {ylabel} over Time"
        )

        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            f"{OUTPUT_DIR}/{condition}_{filename}",
            dpi=200
        )

        plt.close()


def main():

    print("\n=== Sprint 3 ROI-KLT Analysis ===\n")

    data = load_all_data()

    # Save derived frame-level features
    data.to_csv(
        f"{OUTPUT_DIR}/sprint3_derived_features.csv",
        index=False
    )

    summary = create_summary(data)

    print("\n=== SUMMARY ===\n")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # -----------------------------------------------
    # Boxplots
    # -----------------------------------------------

    plot_metric(
        data,
        "eye_speed",
        "Eye Motion Speed",
        "eye_speed_boxplot.png"
    )

    plot_metric(
        data,
        "mouth_speed",
        "Mouth Motion Speed",
        "mouth_speed_boxplot.png"
    )

    plot_metric(
        data,
        "head_speed",
        "Head Motion Speed",
        "head_speed_boxplot.png"
    )

    plot_metric(
        data,
        "tracking_confidence",
        "Tracking Confidence",
        "tracking_confidence_boxplot.png"
    )

    # -----------------------------------------------
    # Time-series plots
    # -----------------------------------------------

    plot_time_series(
        data,
        "eye_speed",
        "Eye Motion Speed",
        "eye_speed_timeseries.png"
    )

    plot_time_series(
        data,
        "mouth_speed",
        "Mouth Motion Speed",
        "mouth_speed_timeseries.png"
    )

    plot_time_series(
        data,
        "head_speed",
        "Head Motion Speed",
        "head_speed_timeseries.png"
    )

    print("\nAnalysis complete.")
    print(f"Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()