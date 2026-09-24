import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FILES = {
    "normal": "outputs/normal_roi_klt.csv",
    "blinking": "outputs/blinking_roi_klt.csv",
    "fake_drowsy": "outputs/fake_drowsy_roi_klt.csv",
    "head_nod": "outputs/head_nod_roi_klt.csv",
}

OUTPUT_DIR = "outputs/sprint3_robust"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROIS = ["left_eye", "right_eye", "mouth", "head"]

MOTION_COLUMNS = [
    "eye_mean_speed",
    "mouth_speed.1",
    "head_speed.1",
]


def load_data():
    data = {}

    for condition, path in FILES.items():
        print(f"Loading {path}")

        df = pd.read_csv(path)

        # Rename duplicated summary columns to clearer names
        df = df.rename(columns={
            "mouth_speed.1": "mouth_motion",
            "head_speed.1": "head_motion",
            "mouth_mean_v.1": "mouth_motion_v",
            "head_mean_v.1": "head_motion_v",
        })

        data[condition] = df

        print(
            f"  frames={len(df)}, "
            f"duration={df['elapsed_sec'].iloc[-1]:.2f}s"
        )

    return data


def add_point_coverage(data):
    """
    Point coverage = current tracked points / initial points.

    This is different from the current tracker confidence.
    It measures how much of the original feature set remains.
    """

    for condition, df in data.items():

        for roi in ROIS:

            point_col = f"{roi}_points"
            coverage_col = f"{roi}_point_coverage"

            initial_points = df[point_col].iloc[0]

            if initial_points > 0:
                df[coverage_col] = (
                    df[point_col] / initial_points
                )
            else:
                df[coverage_col] = 0.0

        coverage_columns = [
            f"{roi}_point_coverage"
            for roi in ROIS
        ]

        df["mean_point_coverage"] = df[
            coverage_columns
        ].mean(axis=1)


def calculate_normal_fb_thresholds(normal_df):
    """
    Use the 95th percentile of NORMAL tracking error
    as a baseline reliability threshold.

    This is an analysis heuristic, not a learned classifier.
    """

    thresholds = {}

    for roi in ROIS:

        column = f"{roi}_fb_error"

        thresholds[roi] = normal_df[column].quantile(0.95)

    return thresholds


def add_reliability_flags(data, thresholds):
    """
    A frame is considered reliable for an ROI if its
    forward-backward error is within the normal-session
    95th percentile.

    all_roi_reliable means every ROI passes.
    """

    for condition, df in data.items():

        roi_flags = []

        for roi in ROIS:

            fb_column = f"{roi}_fb_error"

            flag_column = f"{roi}_reliable"

            df[flag_column] = (
                df[fb_column] <= thresholds[roi]
            )

            roi_flags.append(df[flag_column])

        df["all_roi_reliable"] = np.logical_and.reduce(
            roi_flags
        )


def print_reliability(data, thresholds):

    print("\n=== NORMAL-BASED FB ERROR THRESHOLDS ===")

    for roi, threshold in thresholds.items():
        print(
            f"{roi:12s}: {threshold:.5f}"
        )

    print("\n=== TRACKING RELIABILITY ===")

    rows = []

    for condition, df in data.items():

        row = {
            "condition": condition,
            "reliable_all_roi_pct":
                100 * df["all_roi_reliable"].mean(),
            "mean_point_coverage":
                df["mean_point_coverage"].mean(),
        }

        for roi in ROIS:
            row[f"{roi}_coverage"] = (
                df[f"{roi}_point_coverage"].mean()
            )

        rows.append(row)

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    result.to_csv(
        f"{OUTPUT_DIR}/tracking_reliability.csv",
        index=False
    )


def motion_summary(data, reliable_only=False):

    rows = []

    for condition, df in data.items():

        if reliable_only:
            subset = df[df["all_roi_reliable"]]
        else:
            subset = df

        for metric in [
            "eye_mean_speed",
            "mouth_motion",
            "head_motion",
        ]:

            values = subset[metric].dropna()

            if len(values) == 0:
                continue

            rows.append({
                "condition": condition,
                "metric": metric,
                "reliable_only": reliable_only,
                "n": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),
                "p90": values.quantile(0.90),
                "p95": values.quantile(0.95),
                "p99": values.quantile(0.99),
                "max": values.max(),
            })

    return pd.DataFrame(rows)


def plot_boxplots(data):

    metrics = {
        "eye_mean_speed": "Eye Motion Speed",
        "mouth_motion": "Mouth Motion Speed",
        "head_motion": "Head Motion Speed",
        "mean_point_coverage": "Mean Point Coverage",
    }

    conditions = list(FILES.keys())

    for metric, title in metrics.items():

        values = []
        labels = []

        for condition in conditions:

            if metric not in data[condition]:
                continue

            values.append(
                data[condition][metric].dropna().values
            )

            labels.append(condition)

        if not values:
            continue

        plt.figure(figsize=(10, 6))

        plt.boxplot(
            values,
            tick_labels=labels,
            showmeans=True
        )

        plt.title(title)
        plt.ylabel(title)
        plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        filename = (
            metric.replace("/", "_")
            + "_boxplot.png"
        )

        plt.savefig(
            os.path.join(OUTPUT_DIR, filename),
            dpi=200
        )

        plt.close()


def plot_fb_error(data):

    for roi in ROIS:

        column = f"{roi}_fb_error"

        values = []
        labels = []

        for condition in FILES:

            values.append(
                data[condition][column].dropna().values
            )

            labels.append(condition)

        plt.figure(figsize=(10, 6))

        plt.boxplot(
            values,
            tick_labels=labels,
            showmeans=True
        )

        plt.title(
            f"{roi.replace('_', ' ').title()} "
            "Forward-Backward Error"
        )

        plt.ylabel("Forward-Backward Error")

        plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"{roi}_fb_error_boxplot.png"
            ),
            dpi=200
        )

        plt.close()


def main():

    print("\n=== Sprint 3 Robust Analysis ===\n")

    data = load_data()

    # ------------------------------------------
    # Point coverage
    # ------------------------------------------

    add_point_coverage(data)

    # ------------------------------------------
    # Reliability thresholds
    # ------------------------------------------

    thresholds = calculate_normal_fb_thresholds(
        data["normal"]
    )

    add_reliability_flags(
        data,
        thresholds
    )

    print_reliability(
        data,
        thresholds
    )

    # ------------------------------------------
    # Raw motion summary
    # ------------------------------------------

    raw_summary = motion_summary(
        data,
        reliable_only=False
    )

    raw_summary.to_csv(
        f"{OUTPUT_DIR}/raw_motion_summary.csv",
        index=False
    )

    # ------------------------------------------
    # Reliable motion summary
    # ------------------------------------------

    reliable_summary = motion_summary(
        data,
        reliable_only=True
    )

    reliable_summary.to_csv(
        f"{OUTPUT_DIR}/reliable_motion_summary.csv",
        index=False
    )

    print("\n=== RAW MOTION ===")

    print(
        raw_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n=== RELIABLE-FRAME MOTION ===")

    print(
        reliable_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ------------------------------------------
    # Save derived frame-level data
    # ------------------------------------------

    combined = []

    for condition, df in data.items():

        temp = df.copy()
        temp["condition"] = condition

        combined.append(temp)

    combined = pd.concat(
        combined,
        ignore_index=True
    )

    combined.to_csv(
        f"{OUTPUT_DIR}/sprint3_robust_frame_data.csv",
        index=False
    )

    # ------------------------------------------
    # Plots
    # ------------------------------------------

    plot_boxplots(data)
    plot_fb_error(data)

    print("\nAnalysis complete.")
    print(
        f"Results saved to: {OUTPUT_DIR}/"
    )


if __name__ == "__main__":
    main()