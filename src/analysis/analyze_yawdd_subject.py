from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


INPUT = Path("outputs/YawDD/temporal_analysis/temporal_window_features.csv")
OUTPUT_DIR = Path("outputs/YawDD/subject_analysis")
PLOTS_DIR = OUTPUT_DIR / "plots"

MOTION_MEANS = [
    "eye_motion_mean",
    "mouth_motion_mean",
    "head_motion_mean",
]

MOTION_P95 = [
    "eye_motion_p95",
    "mouth_motion_p95",
    "head_motion_p95",
]

RELIABILITY = [
    "reliable_frame_pct",
    "mean_points",
    "mean_fb_error",
    "fb_error_p95",
]


def existing(columns, candidates):
    return [c for c in candidates if c in columns]


def flatten_columns(columns):
    result = []
    for col in columns:
        if isinstance(col, tuple):
            result.append("_".join(str(x) for x in col if str(x) != ""))
        else:
            result.append(str(col))
    return result


def make_boxplot(subject, column, title):
    if column not in subject.columns:
        return False

    data = []
    labels = []

    for condition, sdf in subject.groupby("condition"):
        values = pd.to_numeric(sdf[column], errors="coerce").dropna().values
        if len(values):
            data.append(values)
            labels.append(condition)

    if not data:
        return False

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, tick_labels=labels)
    plt.ylabel(column)
    plt.title(title + " — subject level")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    safe_name = column.replace("/", "_")
    plt.savefig(PLOTS_DIR / f"{safe_name}.png", dpi=180, bbox_inches="tight")
    plt.close()
    return True


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT}. Run the full YawDD temporal analysis first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT)

    required = ["video_id", "subject_uid", "condition", "window_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_features = MOTION_MEANS + MOTION_P95 + RELIABILITY
    for c in numeric_features:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Window -> video: every video receives equal weight at the next stage.
    group_cols = ["video_id", "subject_uid", "condition"]
    agg = {"window_id": "count", "frames": "sum"}

    for c in existing(df.columns, MOTION_MEANS):
        agg[c] = ["mean", "median", "std"]

    for c in existing(df.columns, MOTION_P95):
        agg[c] = ["mean", "median"]

    for c in existing(df.columns, RELIABILITY):
        agg[c] = ["mean", "median"]

    video = df.groupby(group_cols).agg(agg)

    flat = []
    for col in video.columns:
        if isinstance(col, tuple):
            if col[0] == "window_id":
                flat.append("window_count")
            elif col[0] == "frames":
                flat.append("frame_count")
            else:
                flat.append("_".join(str(x) for x in col if str(x) != ""))
        else:
            flat.append(str(col))
    video.columns = flat
    video = video.reset_index()

    video_path = OUTPUT_DIR / "video_temporal_features.csv"
    video.to_csv(video_path, index=False)

    # Video -> subject x condition: each video receives equal weight.
    subject_group = ["subject_uid", "condition"]

    numeric_cols = [
        c for c in video.columns
        if c not in group_cols
        and pd.api.types.is_numeric_dtype(video[c])
    ]

    subject = (
        video.groupby(subject_group)[numeric_cols]
        .agg(["mean", "median", "std"])
    )
    subject.columns = flatten_columns(subject.columns)
    subject = subject.reset_index()

    counts = (
        video.groupby(subject_group)
        .agg(
            video_count=("video_id", "nunique"),
            window_count=("window_count", "sum"),
        )
        .reset_index()
    )

    subject = counts.merge(subject, on=subject_group, how="left")

    subject_path = OUTPUT_DIR / "subject_condition_features.csv"
    subject.to_csv(subject_path, index=False)

    # Coverage.
    coverage = (
        video.groupby("condition")
        .agg(
            subjects=("subject_uid", "nunique"),
            videos=("video_id", "nunique"),
        )
        .reset_index()
    )

    video_counts = (
        video.groupby("condition")
        .size()
        .rename("video_rows")
        .reset_index()
    )
    coverage = coverage.merge(video_counts, on="condition")
    coverage["mean_videos_per_subject"] = (
        coverage["videos"] / coverage["subjects"]
    )

    coverage.to_csv(
        OUTPUT_DIR / "subject_condition_counts.csv",
        index=False,
    )

    # Subject-level descriptive summary.
    feature_candidates = [
        "eye_motion_mean_mean_mean",
        "mouth_motion_mean_mean_mean",
        "head_motion_mean_mean_mean",
        "eye_motion_p95_mean_mean",
        "mouth_motion_p95_mean_mean",
        "head_motion_p95_mean_mean",
        "reliable_frame_pct_mean_mean",
        "mean_points_mean_mean",
        "mean_fb_error_mean_mean",
        "fb_error_p95_mean_mean",
    ]
    feature_candidates = [
        c for c in feature_candidates if c in subject.columns
    ]

    if feature_candidates:
        condition_summary = (
            subject.groupby("condition")[feature_candidates]
            .agg(["mean", "median", "std"])
        )
        condition_summary.to_csv(
            OUTPUT_DIR / "subject_condition_summary.csv"
        )

    # Subject-level distribution plots.
    plot_specs = [
        ("eye_motion_mean_mean_mean", "Eye temporal motion"),
        ("mouth_motion_mean_mean_mean", "Mouth temporal motion"),
        ("head_motion_mean_mean_mean", "Head temporal motion"),
        ("eye_motion_p95_mean_mean", "Eye temporal P95"),
        ("mouth_motion_p95_mean_mean", "Mouth temporal P95"),
        ("head_motion_p95_mean_mean", "Head temporal P95"),
        ("reliable_frame_pct_mean_mean", "Reliable-frame percentage"),
        ("mean_fb_error_mean_mean", "Mean forward-backward error"),
    ]

    generated = 0
    for column, title in plot_specs:
        if make_boxplot(subject, column, title):
            generated += 1

    print("=" * 72)
    print("YawDD SUBJECT-LEVEL TEMPORAL ANALYSIS")
    print("=" * 72)
    print(f"Input windows       : {len(df)}")
    print(f"Videos represented  : {video['video_id'].nunique()}")
    print(f"Subjects represented: {subject['subject_uid'].nunique()}")
    print()
    print("SUBJECT / CONDITION COVERAGE")
    print(coverage.to_string(index=False))
    print()
    print("SUBJECT-LEVEL COUNTS BY CONDITION")
    print(
        subject.groupby("condition")["subject_uid"]
        .nunique()
        .to_string()
    )
    print()

    if feature_candidates:
        print("SUBJECT-LEVEL DESCRIPTIVE SUMMARY")
        print(
            subject.groupby("condition")[feature_candidates]
            .agg(["mean", "median"])
            .round(4)
            .to_string()
        )

    print()
    print(f"Plots generated: {generated}")
    print("Outputs:")
    print(video_path)
    print(subject_path)
    print(OUTPUT_DIR / "subject_condition_counts.csv")
    print(OUTPUT_DIR / "subject_condition_summary.csv")
    print(PLOTS_DIR)


if __name__ == "__main__":
    main()
