from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


INPUT = Path("outputs/YawDD/subject_analysis/subject_condition_features.csv")
OUTPUT_DIR = Path("outputs/YawDD/subject_analysis")


FEATURES = [
    ("eye_motion", "eye_motion_mean_mean_mean"),
    ("mouth_motion", "mouth_motion_mean_mean_mean"),
    ("head_motion", "head_motion_mean_mean_mean"),
    ("eye_p95", "eye_motion_p95_mean_mean"),
    ("mouth_p95", "mouth_motion_p95_mean_mean"),
    ("head_p95", "head_motion_p95_mean_mean"),
    ("reliable_pct", "reliable_frame_pct_mean_mean"),
    ("mean_points", "mean_points_mean_mean"),
    ("mean_fb_error", "mean_fb_error_mean_mean"),
    ("fb_p95", "fb_error_p95_mean_mean"),
]


def paired_stats(df, condition, feature_name, column):
    if column not in df.columns:
        return None

    sub = df[df["condition"].isin(["Normal", condition])][
        ["subject_uid", "condition", column]
    ].copy()

    pivot = sub.pivot_table(
        index="subject_uid",
        columns="condition",
        values=column,
        aggfunc="mean",
    )

    if "Normal" not in pivot.columns or condition not in pivot.columns:
        return None

    pair = pivot[["Normal", condition]].dropna()
    n = len(pair)

    if n == 0:
        return None

    normal = pair["Normal"].to_numpy(dtype=float)
    other = pair[condition].to_numpy(dtype=float)
    diff = other - normal

    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        p = 1.0
        statistic = 0.0
    else:
        result = wilcoxon(
            normal,
            other,
            alternative="two-sided",
            method="auto",
        )
        statistic = float(result.statistic)
        p = float(result.pvalue)

    # Rank-biserial correlation from signed-rank statistic.
    # Positive means the condition tends to exceed Normal.
    if len(nonzero) > 0:
        abs_diff = np.abs(nonzero)
        ranks = pd.Series(abs_diff).rank(method="average").to_numpy()
        w_plus = ranks[nonzero > 0].sum()
        w_minus = ranks[nonzero < 0].sum()
        denom = w_plus + w_minus
        rank_biserial = (
            float((w_plus - w_minus) / denom)
            if denom else 0.0
        )
    else:
        rank_biserial = 0.0

    return {
        "comparison": f"Normal_vs_{condition}",
        "feature": feature_name,
        "column": column,
        "n_paired_subjects": n,
        "normal_mean": float(np.mean(normal)),
        "condition_mean": float(np.mean(other)),
        "normal_median": float(np.median(normal)),
        "condition_median": float(np.median(other)),
        "mean_difference_condition_minus_normal": float(np.mean(diff)),
        "median_difference_condition_minus_normal": float(np.median(diff)),
        "std_difference": float(np.std(diff, ddof=1)) if n > 1 else np.nan,
        "wilcoxon_statistic": statistic,
        "p_value": p,
        "rank_biserial_correlation": rank_biserial,
    }


def benjamini_hochberg(pvalues):
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(m, dtype=float)

    running = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        value = ranked[i] * m / rank
        running = min(running, value)
        q[order[i]] = min(running, 1.0)

    return q


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT}. Run subject-level analysis first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)

    comparisons = ["Talking", "Yawning", "Talking&Yawning"]
    rows = []

    for condition in comparisons:
        for feature_name, column in FEATURES:
            result = paired_stats(
                df,
                condition,
                feature_name,
                column,
            )
            if result is not None:
                rows.append(result)

    results = pd.DataFrame(rows)

    if results.empty:
        raise RuntimeError("No paired comparisons could be computed.")

    # Correct within each comparison family across the tested features.
    results["q_value_bh"] = np.nan

    for comparison, idx in results.groupby("comparison").groups.items():
        idx = list(idx)
        results.loc[idx, "q_value_bh"] = benjamini_hochberg(
            results.loc[idx, "p_value"].to_numpy()
        )

    results["significant_bh_0_05"] = (
        results["q_value_bh"] < 0.05
    )

    results.to_csv(
        OUTPUT_DIR / "subject_paired_statistics.csv",
        index=False,
    )

    coverage = (
        df.groupby("condition")["subject_uid"]
        .nunique()
        .rename("subjects")
        .reset_index()
    )
    coverage.to_csv(
        OUTPUT_DIR / "statistical_subject_coverage.csv",
        index=False,
    )

    print("=" * 76)
    print("YawDD SUBJECT-LEVEL PAIRED STATISTICS")
    print("=" * 76)
    print()
    print("Paired comparisons use subjects as the statistical unit.")
    print("Test: two-sided Wilcoxon signed-rank.")
    print("Multiple testing: Benjamini-Hochberg within each comparison.")
    print()
    print("SUBJECT COVERAGE")
    print(coverage.to_string(index=False))
    print()
    print("RESULTS")
    display_cols = [
        "comparison",
        "feature",
        "n_paired_subjects",
        "normal_median",
        "condition_median",
        "median_difference_condition_minus_normal",
        "p_value",
        "q_value_bh",
        "rank_biserial_correlation",
        "significant_bh_0_05",
    ]
    print(
        results[display_cols]
        .round(4)
        .to_string(index=False)
    )
    print()
    print("Outputs:")
    print(OUTPUT_DIR / "subject_paired_statistics.csv")
    print(OUTPUT_DIR / "statistical_subject_coverage.csv")


if __name__ == "__main__":
    main()
