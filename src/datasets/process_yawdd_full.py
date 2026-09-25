from pathlib import Path
import subprocess
import sys
import time

import pandas as pd


METADATA = Path("outputs/YawDD/metadata.csv")
TRACKER = Path("src/tracking/track_roi_klt.py")
OUTPUT_DIR = Path("outputs/YawDD/full_csv")
RESULTS_PATH = Path("outputs/YawDD/full_results.csv")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run headless ROI-KLT tracking on the YawDD Mirror videos."
    )
    parser.add_argument("--metadata", default=str(METADATA))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--results", default=str(RESULTS_PATH))
    parser.add_argument(
        "--condition",
        choices=["Normal", "Talking", "Yawning", "Talking&Yawning"],
        default=None,
        help="Process only one Mirror condition.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos whose output CSV already exists.",
    )
    parser.add_argument("--redetect-every", type=int, default=15)
    parser.add_argument("--min-roi-points", type=int, default=8)
    parser.add_argument("--fb-threshold", type=float, default=1.5)
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    results_path = Path(args.results)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}. "
            "Run build_yawdd_metadata.py first."
        )

    df = pd.read_csv(metadata_path)

    # Only the Mirror subset has explicit behavioural condition labels.
    df = df[df["split_type"].eq("Mirror")].copy()

    if args.condition is not None:
        df = df[df["condition"].eq(args.condition)].copy()

    df = df.sort_values(["condition", "video_id"]).reset_index(drop=True)

    start = max(args.start, 0)
    end = len(df) if args.limit is None else min(start + args.limit, len(df))
    df = df.iloc[start:end].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("YawDD FULL MIRROR DATASET — ROI KLT PROCESSING")
    print("=" * 78)
    print(f"Videos selected : {len(df)}")
    print(f"Output directory: {output_dir}")
    print(f"Tracker         : {TRACKER}")
    print()

    results = []

    for i, row in df.iterrows():
        video_id = int(row["video_id"])
        video_path = Path(row["video_path"])
        output_csv = output_dir / f"video_{video_id}.csv"

        record = {
            "video_id": video_id,
            "video_path": str(video_path),
            "subject_uid": row["subject_uid"],
            "gender": row["gender"],
            "glasses": row["glasses"],
            "condition": row["condition"],
            "success": False,
            "output_csv": str(output_csv),
            "return_code": None,
            "elapsed_processing_sec": None,
        }

        print(
            f"[{i + 1}/{len(df)}] "
            f"ID={video_id} | "
            f"Subject={row['subject_uid']} | "
            f"Condition={row['condition']}"
        )

        if args.skip_existing and output_csv.exists():
            print("  SKIP — output already exists")
            record["success"] = True
            record["return_code"] = 0
            results.append(record)
            continue

        if not video_path.exists():
            print(f"  FAILED — input not found: {video_path}")
            record["return_code"] = -1
            results.append(record)
            continue

        command = [
            sys.executable,
            str(TRACKER),
            "--source",
            str(video_path),
            "--output",
            str(output_csv),
            "--redetect-every",
            str(args.redetect_every),
            "--min-roi-points",
            str(args.min_roi_points),
            "--fb-threshold",
            str(args.fb_threshold),
            "--no-display",
        ]

        started = time.time()
        result = subprocess.run(command)
        elapsed = time.time() - started

        record["return_code"] = result.returncode
        record["elapsed_processing_sec"] = elapsed
        record["success"] = result.returncode == 0 and output_csv.exists()

        print(
            f"  {'SUCCESS' if record['success'] else 'FAILED'} "
            f"({elapsed:.1f} s)"
        )
        results.append(record)

        pd.DataFrame(results).to_csv(results_path, index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)

    print()
    print("=" * 78)
    print("PROCESSING COMPLETE")
    print("=" * 78)
    print(
        f"Successful: {int(results_df['success'].sum())}/"
        f"{len(results_df)}"
    )
    print(f"Results: {results_path}")

    if not results_df.empty:
        print()
        print("By condition:")
        print(
            results_df.groupby("condition")["success"]
            .agg(["count", "sum"])
            .rename(columns={"sum": "successful"})
            .to_string()
        )


if __name__ == "__main__":
    main()
