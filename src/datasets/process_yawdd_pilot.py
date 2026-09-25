from pathlib import Path
import subprocess
import sys

import pandas as pd


INPUT = Path("outputs/YawDD/pilot_12.csv")
OUTPUT_DIR = Path("outputs/YawDD/pilot_csv")

TRACKER = Path("src/tracking/track_roi_klt.py")


def main():
    df = pd.read_csv(INPUT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(df)} videos")
    print("=" * 70)

    results = []

    for i, row in df.iterrows():

        video_id = int(row["video_id"])
        video_path = row["video_path"]

        output_csv = OUTPUT_DIR / f"video_{video_id}.csv"

        print(
            f"\n[{i + 1}/{len(df)}] "
            f"ID={video_id} | "
            f"Subject={row['subject_uid']} | "
            f"Condition={row['condition']}"
        )

        print(f"Input : {video_path}")
        print(f"Output: {output_csv}")

        command = [
            sys.executable,
            str(TRACKER),
            "--source",
            video_path,
            "--output",
            str(output_csv),
        ]

        result = subprocess.run(command)

        success = result.returncode == 0 and output_csv.exists()

        results.append({
            "video_id": video_id,
            "subject_uid": row["subject_uid"],
            "condition": row["condition"],
            "success": success,
            "output_csv": str(output_csv),
        })

        if success:
            print("SUCCESS")
        else:
            print("FAILED")

    results_df = pd.DataFrame(results)

    results_path = Path("outputs/YawDD/pilot_results.csv")
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 70)
    print("PILOT COMPLETE")
    print("=" * 70)

    print(
        results_df[
            ["video_id", "subject_uid", "condition", "success"]
        ].to_string(index=False)
    )

    print(
        f"\nSuccessful: {results_df['success'].sum()}/"
        f"{len(results_df)}"
    )

    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()