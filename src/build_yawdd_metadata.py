from pathlib import Path
import re
import pandas as pd


DATA_ROOT = Path("data/YawDD")
OUTPUT = Path("outputs/YawDD/metadata.csv")


def parse_filename(path):
    name = path.name

    # Remove .avi / .avi.avi
    stem = re.sub(r"(\.avi)+$", "", name, flags=re.IGNORECASE)

    parts = stem.split("-")

    subject_id = parts[0]

    gender = None
    glasses = None
    condition = "Unknown"

    if len(parts) >= 2:
        descriptor = parts[1]

        if "Female" in descriptor:
            gender = "Female"
        elif "Male" in descriptor:
            gender = "Male"

        if "SunGlasses" in descriptor:
            glasses = "SunGlasses"
        elif "NoGlasses" in descriptor:
            glasses = "NoGlasses"
        elif "Glasses" in descriptor:
            glasses = "Glasses"

    if len(parts) >= 3:
        condition = "-".join(parts[2:])

        if condition.lower() == "talking&yawning":
            condition = "Talking&Yawning"

    # Determine recording type from directory
    path_str = str(path)

    if "/Mirror/" in path_str:
        split_type = "Mirror"
    elif "/Dash/" in path_str:
        split_type = "Dash"
    else:
        split_type = "Unknown"

    return {
        "subject_id": subject_id,
        "gender": gender,
        "glasses": glasses,
        "condition": condition,
        "split_type": split_type,
    }


def main():
    videos = sorted(
        DATA_ROOT.rglob("*.avi")
    )

    records = []

    for video_id, path in enumerate(videos, start=1):
        info = parse_filename(path)

        records.append({
            "video_id": video_id,
            "video_path": str(path),
            "subject_uid": f"{info['gender']}_{info['subject_id']}",
            **info,
        })

    df = pd.DataFrame(records)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"Found {len(df)} videos")
    print(f"Saved metadata to: {OUTPUT}")

    print("\nVideos by recording type:")
    print(df["split_type"].value_counts())

    print("\nVideos by condition:")
    print(df["condition"].value_counts())

    print("\nFirst 10 records:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()