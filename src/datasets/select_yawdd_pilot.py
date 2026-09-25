import pandas as pd

df = pd.read_csv("outputs/YawDD/metadata.csv")
df = df[df["split_type"] == "Mirror"].copy()

conditions = ["Normal", "Talking", "Yawning", "Talking&Yawning"]

selected = []
used_subjects = set()

for condition in conditions:
    subset = df[df["condition"] == condition].copy()

    # Prefer subjects not already used
    subset = subset[~subset["subject_uid"].isin(used_subjects)]

    # Prefer a mix of genders where possible
    chosen = []

    for gender in ["Female", "Male"]:
        candidates = subset[subset["gender"] == gender]
        if not candidates.empty:
            chosen.append(candidates.iloc[0])

    # Fill remaining slots
    remaining = subset[
        ~subset["subject_uid"].isin([x["subject_uid"] for x in chosen])
    ]

    for _, row in remaining.iterrows():
        if len(chosen) >= 3:
            break
        chosen.append(row)

    chosen = chosen[:3]

    for row in chosen:
        used_subjects.add(row["subject_uid"])
        selected.append(row)

result = pd.DataFrame(selected)

print("\n=== SELECTED PILOT VIDEOS ===")
print(
    result[
        ["video_id", "subject_uid", "gender",
         "glasses", "condition", "video_path"]
    ].to_string(index=False)
)

print("\nTotal selected:", len(result))
print("Unique subjects:", result["subject_uid"].nunique())

result.to_csv("outputs/YawDD/pilot_12.csv", index=False)

print("\nSaved to:")
print("outputs/YawDD/pilot_12.csv")