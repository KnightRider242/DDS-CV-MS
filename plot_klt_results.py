import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"

files = {
    "normal": "normal_klt.csv",
    "blinking": "blinking_klt.csv",
    "fake_drowsy": "fake_drowsy_klt.csv",
    "head_nod": "head_nod_klt.csv",
}

def load_csv(path):
    df = pd.read_csv(path)
    df["time_sec"] = df["timestamp"] - df["timestamp"].iloc[0]
    return df

def plot_feature(feature_name):
    plt.figure(figsize=(12, 6))

    for label, filename in files.items():
        path = os.path.join(OUTPUT_DIR, filename)

        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue

        df = load_csv(path)
        plt.plot(df["time_sec"], df[feature_name], label=label)

    plt.xlabel("Time (seconds)")
    plt.ylabel(feature_name)
    plt.title(f"KLT Motion Feature: {feature_name}")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(OUTPUT_DIR, f"{feature_name}_comparison.png")
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"Saved: {save_path}")

if __name__ == "__main__":
    features_to_plot = [
        "mean_v",
        "mean_speed",
        "std_v",
        "forward_backward_error",
        "tracking_confidence",
        "num_tracked_points",
    ]

    for feature in features_to_plot:
        plot_feature(feature)