import argparse
import csv
import os
import time

import cv2
import numpy as np


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROI_NAMES = ["left_eye", "right_eye", "mouth", "head"]

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

feature_params = dict(
    maxCorners=80,
    qualityLevel=0.01,
    minDistance=5,
    blockSize=7,
)


def parse_source(source):
    try:
        return int(source)
    except ValueError:
        return source


def find_haar_cascade():
    cascade_filename = "haarcascade_frontalface_default.xml"
    candidates = []

    if hasattr(cv2, "data"):
        candidates.append(os.path.join(cv2.data.haarcascades, cascade_filename))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.extend([
            os.path.join(conda_prefix, "share", "opencv4", "haarcascades", cascade_filename),
            os.path.join(conda_prefix, "share", "opencv", "haarcascades", cascade_filename),
            os.path.join(conda_prefix, "Library", "etc", "haarcascades", cascade_filename),
        ])

    candidates.extend([
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    ])

    for path in candidates:
        if path and os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Could not find haarcascade_frontalface_default.xml. "
        "Install OpenCV with: pip install opencv-python"
    )


cascade_path = find_haar_cascade()
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise RuntimeError(f"Could not load Haar cascade from: {cascade_path}")

print(f"Loaded Haar cascade from: {cascade_path}")


def largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda r: r[2] * r[3])


def detect_face(gray):
    """
    Robust Haar face detection.
    Histogram equalisation helps under weak or uneven lighting.
    """
    gray_eq = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60),
    )

    return largest_face(faces)


def wait_for_initial_face(cap):
    """
    Haar detection can miss the first webcam frame.
    This waits until the face is visible instead of crashing.
    """
    print("Waiting for face detection. Look at the camera...")

    while True:
        ret, frame = cap.read()

        if not ret:
            raise RuntimeError("Could not read frame while waiting for face.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_box = detect_face(gray)

        display = frame.copy()

        cv2.putText(
            display,
            "Waiting for face... look at camera",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        if face_box is not None:
            x, y, w, h = face_box
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                display,
                "Face detected. Starting...",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Sprint 3: ROI KLT Tracker", display)
            cv2.waitKey(500)
            print("Face detected. Starting ROI tracking.")
            return frame, gray, face_box

        cv2.imshow("Sprint 3: ROI KLT Tracker", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit("Stopped while waiting for face.")


def clip_rect(rect, width, height):
    x, y, w, h = rect

    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))

    return x, y, w, h


def get_face_rois(face_box, frame_shape):
    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = face_box

    rois = {
        "left_eye": (
            x + int(0.10 * w),
            y + int(0.22 * h),
            int(0.35 * w),
            int(0.22 * h),
        ),
        "right_eye": (
            x + int(0.55 * w),
            y + int(0.22 * h),
            int(0.35 * w),
            int(0.22 * h),
        ),
        "mouth": (
            x + int(0.25 * w),
            y + int(0.62 * h),
            int(0.50 * w),
            int(0.25 * h),
        ),
        "head": (
            x,
            y,
            w,
            h,
        ),
    }

    return {
        name: clip_rect(rect, frame_w, frame_h)
        for name, rect in rois.items()
    }


def create_rect_mask(gray, rect):
    mask = np.zeros_like(gray)
    x, y, w, h = rect
    mask[y:y + h, x:x + w] = 255
    return mask


def detect_roi_points(gray, rect):
    mask = create_rect_mask(gray, rect)
    pts = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
    return pts


def initialize_all_roi_points(gray, rois):
    roi_points = {}
    for name, rect in rois.items():
        roi_points[name] = detect_roi_points(gray, rect)
    return roi_points


def empty_metrics(prefix):
    return {
        f"{prefix}_points": 0,
        f"{prefix}_mean_u": 0.0,
        f"{prefix}_mean_v": 0.0,
        f"{prefix}_std_u": 0.0,
        f"{prefix}_std_v": 0.0,
        f"{prefix}_speed": 0.0,
        f"{prefix}_fb_error": 0.0,
        f"{prefix}_confidence": 0.0,
    }


def rename_metrics(metrics, roi_name):
    renamed = {}
    for key, value in metrics.items():
        renamed[key.replace("tmp", roi_name)] = value
    return renamed


def track_roi(prev_gray, gray, prev_pts, fb_threshold):
    if prev_pts is None or len(prev_pts) == 0:
        return None, empty_metrics("tmp")

    next_pts, st_forward, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_pts, None, **lk_params
    )

    if next_pts is None or st_forward is None:
        return None, empty_metrics("tmp")

    back_pts, st_backward, _ = cv2.calcOpticalFlowPyrLK(
        gray, prev_gray, next_pts, None, **lk_params
    )

    if back_pts is None or st_backward is None:
        return None, empty_metrics("tmp")

    fb_error = np.linalg.norm(prev_pts - back_pts, axis=2).reshape(-1)

    good = (
        (st_forward.reshape(-1) == 1)
        & (st_backward.reshape(-1) == 1)
        & (fb_error < fb_threshold)
    )

    old_good = prev_pts[good]
    new_good = next_pts[good]

    if len(old_good) == 0:
        return None, empty_metrics("tmp")

    displacement = new_good.reshape(-1, 2) - old_good.reshape(-1, 2)

    u = displacement[:, 0]
    v = displacement[:, 1]
    speed = np.sqrt(u ** 2 + v ** 2)

    metrics = {
        "tmp_points": int(len(old_good)),
        "tmp_mean_u": float(np.mean(u)),
        "tmp_mean_v": float(np.mean(v)),
        "tmp_std_u": float(np.std(u)),
        "tmp_std_v": float(np.std(v)),
        "tmp_speed": float(np.mean(speed)),
        "tmp_fb_error": float(np.mean(fb_error[good])),
        "tmp_confidence": float(len(old_good) / len(prev_pts)),
    }

    return new_good.reshape(-1, 1, 2), metrics


def draw_roi(frame, rect, label):
    x, y, w, h = rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
    cv2.putText(
        frame,
        label,
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )


def draw_points(frame, pts):
    if pts is None:
        return

    for p in pts:
        x, y = p.ravel()
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)


def safe_mean(values):
    valid = [v for v in values if v is not None]
    if len(valid) == 0:
        return 0.0
    return float(np.mean(valid))


def build_header():
    base = ["timestamp", "elapsed_sec", "frame_idx", "face_detected"]

    roi_columns = []
    for name in ROI_NAMES:
        roi_columns.extend([
            f"{name}_points",
            f"{name}_mean_u",
            f"{name}_mean_v",
            f"{name}_std_u",
            f"{name}_std_v",
            f"{name}_speed",
            f"{name}_fb_error",
            f"{name}_confidence",
        ])

    summary = [
        "eye_mean_speed",
        "eye_mean_v",
        "mouth_speed",
        "mouth_mean_v",
        "head_speed",
        "head_mean_v",
        "overall_tracking_confidence",
    ]

    return base + roi_columns + summary


def build_row(timestamp, start_time, frame_idx, face_detected, all_metrics):
    row = {
        "timestamp": timestamp,
        "elapsed_sec": timestamp - start_time,
        "frame_idx": frame_idx,
        "face_detected": int(face_detected),
    }

    confidences = []

    for name in ROI_NAMES:
        metrics = all_metrics.get(name, empty_metrics(name))
        row.update(metrics)
        confidences.append(metrics.get(f"{name}_confidence", 0.0))

    row["eye_mean_speed"] = safe_mean([
        row["left_eye_speed"],
        row["right_eye_speed"],
    ])

    row["eye_mean_v"] = safe_mean([
        row["left_eye_mean_v"],
        row["right_eye_mean_v"],
    ])

    row["mouth_speed"] = row["mouth_speed"]
    row["mouth_mean_v"] = row["mouth_mean_v"]

    row["head_speed"] = row["head_speed"]
    row["head_mean_v"] = row["head_mean_v"]

    row["overall_tracking_confidence"] = safe_mean(confidences)

    return row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="0",
        help="Camera index or video path",
    )

    parser.add_argument(
        "--output",
        default=os.path.join(OUTPUT_DIR, "roi_klt_motion_log.csv"),
        help="Output CSV path",
    )

    parser.add_argument(
        "--redetect-every",
        type=int,
        default=15,
        help="How often to redetect face and update ROI boxes",
    )

    parser.add_argument(
        "--min-roi-points",
        type=int,
        default=8,
        help="Minimum points before reinitialising a ROI",
    )

    parser.add_argument(
        "--fb-threshold",
        type=float,
        default=1.5,
        help="Forward-backward error threshold",
    )

    args = parser.parse_args()

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    frame, prev_gray, face_box = wait_for_initial_face(cap)

    rois = get_face_rois(face_box, frame.shape)
    roi_points = initialize_all_roi_points(prev_gray, rois)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    frame_idx = 0

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=build_header())
        writer.writeheader()

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_idx += 1
            timestamp = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_detected = False

            if frame_idx % args.redetect_every == 0:
                new_face_box = detect_face(gray)

                if new_face_box is not None:
                    face_box = new_face_box
                    rois = get_face_rois(face_box, frame.shape)
                    face_detected = True

            all_metrics = {}
            updated_roi_points = {}

            for name in ROI_NAMES:
                prev_pts = roi_points.get(name)

                new_pts, metrics = track_roi(
                    prev_gray,
                    gray,
                    prev_pts,
                    fb_threshold=args.fb_threshold,
                )

                metrics = rename_metrics(metrics, name)

                if new_pts is None or len(new_pts) < args.min_roi_points:
                    new_pts = detect_roi_points(gray, rois[name])

                updated_roi_points[name] = new_pts
                all_metrics[name] = metrics

            roi_points = updated_roi_points

            row = build_row(
                timestamp=timestamp,
                start_time=start_time,
                frame_idx=frame_idx,
                face_detected=face_detected,
                all_metrics=all_metrics,
            )

            writer.writerow(row)

            for name, rect in rois.items():
                draw_roi(frame, rect, name)
                draw_points(frame, roi_points.get(name))

            cv2.putText(
                frame,
                (
                    f"eye_speed={row['eye_mean_speed']:.3f} | "
                    f"mouth={row['mouth_speed']:.3f} | "
                    f"head_v={row['head_mean_v']:.3f}"
                ),
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Sprint 3: ROI KLT Tracker", frame)

            prev_gray = gray.copy()

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Saved ROI KLT log to: {args.output}")


if __name__ == "__main__":
    main()