import cv2
import csv
import time
import os
import numpy as np

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_path = os.path.join(OUTPUT_DIR, "klt_motion_log.csv")

# cascade_path = "/home/amathew052/anaconda3/envs/drowsy-cv/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

# face_cascade = cv2.CascadeClassifier(cascade_path)

# if face_cascade.empty():
#     raise RuntimeError(f"Could not load Haar cascade from: {cascade_path}")

# print(f"Loaded Haar cascade from: {cascade_path}")

def find_haar_cascade():
    cascade_filename = "haarcascade_frontalface_default.xml"

    candidates = []

    # Works for pip opencv-python
    if hasattr(cv2, "data"):
        candidates.append(os.path.join(cv2.data.haarcascades, cascade_filename))

    # Works for Conda / Linux / WSL
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.extend([
            os.path.join(conda_prefix, "share", "opencv4", "haarcascades", cascade_filename),
            os.path.join(conda_prefix, "share", "opencv", "haarcascades", cascade_filename),
            os.path.join(conda_prefix, "Library", "etc", "haarcascades", cascade_filename),  # Windows Conda
        ])

    # Common Linux locations
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

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

feature_params = dict(
    maxCorners=120,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7,
)


def largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda r: r[2] * r[3])


def detect_face(gray):
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80),
    )
    return largest_face(faces)


def create_face_mask(gray, face_box):
    mask = np.zeros_like(gray)
    if face_box is None:
        return mask

    x, y, w, h = face_box

    # Track mainly the face region.
    # Later we will split this into eye, mouth, and nose ROIs.
    mask[y:y + h, x:x + w] = 255

    return mask


def draw_tracks(frame, old_pts, new_pts, good_mask):
    for old, new, is_good in zip(old_pts, new_pts, good_mask):
        if not is_good:
            continue

        x0, y0 = old.ravel()
        x1, y1 = new.ravel()

        cv2.arrowedLine(
            frame,
            (int(x0), int(y0)),
            (int(x1), int(y1)),
            (0, 255, 0),
            1,
            tipLength=0.3,
        )

    return frame


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame.")

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_box = detect_face(prev_gray)
    mask = create_face_mask(prev_gray, face_box)

    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **feature_params)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "num_tracked_points",
            "mean_u",
            "mean_v",
            "std_u",
            "std_v",
            "mean_speed",
            "forward_backward_error",
            "tracking_confidence"
        ])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_pts is None or len(prev_pts) < 20:
                face_box = detect_face(gray)
                mask = create_face_mask(gray, face_box)
                prev_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
                prev_gray = gray.copy()
                continue

            next_pts, st_forward, err_forward = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, **lk_params
            )

            back_pts, st_backward, err_backward = cv2.calcOpticalFlowPyrLK(
                gray, prev_gray, next_pts, None, **lk_params
            )

            if next_pts is None or back_pts is None:
                prev_pts = None
                prev_gray = gray.copy()
                continue

            fb_error = np.linalg.norm(prev_pts - back_pts, axis=2).reshape(-1)

            good = (
                (st_forward.reshape(-1) == 1)
                & (st_backward.reshape(-1) == 1)
                & (fb_error < 1.5)
            )

            old_good = prev_pts[good]
            new_good = next_pts[good]

            if len(old_good) > 0:
                displacement = new_good.reshape(-1, 2) - old_good.reshape(-1, 2)

                u = displacement[:, 0]
                v = displacement[:, 1]
                speed = np.sqrt(u ** 2 + v ** 2)

                mean_u = float(np.mean(u))
                mean_v = float(np.mean(v))
                std_u = float(np.std(u))
                std_v = float(np.std(v))
                mean_speed = float(np.mean(speed))
                mean_fb_error = float(np.mean(fb_error[good]))
                confidence = float(len(old_good) / len(prev_pts))

                writer.writerow([
                    timestamp,
                    len(old_good),
                    mean_u,
                    mean_v,
                    std_u,
                    std_v,
                    mean_speed,
                    mean_fb_error,
                    confidence
                ])

                frame = draw_tracks(frame, prev_pts, next_pts, good)

                cv2.putText(
                    frame,
                    f"Tracked: {len(old_good)} | mean_v: {mean_v:.3f} | conf: {confidence:.2f}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                prev_pts = new_good.reshape(-1, 1, 2)
            else:
                prev_pts = None

            cv2.imshow("KLT Facial Motion Tracker", frame)

            prev_gray = gray.copy()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Saved motion log to: {csv_path}")


if __name__ == "__main__":
    main()