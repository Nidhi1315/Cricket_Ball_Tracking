
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import math
import time

# --------------------------------------------------
# PATHS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
INPUT_VIDEO = ROOT / "data" / "input_videos" / "cricket6.mp4"
OUTPUT_DIR = ROOT / "data" / "output_videos"
OUTPUT_VIDEO = OUTPUT_DIR / "automatic_ball_path6.mp4"
OUTPUT_IMAGE = OUTPUT_DIR / "automatic_ball_trajectory6.jpg"

# --------------------------------------------------
# DETECTOR SETTINGS
# --------------------------------------------------
MODEL_NAME = "yolo11x.pt"   # Pretrained COCO model
BALL_CLASS_ID = 32          # COCO: sports ball
CONFIDENCE = 0.12
IMAGE_SIZE = 1280

# Tracking settings
MAX_GAP = 5
MIN_CONFIRMATIONS = 2
MIN_BOX_SIZE = 3
MAX_BOX_SIZE = 160

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_size(box):
    x1, y1, x2, y2 = box
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def valid_ball_box(box, frame_width, frame_height):
    x1, y1, x2, y2 = map(float, box)
    w = x2 - x1
    h = y2 - y1

    if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
        return False

    if w > MAX_BOX_SIZE or h > MAX_BOX_SIZE:
        return False

    # A ball is approximately round. This rejects many bottles
    # and long objects, though it cannot reject every false positive.
    ratio = w / max(h, 1.0)
    if ratio < 0.45 or ratio > 2.2:
        return False

    cx, cy = box_center(box)
    if not (0 <= cx < frame_width and 0 <= cy < frame_height):
        return False

    return True


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def draw_dashed_line(frame, p1, p2, color, thickness=2, dash=8):
    x1, y1 = map(int, p1)
    x2, y2 = map(int, p2)
    length = distance((x1, y1), (x2, y2))

    if length < 1:
        return

    dx = (x2 - x1) / length
    dy = (y2 - y1) / length

    pos = 0
    while pos < length:
        end = min(pos + dash, length)
        a = (int(x1 + dx * pos), int(y1 + dy * pos))
        b = (int(x1 + dx * end), int(y1 + dy * end))
        cv2.line(frame, a, b, color, thickness, cv2.LINE_AA)
        pos += dash * 2


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Input video not found:\n{INPUT_VIDEO}\n"
            "Put your video at data/input_videos/cricket.mp4"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading pretrained YOLO model...")
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Save MP4 output.
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {OUTPUT_VIDEO}")

    # Track state
    active = False
    track_points = []       # (frame_index, x, y)
    last_box = None
    last_frame = -1
    velocity = (0.0, 0.0)
    hit_count = 0
    missed = 0

    # Candidate history used to avoid starting a track on one isolated detection.
    pending = []
    frame_index = 0
    last_output_frame = None
    detected_frames = 0

    print(f"Processing {width}x{height} video at {fps:.2f} FPS...")
    started = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Run pretrained detector. No motion-contour fallback is used.
        result = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            classes=[BALL_CLASS_ID],
            verbose=False
        )[0]

        candidates = []
        if result.boxes is not None:
            for det in result.boxes:
                conf = float(det.conf[0].item())
                box = det.xyxy[0].cpu().numpy().astype(float)

                if not valid_ball_box(box, width, height):
                    continue

                candidates.append({
                    "box": box,
                    "center": box_center(box),
                    "conf": conf
                })

        # --------------------------------------------------
        # Choose the detection that best matches the current track.
        # --------------------------------------------------
        chosen = None

        if active and last_box is not None:
            previous_center = box_center(last_box)
            frames_since = max(1, frame_index - last_frame)

            predicted = (
                previous_center[0] + velocity[0] * frames_since,
                previous_center[1] + velocity[1] * frames_since
            )

            # Allow larger movement after a missed detection, but limit
            # the gate so unrelated objects are less likely to be linked.
            diagonal = math.hypot(width, height)
            gate = min(0.16 * diagonal, max(70.0, 65.0 * frames_since))

            best_score = float("inf")

            for cand in candidates:
                d = distance(cand["center"], predicted)

                old_w, old_h = box_size(last_box)
                new_w, new_h = box_size(cand["box"])

                size_ratio_w = max(old_w, new_w) / max(1.0, min(old_w, new_w))
                size_ratio_h = max(old_h, new_h) / max(1.0, min(old_h, new_h))

                # Reject large size changes; they often indicate a different object.
                if size_ratio_w > 4.0 or size_ratio_h > 4.0:
                    continue

                if d > gate:
                    continue

                score = d + 10.0 * abs(math.log(size_ratio_w))
                score += 10.0 * abs(math.log(size_ratio_h))
                score -= 15.0 * cand["conf"]

                if score < best_score:
                    best_score = score
                    chosen = cand

        # --------------------------------------------------
        # If no track exists, look for a candidate seen repeatedly.
        # --------------------------------------------------
        if not active:
            updated_pending = []

            for cand in candidates:
                matched = None
                for old in pending:
                    age = frame_index - old["frame"]
                    if age <= 4 and distance(cand["center"], old["center"]) < max(
                        70.0, 0.08 * math.hypot(width, height)
                    ):
                        matched = old
                        break

                if matched is not None:
                    matched["hits"] += 1
                    matched["frame"] = frame_index
                    matched["center"] = cand["center"]
                    matched["box"] = cand["box"]
                    matched["conf"] = cand["conf"]

                    if matched["hits"] >= MIN_CONFIRMATIONS:
                        chosen = cand
                        active = True
                        hit_count = matched["hits"]
                        track_points = [
                            (frame_index - 1,
                             matched["center"][0],
                             matched["center"][1]),
                            (frame_index,
                             cand["center"][0],
                             cand["center"][1])
                        ]
                        last_box = cand["box"]
                        last_frame = frame_index
                        velocity = (0.0, 0.0)
                        missed = 0
                        break
                    else:
                        updated_pending.append(matched)
                else:
                    updated_pending.append({
                        "frame": frame_index,
                        "center": cand["center"],
                        "box": cand["box"],
                        "conf": cand["conf"],
                        "hits": 1
                    })

            pending = updated_pending[-30:]

        # --------------------------------------------------
        # Update confirmed track.
        # --------------------------------------------------
        if active and chosen is not None:
            center = chosen["center"]

            if last_box is not None and last_frame < frame_index:
                old_center = box_center(last_box)
                dt = max(1, frame_index - last_frame)
                new_velocity = (
                    (center[0] - old_center[0]) / dt,
                    (center[1] - old_center[1]) / dt
                )

                # Smooth velocity instead of reacting to every detection.
                velocity = (
                    0.55 * velocity[0] + 0.45 * new_velocity[0],
                    0.55 * velocity[1] + 0.45 * new_velocity[1]
                )

            track_points.append((frame_index, center[0], center[1]))
            last_box = chosen["box"]
            last_frame = frame_index
            missed = 0
            hit_count += 1
            detected_frames += 1

        elif active:
            missed += 1
            # Keep the track briefly while waiting for the detector to see
            # the ball again. Predicted positions are not added to the trail.
            if missed > MAX_GAP:
                active = False
                last_box = None
                velocity = (0.0, 0.0)
                pending = []

        # --------------------------------------------------
        # Draw only confirmed positions.
        # Connect points only when the gap is short.
        # --------------------------------------------------
        display = frame.copy()

        if len(track_points) >= 2:
            for i in range(1, len(track_points)):
                f1, x1, y1 = track_points[i - 1]
                f2, x2, y2 = track_points[i]

                gap = f2 - f1
                if 0 < gap <= MAX_GAP + 1:
                    p1 = (int(x1), int(y1))
                    p2 = (int(x2), int(y2))

                    if gap == 1:
                        cv2.line(
                            display, p1, p2, (0, 255, 0), 3,
                            cv2.LINE_AA
                        )
                    else:
                        # Dashed connector makes a missed-detection gap visible.
                        draw_dashed_line(
                            display, p1, p2, (0, 200, 255), 2
                        )

        # Draw confirmed ball positions.
        for _, x, y in track_points[-100:]:
            cv2.circle(
                display, (int(x), int(y)), 4,
                (0, 255, 0), -1, cv2.LINE_AA
            )

        if chosen is not None and active:
            x1, y1, x2, y2 = map(int, chosen["box"])
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display, "Ball",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2, cv2.LINE_AA
            )

        status = "TRACKING" if active else "SEARCHING FOR BALL"
        cv2.putText(
            display, status, (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 0) if active else (0, 165, 255),
            2, cv2.LINE_AA
        )

        writer.write(display)
        last_output_frame = display.copy()

        frame_index += 1
        if frame_index % 30 == 0:
            elapsed = time.time() - started
            print(
                f"Frames: {frame_index}/{total_frames or '?'} | "
                f"Ball detections: {detected_frames} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    cap.release()
    writer.release()

    if last_output_frame is not None:
        cv2.imwrite(str(OUTPUT_IMAGE), last_output_frame)

    print("\nFinished.")
    print(f"Output video: {OUTPUT_VIDEO}")
    print(f"Trajectory preview: {OUTPUT_IMAGE}")
    print(f"Frames processed: {frame_index}")
    print(f"Confirmed ball detections: {detected_frames}")

    if detected_frames == 0:
        print(
            "\nWARNING: No ball detections were confirmed. "
            "The pretrained model may not recognize the cricket ball "
            "in this footage. Do not interpret the output as a successful track."
        )


if __name__ == "__main__":
    main()
