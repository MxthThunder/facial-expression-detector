import cv2
from deepface import DeepFace
import numpy as np
import time
import sys
import argparse
from collections import deque

# ── Emotion config ─────────────────────────────────────────────────────────────
ALL_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

EMOTION_COLORS = {
    "happy":    (0, 220, 100),
    "sad":      (255, 120, 50),
    "angry":    (30, 30, 220),
    "surprise": (0, 210, 255),
    "fear":     (180, 0, 200),
    "disgust":  (30, 180, 30),
    "neutral":  (180, 180, 180),
}

EMOTION_LABELS = {
    "happy":    "HAPPY",
    "sad":      "SAD",
    "angry":    "ANGRY",
    "surprise": "SURPRISED",
    "fear":     "FEAR",
    "disgust":  "DISGUST",
    "neutral":  "NEUTRAL",
}

# ── Smoothing & stability ──────────────────────────────────────────────────────
SMOOTH_ALPHA      = 0.12   # EMA weight — lower = smoother transitions
CONSISTENCY_K     = 8     # frames new dominant must hold before label switches
HYSTERESIS_MARGIN = 8.0   # new emotion must lead current by this % to trigger switch
ANALYSIS_W        = 480   # width to downscale frame to before DeepFace (more stable)

# ── Camera detection ───────────────────────────────────────────────────────────
HP_TRUEVSION_INDEX = 1   # default camera

def scan_cameras(max_idx=8):
    """Return list of (index, width, height, fps) for every working camera."""
    found = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                found.append((i, w, h, fps))
            cap.release()
    return found


def pick_camera(preselected=None):
    """
    Terminal camera picker.
    Lists available cameras by their actual index number.
    User types the index they want (e.g. 0, 1, 2...).
    Pressing Enter uses HP TrueVision (index 1) as the default.
    Pass preselected=<int> to skip the prompt entirely.
    """
    print("\n" + "═" * 54)
    print("  😊  Facial Expression Detector — Camera Setup")
    print("═" * 54)
    print("  Scanning available cameras...\n")

    cameras = scan_cameras()

    if not cameras:
        print("  [ERROR] No cameras found. Exiting.")
        return None

    valid_indices = []
    for (cam_idx, w, h, fps) in cameras:
        label = ""
        if cam_idx == 1:
            label = "  ← HP TrueVision FHD  [default]"
        elif cam_idx == 0:
            label = "  ← OMEN Cam & Voice"
        fps_str = f"{int(fps)}fps" if fps > 0 else "?"
        print(f"  Camera {cam_idx} — {w}x{h} @ {fps_str}{label}")
        valid_indices.append(cam_idx)

    # Pick the default: HP TrueVision (1) if available, else first found
    default_idx = HP_TRUEVSION_INDEX if HP_TRUEVSION_INDEX in valid_indices else valid_indices[0]

    # If caller pre-selected (via --camera flag), honour it
    if preselected is not None:
        if preselected in valid_indices:
            print(f"\n  ✔  Using Camera {preselected} (via --camera flag)")
            print("═" * 54 + "\n")
            return preselected
        else:
            print(f"  ⚠  Camera {preselected} not found, falling back to default ({default_idx}).")

    print(f"\n  Type the camera number you want, then press Enter.")
    print(f"  Press Enter alone to use the default (Camera {default_idx}).")
    print("═" * 54)

    while True:
        try:
            raw = input(f"  Your choice {valid_indices} (default={default_idx}): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if raw == "":
            chosen = default_idx
        else:
            try:
                chosen = int(raw)
            except ValueError:
                print(f"  ⚠  Please type one of: {valid_indices}")
                continue

        if chosen in valid_indices:
            _, w, h, _ = next(c for c in cameras if c[0] == chosen)
            print(f"\n  ✔  Using Camera {chosen} ({w}x{h})")
            print("═" * 54 + "\n")
            return chosen
        else:
            print(f"  ⚠  Invalid choice. Please type one of: {valid_indices}")


# ── Drawing helpers ────────────────────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, thickness=2, r=15, filled=False):
    x1, y1 = pt1
    x2, y2 = pt2
    if filled:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def draw_emotion_panel(frame, emotions, panel_x, panel_y, bar_width=215, bar_height=22, gap=8):
    """
    Always draws ALL 7 emotions sorted by score (highest first),
    each with a coloured fill bar and a percentage label.
    """
    sorted_emo = sorted(emotions.items(), key=lambda e: e[1], reverse=True)

    for i, (emo, score) in enumerate(sorted_emo):
        color  = EMOTION_COLORS.get(emo, (180, 180, 180))
        ty     = panel_y + i * (bar_height + gap)

        # ── Background track ──────────────────────────────────────────────────
        cv2.rectangle(frame, (panel_x, ty),
                      (panel_x + bar_width, ty + bar_height), (35, 35, 35), -1)

        # ── Coloured fill ─────────────────────────────────────────────────────
        filled_w = int(bar_width * min(score, 100.0) / 100)
        if filled_w > 0:
            cv2.rectangle(frame, (panel_x, ty),
                          (panel_x + filled_w, ty + bar_height), color, -1)

        # ── Border ────────────────────────────────────────────────────────────
        cv2.rectangle(frame, (panel_x, ty),
                      (panel_x + bar_width, ty + bar_height), (75, 75, 75), 1)

        # ── Label: EMOTION    XX.X% ───────────────────────────────────────────
        label = f"{emo.upper():<8}  {score:5.1f}%"
        cv2.putText(frame, label,
                    (panel_x + 6, ty + bar_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (245, 245, 245), 1, cv2.LINE_AA)


def preprocess_frame(frame):
    """CLAHE contrast boost + unsharp mask for better face detection."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur  = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharp = cv2.addWeighted(enhanced, 1.4, blur, -0.4, 0)
    return sharp


# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    # ── Camera selection (optional --camera flag) ─────────────────────────────
    parser = argparse.ArgumentParser(description="Facial Expression Detector")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index to use (skips the prompt)")
    args = parser.parse_args()

    cam_idx = pick_camera(preselected=args.camera)
    if cam_idx is None:
        return

    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)

    # ── Camera quality settings ───────────────────────────────────────────────
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,     1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    cap.set(cv2.CAP_PROP_BRIGHTNESS,  130)
    cap.set(cv2.CAP_PROP_CONTRAST,    132)
    cap.set(cv2.CAP_PROP_SHARPNESS,   160)
    cap.set(cv2.CAP_PROP_SATURATION,  128)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,    1)   # minimal capture latency

    # ── Smoothed emotion state ────────────────────────────────────────────────
    smoothed        = {e: 0.0 for e in ALL_EMOTIONS}
    dominant_buf    = deque(maxlen=CONSISTENCY_K)
    stable_dominant = "neutral"
    last_region     = {}
    display_emotions = {e: 0.0 for e in ALL_EMOTIONS}

    last_analysis_time = 0.0
    analysis_interval  = 0.18   # ~5-6 DeepFace calls/sec (more stable than 0.15)

    fps_timer   = time.time()
    fps         = 0
    frame_count = 0

    BACKENDS    = ["ssd", "opencv", "haarcascade"]
    backend_idx = 0

    print("[INFO] Starting — press Q in the window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read from camera.")
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        now   = time.time()

        # ── FPS counter ───────────────────────────────────────────────────────
        frame_count += 1
        if now - fps_timer >= 1.0:
            fps         = frame_count
            frame_count = 0
            fps_timer   = now

        # ── DeepFace analysis (periodic) ──────────────────────────────────────
        if now - last_analysis_time >= analysis_interval:
            last_analysis_time = now
            try:
                # ── Downscale for faster, more consistent analysis ────────────
                scale  = ANALYSIS_W / max(frame.shape[1], 1)
                small  = cv2.resize(frame,
                                    (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                                    interpolation=cv2.INTER_AREA)
                proc   = preprocess_frame(small)

                results = DeepFace.analyze(
                    proc,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend=BACKENDS[backend_idx],
                    silent=True,
                )
                result       = results[0] if isinstance(results, list) else results
                raw_emotions = result.get("emotion", {})

                # Scale region coords back to full-resolution frame
                reg = result.get("region", {})
                if reg and scale != 1.0:
                    inv = 1.0 / scale
                    last_region = {
                        "x": int(reg.get("x", 0) * inv),
                        "y": int(reg.get("y", 0) * inv),
                        "w": int(reg.get("w", 0) * inv),
                        "h": int(reg.get("h", 0) * inv),
                    }
                else:
                    last_region = reg

                # ── EMA smoothing ─────────────────────────────────────────────
                for emo in ALL_EMOTIONS:
                    raw_val      = raw_emotions.get(emo, 0.0)
                    smoothed[emo] = SMOOTH_ALPHA * raw_val + (1 - SMOOTH_ALPHA) * smoothed[emo]

                # Normalise to 100 %
                total = sum(smoothed.values()) or 1.0
                display_emotions = {e: (v / total) * 100 for e, v in smoothed.items()}

                # ── Hysteresis + consistency gate ─────────────────────────────
                # Find the top two emotions in the smoothed scores
                sorted_s    = sorted(smoothed.items(), key=lambda x: x[1], reverse=True)
                top_emo     = sorted_s[0][0]
                top_score   = sorted_s[0][1]
                second_score = sorted_s[1][1] if len(sorted_s) > 1 else 0.0
                lead        = top_score - second_score

                # Only push to buffer if it has a meaningful lead over second place
                if lead >= HYSTERESIS_MARGIN:
                    dominant_buf.append(top_emo)
                # else: don't update the buffer — keep existing label stable

                # Switch stable label only when the buffer is fully consistent
                if len(dominant_buf) == CONSISTENCY_K and len(set(dominant_buf)) == 1:
                    stable_dominant = dominant_buf[-1]

            except Exception:
                total = sum(smoothed.values()) or 1.0
                display_emotions = {e: (v / total) * 100 for e, v in smoothed.items()}

        else:
            total = sum(smoothed.values()) or 1.0
            display_emotions = {e: (v / total) * 100 for e, v in smoothed.items()}

        # ── Side panel dark overlay ───────────────────────────────────────────
        panel_w = 270
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # ── Face bounding box + emotion tag ───────────────────────────────────
        if last_region:
            fx = last_region.get("x", 0)
            fy = last_region.get("y", 0)
            fw = last_region.get("w", 0)
            fh = last_region.get("h", 0)
            box_color = EMOTION_COLORS.get(stable_dominant, (180, 180, 180))

            if fw > 0 and fh > 0:
                draw_rounded_rect(frame, (fx, fy), (fx + fw, fy + fh),
                                  box_color, thickness=3, r=14)

                tag      = EMOTION_LABELS.get(stable_dominant, stable_dominant.upper())
                tag_size, _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
                tx       = fx + (fw - tag_size[0]) // 2
                ty_tag   = max(fy - 14, 24)

                draw_rounded_rect(frame,
                                  (tx - 10, ty_tag - tag_size[1] - 8),
                                  (tx + tag_size[0] + 10, ty_tag + 6),
                                  box_color, r=10, filled=True)
                cv2.putText(frame, tag, (tx, ty_tag),
                            cv2.FONT_HERSHEY_DUPLEX, 0.85, (10, 10, 10), 2, cv2.LINE_AA)

        # ── Emotions panel — ALL 7 always visible ─────────────────────────────
        panel_x = w - panel_w + 12

        # Header
        cv2.putText(frame, "EMOTION ANALYSIS",
                    (panel_x, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (panel_x, 46), (w - 12, 46), (85, 85, 85), 1)

        draw_emotion_panel(frame, display_emotions,
                           panel_x=panel_x, panel_y=58)

        # Dominant label at bottom of panel
        dom_color = tuple(int(c) for c in EMOTION_COLORS.get(stable_dominant, (180, 180, 180)))
        dom_label = f"> {EMOTION_LABELS.get(stable_dominant, stable_dominant.upper())}"
        cv2.putText(frame, dom_label,
                    (panel_x, h - 54),
                    cv2.FONT_HERSHEY_DUPLEX, 0.70, dom_color, 2, cv2.LINE_AA)

        # Divider above dominant label
        cv2.line(frame, (panel_x, h - 62), (w - 12, h - 62), (85, 85, 85), 1)

        # ── FPS + hints ───────────────────────────────────────────────────────
        cv2.putText(frame, f"FPS: {fps}",
                    (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (130, 130, 130), 1, cv2.LINE_AA)
        cv2.putText(frame, "Q  quit",
                    (12, h - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (130, 130, 130), 1, cv2.LINE_AA)

        cv2.imshow("Facial Expression Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detector stopped.")


if __name__ == "__main__":
    run()
