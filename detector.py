import cv2
from deepface import DeepFace
import numpy as np
import time

# ── Emotion color map ──────────────────────────────────────────────────────────
EMOTION_COLORS = {
    "happy":     (0, 255, 128),
    "sad":       (255, 100, 50),
    "angry":     (0, 0, 255),
    "surprise":  (0, 200, 255),
    "fear":      (180, 0, 180),
    "disgust":   (0, 180, 0),
    "neutral":   (200, 200, 200),
}

EMOJI_MAP = {
    "happy":    "😊",
    "sad":      "😢",
    "angry":    "😠",
    "surprise": "😲",
    "fear":     "😨",
    "disgust":  "🤢",
    "neutral":  "😐",
}

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

def draw_emotion_bar(frame, emotions, x, y, bar_width=200, bar_height=18, gap=6):
    """Draw a vertical stack of labelled emotion confidence bars."""
    sorted_emo = sorted(emotions.items(), key=lambda e: e[1], reverse=True)
    for i, (emo, score) in enumerate(sorted_emo):
        color = EMOTION_COLORS.get(emo, (200, 200, 200))
        label = f"{emo.upper():<9} {score:5.1f}%"
        ty = y + i * (bar_height + gap)
        # Background track
        cv2.rectangle(frame, (x, ty), (x + bar_width, ty + bar_height), (50, 50, 50), -1)
        # Filled bar
        filled_w = int(bar_width * score / 100)
        cv2.rectangle(frame, (x, ty), (x + filled_w, ty + bar_height), color, -1)
        # Label
        cv2.putText(frame, label, (x + 5, ty + bar_height - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

def run():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_result = None
    analysis_interval = 0.18   # seconds between DeepFace calls
    last_analysis_time = 0
    fps_timer = time.time()
    fps = 0
    frame_count = 0

    print("[INFO] Starting Facial Expression Detector — press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read from camera.")
            break

        frame = cv2.flip(frame, 1)   # mirror view
        h, w = frame.shape[:2]
        now = time.time()

        # ── FPS counter ──────────────────────────────────────────────────────
        frame_count += 1
        if now - fps_timer >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_timer = now

        # ── Run DeepFace analysis periodically ───────────────────────────────
        if now - last_analysis_time >= analysis_interval:
            last_analysis_time = now
            try:
                results = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv",
                    silent=True,
                )
                last_result = results[0] if isinstance(results, list) else results
            except Exception:
                last_result = None

        # ── Dark overlay for HUD panel ───────────────────────────────────────
        overlay = frame.copy()
        panel_w = 250
        cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # ── Draw face box + dominant emotion ─────────────────────────────────
        if last_result:
            region = last_result.get("region", {})
            fx = region.get("x", 0)
            fy = region.get("y", 0)
            fw = region.get("w", 0)
            fh = region.get("h", 0)

            dominant = last_result.get("dominant_emotion", "neutral")
            emotions = last_result.get("emotion", {})
            box_color = EMOTION_COLORS.get(dominant, (200, 200, 200))

            if fw > 0 and fh > 0:
                draw_rounded_rect(frame, (fx, fy), (fx + fw, fy + fh),
                                  box_color, thickness=3, r=14)

                # Emotion tag above the box
                tag = f"{EMOJI_MAP.get(dominant,'?')} {dominant.upper()}"
                tag_size, _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
                tx = fx + (fw - tag_size[0]) // 2
                ty_tag = max(fy - 12, 20)

                # Tag background
                draw_rounded_rect(frame,
                                  (tx - 8, ty_tag - tag_size[1] - 6),
                                  (tx + tag_size[0] + 8, ty_tag + 4),
                                  box_color, r=8, filled=True)
                cv2.putText(frame, tag, (tx, ty_tag),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

            # ── Emotion bars in side panel ────────────────────────────────────
            draw_emotion_bar(frame, emotions,
                             x=w - panel_w + 15,
                             y=80)

        # ── Panel header ─────────────────────────────────────────────────────
        cv2.putText(frame, "EMOTIONS", (w - panel_w + 15, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (w - panel_w + 15, 58), (w - 15, 58), (100, 100, 100), 1)

        # ── FPS ───────────────────────────────────────────────────────────────
        cv2.putText(frame, f"FPS: {fps}", (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)

        # ── Quit hint ─────────────────────────────────────────────────────────
        cv2.putText(frame, "Press Q to quit", (12, h - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow("Facial Expression Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detector stopped.")

if __name__ == "__main__":
    run()
