import cv2
import numpy as np
import requests
import time
import threading
import os
from collections import deque
from mediapipe.python.solutions import pose as mp_pose_module
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions.pose import PoseLandmark

# ─── Configuration ───────────────────────────────────────────────
FLASK_URL        = "http://127.0.0.1:5000/alert"
MOVEMENT_HISTORY = deque(maxlen=60)
ALERT_COOLDOWN   = 8
last_alert_time  = 0

# ── Strict Thresholds ─────────────────────────────────────────────
MIN_AVG_MOVEMENT  = 0.40
MIN_RAPID_CHANGES = 0.80
MIN_MAX_MOVEMENT  = 0.60
MIN_STD           = 0.10
MIN_SCORE         = 65.0
MIN_FRAMES        = 40

# ── Alarm State ───────────────────────────────────────────────────
alarm_active    = False
alarm_thread    = None
alarm_stop_flag = threading.Event()

# ─── MediaPipe Setup ─────────────────────────────────────────────
pose    = mp_pose_module.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_pose = mp_pose_module

# ─── Key Body Points ─────────────────────────────────────────────
KEY_POINTS = [
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_HIP,
    PoseLandmark.RIGHT_HIP,
    PoseLandmark.LEFT_KNEE,
    PoseLandmark.RIGHT_KNEE,
]


# ─── Alarm System ────────────────────────────────────────────────

def generate_alarm_sound():
    """
    Generate emergency alarm using pygame.
    Plays a loud beeping sound continuously until stopped.
    """
    try:
        import pygame
        import numpy as np

        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

        sample_rate = 44100
        duration    = 0.3   # each beep duration
        freq1       = 880   # high tone Hz
        freq2       = 440   # low tone Hz

        def make_beep(freq, dur):
            t      = np.linspace(0, dur, int(sample_rate * dur), False)
            wave   = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            sound  = pygame.sndarray.make_sound(wave)
            return sound

        beep_high = make_beep(freq1, duration)
        beep_low  = make_beep(freq2, duration)

        print("🔊 ALARM STARTED — Press 'S' to stop!")

        while not alarm_stop_flag.is_set():
            beep_high.play()
            time.sleep(duration + 0.05)
            beep_low.play()
            time.sleep(duration + 0.05)

        pygame.mixer.stop()
        print("🔕 Alarm stopped.")

    except ImportError:
        # Fallback: use system bell if pygame not available
        print("🔊 ALARM! ALARM! ALARM! (install pygame for sound)")
        while not alarm_stop_flag.is_set():
            print("\a🚨 SEIZURE EMERGENCY! 🚨", flush=True)
            time.sleep(0.5)


def start_alarm():
    """Start alarm in background thread"""
    global alarm_active, alarm_thread, alarm_stop_flag

    if alarm_active:
        return  # Already running

    alarm_stop_flag.clear()
    alarm_active  = True
    alarm_thread  = threading.Thread(
        target=generate_alarm_sound,
        daemon=True
    )
    alarm_thread.start()


def stop_alarm():
    """Stop the alarm"""
    global alarm_active

    if not alarm_active:
        return

    alarm_stop_flag.set()
    alarm_active = False
    print("🔕 Alarm stopped by user.")


# ─── Movement Calculation ────────────────────────────────────────

def calculate_movement(landmarks, prev_landmarks):
    """
    Calculate body movement between 2 frames.
    Only counts clearly visible landmarks.
    Returns: 0.0 (still) to 1.0 (max)
    """
    if prev_landmarks is None:
        return 0.0

    total = 0.0
    count = 0

    for point in KEY_POINTS:
        try:
            curr = landmarks[point.value]
            prev = prev_landmarks[point.value]
            if curr.visibility > 0.7 and prev.visibility > 0.7:
                dist   = np.sqrt(
                    (curr.x - prev.x)**2 +
                    (curr.y - prev.y)**2
                )
                total += dist
                count += 1
        except Exception:
            continue

    if count == 0:
        return 0.0

    return min((total / count) * 100, 1.0)


def analyze_seizure_pattern(movement_history):
    """
    Detect real seizure vs normal movement.
    All 5 conditions must pass.
    """
    if len(movement_history) < MIN_FRAMES:
        return False, 0.0, 0.0, 0.0

    history      = list(movement_history)
    avg_movement = np.mean(history)
    max_movement = np.max(history)
    std_movement = np.std(history)

    changes = sum(
        1 for i in range(1, len(history))
        if abs(history[i] - history[i-1]) > 0.08
    )
    rapid_changes = changes / len(history)

    seizure_score = round(
        (avg_movement * 0.35 + rapid_changes * 0.65) * 100, 1
    )

    is_seizure = (
        avg_movement  > MIN_AVG_MOVEMENT  and
        rapid_changes > MIN_RAPID_CHANGES and
        max_movement  > MIN_MAX_MOVEMENT  and
        std_movement  > MIN_STD           and
        seizure_score > MIN_SCORE
    )

    return is_seizure, seizure_score, rapid_changes, avg_movement


# ─── Flask Communication ─────────────────────────────────────────

def send_to_flask(movement_intensity, rapid_changes, seizure_score):
    """Send movement data to Flask with cooldown"""
    global last_alert_time
    now = time.time()

    if now - last_alert_time < ALERT_COOLDOWN:
        return None

    try:
        response = requests.post(
            FLASK_URL,
            json={
                "movement_intensity": round(float(movement_intensity), 4),
                "rapid_changes"     : round(float(rapid_changes),      4),
                "seizure_score"     : float(seizure_score)
            },
            timeout=3
        )
        last_alert_time = now
        return response.json()

    except requests.exceptions.ConnectionError:
        print("❌ Flask not running! Start app.py first.")
        return None
    except Exception as e:
        print(f"❌ Flask error: {e}")
        return None


# ─── Draw Overlay ────────────────────────────────────────────────

def draw_overlay(frame, is_seizure, movement_score,
                 seizure_score, rapid_changes,
                 flask_result, frames_collected):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 195), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Warmup
    if frames_collected < MIN_FRAMES:
        cv2.putText(frame,
                    f"Warming up... ({frames_collected}/{MIN_FRAMES})",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 165, 255), 2)
        cv2.putText(frame, "Q:Quit  S:Stop Alarm",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (150, 150, 150), 1)
        return frame

    # ── Flashing red border when seizure ─────────────────────────
    if is_seizure or alarm_active:
        flash = int(time.time() * 4) % 2 == 0
        if flash:
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 8)

    # ── Main status ───────────────────────────────────────────────
    if is_seizure or alarm_active:
        cv2.putText(frame, "SEIZURE DETECTED!",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0, 0, 255), 3)
        # Alarm indicator
        alarm_text = "🔊 ALARM ON — Press S to stop"
        cv2.putText(frame, "ALARM ON — Press S to stop",
                    (10, 195), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "NORMAL",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0, 255, 0), 2)

    # ── Movement bar ──────────────────────────────────────────────
    fill1 = int(200 * min(movement_score * 4, 1.0))
    col1  = (0, 0, 255) if movement_score > MIN_AVG_MOVEMENT else (0, 200, 0)
    cv2.rectangle(frame, (10, 52), (210, 65),        (50,50,50), -1)
    cv2.rectangle(frame, (10, 52), (10+fill1, 65),   col1,       -1)
    cv2.putText(frame, f"Movement    : {movement_score:.3f}",
                (215, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    # ── Rapid changes bar ─────────────────────────────────────────
    fill2 = int(200 * min(rapid_changes, 1.0))
    col2  = (0, 0, 255) if rapid_changes > MIN_RAPID_CHANGES else (0, 200, 0)
    cv2.rectangle(frame, (10, 73), (210, 86),        (50,50,50), -1)
    cv2.rectangle(frame, (10, 73), (10+fill2, 86),   col2,       -1)
    cv2.putText(frame, f"Rapid Change: {rapid_changes:.2f}",
                (215, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    # ── Seizure score ─────────────────────────────────────────────
    sc = (0, 0, 255) if seizure_score > MIN_SCORE else (255,255,255)
    cv2.putText(frame, f"Seizure Score: {seizure_score}%",
                (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sc, 1)

    # ── AI result ─────────────────────────────────────────────────
    if flask_result:
        result   = flask_result.get('result', '').upper()
        conf     = flask_result.get('confidence', 0)
        aic      = (0, 0, 255) if result == 'SEIZURE' else (0, 255, 0)
        cv2.putText(frame, f"AI Model : {result} ({conf}%)",
                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, aic, 2)

        if is_seizure and result == 'SEIZURE':
            msg, color = "CONFIRMED SEIZURE!", (0, 0, 255)
        elif is_seizure or result == 'SEIZURE':
            msg, color = "POSSIBLE SEIZURE — Monitor!", (0, 165, 255)
        else:
            msg, color = "All Clear", (0, 255, 0)

        cv2.putText(frame, msg,
                    (10, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # ── Controls ──────────────────────────────────────────────────
    cv2.putText(frame, "Q:Quit  S:Stop Alarm",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (150, 150, 150), 1)

    return frame


# ─── Main Detection Loop ─────────────────────────────────────────

def run_detection(source=0):
    print(f"\n🎥 Starting detection...")
    print(f"   Source   : {'Webcam' if source == 0 else source}")
    print(f"   Flask URL: {FLASK_URL}")
    print("─" * 45)
    print("   Q / ESC = Quit")
    print("   S       = Stop Alarm")
    print("─" * 45)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Cannot open source!")
        return

    prev_landmarks   = None
    flask_result     = None
    frame_count      = 0
    frames_with_body = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Done!" if source != 0 else "❌ Camera lost!")
            break

        frame_count += 1
        if source == 0:
            frame = cv2.flip(frame, 1)

        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb)

        movement_score = 0.0
        seizure_score  = 0.0
        rapid_changes  = 0.0
        is_seizure     = False

        if pose_results.pose_landmarks:
            frames_with_body += 1

            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(0, 255, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2)
            )

            landmarks      = pose_results.pose_landmarks.landmark
            movement_score = calculate_movement(landmarks, prev_landmarks)
            MOVEMENT_HISTORY.append(movement_score)
            prev_landmarks = landmarks

            is_seizure, seizure_score, rapid_changes, avg_mov = \
                analyze_seizure_pattern(MOVEMENT_HISTORY)

            # ── Seizure detected — start alarm ────────────────────
            if is_seizure:
                start_alarm()   # starts only if not already running

                result = send_to_flask(avg_mov, rapid_changes, seizure_score)
                if result:
                    flask_result = result
                    prediction   = result.get('result', '').upper()
                    confidence   = result.get('confidence', 0)
                    icon         = "🚨" if prediction == "SEIZURE" else "✅"
                    print(f"{icon} {prediction} | "
                          f"Movement: {movement_score:.3f} | "
                          f"Rapid: {rapid_changes:.2f} | "
                          f"Score: {seizure_score}% | "
                          f"Conf: {confidence}%")

        else:
            prev_landmarks = None
            cv2.putText(frame,
                        "No body detected — step back",
                        (10, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 165, 255), 2)

        frame = draw_overlay(
            frame, is_seizure, movement_score,
            seizure_score, rapid_changes,
            flask_result, frames_with_body
        )

        cv2.imshow("Seizure Detection System", frame)

        # ── Key controls ──────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:      # Quit
            stop_alarm()
            print("\n👋 Stopped.")
            break
        elif key == ord('s') or key == ord('S'):  # Stop alarm
            stop_alarm()

    cap.release()
    cv2.destroyAllWindows()
    stop_alarm()
    print(f"📊 Total frames: {frame_count}")


# ─── Entry Point ─────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 45)
    print("  Seizure Detection + Emergency Alarm")
    print("=" * 45)
    print()
    print("  1 — Live Webcam")
    print("  2 — Video File")
    print()

    choice = input("Select (1 or 2): ").strip()

    if choice == '1':
        run_detection(source=0)
    elif choice == '2':
        path = input("Video file path: ").strip()
        if path:
            run_detection(source=path)
        else:
            print("❌ No path entered!")
    else:
        print("❌ Invalid choice!")