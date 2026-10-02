from flask import Flask, request, jsonify, send_from_directory
from sklearn.ensemble import RandomForestClassifier
import time

app = Flask(__name__)

# ─── Emergency Configuration ─────────────────────────────────────
EMERGENCY_NUMBER  = "+923403980046"    # Parents phone number
seizure_detected  = False              # Global seizure state
seizure_timestamp = ""                 # Time when seizure was detected


# ─── AI Model Training Data ───────────────────────────────────────
# Features: [movement_intensity, rapid_changes, seizure_score]
#
# movement_intensity : 0.0 (still) to 1.0 (maximum movement)
# rapid_changes      : 0.0 (smooth) to 1.0 (very rapid shaking)
# seizure_score      : 0.0 (normal) to 100.0 (definite seizure)
#
# Key rule:
#   Normal walking  = high movement + LOW  rapid changes
#   Seizure/shaking = high movement + HIGH rapid changes

X_train = [
    # ── Completely still ─────────────────────────────────────────
    [0.00, 0.00,  0.0],
    [0.01, 0.05,  1.0],
    [0.02, 0.05,  2.0],
    [0.03, 0.10,  3.0],
    [0.04, 0.10,  4.0],
    [0.05, 0.10,  5.0],

    # ── Normal slow movement (talking, breathing, sitting) ────────
    [0.06, 0.15,  6.0],
    [0.07, 0.15,  7.0],
    [0.08, 0.20,  8.0],
    [0.09, 0.20,  9.0],
    [0.10, 0.20, 10.0],
    [0.12, 0.25, 12.0],
    [0.14, 0.25, 14.0],

    # ── Normal fast movement (walking, waving) — NOT seizure ──────
    [0.30, 0.20, 20.0],
    [0.40, 0.20, 22.0],
    [0.50, 0.25, 25.0],
    [0.60, 0.25, 27.0],
    [0.70, 0.20, 28.0],
    [0.80, 0.20, 30.0],
    [0.90, 0.25, 32.0],
    [0.95, 0.25, 33.0],
    [1.00, 0.20, 30.0],

    # ── Some direction changes — still NOT seizure ────────────────
    [0.30, 0.35, 25.0],
    [0.40, 0.40, 28.0],
    [0.50, 0.40, 30.0],
    [0.60, 0.45, 33.0],

    # ── Seizure patterns (high movement + very rapid changes) ──────
    [0.25, 0.70, 52.0],
    [0.30, 0.72, 55.0],
    [0.35, 0.75, 60.0],
    [0.40, 0.75, 63.0],
    [0.45, 0.78, 67.0],
    [0.50, 0.80, 72.0],
    [0.55, 0.82, 76.0],
    [0.60, 0.82, 79.0],
    [0.65, 0.85, 82.0],
    [0.70, 0.85, 85.0],
    [0.75, 0.88, 87.0],
    [0.80, 0.88, 89.0],
    [0.85, 0.90, 91.0],
    [0.90, 0.90, 93.0],
    [0.95, 0.92, 95.0],
    [1.00, 1.00, 100.0],
]

y_train = [
    # Still (6 samples)
    "normal", "normal", "normal", "normal", "normal", "normal",
    # Slow movement (7 samples)
    "normal", "normal", "normal", "normal", "normal", "normal", "normal",
    # Fast movement — not seizure (9 samples)
    "normal", "normal", "normal", "normal", "normal",
    "normal", "normal", "normal", "normal",
    # Some direction change — not seizure (4 samples)
    "normal", "normal", "normal", "normal",
    # Seizure patterns (16 samples)
    "seizure", "seizure", "seizure", "seizure",
    "seizure", "seizure", "seizure", "seizure",
    "seizure", "seizure", "seizure", "seizure",
    "seizure", "seizure", "seizure", "seizure",
]

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print("✅ AI Model trained and ready!")
print(f"   Total training samples : {len(X_train)}")
print(f"   Classes                : {model.classes_}")
print(f"   Emergency number       : {EMERGENCY_NUMBER}")


# ─── Routes ──────────────────────────────────────────────────────

@app.route('/')
def home():
    """Home route — confirms server is running"""
    return jsonify({
        "status"   : "Seizure Detection Server is running!",
        "endpoints": {
            "POST /alert"        : "Send movement data for seizure detection",
            "GET  /status_check" : "Check if seizure is currently detected",
            "POST /reset"        : "Reset seizure state after alert handled",
            "GET  /status"       : "Check server and model status",
            "GET  /test"         : "Open mobile test page in browser"
        }
    })


@app.route('/test')
def test_page():
    """Serve the mobile test HTML page"""
    return send_from_directory('.', 'test.html')


@app.route('/alert', methods=['POST'])
def alert():
    """
    Main seizure detection endpoint.
    Receives movement data from camera_detection.py
    and runs AI prediction.

    Request JSON:
    {
        "movement_intensity" : 0.35,
        "rapid_changes"      : 0.75,
        "seizure_score"      : 65.0
    }

    Response JSON:
    {
        "result"     : "seizure" or "normal",
        "confidence" : 88.0
    }
    """
    global seizure_detected, seizure_timestamp

    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Extract features from request
        movement_intensity = float(data.get('movement_intensity', 0.0))
        rapid_changes      = float(data.get('rapid_changes',      0.0))
        seizure_score      = float(data.get('seizure_score',      0.0))

        # Run AI prediction
        features   = [[movement_intensity, rapid_changes, seizure_score]]
        prediction = model.predict(features)[0]
        proba      = model.predict_proba(features)[0]
        confidence = round(max(proba) * 100, 1)

        # Update global seizure state
        if prediction == "seizure":
            seizure_detected  = True
            seizure_timestamp = time.strftime("%H:%M:%S")
        else:
            seizure_detected  = False

        # Print result to terminal
        icon = "🚨" if prediction == "seizure" else "✅"
        print(f"{icon} {prediction.upper():8s} | "
              f"Movement: {movement_intensity:.3f} | "
              f"Rapid: {rapid_changes:.2f} | "
              f"Score: {seizure_score:.1f}% | "
              f"Confidence: {confidence}%")

        return jsonify({
            "result"    : prediction,
            "confidence": confidence
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/status_check', methods=['GET'])
def status_check():
    """
    Automate app polls this endpoint every 10 seconds.
    If seizure is True, Automate triggers a phone call.

    Response JSON:
    {
        "seizure"   : true or false,
        "call"      : "+92" or "",
        "message"   : "SEIZURE EMERGENCY!" or "Normal",
        "timestamp" : "08:45:32"
    }
    """
    return jsonify({
        "seizure"  : seizure_detected,
        "call"     : EMERGENCY_NUMBER if seizure_detected else "",
        "message"  : "SEIZURE EMERGENCY! Call Now!" if seizure_detected else "Normal",
        "timestamp": seizure_timestamp if seizure_detected else time.strftime("%H:%M:%S")
    })


@app.route('/reset', methods=['POST'])
def reset():
    """
    Reset seizure state after alert has been handled.
    Called by Automate app after making the emergency call.
    """
    global seizure_detected
    seizure_detected = False
    print("[RESET] Seizure state has been reset to normal.")
    return jsonify({"status": "Seizure state reset successfully"})


@app.route('/status', methods=['GET'])
def status():
    """Check current server and model status"""
    return jsonify({
        "server"          : "online",
        "model"           : "RandomForestClassifier",
        "trees"           : 200,
        "features"        : ["movement_intensity", "rapid_changes", "seizure_score"],
        "seizure_active"  : seizure_detected,
        "emergency_number": EMERGENCY_NUMBER
    })


# ─── Start Server ─────────────────────────────────────────────────

if __name__ == '__main__':
    print()
    print("=" * 50)
    print("   Seizure Detection Server")
    print("=" * 50)
    print(f"   Local URL    : http://127.0.0.1:5000")
    print(f"   Network URL  : http://192.168.100.169:5000")
    print(f"   Mobile test  : http://127.0.0.1:5000/test")
    print(f"   Automate poll: /status_check")
    print("=" * 50)
    print()
    app.run(host='0.0.0.0', port=5000, debug=True)