import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# FAKE DATA (same as your model)
# -----------------------------
np.random.seed(42)

normal_heart = np.random.normal(70, 10, 200)
normal_movement = np.random.normal(1, 0.5, 200)
normal_label = np.zeros(200)

seizure_heart = np.random.normal(120, 15, 200)
seizure_movement = np.random.normal(5, 1.5, 200)
seizure_label = np.ones(200)

heart = np.concatenate([normal_heart, seizure_heart])
movement = np.concatenate([normal_movement, seizure_movement])
labels = np.concatenate([normal_label, seizure_label])

data = pd.DataFrame({
    "heart_rate": heart,
    "movement": movement,
    "label": labels
})

# -----------------------------
# TRAIN MODEL
# -----------------------------
X = data[["heart_rate", "movement"]]
y = data["label"]

model = RandomForestClassifier()
model.fit(X, y)

print("System Started... Monitoring Patient 🔍")

# -----------------------------
# OFFLINE ALARM FUNCTION
# -----------------------------
def alarm():
    print("\n🚨 SEIZURE ALERT! 🚨")
    print("🔊 ALARM TRIGGERED!")

    # simple beep sound (Linux)
    for _ in range(5):
        print("\a")  # system beep
        time.sleep(0.5)

# -----------------------------
# REAL-TIME LOOP
# -----------------------------
while True:

    # simulate live sensor data
    heart_rate = np.random.normal(75, 20)
    movement = np.random.normal(2, 1)

    input_data = pd.DataFrame([[heart_rate, movement]],
                              columns=["heart_rate", "movement"])

    prediction = model.predict(input_data)

    print(f"Heart: {heart_rate:.2f}, Movement: {movement:.2f}")

    if prediction[0] == 1:
        alarm()

    time.sleep(2)  # delay for real-time simulation