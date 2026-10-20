# AI-Based Seizure Detection System

An AI-assisted seizure detection system designed to support epilepsy patient safety through real-time camera-based movement analysis and machine learning.

## 📌 Project Overview

This project is an AI-based healthcare monitoring prototype that detects seizure-like movements using a camera and computer vision.

The system uses **OpenCV** and **MediaPipe Pose** to track human body movements and extract movement-related features from body landmarks. These features are analyzed using a **Random Forest machine learning model** to identify abnormal seizure-like movement patterns.

The system can work with:

- 🎥 Live camera
- 📹 Recorded seizure videos

When seizure-like activity is detected, the system can trigger an emergency alarm and send an alert to the connected monitoring system.

---

## 🚀 Key Features

- Real-time camera-based seizure-like movement detection
- Recorded video analysis
- Human pose detection using MediaPipe
- Body movement analysis
- Temporal movement pattern analysis
- Random Forest machine learning classification
- Seizure score and model confidence
- Local emergency alarm
- Flask backend API
- Alert and status monitoring
- Mobile notification/call automation integration
- HTML testing interface

---

## 🧠 How It Works

```text
Camera / Recorded Video
          ↓
       OpenCV
          ↓
    MediaPipe Pose
          ↓
 Body Landmark Tracking
          ↓
Movement Feature Extraction
          ↓
 Seizure Pattern Analysis
          ↓
 Random Forest ML Model
          ↓
 Seizure-like Activity?
       ↙          ↘
     No            Yes
     ↓              ↓
 Continue      Emergency Alarm
                    ↓
              Flask Backend
                    ↓
          Alert / Notification
```

---

## 🔬 AI & Machine Learning

The system uses human body pose landmarks to analyze movement patterns over time.

Important body landmarks include:

* Shoulders
* Elbows
* Wrists
* Hips
* Knees

The system analyzes movement-related features such as:

* Average movement
* Maximum movement
* Movement variation
* Rapid movement changes
* Temporal movement patterns
* Seizure pattern score

These features are evaluated using a **Random Forest Classifier** to classify seizure-like movement patterns.

---

## 🛠️ Technologies Used

| Technology      | Purpose                     |
| --------------- | --------------------------- |
| Python          | Core programming            |
| OpenCV          | Camera and video processing |
| MediaPipe       | Human pose estimation       |
| Scikit-learn    | Machine learning            |
| Random Forest   | Seizure classification      |
| Flask           | Backend API                 |
| NumPy           | Numerical processing        |
| Joblib          | Model handling               |
| Pygame          | Emergency alarm             |
| HTML/JavaScript | Testing interface           |

---

## 📁 Project Structure

```text
AI-Seizure-Detection-System/
│
├── app.py
├── camera_detection.py
├── offline_alarm.py
├── seizure_model.py
├── server.py
├── test.html
├── test_alert.py
├── test_telegram.py
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/Zahidali37/AI-Seizure-Detection-System.git
```

### Enter the Project Directory

```bash
cd AI-Seizure-Detection-System
```

### Create a Virtual Environment

```bash
python3 -m venv venv
```

### Activate the Virtual Environment

#### Linux/macOS

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install opencv-python mediapipe scikit-learn flask numpy joblib pygame requests
```

---

## ▶️ Running the Project

### Start the Flask Backend

```bash
python app.py
```

### Start Camera Detection

```bash
python camera_detection.py
```

The system can be tested using a live camera or recorded video, depending on the configured input.

---

## 🚨 Emergency Alert System

When seizure-like activity is detected, the prototype can:

1. Detect abnormal movement patterns
2. Calculate a seizure-related score
3. Display detection information
4. Activate a local emergency alarm
5. Send an alert to the backend
6. Communicate with the configured notification system

---

## 🧪 Testing & Evaluation

The prototype can be tested using recorded seizure videos and live camera input.

The system is designed to identify abnormal movement patterns and generate seizure-like activity alerts.

For larger-scale evaluation, important metrics include:

* Accuracy
* Sensitivity
* Specificity
* Precision
* Recall
* F1-score
* False-positive rate
* False-negative rate
* Detection latency

Further evaluation using larger and clinically appropriate datasets is required for medical validation.

---

## 📊 Current Development Status

**Status: Working Prototype / Proof of Concept**

This project demonstrates the combination of:

**Computer Vision + Human Pose Estimation + Machine Learning + Emergency Alerts**

The current system is intended for:

* Educational purposes
* Research
* AI/ML experimentation
* Computer vision experimentation
* Prototype development

---

## 🔮 Future Improvements

### 📶 Wearable Integration (Smartwatch/Band)

A key future direction for this project is integrating a smartwatch or wearable band 
to extend seizure monitoring beyond camera-based environments (home/hospital) to 
anywhere the patient goes.

Planned capabilities include:

* Continuous heart rate (HR) and blood pressure (BP) monitoring via wearable sensors
* Patient-specific threshold configuration for abnormal HR/BP patterns
* Automatic detection when readings exceed safe limits
* Instant emergency call/SMS notification to the patient's guardian or caregiver
* Real-time GPS live location sharing during an emergency alert
* Continuous monitoring independent of location — not limited to camera coverage

This wearable-based approach would allow the system to protect patients even when 
they are alone, outdoors, or away from any camera-monitored environment, making 
seizure detection more accessible and practical for everyday use.

---

## 🤝 Research & Collaboration

This project can be further developed through collaboration with:

* Hospitals
* Neurologists
* Healthcare professionals
* Medical-device companies
* Healthcare technology companies
* Universities
* Clinical research centers
* AI/ML research organizations

Potential collaboration areas include:

* Technical evaluation
* Medical review
* Dataset development
* Clinical validation
* Hospital pilot studies
* AI model improvement
* Research collaboration

---

## 🔐 Privacy & Security

Because the system processes camera and video data, privacy is an important consideration.

A production-ready implementation should consider:

* Secure data transmission
* Access control
* Encryption
* Minimal data retention
* Local/edge processing
* Protection of patient information
* Appropriate consent procedures
* Healthcare privacy requirements

---

## ⚠️ Medical Disclaimer

This project is a **research and educational prototype** and is **not a clinically validated medical diagnostic device**.

The system is designed to identify **seizure-like movement patterns** and should not be used as a substitute for professional medical diagnosis, treatment, or emergency medical services.

Clinical testing, validation, ethical approval, privacy assessment, and applicable regulatory evaluation are required before real-world medical deployment.

---

## 👨‍💻 Author

**Zahid Ali & M Haroon Abbas**

Computer Science Student
University of Sindh, Laar Campus Badin, Pakistan

---

## 📄 License

This project is currently provided for **research and educational purposes**.

---

## ⭐ Project Vision

The long-term goal of this project is to explore how **computer vision, artificial intelligence, and real-time monitoring** can be combined to develop accessible technologies that may help improve safety and emergency response for people experiencing seizure-like events.
