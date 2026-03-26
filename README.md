# 🛰️ AI-Based Real-Time Anomaly Detection for Deep Space Missions

> An autonomous AI system that detects anomalies in spacecraft telemetry and makes real-time decisions without ground intervention.

---

## 📌 Project Overview

Deep space missions like **Chandrayaan-3**, **Aditya-L1**, and **Mars Perseverance Rover** face communication delays of 5–20 minutes. Waiting for Earth-based decisions during critical failures is impossible.

This project simulates an **onboard AI system** that:
- ✅ Detects anomalies in real-time telemetry
- ✅ Predicts potential subsystem failures
- ✅ Recommends corrective actions autonomously
- ✅ Alerts ground control only for critical events

---

## 🧠 Models Implemented

| Model | Accuracy | ROC-AUC | Precision (Anomaly) | Recall (Anomaly) |
|-------|----------|---------|---------------------|------------------|
| Isolation Forest | **91%** | 0.83 | 0.69 | 0.73 |
| Autoencoder | 89% | **0.85** | 0.72 | 0.70 |

---

## 📊 Dashboard Preview

The **Streamlit-based Mission Control Dashboard** provides:
- Live telemetry plots
- Real-time anomaly alerts
- Subsystem status with RUL (Remaining Useful Life)
- Autonomous decision log

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| ML Models | Scikit-learn, TensorFlow/Keras |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Data | NASA CMAPSS Turbofan Dataset |

---

## 📁 Project Structure

space_anomaly_project/
│
├── data/raw/ # NASA CMAPSS dataset (10+ files)
├── src/
│ ├── utils/ # Data loader & preprocessing
│ └── models/ # Isolation Forest & Autoencoder
├── dashboard/ # Streamlit mission control app
├── results/figures/ # Confusion matrices & training plots
├── README.md
├── TECHNICAL_REPORT.md
├── SUBMISSION_CHECKLIST.md
└── requirements.txt

text

---

## 🚀 How to Run

### 1. Activate Virtual Environment
```bash
venv\Scripts\activate
2. Run Anomaly Detection Models
bash
python src\models\anomaly_detector.py
python src\models\autoencoder.py
3. Launch Mission Control Dashboard
bash
streamlit run dashboard\app.py
Dashboard will open at http://localhost:8501

📈 Results
Isolation Forest: 91% accuracy, detects 73% of anomalies

Autoencoder: ROC-AUC 0.85, better at capturing subtle degradation patterns

Autonomous Decision System: Classifies risk into NORMAL / WARNING / CRITICAL with appropriate actions

🔮 Future Mission Extensions
Mission	Extension
Aditya-L1	Integrate solar wind & flare prediction models
Interplanetary	Simulate communication latency & queue decisions
RL-Based	Train reinforcement learning agent for optimal corrective actions
👨‍💻 Author
ALLANKI VV MANIKANTA SAI
GitHub: Mario5132
Project for: India Space Academy

📄 License
This project is submitted as part of the Artificial Intelligence & Machine Learning in Space Exploration coursework.

text

---


---




