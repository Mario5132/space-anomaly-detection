\# Technical Report: AI-Based Anomaly Detection for Deep Space Missions



\## 1. Abstract

Deep space missions face communication delays of 5-20 minutes, making Earth-dependent decisions impossible. This project develops an autonomous AI system that detects anomalies in real-time telemetry and recommends corrective actions without ground intervention.



\## 2. Introduction

\### 2.1 Problem Statement

Spacecraft in deep space cannot wait for ground control to analyze every anomaly. The system must:

\- Detect anomalies within seconds

\- Predict potential failures

\- Execute autonomous decisions

\- Alert ground only for critical events



\### 2.2 Why This Matters

Missions like Chandrayaan-3, Aditya-L1, and Mars Perseverance Rover demonstrated the need for onboard autonomy. This project simulates that capability using NASA's turbofan degradation dataset.



\## 3. Methodology



\### 3.1 Dataset

NASA CMAPSS dataset contains 100 engines with 21 sensors each, run until failure. Each engine represents a spacecraft subsystem.



\### 3.2 Preprocessing

\- Removed constant sensors (no variation)

\- Added RUL (Remaining Useful Life) column

\- Created anomaly labels: RUL ≤ 30 cycles = anomaly

\- Standardized features for neural networks



\### 3.3 Model 1: Isolation Forest

Unsupervised algorithm that isolates anomalies instead of profiling normal data. Trained on all data with contamination = anomaly percentage (15.03%).



\### 3.4 Model 2: Autoencoder

Neural network trained to reconstruct normal data. Reconstruction error becomes anomaly score. Architecture:

\- Input: 18 features

\- Encoder: 32 → 16 → 8

\- Decoder: 16 → 32 → 18

\- Loss: MSE, Optimizer: Adam



\### 3.5 Autonomous Decision System

Based on anomaly score threshold:

\- Score > -0.1: NORMAL → Continue

\- Score -0.3 to -0.1: WARNING → Isolate, alert ground

\- Score < -0.3: CRITICAL → Shutdown, switch redundancy



\## 4. Results



\### 4.1 Isolation Forest

\- Accuracy: 91%

\- ROC-AUC: 0.83

\- True Positives: 450 anomalies detected

\- False Positives: 204

\- False Negatives: 170 (anomalies missed)



\### 4.2 Autoencoder

\- ROC-AUC: 0.85

\- Better at distinguishing subtle anomalies

\- Training time: 2 minutes



\### 4.3 Dashboard Output

Interactive Streamlit dashboard showing:

\- Real-time telemetry plots

\- Current system status

\- Anomaly timeline

\- Autonomous decision log



\## 5. Conclusion

The AI system successfully detects 73% of anomalies with 91% accuracy. The dashboard provides mission control with real-time visibility while the autonomous decision layer handles immediate threats without ground intervention.



\## 6. Future Work

\- Deploy on edge hardware (Raspberry Pi)

\- Add reinforcement learning for optimal actions

\- Integrate with Aditya-L1 solar storm data

\- Simulate communication latency delays

