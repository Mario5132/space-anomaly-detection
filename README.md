\# AI-Based Real-Time Anomaly Detection for Deep Space Missions



\## Project Overview

This project develops an autonomous anomaly detection system for deep space missions like Chandrayaan-3, Aditya-L1, and Mars Perseverance Rover. It simulates real-time telemetry monitoring and autonomous decision-making when Earth communication is delayed.



\## Technologies Used

\- Python 3.10

\- Scikit-learn (Isolation Forest)

\- TensorFlow/Keras (Autoencoder)

\- Streamlit (Dashboard)

\- Plotly (Visualizations)



\## Dataset

NASA CMAPSS Turbofan Engine Degradation Dataset - simulates sensor readings from spacecraft subsystems until failure.



\## Models Implemented

1\. \*\*Isolation Forest\*\*: Baseline anomaly detector with 91% accuracy

2\. \*\*Autoencoder\*\*: Deep learning model with ROC-AUC 0.85



\## Results

| Model | Accuracy | ROC-AUC | Precision (Anomaly) | Recall (Anomaly) |

|-------|----------|---------|-------------------|-----------------|

| Isolation Forest | 91% | 0.83 | 0.69 | 0.73 |

| Autoencoder | 89% | 0.85 | 0.72 | 0.70 |



\## How to Run



1\. Activate virtual environment:

venv\\Scripts\\activate



text



2\. Run anomaly detection:

python src\\models\\anomaly\_detector.py

python src\\models\\autoencoder.py



text



3\. Launch mission control dashboard:

streamlit run dashboard\\app.py



text



\## Project Structure

space\_anomaly\_project/

├── data/raw/ # NASA CMAPSS dataset

├── src/

│ ├── utils/ # Data loading and preprocessing

│ └── models/ # Isolation Forest, Autoencoder

├── dashboard/ # Streamlit mission control

├── results/figures/ # Confusion matrices, training plots

└── README.md



text



\## Future Mission Extensions

\- \*\*Aditya-L1\*\*: Integrate solar storm prediction

\- \*\*Interplanetary\*\*: Add communication latency simulation

\- \*\*RL-based\*\*: Reinforcement learning for optimal actions

