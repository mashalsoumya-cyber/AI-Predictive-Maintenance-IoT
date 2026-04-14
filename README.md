# 🚀 AI-Powered Predictive Maintenance for IoT Devices
 
## 📌 Project Overview
This project is an AI-based system that predicts machine failures using IoT sensor data such as **temperature, vibration, and current**.

The goal is to **detect potential failures before they happen**, helping industries reduce downtime and maintenance costs.

---

## 🎯 Problem Statement
Traditional maintenance is reactive — machines are fixed only after failure.

This project solves that by:
- Predicting failures in advance
- Reducing downtime
- Improving operational efficiency
- Saving maintenance cost

---

## 🏭 Industry Relevance
Used in:
- Manufacturing plants
- Power plants
- Automotive industry
- Aviation industry
- Smart factories (Industry 4.0)

Companies like:
**Siemens, GE, Tesla, IBM, Bosch** use similar systems.

---

## ⚙️ Tech Stack

- **Language:** Python 3.10
- **Libraries:**
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib
  - Joblib

---

## 🧠 Machine Learning Model

- Model Used: **Random Forest Classifier**
- Task: Binary Classification
  - `0 → No Failure`
  - `1 → Failure`

---

## 📊 Features Used

- Temperature
- Vibration
- Current

---

## 🏗️ Project Structure
AI-Predictive-Maintenance-IoT/
│
├── data/ # Dataset
├── src/ # Source code
│ ├── preprocess.py
│ ├── train.py
│ ├── predict.py
│
├── models/ # Saved ML model
├── outputs/ # Generated graphs
├── images/ # Project visuals (for README)
├── main.py # Main execution file
├── requirements.txt
└── README.md

---

## 🔁 Workflow

1. Load dataset
2. Preprocess data
3. Train ML model
4. Evaluate performance
5. Predict failure
6. Generate graphs

---

## ▶️ How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/your-username/AI-Predictive-Maintenance-IoT.git
cd AI-Predictive-Maintenance-IoT








## 📊 Project Outputs

![Temperature](outputs/failure_vs_temperature.png)
![Vibration](outputs/failure_vs_vibration.png)
![Confusion Matrix](outputs/confusion_matrix.png)
