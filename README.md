⚠️ **Superseded by [AeroTwin-Edge](https://github.com/mmeetbe24-eng/Project_Aerotwin_Edge)** — this was the original LSTM-based prototype. The redesigned GRU-Attention version is smaller, faster, and deployment-optimized.
# ✈️ Turbofan Engine RUL Prediction — Predictive Maintenance AI

> A deep learning system using LSTM networks to predict the **Remaining Useful Life (RUL)** of aircraft turbofan engines, with uncertainty estimation, transfer learning, and TFLite deployment.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-LSTM-red?logo=keras&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-NASA%20CMAPSS-green)
![Status](https://img.shields.io/badge/Status-EST%20Project-yellow)

---

## 📌 Project Overview

This project implements a **predictive maintenance system** for jet engines using the NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. The model predicts how many operational cycles remain before an engine requires maintenance — enabling proactive scheduling instead of reactive repairs.

The system goes beyond basic regression by adding:
- **Bayesian-style uncertainty estimation** via MC Dropout
- **Transfer learning** from FD001 → FD004 (simple → complex conditions)
- **Explainable AI** through sensor importance correlation
- **Edge deployment** via TensorFlow Lite conversion

---

## 🎯 Key Features

| Feature | Details |
|---|---|
| Architecture | 2-layer Stacked LSTM |
| Loss Function | Huber Loss (robust to outliers) |
| Imbalance Handling | `scale_pos_weight`, class weighting |
| Uncertainty | MC Dropout (50 forward passes) |
| Transfer Learning | FD001 → FD004 domain adaptation |
| Deployment | TFLite + fallback `.h5` export |
| Evaluation | 10-parameter report incl. NASA S-Score |

---

## 📁 Repository Structure

```
✈️ turbofan-rul-prediction/
├── AI_EST_PROJECT_fixed.ipynb     # Main notebook (full pipeline)
└── README.md                      # You are here
```

> **Dataset not included** — download instructions below.

---

## 🗂️ Dataset — NASA CMAPSS

This project uses the **Turbofan Engine Degradation Simulation Dataset** from NASA.

**Download:** [NASA CMAPSS Dataset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

After downloading, place the files in your Google Drive at:
```
MyDrive/AI-PRESENTATION_EST/6.+Turbofan+Engine+Degradation+Simulation+Data+Set/CMAPSSData/
```

Files used:
```
train_FD001.txt   test_FD001.txt   RUL_FD001.txt
train_FD002.txt   test_FD002.txt   RUL_FD002.txt
train_FD004.txt   test_FD004.txt   RUL_FD004.txt
```

| Dataset | Operating Conditions | Fault Modes |
|---------|---------------------|-------------|
| FD001 | 1 | 1 (simple) |
| FD002 | 6 | 1 |
| FD004 | 6 | 2 (complex) |

---

## ⚙️ Feature Engineering

**Removed constant sensors** (no information in FD001):
`s_1, s_5, s_6, s_10, s_16, s_18, s_19`

**17 features used:**
```python
features = ['setting_1', 'setting_2', 'setting_3',
            's_2', 's_3', 's_4', 's_7', 's_8', 's_9',
            's_11', 's_12', 's_13', 's_14', 's_15',
            's_17', 's_20', 's_21']
```

**RUL Target Engineering:**
```python
RUL = max_cycle - current_cycle
RUL_clipped = RUL.clip(upper=125)   # Piecewise linear — stabilizes LSTM training
```

**Sliding Window:** sequences of 50 cycles → predict RUL at cycle 50

---

## 🧠 Model Architecture

```
Input: (50 time steps × 17 features)
        │
        ▼
  LSTM(128, tanh) + return_sequences=True
        │
   Dropout(0.2)
        │
        ▼
   LSTM(64, tanh)
        │
   Dropout(0.2)
        │
        ▼
   Dense(32, relu)
        │
        ▼
   Dense(1, linear)   ← Raw RUL prediction (cycles)
```

**Optimizer:** Adam (lr=0.001)
**Loss:** Huber (robust to outlier RUL values)
**Callbacks:** EarlyStopping (patience=10) + ModelCheckpoint

---

## 📊 Evaluation — 10 Parameters

The model is evaluated on both regression and classification metrics:

| # | Parameter | Type |
|---|-----------|------|
| 1 | RMSE | Regression |
| 2 | MAE | Regression |
| 3 | R² Score | Regression |
| 4 | Binary Accuracy (RUL ≤ 30) | Classification |
| 5 | Precision | Classification |
| 6 | Recall | Classification |
| 7 | F1-Score | Classification |
| 8 | AUC-ROC | Classification |
| 9 | Inference Latency (ms) | Performance |
| 10 | Confusion Matrix | Visualization |

> Classification threshold: **RUL ≤ 30 cycles = Maintenance Required**

Also includes the **NASA S-Score** for FD002 vs FD004 cross-domain comparison.

---

## 🔁 Transfer Learning Pipeline

```
Train on FD001 (simple, 1 operating condition)
           │
           ▼
    Save best_model.keras
           │
           ▼
  Freeze LSTM layers (preserve engine physics)
           │
           ▼
  Fine-tune Dense layers on FD004
  (complex, 6 conditions, 2 fault modes)
           │
           ▼
  Save universal_engine_model.keras
```

Fine-tuning uses `lr=0.0001` (10x smaller) to prevent catastrophic forgetting.

---

## 🎲 Uncertainty Estimation

Uses **MC Dropout** (Monte Carlo Dropout) — runs 50 forward passes with dropout active to simulate a Bayesian ensemble:

```python
predictions = [model(X, training=True) for _ in range(50)]
mean_rul = np.mean(predictions)
uncertainty = np.std(predictions)
lower_bound = mean_rul - 1.96 * uncertainty  # 95% confidence
```

Output example:
```
--- RELIABILITY REPORT ---
Actual RUL:       42.0 cycles
Predicted RUL:    38.7 cycles
Uncertainty (±):  4.2 cycles
DECISION: Schedule maintenance within next 30 cycles for safety.
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/turbofan-rul-prediction.git
cd turbofan-rul-prediction
```

### 2. Open in Google Colab
Upload `AI_EST_PROJECT_fixed.ipynb` to [Google Colab](https://colab.research.google.com/) and run all cells top to bottom.

### 3. Install dependencies
The first cell handles this automatically. Manually:
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

### 4. Set up dataset
Download the NASA CMAPSS dataset and place in Google Drive as shown in the Dataset section above.

---

## 📦 Model Export

The notebook exports the final model in two formats:

| Format | File | Use Case |
|--------|------|----------|
| TFLite | `predictive_maintenance_model.tflite` | Edge/mobile deployment |
| Keras H5 | `final_model_backup.h5` | Fallback / further training |

---

## 👤 Author

| Name | Enrollment No. |
|------|---------------|
| Meet | 1024030343 |

**Submitted to:** Mrs. Neeru Jindal
**Course:** Artificial Intelligence (EST)

---

## 📄 License

This project is submitted as part of an academic course. Dataset credit: NASA Ames Research Center.
