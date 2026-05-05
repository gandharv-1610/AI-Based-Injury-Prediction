# 🦶 GaitScan AI — AI-Based Gait Injury Prediction

> **Clinical plantar pressure analysis powered by a Squeeze-and-Excitation ResNet (SE-ResNet) deep learning model.**  
> Detects 6 gait pathologies from 128×48 foot pressure matrices and generates structured clinical reports.

---

## 🎯 What This Project Does

GaitScan AI reads a CSV export from any plantar pressure plate sensor and:

1. **Classifies** the gait pattern into one of 6 clinical categories
2. **Scores** injury risk 0–100 using biomechanics, BMI, and age
3. **Generates** a full clinical report — insights, referrals, exercises, prognosis
4. **Visualises** 2D heatmaps, 3D pressure surfaces, and foot zone breakdowns

---

## 🦴 Gait Classes Detected

| Class | Clinical Name | Typical Cause |
|-------|--------------|---------------|
| `normal` | Normal Bipedal Gait | Healthy, symmetrical gait |
| `antalgic` | Pain-Avoidance Gait | Shortened stance to avoid pain |
| `lurching` | Gluteus Maximus Gait | Weak hip extensors |
| `steppage` | Foot-Drop Gait | Tibialis anterior / peroneal nerve weakness |
| `stiff-legged` | Quadriceps Gait | Reduced knee flexion during swing |
| `trendelenburg` | Hip Abductor Insufficiency | Gluteus medius weakness |

---

## 📂 Project Structure

```
AI-Based-Injury-Prediction-main/
│
├── app.py                    # Streamlit web application (4-tab clinical dashboard)
├── train_dl.py               # SE-ResNet deep learning training pipeline
├── train.py                  # Random Forest baseline training script
├── requirements.txt          # Python dependencies
│
├── Pressure_Data/            # 1,440 CSV pressure matrix files
│   └── subjectX_gaittype_trialY_pressure.csv
│
├── models/
│   ├── pressure_dl_model.pth # Trained SE-ResNet weights (~64 MB)
│   └── pressure_model.pkl    # Trained Random Forest fallback (~7 MB)
│
└── utils/
    ├── dl_model.py           # SE-ResNet CNN architecture
    ├── data_loader.py        # Dataset loading + augmentation
    ├── feature_extractor.py  # 16 biomechanical feature calculations
    └── medical_engine.py     # Clinical report generation engine
```

---

## 🧠 Deep Learning Model — SE-ResNet

The primary model is a **Squeeze-and-Excitation ResNet** that operates on raw 128×48 pressure matrices:

```
Input: (B, 1, 128, 48)
  → Stem (Conv 7×7, stride-2, MaxPool)
  → Stage 1: 2× SEResBlock  64→ 64
  → Stage 2: 2× SEResBlock  64→128
  → Stage 3: 2× SEResBlock 128→256
  → Stage 4: 3× SEResBlock 256→512
  → AdaptiveAvgPool → Classifier Head (512→256→128→6)
Output: 6-class logits
```

**SE blocks** learn *which feature channels matter most* for each gait class — ideal for structured pressure maps where heel, midfoot, and forefoot carry distinct diagnostic information.

---

## 🧬 Biomechanical Features Extracted

16 features are computed from every pressure matrix for display and the RF fallback model:

| Feature | Description |
|---------|-------------|
| `total_pressure` | Sum of all sensor values |
| `balance` | Left/Right asymmetry ratio |
| `cop_x`, `cop_y` | Centre of Pressure coordinates |
| `max_pressure` | Peak sensor reading |
| `variance`, `std_dev` | Statistical spread |
| `skewness`, `kurtosis` | Distribution shape |
| `forefoot/midfoot/heel_ratio` | Pressure zone ratios |
| `left/right_ratio` | Bilateral load ratios |
| `peak_pressure_index` | Peak-to-mean non-uniformity |
| `contact_area` | Fraction of active sensor cells |

---

## 📊 Dataset

- **12 subjects** × **6 gait types** × **20 trials** = **1,440 raw CSV files**
- Each file: **128 rows × 48 columns** of pressure values (Newtons)
- Naming convention: `subject{N}_{gaittype}_trial{T}_pressure.csv`
- With augmentation: **5,760 training samples** (4× expansion)

---

## ⚙️ Training Pipeline Highlights

`train_dl.py` uses **10 advanced techniques** to maximise accuracy:

| Technique | Purpose |
|-----------|---------|
| SE-ResNet architecture | Channel attention for pressure zones |
| 4× offline augmentation | H-flip, noise, brightness variation |
| Mixup augmentation (α=0.4) | Smooth decision boundaries |
| Runtime batch augmentation | On-the-fly noise + brightness |
| OneCycleLR scheduler | Warm-up + cosine annealing |
| Label smoothing (0.1) | Prevents overconfident softmax |
| AdamW + weight decay (1e-3) | Decoupled L2 regularisation |
| Gradient clipping (max=2.0) | Prevents exploding gradients |
| Test-Time Augmentation (TTA) | +1–3% accuracy at inference |
| Early stopping (patience=10) | Avoids overfitting |

---

## 🖥️ Frontend — 4-Tab Streamlit Dashboard

The web app features a **premium dark-mode glassmorphism** design with animated floating orbs, gradient text, and glass cards.

| Tab | What You See |
|-----|-------------|
| 🩺 **Clinical Report** | Risk gauge (0–100), gait classification, confidence %, clinical insights, specialist referral |
| 🔬 **Biomechanics** | 2D heatmap, 3D interactive surface, foot zone %, bilateral load donut chart |
| 📊 **Data Analysis** | Confidence bars for all 6 classes, all 16 feature metrics, raw matrix viewer, JSON report download |
| 💊 **Medical Insights** | Condition description, affected regions, recommendations, physiotherapy exercises, prognosis |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model

**Deep Learning (recommended, ~15–40 min on CPU):**
```bash
python train_dl.py
```

**Random Forest (fast fallback, ~2 min):**
```bash
python train.py
```

### 3. Launch the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### 4. Upload & Analyse
- Upload any CSV from the `Pressure_Data/` folder
- Fill in patient details (name, age, sex, height, weight)
- Explore all 4 tabs
- Download the full clinical report as JSON

> 📄 See `HOW_TO_RUN.txt` for the complete step-by-step setup guide including virtual environment setup, troubleshooting, and system requirements.  
> 📄 See `PROJECT_DOCUMENTATION.md` for full technical documentation of every component.

---

## 🏥 Medical Disclaimer

This system is developed for **educational and research purposes only**. All clinical insights, risk scores, recommendations, and specialist referrals are AI-generated informational outputs and **do not constitute medical advice**. Always consult a qualified healthcare professional for diagnosis and treatment.

---

*12 Subjects | 6 Gait Classes | 1,440 Pressure Samples | SE-ResNet Deep Learning*
