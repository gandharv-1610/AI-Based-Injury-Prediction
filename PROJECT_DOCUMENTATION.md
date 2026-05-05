# 🦶 GaitScan AI — Complete Project Documentation

> **AI-Based Gait Injury Prediction System**  
> A clinical-grade, deep learning-powered plantar pressure analysis platform for detecting and classifying gait pathologies.

---

## 📑 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Dataset](#3-dataset)
4. [Feature Engineering](#4-feature-engineering)
5. [Machine Learning Model (Random Forest)](#5-machine-learning-model-random-forest)
6. [Deep Learning Model (SE-ResNet CNN)](#6-deep-learning-model-se-resnet-cnn)
7. [Training Pipeline (train_dl.py)](#7-training-pipeline-train_dlpy)
8. [Medical Engine](#8-medical-engine)
9. [Frontend — Streamlit Application](#9-frontend--streamlit-application)
10. [Tab-by-Tab UI Walkthrough](#10-tab-by-tab-ui-walkthrough)
11. [Full Data Flow: Upload → Diagnosis](#11-full-data-flow-upload--diagnosis)
12. [Model Performance & Results](#12-model-performance--results)
13. [Installation & Running](#13-installation--running)
14. [File Reference](#14-file-reference)

---

## 1. Project Overview

**GaitScan AI** is an end-to-end clinical AI system that analyses plantar pressure data (foot pressure matrices) to:

- **Classify** a patient's gait type from 6 categories: Normal, Antalgic, Lurching, Steppage, Stiff-Legged, or Trendelenburg
- **Score** the patient's injury risk on a 0–100 scale incorporating biomechanics, BMI, and age
- **Generate** a full clinical report: insights, recommendations, specialist referrals, physiotherapy exercises, and prognosis

### Why This Matters

Abnormal gait patterns are early indicators of neurological disorders, muscle weakness, joint damage, and nerve injury. Traditionally, gait analysis requires expensive motion capture labs. This system provides clinical-grade analysis from a simple CSV export from any pressure-plate sensor.

### Gait Classes Detected

| Class | Clinical Name | Root Cause |
|-------|--------------|------------|
| `normal` | Normal Bipedal Gait | Healthy, symmetrical gait |
| `antalgic` | Pain-Avoidance Gait | Reduced stance to avoid pain |
| `lurching` | Gluteus Maximus Gait | Weak hip extensors |
| `steppage` | Foot-Drop Gait | Tibialis anterior weakness / peroneal nerve |
| `stiff-legged` | Quadriceps Gait | Reduced knee flexion in swing |
| `trendelenburg` | Hip Abductor Insufficiency | Gluteus medius weakness |

---

## 2. Project Structure

```
AI-Based-Injury-Prediction-main/
│
├── app.py                    # Main Streamlit web application (872 lines)
├── train.py                  # Random Forest training script
├── train_dl.py               # SE-ResNet deep learning training pipeline
├── requirements.txt          # Python dependencies
│
├── Pressure_Data/            # Dataset — 1,440 CSV pressure files
│   ├── subject1_normal_trial1_pressure.csv
│   ├── subject1_antalgic_trial1_pressure.csv
│   └── ...  (12 subjects × 6 gait types × 20 trials = 1,440 files)
│
├── models/
│   ├── pressure_dl_model.pth  # Trained SE-ResNet weights (~64 MB)
│   └── pressure_model.pkl     # Trained Random Forest (~7 MB)
│
└── utils/
    ├── data_loader.py         # Dataset loading + augmentation
    ├── dl_model.py            # SE-ResNet CNN architecture
    ├── feature_extractor.py   # 16 biomechanical feature calculations
    └── medical_engine.py      # Clinical report generation engine
```

---

## 3. Dataset

### Structure

The dataset lives in `Pressure_Data/` and contains **1,440 CSV files**.

**Naming convention:** `subjectX_gaittype_trialY_pressure.csv`

- **12 subjects** (subject1 through subject12)
- **6 gait types** per subject: normal, antalgic, lurching, steppage, stiff-legged, trendelenburg
- **20 trials** per gait type per subject
- Total: 12 × 6 × 20 = **1,440 raw samples**

### Pressure Matrix Format

Each CSV file is a **128 × 48 numerical matrix**:
- **128 rows** → foot length axis (heel to toe)
- **48 columns** → foot width axis (left half = cols 0–23, right half = cols 24–47)
- Each cell value = pressure reading in Newtons (N) at that sensor point

### Foot Zone Layout

```
Rows   0 –  42  →  Forefoot (toes & metatarsal heads)
Rows  43 –  84  →  Midfoot  (arch region)
Rows  85 – 127  →  Heel     (calcaneus / heel bone)

Cols   0 –  23  →  Left foot half
Cols  24 –  47  →  Right foot half
```

### Data Augmentation

When `augment=True` in `load_dl_dataset()`, each raw sample generates **3 additional variants**, bringing total training samples to **1,440 × 4 = 5,760**:

| Augmentation | How | Purpose |
|---|---|---|
| Horizontal Flip | `np.fliplr()` | Simulates left/right foot swap |
| Gaussian Noise | σ = 0.02 added | Models sensor noise |
| Brightness Scale | Random ±15% intensity | Models calibration variation |

Additionally during training, **runtime augmentation** is applied on every batch:
- Random H-flip with 50% probability
- Brightness perturbation ±10%
- Gaussian noise σ = 0.015

---

## 4. Feature Engineering

**File:** `utils/feature_extractor.py`

The function `extract_features(pressure_matrix)` computes **16 biomechanical features** from the raw 128×48 matrix. These features are used by the Random Forest model and for display in the UI.

### Complete Feature List

| Feature | Formula / Description |
|---------|----------------------|
| `total_pressure` | Sum of all cell values — overall plantar load |
| `balance` | `|left_sum - right_sum| / total` — L/R asymmetry (0=perfect) |
| `cop_x` | Centre of Pressure X-coordinate (weighted column mean) |
| `cop_y` | Centre of Pressure Y-coordinate (weighted row mean) |
| `max_pressure` | Peak pressure value anywhere on the foot |
| `variance` | Statistical variance of all pressure values |
| `std_dev` | Standard deviation of pressure distribution |
| `skewness` | 3rd moment — distribution tail direction |
| `kurtosis` | 4th moment (excess) — peakedness vs Gaussian |
| `forefoot_ratio` | `sum(rows 0–42) / total` |
| `midfoot_ratio` | `sum(rows 43–84) / total` |
| `heel_ratio` | `sum(rows 85–127) / total` |
| `left_ratio` | `sum(cols 0–23) / total` |
| `right_ratio` | `sum(cols 24–47) / total` |
| `peak_pressure_index` | `max_pressure / mean_pressure` — non-uniformity |
| `contact_area` | Fraction of non-zero cells (foot contact coverage) |

### Centre of Pressure Calculation

```python
y_idx, x_idx = np.indices(pressure_matrix.shape)
cop_x = sum(matrix * x_idx) / total_pressure
cop_y = sum(matrix * y_idx) / total_pressure
```

This gives the weighted centroid of the pressure distribution — a clinically critical metric for gait analysis.

---

## 5. Machine Learning Model (Random Forest)

**File:** `train.py` | **Saved:** `models/pressure_model.pkl`

### How It Works

1. Loads all 1,440 CSVs via `load_dataset()`
2. Extracts 16 features per file using `extract_features()`
3. Builds a labelled DataFrame (16 features + label column)
4. Splits 80/20 train/test (stratified by class)
5. Trains `RandomForestClassifier(n_estimators=100)`
6. Saves to `models/pressure_model.pkl` via `joblib`

### When It's Used

The Random Forest is the **fallback model**. If the DL model file is missing or incompatible, the app automatically loads the RF model. The RF uses only 6 core features: `total_pressure`, `balance`, `cop_x`, `cop_y`, `max_pressure`, `variance`.

### Training Command
```bash
python train.py
```

---

## 6. Deep Learning Model (SE-ResNet CNN)

**File:** `utils/dl_model.py` | **Saved:** `models/pressure_dl_model.pth`

This is the **primary, high-accuracy model**. It operates directly on the raw 128×48 pressure matrix (no hand-crafted features needed).

### Architecture: SE-ResNet (Squeeze-and-Excitation ResNet)

The model is a custom ResNet with **Squeeze-and-Excitation (SE) channel attention** blocks at every residual stage.

```
Input: (B, 1, 128, 48)   — batch of normalised pressure maps
│
├── Stem
│     Conv2d(1→64, 7×7, stride=2) → BN → ReLU → MaxPool   →  (B, 64, 32, 12)
│
├── Stage 1 — 2× SEResBlock(64→64,   stride=1)              →  (B,  64, 32, 12)
├── Stage 2 — 2× SEResBlock(64→128,  stride=2)              →  (B, 128, 16,  6)
├── Stage 3 — 2× SEResBlock(128→256, stride=2)              →  (B, 256,  8,  3)
├── Stage 4 — 3× SEResBlock(256→512, stride=2)              →  (B, 512,  4,  2)
│
├── AdaptiveAvgPool2d(1×1)                                   →  (B, 512, 1, 1)
│
└── Classifier Head
      BN1d(512) → Linear(512→256) → ReLU → Dropout(0.3)
      → Linear(256→128) → ReLU → Dropout(0.15)
      → Linear(128→6)
      
Output: (B, 6)  — raw logits for 6 gait classes
```

**Total parameters:** ~11 million

### Squeeze-and-Excitation Block

The SE block lets the network learn **which channels (feature maps) matter most** for each gait class:

```
Input Feature Map (B, C, H, W)
    ↓ AdaptiveAvgPool2d → (B, C, 1, 1)   [Squeeze]
    ↓ Flatten → (B, C)
    ↓ Linear(C → C//16) → ReLU            [Excite - compress]
    ↓ Linear(C//16 → C) → Sigmoid         [Excite - expand]
    ↓ Reshape → (B, C, 1, 1)
    ↓ Multiply with original feature map  [Scale]
```

This is especially powerful for pressure maps because heel, midfoot, and forefoot channels carry distinct information for each gait type.

### Residual Block (SEResBlock)

Each block uses **pre-activation** (BN→ReLU before convolution) for better gradient flow:

```
Input x
├── BN → ReLU → Conv(3×3, stride) → BN → ReLU → Conv(3×3) → SE Block
└── Shortcut: Conv(1×1) if dims change, else Identity
→ Add both paths → Output
```

### Weight Initialisation

- **Conv2d:** Kaiming Normal (fan_out, relu mode)
- **BatchNorm:** weights=1, bias=0
- **Linear:** Xavier Uniform

---

## 7. Training Pipeline (`train_dl.py`)

**File:** `train_dl.py`

This is the high-accuracy training pipeline incorporating 10 advanced techniques.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 60 |
| Batch Size | 32 |
| Optimizer | AdamW |
| Max Learning Rate | 5e-4 |
| Weight Decay | 1e-3 |
| Scheduler | OneCycleLR |
| Loss Function | CrossEntropyLoss (label_smoothing=0.1) |
| Gradient Clipping | max_norm=2.0 |
| Val Split | 15% |
| Early Stopping Patience | 10 epochs |

### 10 Advanced Techniques

#### 1. SE-ResNet Architecture
Channel attention via SE blocks (described above) lets the model selectively focus on clinically relevant pressure zones.

#### 2. Offline Data Augmentation (4× expansion)
The data loader generates 3 augmented copies per sample, resulting in 5,760 training samples from 1,440 originals.

#### 3. Mixup Augmentation
Interpolates two random training samples and their labels:
```
x_mixed = λ·x_i + (1-λ)·x_j
loss = λ·CE(logits, y_i) + (1-λ)·CE(logits, y_j)
```
Lambda is sampled from Beta(0.4, 0.4). Mixup smooths decision boundaries and prevents overconfident predictions. **Disabled in the final 15 epochs** to allow the model to fine-tune on clean samples.

#### 4. Runtime Augmentation
Applied during every training batch (on-the-fly, GPU-compatible):
- Random horizontal flip (50% probability)
- Brightness perturbation ±10%
- Gaussian noise σ=0.015

#### 5. OneCycleLR Scheduler
Warm-up for 30% of total steps, then cosine annealing to `LR_max / 10,000`. This aggressive schedule finds better optima than a fixed learning rate.

#### 6. Label Smoothing (0.1)
Prevents the softmax from becoming overconfident. True class gets target `0.9` instead of `1.0`, distributing 0.1 across all other classes.

#### 7. AdamW + Weight Decay (L2=1e-3)
Decoupled weight decay (not applied to Adam's momentum terms) provides cleaner regularisation vs standard Adam + L2.

#### 8. Gradient Clipping (max_norm=2.0)
Prevents exploding gradients during high-LR warmup phase on CPU training.

#### 9. Test-Time Augmentation (TTA)
During final evaluation, each sample is augmented 4 times and softmax probabilities are averaged. This reduces prediction variance and consistently adds +1–3% accuracy.

#### 10. Early Stopping (patience=10)
Stops training if validation accuracy doesn't improve for 10 consecutive epochs. Best weights are immediately saved to disk on every improvement (safe against Ctrl+C interruption).

### Training Loop Flow

```
for epoch in 1..60:
    ├── If epoch ≤ 45: use Mixup
    │   else: pure CE loss (fine-tuning phase)
    ├── train_one_epoch()
    │     for each batch:
    │       ├── runtime_augment(x)
    │       ├── mixup_batch(x, y) if Mixup
    │       ├── forward pass → logits
    │       ├── compute loss
    │       ├── backward + clip_grad_norm
    │       └── scheduler.step() (per-step, not per-epoch)
    ├── evaluate() on val set
    ├── if val_acc improved: save model
    └── if patience exceeded: early stop
```

### Training Command
```bash
python train_dl.py
```

---

## 8. Medical Engine

**File:** `utils/medical_engine.py`

The medical engine translates AI predictions into structured clinical reports.

### MedicalReport Dataclass

```python
@dataclass
class MedicalReport:
    gait_type:           str
    injury_risk_score:   int          # 0–100
    risk_category:       str          # Low | Moderate | High | Critical
    risk_color:          str          # Hex colour for UI
    primary_condition:   str
    affected_regions:    List[str]
    short_description:   str
    clinical_insights:   List[str]    # 5 clinical observations
    recommendations:     List[str]    # 5 clinical recommendations
    exercises:           List[dict]   # 3 physiotherapy exercises
    specialist_referral: str
    follow_up_urgency:   str
    prognosis:           str
    disclaimer:          str
```

### Risk Score Calculation

The injury risk score (0–100) is computed from 4 factors:

```
score = (base_risk + balance_penalty + bmi_mod + age_mod) × confidence
        + 50 × (1 - confidence)
```

| Factor | Range | Logic |
|--------|-------|-------|
| `base_risk` | By gait type | Normal=8, Antalgic=72, Steppage=82, Trendelenburg=68, Lurching=65, Stiff-legged=58 |
| `balance_penalty` | 0–15 pts | `min(15, asymmetry × 50)` |
| `bmi_mod` | 0–10 pts | BMI≥30→+10, BMI≥25→+5, else 0 |
| `age_mod` | 0–8 pts | Age≥65→+8, Age≥45→+4, else 0 |
| Confidence | Pulls toward 50 | Low confidence → uncertain, pulls score toward neutral |

### Risk Categories

| Score | Category | Colour |
|-------|----------|--------|
| 0–20 | Low | 🟢 Emerald `#10b981` |
| 21–45 | Moderate | 🟡 Amber `#f59e0b` |
| 46–70 | High | 🔴 Red `#ef4444` |
| 71–100 | Critical | 🟣 Purple `#9333ea` |

### Clinical Knowledge Base

Each gait type has a hardcoded knowledge base entry (`_KB` dictionary) containing:
- **Primary condition** name
- **Affected anatomical regions**
- **Short clinical description**
- **5 clinical insights** (biomechanical observations)
- **5 recommendations** (treatment/management steps)
- **3 physiotherapy exercises** (name, description, sets/reps)
- **Specialist referral** (who to see and how urgently)
- **Follow-up urgency** (48 hours to 12 months)
- **Prognosis** (expected outcome with treatment)

---

## 9. Frontend — Streamlit Application

**File:** `app.py` (872 lines)

### Design System

The UI uses a **premium dark-mode glassmorphism** design built entirely with custom CSS injected via `st.markdown(unsafe_allow_html=True)`.

#### Visual Design Features

| Element | Implementation |
|---------|----------------|
| Background | Radial gradient: `#0c1445 → #0f172a → #0a0a1a` |
| Animated Orbs | 3 fixed `div.orb` elements with `@keyframes float` (12s loop) |
| Typography | Google Fonts "Outfit" (300–800 weight) |
| Glass Cards | `rgba(255,255,255,0.04)` bg + `backdrop-filter: blur(12px)` |
| Risk Borders | Left border colour-coded: green/amber/red/purple |
| Gradient Text | `linear-gradient(90deg, #38bdf8, #818cf8, #c084fc)` |
| Metric Values | Gradient text via `-webkit-background-clip: text` |

#### CSS Component Classes

- `.glass-card` — frosted-glass card container
- `.risk-low/moderate/high/critical` — coloured left border
- `.risk-badge` — animated pulsing risk level badge
- `.gauge-ring` — circular progress ring using `conic-gradient`
- `.insight-item` — clinical insight row with icon
- `.ex-card` — physiotherapy exercise card
- `.conf-bar-wrap` — horizontal confidence bar
- `.info-box / .warning-box / .emergency-box` — contextual alert boxes

### Sidebar

The sidebar contains:
1. **GaitScan AI branding** with gradient text logo
2. **CSV file uploader** — accepts 128×48 pressure matrix files
3. **Patient Profile form** — Name, Age, Sex, Height (cm), Weight (kg)
4. **Model status indicator** — shows active backend (DL or RF)

### Model Loading (`@st.cache_resource`)

```python
# Priority 1: Load SE-ResNet DL model
if os.path.exists('models/pressure_dl_model.pth'):
    model = PressureCNN(num_classes=6)
    model.load_state_dict(torch.load(...))
    return model, "DL"

# Priority 2: Fallback to Random Forest
if os.path.exists('models/pressure_model.pkl'):
    return joblib.load(rf_path), "RF"
```

If the DL weights are incompatible with the current architecture, the stale `.pth` file is **automatically deleted** and the RF fallback is used.

### Inference Logic

**DL Model path:**
```python
# Normalise: (matrix - min) / (max - min)
tensor = torch.tensor(norm_matrix).unsqueeze(0).unsqueeze(0)  # → (1,1,128,48)
logits = model(tensor)
probs = torch.softmax(logits, dim=1).squeeze().numpy()
prediction = LABELS[np.argmax(probs)]
confidence = max(probs)
```

**RF Model path:**
```python
feature_df = pd.DataFrame([{6 core features}])
prediction = model.predict(feature_df)[0]
probabilities = model.predict_proba(feature_df)[0]
confidence = max(probabilities)
```

---

## 10. Tab-by-Tab UI Walkthrough

After uploading a CSV, the app shows **4 tabs**:

### Tab 1: 🩺 Clinical Report

**What it shows:**
- **Patient header card** — name, sex, age, height, weight, BMI, active model backend
- **Risk gauge ring** — animated circular gauge (0–100) coloured by severity
- **Risk badge** — pulsing "Low/Moderate/High/Critical Risk" label
- **AI classification card** — predicted gait type, primary condition, confidence %, L/R asymmetry
- **4 biometric metrics** — BMI, BMR (Mifflin-St Jeor equation), Peak Pressure, Contact Area
- **5 clinical insights** — gait-type-specific observations pulled from knowledge base
- **Specialist referral** + **Follow-up urgency** side-by-side cards
- **AI disclaimer** warning box

### Tab 2: 🔬 Biomechanics

**What it shows:**
- **2D Pressure Heatmap** (Plasma colorscale, Plotly `imshow`) with COP marker (white × symbol)
- **3D Topographic Surface** (interactive Plotly `go.Surface`, draggable/rotatable)
- **Foot Zone Pressure** — 3 glass cards showing Forefoot / Midfoot / Heel % of total pressure
- **Bilateral Load Distribution** — donut chart (left blue / right red) + bar chart

### Tab 3: 📊 Data Analysis

**What it shows:**
- **Model Confidence Bar Chart** — horizontal bars for all 6 classes sorted by probability (Viridis colorscale)
- **Biomechanical Feature Metrics** — all 16 extracted features displayed as `st.metric` widgets
- **Raw Matrix Expander** — `st.dataframe` showing the full 128×48 pressure matrix
- **Download Clinical Report** — exports complete report as JSON (patient info, gait type, risk, insights, exercises, features, etc.)

### Tab 4: 💊 Medical Insights

**What it shows:**
- **Detected Condition card** — full condition name + description
- **Affected Anatomical Regions** — pill-shaped badges (e.g., "Lateral ankle", "Gluteus medius")
- **5 Clinical Recommendations** — treatment/management steps with icons
- **3 Physiotherapy Exercises** — exercise cards with name, description, sets/reps
- **Prognosis** — expected outcome statement
- **Emergency Warning** (High/Critical only) — red alert with emergency symptoms to watch for
- **AI Disclaimer**

---

## 11. Full Data Flow: Upload → Diagnosis

```
User uploads CSV
        │
        ▼
np.loadtxt(file, delimiter=',')
→ Validate shape (must be 2D)
        │
        ├──────────────────────────────────────────┐
        │  DL Path                                 │ RF Path
        ▼                                          ▼
Normalise [0,1]                        extract_features(matrix)
Reshape → (1,1,128,48) tensor          → 6-feature DataFrame
        │                                          │
        ▼                                          ▼
PressureCNN.forward()                  RF.predict() + predict_proba()
Softmax → 6 class probs                → prediction + probabilities
        │                                          │
        └──────────────┬───────────────────────────┘
                       │
                       ▼
              prediction (gait type)
              confidence (0.0–1.0)
              probabilities (6 values)
                       │
                       ▼
              extract_features(matrix) ← always run for UI display
                       │
                       ▼
              generate_report(gait_type, features, age, weight, height, confidence)
              → MedicalReport dataclass
                       │
                       ▼
              Compute BMI, BMR from patient sidebar inputs
                       │
                       ▼
              Render 4 Tabs with all visualisations, metrics, and clinical data
```

---

## 12. Model Performance & Results

### Deep Learning Model (SE-ResNet)
- **Architecture:** SE-ResNet with 4 residual stages, ~11M parameters
- **Training samples:** 5,760 (1,440 originals × 4 augmentation factor)
- **Validation split:** 15%
- **Best validation accuracy:** Tracked per-run; target >90% with full pipeline
- **Techniques:** Mixup + OneCycleLR + Label Smoothing + TTA

### Random Forest Model (Baseline)
- **Features:** 16 biomechanical features (6 used during inference)
- **Estimators:** 100 trees
- **Training samples:** 1,440
- **Typical accuracy:** ~85–88% on held-out test set (80/20 stratified split)

### Class Difficulty

The hardest-to-distinguish classes are typically:
- **Lurching vs Trendelenburg** (both show trunk compensation)
- **Stiff-legged vs Normal** (subtle swing-phase difference in pressure data)
- **Antalgic** is usually the easiest (highly asymmetric pressure signature)

---

## 13. Installation & Running

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` contents:**
```
numpy
pandas
scikit-learn
matplotlib
streamlit
joblib
torch
torchvision
plotly
```

### Train the Models

**Option A — Deep Learning (recommended, ~15–30 min on CPU):**
```bash
python train_dl.py
```

**Option B — Random Forest (fast, ~2 min):**
```bash
python train.py
```

### Run the Web Application

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Using the App

1. Upload any CSV file from `Pressure_Data/` folder (or from a real pressure plate)
2. Fill in patient details in the sidebar (Name, Age, Sex, Height, Weight)
3. Browse the 4 tabs for clinical report, biomechanics, data analysis, and medical insights
4. Download the full JSON report using the button in Tab 3

---

## 14. File Reference

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `app.py` | Streamlit UI (872 lines) | `load_ml_model()`, `_gauge_html()`, `_badge_html()`, Tab rendering |
| `train_dl.py` | DL training pipeline | `main()`, `mixup_batch()`, `runtime_augment()`, `tta_predict()`, `train_one_epoch()`, `evaluate()` |
| `train.py` | RF baseline training | `main()` |
| `utils/dl_model.py` | SE-ResNet architecture | `PressureCNN`, `SEResBlock`, `SEBlock` |
| `utils/data_loader.py` | Dataset loading + augmentation | `load_dl_dataset()`, `load_dataset()`, `_augment()`, `_normalize()` |
| `utils/feature_extractor.py` | Biomechanical feature computation | `extract_features()` |
| `utils/medical_engine.py` | Clinical report generation | `generate_report()`, `_compute_risk_score()`, `_risk_category()`, `MedicalReport` |
| `models/pressure_dl_model.pth` | Trained SE-ResNet weights | ~64 MB, PyTorch state_dict |
| `models/pressure_model.pkl` | Trained Random Forest | ~7 MB, joblib-serialized sklearn model |
| `Pressure_Data/` | Training data | 1,440 CSV files (128×48 each) |

---

## ⚠️ Medical Disclaimer

This system is developed for **educational and research purposes only**. All clinical insights, risk scores, recommendations, and specialist referrals generated by this system are AI-based informational outputs and **do not constitute medical advice**. Always consult a qualified healthcare professional for diagnosis, treatment, or any medical decisions.

---

*Documentation generated for GaitScan AI — AI-Based Gait Injury Prediction System*  
*12 Subjects | 6 Gait Classes | 1,440 Pressure Samples | SE-ResNet Deep Learning*
