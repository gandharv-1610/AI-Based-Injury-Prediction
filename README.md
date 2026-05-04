# Plantar Pressure-Based Gait & Injury Detection System

## 👣 Project Overview
This project is an end-to-end Machine Learning AI system designed to analyze plantar pressure data. By reading dynamic CSV grid matrices containing pressure plate foot distributions, the system automatically extracts crucial balance and gait features, trains a Random Forest Classifier, and diagnoses different walking patterns (e.g., normal, antalgic, lurching, steppage, stiff-legged, trendelenburg) alongside predicting potential injury risks.

Additionally, we provide a robust and visual Streamlit web application allowing users to upload a single pressure matrix file and instantly receive pressure heat maps, Center of Pressure (COP) analysis, diagnostic results, and feature metrics.

## 📂 Project Structure
```text
project2/
│
├── Pressure_Data/            # Contains 1,440 gait pressure CSV files
├── models/
│   └── pressure_model.pkl    # Pre-trained Random Forest model
├── utils/
│   ├── feature_extractor.py  # Algorithms for COP and Pressure calculations
│   └── data_loader.py        # Automated directory parsing and dataset structuring
├── app.py                    # Streamlit visual dashboard
├── train.py                  # ML model training execution pipeline
└── requirements.txt          # Python library dependencies
```

## 🧠 Extracted Features
For every 128x48 matrix parsed, the extraction algorithms (`utils/feature_extractor.py`) gather:
1. **Total Pressure:** The total force applied across the entire matrix.
2. **Left/Right Balance Ratio:** Computes total pressure independently mapped to the Left Foot vs. Right Foot grids and establishes a proportional imbalance variable.
3. **Center of Pressure (COP):** Extract spatial weights mapped to `X` and `Y` coordinates showing the functional median pressure node.
4. **Max Pressure:** The highest stress node value recorded.
5. **Variance:** Broadness or tightness of the pressure grid map.

## ⚙️ Model Training
The automated training script (`train.py`):
1. **Reads** all `subjectX_gaitType_trialY_pressure.csv` files automatically.
2. **Assigns Classification Labels** directly based on string patterns mapped in the dataset (e.g., normal, antalgic).
3. **Splits** data for standard `80% train / 20% test` sampling.
4. **Trains a core Random Forest Classifier**. 
5. Outputs metrics (Accuracy, Confusion Matrix) and saves `pressure_model.pkl` serialization straight into the `models/` directory for fast external inference speeds.

## 💻 How to Run Locally

### 1. Prerequisites
Ensure you have `Python 3.8+` installed on your machine. Install required module dependencies using:
```bash
pip install -r requirements.txt
```

### 2. (Optional) Modifying and Retraining the System
If you update `Pressure_Data/` with new test subject files or modify data extraction formulas, you will need to recompile the ML Model. Execute:
```bash
python train.py
```
*(Wait until it reads the dataset and successfully saves to models/pressure_model.pkl)*

### 3. Launch the App Interface
Host the dashboard visualizing tool:
```bash
streamlit run app.py
```

This will automatically open your default web browser tracking to `http://localhost:8501`. 
* Use the sidebar to **Upload a Pressure CSV**.
* Instantly view the Gait Diagnosis, calculated injury risk, mapped colored heat grids, and dynamic charts mapping structural footing!
