import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils.data_loader import load_dataset

def main():
    data_dir = 'Pressure_Data'
    model_save_path = 'models/pressure_model.pkl'
    
    # 1. Load dataset
    print(f"Loading dataset from {data_dir}...")
    df = load_dataset(data_dir)
    
    if df.empty:
        print("Failed to load dataset. Make sure the data directory contains well-formatted CSVs.")
        return
        
    print(f"Dataset loaded. Total shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # 2. Prepare Features (X) and Labels (y)
    X = df.drop(columns=['label'])
    y = df['label']
    
    # 3. Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Train Model
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate Model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # 6. Save Model
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump(model, model_save_path)
    print(f"\nModel saved successfully to {model_save_path}")

if __name__ == '__main__':
    main()
