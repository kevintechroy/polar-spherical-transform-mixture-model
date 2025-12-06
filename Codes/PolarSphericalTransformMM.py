import time
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
# from PSTMM import PolarSphericalClassifier
from PSTMM_v2 import PolarSphericalClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================================
#  HELPER: PREPROCESSING & RUNNER
# ============================================================================

def find_dataset_folder(start_dir):
    cur = start_dir
    while True:
        candidate = os.path.join(cur, "Datasets")
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent

def run_pipeline(dataset_name, dataset_path, target_col=None, test_size=0.5, 
                        scaling_factor=10.0, random_state=42):
    print(f"\n[INFO] Loading dataset: {dataset_path}")
    try:
        data = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {dataset_path}")
        return

    # --- 1. CLEANING & ENCODING ---
    # Separate X (features) and y (target)
    if target_col:
        X_raw = data.drop(columns=[target_col])
        y_raw = data[target_col]
    else:
        X_raw = data.iloc[:, :-1]
        y_raw = data.iloc[:, -1]

    # Impute Missing Values (Numerical -> Mean, Categorical -> Mode)
    # First split columns by type
    num_cols = X_raw.select_dtypes(include=[np.number]).columns
    cat_cols = X_raw.select_dtypes(exclude=[np.number]).columns

    # Fill numeric missing values
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy='mean')
        X_raw[num_cols] = imputer_num.fit_transform(X_raw[num_cols])

    # Fill categorical missing values & Label Encode
    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X_raw[cat_cols] = imputer_cat.fit_transform(X_raw[cat_cols])
        
        # Encode text to numbers
        le = LabelEncoder()
        for col in cat_cols:
            print(f"       Encoding column '{col}'...")
            X_raw[col] = le.fit_transform(X_raw[col].astype(str))

    # Convert X to float for the model
    X = X_raw.values.astype(float)
    
    # Process Target Y (Label Encode if it's text like 'ckd', 'notckd')
    y_le = LabelEncoder()
    y = y_le.fit_transform(y_raw.astype(str))
    
    num_classes = len(np.unique(y))
    print(f"\n       Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"       Classes: {num_classes} {y_le.classes_}")

    # --- 2. TRAIN & EVALUATE ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"\n[INFO] Training Polar-Spherical Model (Scaling Factor={scaling_factor})...")
    model = PolarSphericalClassifier(
        n_components=num_classes,
        scaling_factor=scaling_factor,
        covariance_type="diag",
        max_iter=200,
        n_init=2
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start
    print(f"\n[INFO] Training finished in {duration:.2f} seconds.")

    cluster_labels = model.predict(X_test)
    
    # Map clusters to actual labels
    label_mapping = {}
    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        if mask.sum() > 0:
            vals, counts = np.unique(y_test[mask], return_counts=True)
            label_mapping[c] = vals[np.argmax(counts)]

    y_pred = np.array([label_mapping.get(c, -1) for c in cluster_labels])

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n" + "="*40)
    print(f" Model Evaluation Metrics:")
    print("="*40)
    print("Accuracy :", 100 * accuracy)
    print("Precision:", 100 * precision)
    print("Recall   :", 100 * recall)
    print("F1 Score :", 100 * f1, "\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 4))
    plt.title(f"Confusion Matrix (Scaling={scaling_factor})\nAcc: {accuracy*100:.2f}%")    
    plt.title("Confusion Matrix (Polar-Spherical Transform Mixture Model)", fontsize=16, pad=30)
    plt.text(
        0.5, 1.05,
        f"Accuracy on {dataset_name}: {accuracy*100:.2f}, with Scaling Factor = {scaling_factor}",
        ha="center",
        fontsize=12,
        color="gray",
        transform=plt.gca().transAxes,
    )
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=y_le.classes_, yticklabels=y_le.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.show()

# ============================================================================
#  MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = find_dataset_folder(current_script_dir)
    
    print("\n" + "-"*50)
    print(f"Detected Datasets Directory: {datasets_dir}")
    print("-"*50)
    
    dataset_name_input = (input("Enter Dataset Name (e.g. Iris, Wine, Insurance): ").strip()).title()
    
    if not dataset_name_input.lower().endswith(".csv"):
        # Try adding " Dataset" if user typed just "Iris"
        potential_name = dataset_name_input + " Dataset.csv"
        potential_path = os.path.join(datasets_dir, potential_name)
        
        if os.path.exists(potential_path):
            dataset_path = potential_path
        else:
            # Try just adding .csv
            dataset_path = os.path.join(datasets_dir, dataset_name_input + ".csv")
    else:
        dataset_path = os.path.join(datasets_dir, dataset_name_input)
        
    dataset_path = os.path.normpath(dataset_path)
    
    # Run Pipeline
    run_pipeline(dataset_name_input, dataset_path, scaling_factor=10.0)
