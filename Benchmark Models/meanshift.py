import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def run_meanshift_pipeline(dataset_name, dataset_path, target_col=None, test_size=0.5, random_state=42):
    print(f"\n[INFO] Loading dataset: {dataset_path}")
    try:
        data = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {dataset_path}")
        return

    if target_col:
        X_raw = data.drop(columns=[target_col])
        y_raw = data[target_col]
    else:
        X_raw = data.iloc[:, :-1]
        y_raw = data.iloc[:, -1]

    num_cols = X_raw.select_dtypes(include=[np.number]).columns
    cat_cols = X_raw.select_dtypes(exclude=[np.number]).columns

    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy='mean')
        X_raw[num_cols] = imputer_num.fit_transform(X_raw[num_cols])
    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X_raw[cat_cols] = imputer_cat.fit_transform(X_raw[cat_cols])
        le = LabelEncoder()
        for col in cat_cols:
            X_raw[col] = le.fit_transform(X_raw[col].astype(str))

    X = X_raw.values.astype(float)
    y_le = LabelEncoder()
    y = y_le.fit_transform(y_raw.astype(str))
    num_classes = len(np.unique(y))
    print(f"       Classes: {num_classes} {y_le.classes_}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("[INFO] Estimating bandwidth...")
    try:
        bandwidth = estimate_bandwidth(X_train, quantile=0.2, n_samples=min(300, len(X_train)))
    except ValueError:
        bandwidth = 0.0
    if bandwidth <= 0.0001: bandwidth = 1.0
    else: print(f"       Bandwidth: {bandwidth:.4f}")

    print("[INFO] Training Mean Shift...")
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    model.fit(X_train)
    cluster_labels = model.predict(X_test)

    # Majority Vote Mapping
    unique_clusters = np.unique(cluster_labels)
    label_mapping = {}
    for c in unique_clusters:
        idx = np.where(cluster_labels == c)[0]
        if len(idx) == 0: continue
        vals, counts = np.unique(y_test[idx], return_counts=True)
        label_mapping[c] = vals[np.argmax(counts)]
    y_pred = np.array([label_mapping.get(c, -1) for c in cluster_labels])

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n" + "="*40)
    print(" MODEL RESULTS (Mean Shift)")
    print("="*40)
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall   : {rec*100:.2f}%")
    print(f"F1 Score : {f1*100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.title(f"Confusion Matrix (Mean Shift)\nAcc: {acc*100:.2f}% on {dataset_name}")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=y_le.classes_, yticklabels=y_le.classes_)
    plt.xlabel("Predicted Label (Mapped)")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_script_dir
    while not os.path.exists(os.path.join(project_root, "Datasets")):
        parent = os.path.dirname(project_root)
        if parent == project_root: print("[ERROR] Datasets folder not found."); exit()
        project_root = parent
    datasets_dir = os.path.join(project_root, "Datasets")
    
    print("\n" + "-"*50)
    print(f"Detected Datasets Directory: {datasets_dir}")
    print("-"*50)
    dataset_name_input = input("Enter Dataset Name: ").strip()
    if not dataset_name_input.lower().endswith(".csv"):
        potential_name = dataset_name_input + " Dataset.csv"
        if os.path.exists(os.path.join(datasets_dir, potential_name)):
            dataset_path = os.path.join(datasets_dir, potential_name)
        else:
            dataset_path = os.path.join(datasets_dir, dataset_name_input + ".csv")
    else:
        dataset_path = os.path.join(datasets_dir, dataset_name_input)
    
    run_meanshift_pipeline(dataset_name_input, os.path.normpath(dataset_path))
