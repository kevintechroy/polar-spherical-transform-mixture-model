import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns

def run_kmeans_pipeline(dataset_name, dataset_path, target_col=None, test_size=0.5, random_state=42):
    print(f"\n[INFO] Loading dataset: {dataset_path}")
    try:
        data = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {dataset_path}")
        return

    # --- CLEANING & ENCODING ---
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

    # --- SPLIT & SCALE ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- K-MEANS ---
    print(f"[INFO] Training K-Means (K={num_classes})...")
    model = KMeans(n_clusters=num_classes, random_state=random_state, n_init="auto")
    model.fit(X_train)
    cluster_labels = model.predict(X_test)

    # --- MAPPING ---
    def map_clusters_to_labels(y_true, y_clusters):
        clusters = np.unique(y_clusters)
        classes = np.unique(y_true)
        cost = np.zeros((len(clusters), len(classes)), dtype=int)
        for i, c in enumerate(clusters):
            for j, cls in enumerate(classes):
                mask = (y_clusters == c)
                cost[i, j] = np.sum(y_true[mask] != cls)
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = {}
        for r, c in zip(row_ind, col_ind):
             if r < len(clusters) and c < len(classes):
                mapping[clusters[r]] = classes[c]
        return mapping

    label_mapping = map_clusters_to_labels(y_test, cluster_labels)
    y_pred = np.array([label_mapping.get(c, -1) for c in cluster_labels])

    # --- EVALUATION ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n" + "="*40)
    print(" MODEL RESULTS (K-Means)")
    print("="*40)
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall   : {rec*100:.2f}%")
    print(f"F1 Score : {f1*100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.title(f"Confusion Matrix (K-Means)\nAcc: {acc*100:.2f}% on {dataset_name}")
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
        if parent == project_root: 
            print("[ERROR] Datasets folder not found."); exit()
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
    
    run_kmeans_pipeline(dataset_name_input, os.path.normpath(dataset_path))
