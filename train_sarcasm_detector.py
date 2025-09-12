# train_sarcasm_detector.py
import os, joblib
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns

FEATURE_DIR = './features'
OUT_DIR = './models'
os.makedirs(OUT_DIR, exist_ok=True)


print("[INFO] Loading feature matrix and metadata...")
X = sparse.load_npz(os.path.join(FEATURE_DIR, 'X_all.npz'))
meta = pd.read_csv(os.path.join(FEATURE_DIR, 'meta.csv'))
print(f"[INFO] Feature matrix shape: {X.shape}, Meta shape: {meta.shape}")
 # If you kept source file ordering consistent, meta aligns with X rows


print("[INFO] Creating target labels for sarcasm detection...")
target_col = 'sarcasm_label'  # if you annotated manually
if target_col in meta.columns:
    y = meta[target_col].astype(int)
else:
    y = meta['weak_sarcasm'].astype(int)

class_dist = pd.Series(y).value_counts().sort_index()
print("Class counts:\n", class_dist)

# Plot class distribution
plt.figure(figsize=(6,4))
class_dist.plot(kind='bar', color=['gray','red'])
plt.title('Sarcasm Class Distribution (0=Not Sarcastic, 1=Sarcastic)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'sarcasm_class_distribution.png'))
plt.close()
print(f"[INFO] Saved sarcasm class distribution plot to {OUT_DIR}/sarcasm_class_distribution.png")


print("[INFO] Training and cross-validating LinearSVC sarcasm model...")
clf = LinearSVC(max_iter=10000, class_weight='balanced')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
preds = cross_val_predict(clf, X, y, cv=cv, method='predict')
print("[INFO] Classification report:")
print(classification_report(y, preds, digits=4))

# Confusion matrix plot
cm = confusion_matrix(y, preds, labels=sorted(class_dist.index))
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(class_dist.index), yticklabels=sorted(class_dist.index))
plt.title('Sarcasm Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'sarcasm_confusion_matrix.png'))
plt.close()
print(f"[INFO] Saved confusion matrix plot to {OUT_DIR}/sarcasm_confusion_matrix.png")

print("[INFO] Fitting model on all data and saving...")
clf.fit(X, y)
joblib.dump(clf, os.path.join(OUT_DIR, 'sarcasm_linsvc.joblib'))
print("[INFO] Saved sarcasm model to", OUT_DIR)
