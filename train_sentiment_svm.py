# train_sentiment_svm.py
import os, joblib, numpy as np, pandas as pd
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from utils_lexicon import vader_scores
import matplotlib.pyplot as plt
import seaborn as sns

FEATURE_DIR = './features'
MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)


print("[INFO] Loading feature matrix and metadata...")
X_all = sparse.load_npz(os.path.join(FEATURE_DIR, 'X_all.npz'))
meta = pd.read_csv(os.path.join(FEATURE_DIR, 'meta.csv'))
print(f"[INFO] Feature matrix shape: {X_all.shape}, Meta shape: {meta.shape}")


print("[INFO] Computing VADER compound scores and appending as feature...")
vader_compound = meta['text_clean'].fillna('').astype(str).apply(lambda t: vader_scores(t)['compound'])
vader_compound = vader_compound.fillna(0.0).values.reshape(-1,1)
vader_sparse = sparse.csr_matrix(vader_compound)
X = sparse.hstack([X_all, vader_sparse], format='csr')
print("[INFO] Final X shape:", X.shape)


print("[INFO] Creating sentiment labels...")
if 'sentiment' in meta.columns:
    y = meta['sentiment'].astype(str)
else:
    # No gold sentiment labels found. Here we suggest a weak rule (only for bootstrapping).
    # Use VADER compound thresholds to create 3 classes - REMOVE and replace with human labels ASAP.
    vc = vader_compound.ravel()
    y = pd.cut(vc, bins=[-1.1, -0.05, 0.05, 1.1], labels=['neg','neu','pos']).astype(str)
    print("WARNING: using weak VADER-based labels. Replace with human annotated labels for production.")

class_dist = pd.Series(y).value_counts()
print("Class distribution:\n", class_dist)

# Plot class distribution
plt.figure(figsize=(6,4))
class_dist.plot(kind='bar', color=['red','gray','green'])
plt.title('Sentiment Class Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'sentiment_class_distribution.png'))
plt.close()
print(f"[INFO] Saved sentiment class distribution plot to {MODEL_DIR}/sentiment_class_distribution.png")

clf = LinearSVC(max_iter=10000, class_weight='balanced')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
preds = cross_val_predict(clf, X, y, cv=cv, method='predict')

print("[INFO] Training and cross-validating LinearSVC sentiment model...")
clf = LinearSVC(max_iter=10000, class_weight='balanced')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
preds = cross_val_predict(clf, X, y, cv=cv, method='predict')
print("[INFO] Classification report:")
print(classification_report(y, preds, digits=4))

# Confusion matrix plot
cm = confusion_matrix(y, preds, labels=sorted(class_dist.index))
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(class_dist.index), yticklabels=sorted(class_dist.index))
plt.title('Sentiment Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'sentiment_confusion_matrix.png'))
plt.close()
print(f"[INFO] Saved confusion matrix plot to {MODEL_DIR}/sentiment_confusion_matrix.png")

print("[INFO] Fitting model on all data and saving...")
clf.fit(X, y)
joblib.dump(clf, os.path.join(MODEL_DIR, 'sentiment_linsvc.joblib'))
print("[INFO] Saved sentiment model to", MODEL_DIR)
