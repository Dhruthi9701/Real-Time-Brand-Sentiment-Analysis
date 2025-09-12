# build_feature_matrix.py
import os, glob, joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy import sparse

PROCESSED_DIR = './processed'
OUT_DIR = './features'
os.makedirs(OUT_DIR, exist_ok=True)

# choose files to include
files = glob.glob(os.path.join(PROCESSED_DIR, '*.processed.csv'))
print("Files:", files)


# read and concat, and assign brand from filename
dfs = []
for f in files:
    df = pd.read_csv(f)
    df['source_file'] = os.path.basename(f)
    # Assign brand based on filename
    fname = os.path.basename(f).lower()
    if 'chanel' in fname:
        df['brand'] = 'Chanel'
    elif 'gucci' in fname:
        df['brand'] = 'Gucci'
    elif 'hermes' in fname:
        df['brand'] = 'Hermes'
    else:
        df['brand'] = 'Unknown'
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
print("Total rows:", len(df))

# minimal cleaning
df['text_clean'] = df['text_clean'].fillna('').astype(str)
numeric_cols = ['exclam_count','question_count','ellipsis_count','uppercase_ratio','elongated_count','has_hashtag','has_mention','quote_flag','emoji_count']
for c in numeric_cols:
    if c not in df.columns:
        df[c] = 0
df[numeric_cols] = df[numeric_cols].fillna(0)

# TF-IDF (fit on combined dataset)
tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_features=50000)
X_tfidf = tfidf.fit_transform(df['text_clean'])
print("TF-IDF shape:", X_tfidf.shape)

# numeric scaler
scaler = StandardScaler()
X_num = scaler.fit_transform(df[numeric_cols])

# combine sparse + dense
X_num_sparse = sparse.csr_matrix(X_num)
X = sparse.hstack([X_tfidf, X_num_sparse], format='csr')
print("Combined X shape:", X.shape)

# save artifacts & matrix
joblib.dump(tfidf, os.path.join(OUT_DIR, 'tfidf.joblib'))
joblib.dump(scaler, os.path.join(OUT_DIR, 'scaler.joblib'))
sparse.save_npz(os.path.join(OUT_DIR, 'X_all.npz'), X)


# Save meta.csv with all relevant columns, including date/time if present
meta_cols = ['text_raw', 'text_clean', 'brand']
date_col = None
for cand in ['date', 'created_utc', 'timestamp', 'time', 'datetime']:
    if cand in df.columns:
        date_col = cand
        break
if date_col:
    meta_cols.append(date_col)
else:
    # Add a dummy date column if none found
    df['date'] = pd.Timestamp.today().normalize()
    meta_cols.append('date')
if 'weak_sarcasm' in df.columns:
    meta_cols.append('weak_sarcasm')
if 'product' in df.columns:
    meta_cols.append('product')
if 'source_file' in df.columns:
    meta_cols.append('source_file')
df[meta_cols].to_csv(os.path.join(OUT_DIR, 'meta.csv'), index=False)

print("Saved feature matrix and artifacts to", OUT_DIR)
