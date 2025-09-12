import os, joblib, pandas as pd, numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from utils_lexicon import vader_scores

FEATURE_DIR = './features'
MODEL_DIR = './models'
OUT_DIR = './analysis'
os.makedirs(OUT_DIR, exist_ok=True)

# --- Efficient unified preprocessing ---
# Load data & model
X = sparse.load_npz(os.path.join(FEATURE_DIR, 'X_all.npz'))
meta = pd.read_csv(os.path.join(FEATURE_DIR, 'meta.csv'))
clf = joblib.load(os.path.join(MODEL_DIR, 'sentiment_linsvc.joblib'))

# Predict sentiment and compute VADER only once
if 'pred_sentiment' not in meta.columns:
    print('[DEBUG] Predicting sentiment for all rows...')
    vader_compound = meta['text_clean'].fillna('').astype(str).apply(lambda t: vader_scores(t)['compound'])
    vader_compound = vader_compound.fillna(0.0).values.reshape(-1,1)
    vader_sparse = sparse.csr_matrix(vader_compound)
    X_full = sparse.hstack([X, vader_sparse], format='csr')
    print(f"[INFO] Input feature matrix shape after VADER: {X_full.shape}")
    y_pred = clf.predict(X_full)
    meta['pred_sentiment'] = y_pred
else:
    # If already present, just stack VADER for downstream use
    vader_compound = meta['text_clean'].fillna('').astype(str).apply(lambda t: vader_scores(t)['compound'])
    vader_compound = vader_compound.fillna(0.0).values.reshape(-1,1)
    vader_sparse = sparse.csr_matrix(vader_compound)
    X_full = sparse.hstack([X, vader_sparse], format='csr')


# Only summarize by brand 
print('[DEBUG] Creating brand-level sentiment summary...')
brands = ['Hermes', 'Gucci', 'Chanel']
brand_summary = meta[meta['brand'].isin(brands)].groupby('brand')['pred_sentiment'].value_counts().unstack(fill_value=0).reset_index()
def get_pct(row, label):
    total = row.get('neg', 0) + row.get('pos', 0) + row.get('neu', 0)
    if total == 0:
        return 0
    return 100 * row.get(label, 0) / total
brand_summary['neg_pct'] = brand_summary.apply(lambda r: get_pct(r, 'neg'), axis=1)
brand_summary['pos_pct'] = brand_summary.apply(lambda r: get_pct(r, 'pos'), axis=1)
brand_summary = brand_summary.sort_values(['brand'])
brand_summary.to_csv(os.path.join(OUT_DIR, 'product_sentiment_summary.csv'), index=False)
print("Saved product sentiment summary (brand-level) ->", os.path.join(OUT_DIR, 'product_sentiment_summary.csv'))

# Plot time series of mentions for Hermes, Gucci, Chanel from eda_outputs_<brand>/mentions_by_month.csv
import matplotlib.pyplot as plt
MODELS_DIR = './models'
os.makedirs(MODELS_DIR, exist_ok=True)

print('[DEBUG] Loading monthly mentions data for all brands...')
hermes = pd.read_csv('eda_outputs_hermes/mentions_by_month.csv')
gucci = pd.read_csv('eda_outputs_gucci/gucci_mentions_by_month.csv')
chanel = pd.read_csv('eda_outputs_chanel/mentions_by_month.csv')

# Standardize column names
hermes.columns = ['month', 'hermes']
gucci.columns = ['month', 'gucci']
chanel.columns = ['month', 'chanel']

# Merge on month
df = pd.merge(hermes, gucci, on='month', how='outer')
df = pd.merge(df, chanel, on='month', how='outer')
df = df.sort_values('month')

# Plot
print('[DEBUG] Plotting mentions time series for all brands...')
plt.figure(figsize=(12,5))
plt.plot(df['month'], df['hermes'], marker='o', label='Hermes', color='#888888')
plt.plot(df['month'], df['gucci'], marker='o', label='Gucci', color='#f28e2b')
plt.plot(df['month'], df['chanel'], marker='o', label='Chanel', color='#4e79a7')
plt.title('Mentions per Month by Brand')
plt.xlabel('Month')
plt.ylabel('Mentions')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
mentions_plot_path = os.path.join(MODELS_DIR, 'brand_mentions_timeseries.png')
plt.savefig(mentions_plot_path)
plt.close()
print(f"[DEBUG] Saved plot -> {mentions_plot_path}")

# Plot positive and negative comment percentages for each brand as bar graphs
print('[DEBUG] Plotting positive and negative comment percentages for each brand...')
MODELS_DIR = './models'
os.makedirs(MODELS_DIR, exist_ok=True)

brands = ['Hermes', 'Gucci', 'Chanel']
# Calculate brand-level sentiment summary
group_col = None
for cand in ('product','brand','source_file'):
    if cand in meta.columns:
        group_col = cand
        break
if group_col is None:
    group_col = 'text_clean' if 'text_clean' in meta.columns else 'text_raw'
meta[group_col] = meta[group_col].astype(str)

summary = meta.groupby(group_col)['pred_sentiment'].value_counts().unstack(fill_value=0)
summary['total'] = summary.sum(axis=1)
if 'pos' in summary.columns and 'neg' in summary.columns:
    summary['neg_pct'] = 100 * summary['neg'] / summary['total']
    summary['pos_pct'] = 100 * summary['pos'] / summary['total']
else:
    cols = list(summary.columns)
    if len(cols) >= 2:
        summary['neg_pct'] = 100 * summary[cols[0]] / summary['total']
        summary['pos_pct'] = 100 * summary[cols[-1]] / summary['total']
    else:
        summary['neg_pct'] = 0
        summary['pos_pct'] = 0

# Only keep rows for Hermes, Gucci, Chanel (case-insensitive match)
summary_brands = summary.loc[[b for b in summary.index if b.lower() in [x.lower() for x in brands]]]

# Plot negative percentage
plt.figure(figsize=(7,5))
plt.bar(summary_brands.index, summary_brands['neg_pct'], color='red')
plt.ylabel('Percentage of Negative Comments')
plt.xlabel('Brand')
plt.title('Percentage of Negative Comments by Brand')
plt.ylim(0, 100)
plt.tight_layout()
neg_plot_path = os.path.join(MODELS_DIR, 'brand_negative_percentage.png')
plt.savefig(neg_plot_path)
plt.close()
print(f"[DEBUG] Saved plot -> {neg_plot_path}")

# Plot positive percentage
plt.figure(figsize=(7,5))
plt.bar(summary_brands.index, summary_brands['pos_pct'], color='green')
plt.ylabel('Percentage of Positive Comments')
plt.xlabel('Brand')
plt.title('Percentage of Positive Comments by Brand')
plt.ylim(0, 100)
plt.tight_layout()
pos_plot_path = os.path.join(MODELS_DIR, 'brand_positive_percentage.png')
plt.savefig(pos_plot_path)
plt.close()
print(f"[DEBUG] Saved plot -> {pos_plot_path}")

print("Analysis complete. Check the ./models folder for brand mentions time series visualization.")
