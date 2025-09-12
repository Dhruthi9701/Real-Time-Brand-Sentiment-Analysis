import os, re
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from wordcloud import WordCloud

# ---------- CONFIG ----------

CSV_FILE = "gucci_cleaned.csv"   # change this for other brands
OUTPUT_DIR = "E:\Brand sentiment analysis\Brand-Sentiment-Analysis\eda_outputs_gucci"
MIN_WORD_DF = 5
MIN_BIGRAM_DF = 3
TOP_N = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- helpers ----------
def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = re.sub(r"http\S+", " ", s)            # URLs
    s = re.sub(r"u\/\w+", " ", s)             # usernames
    s = re.sub(r"r\/\w+", " ", s)             # subreddit refs
    s = re.sub(r"[^A-Za-z0-9\s']", " ", s)    # punctuation
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

# stopwords
try:
    import nltk
    nltk.data.find("corpora/stopwords")
    from nltk.corpus import stopwords as nltk_stopwords
    STOPWORDS = set(nltk_stopwords.words("english"))
except Exception:
    STOPWORDS = set(ENGLISH_STOP_WORDS)

df = pd.read_csv(CSV_FILE)

# ---------- load ----------
print("Loading CSV:", CSV_FILE)
print("Rows loaded:", len(df))

# ensure created_utc → datetime
if "created_utc" in df.columns:
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    df["date"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df = df.dropna(subset=["date"])
else:
    raise ValueError("❌ created_utc column missing from CSV")

# ensure score is numeric
df["score"] = pd.to_numeric(df.get("score", 0), errors="coerce").fillna(0).astype(int)

# drop empty texts
df["text"] = df["text"].astype(str)
df = df[df["text"].str.strip() != ""]

# derived fields
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.to_period("M")


print("Prepared DataFrame with columns:", df.columns.tolist())

# ---------- BASIC COUNTS ----------
print("\n=== BASIC COUNTS ===")
print("Total mentions:", len(df))
print("Unique subreddits:", df["subreddit"].nunique())


# Monthly Usage
ts = df.groupby("month").size().sort_index()
fig_mentions, ax_mentions = plt.subplots(figsize=(10,4))
ax_mentions.plot(ts.index.astype(str), ts.values, marker="o")
plt.xticks(rotation=45, ha="right")
plt.title("Mentions per month")
plt.tight_layout()

# ---------- TEXT ANALYSIS ----------
print("\n=== TEXT ANALYSIS ===")
df["clean_text"] = df["text"].apply(clean_text)


# Top Words used
vec = CountVectorizer(stop_words=list(STOPWORDS), min_df=MIN_WORD_DF, token_pattern=r"(?u)\b\w+\b")
X = vec.fit_transform(df["clean_text"])
word_freq = dict(zip(vec.get_feature_names_out(), X.sum(axis=0).A1))
top_words = Counter(word_freq).most_common(TOP_N)
df_top_words = pd.DataFrame(top_words, columns=["word","count"])

# Word Charts
words, counts = zip(*top_words)
fig_words, ax_words = plt.subplots(figsize=(8,6))
ax_words.barh(list(words)[::-1], list(counts)[::-1])
plt.title("Top Words")
plt.tight_layout()



# Bigram Charts
vec2 = CountVectorizer(stop_words=list(STOPWORDS), ngram_range=(2,2), min_df=MIN_BIGRAM_DF)
X2 = vec2.fit_transform(df["clean_text"])
bigram_freq = dict(zip(vec2.get_feature_names_out(), X2.sum(axis=0).A1))
top_bigrams = Counter(bigram_freq).most_common(TOP_N)
df_top_bigrams = pd.DataFrame(top_bigrams, columns=["bigram","count"])

# Bigram bar-chart
bigrams, bcounts = zip(*top_bigrams)
fig_bigrams, ax_bigrams = plt.subplots(figsize=(8,6))
ax_bigrams.barh(list(bigrams)[::-1], list(bcounts)[::-1])
plt.title("Top Bigrams")
plt.tight_layout()



# ---------- SENTIMENT ----------
print("\n=== SENTIMENT ===")
analyzer = SentimentIntensityAnalyzer()
tqdm.pandas()

def vader_row(s):
    try: return analyzer.polarity_scores(s)
    except: return {"neg":0.0,"neu":1.0,"pos":0.0,"compound":0.0}

v = df["clean_text"].progress_apply(vader_row)
v_df = pd.DataFrame(list(v))
df = pd.concat([df.reset_index(drop=True), v_df.reset_index(drop=True)], axis=1)

def label_from_compound(c):
    if c >= 0.05: return "positive"
    if c <= -0.05: return "negative"
    return "neutral"

df["sentiment_label"] = df["compound"].apply(label_from_compound)
df["weighted_compound"] = df["compound"] * np.log1p(df["score"].abs())

sent_counts = df["sentiment_label"].value_counts()
fig_sentiment, ax_sentiment = plt.subplots()
sent_counts.plot.bar(ax=ax_sentiment, color=["green","red","gray"])
plt.title("Sentiment distribution (VADER)")
plt.tight_layout()


# ---------- CATEGORY ANALYSIS ----------
cat_sent = None
fig_cat_sent = None
if "category" in df.columns:
    cat_sent = df.groupby(["category","sentiment_label"]).size().unstack(fill_value=0)
    fig_cat_sent, ax_cat_sent = plt.subplots(figsize=(8,6))
    cat_sent.plot(kind="bar", stacked=True, ax=ax_cat_sent, color={"positive":"green","negative":"red","neutral":"gray"})
    plt.title("Sentiment by Category")
    plt.ylabel("Count")
    plt.tight_layout()



# ---------- SAVE OUTPUTS ----------
# Save CSV outputs

# Save CSV outputs with 'gucci_' prefix
df_top_words.to_csv(os.path.join(OUTPUT_DIR, "gucci_top_words.csv"), index=False)
df_top_bigrams.to_csv(os.path.join(OUTPUT_DIR, "gucci_top_bigrams.csv"), index=False)
sent_counts.to_csv(os.path.join(OUTPUT_DIR, "gucci_sentiment_counts.csv"))
ts.to_csv(os.path.join(OUTPUT_DIR, "gucci_mentions_by_month.csv"))
if cat_sent is not None:
    cat_sent.to_csv(os.path.join(OUTPUT_DIR, "gucci_category_sentiment.csv"))

# Save figures

# Save figures with 'gucci_' prefix
fig_mentions.savefig(os.path.join(OUTPUT_DIR, "gucci_mentions_by_month.png"))
fig_words.savefig(os.path.join(OUTPUT_DIR, "gucci_top_words.png"))
fig_bigrams.savefig(os.path.join(OUTPUT_DIR, "gucci_top_bigrams.png"))
fig_sentiment.savefig(os.path.join(OUTPUT_DIR, "gucci_sentiment_distribution.png"))
if fig_cat_sent is not None:
    fig_cat_sent.savefig(os.path.join(OUTPUT_DIR, "gucci_category_sentiment.png"))

print("\n✅ ALL DONE. Outputs saved to:", OUTPUT_DIR)


