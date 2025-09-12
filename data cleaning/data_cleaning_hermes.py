import pandas as pd
import re
from tqdm import tqdm

tqdm.pandas(desc="Cleaning", mininterval=0.1, ascii=True)

# ----------- STEP 1: Load Data ------------
cols_needed = ["text", "subreddit", "score", "category", "title","created_utc"]
df = pd.read_json(r"E:\Brand sentiment analysis\Brand-Sentiment-Analysis\hermes_hits.ndjson", lines=True)
df = df[[col for col in cols_needed if col in df.columns]].copy()

# ----------- STEP 2: Clean Functions ------------
emoji_pattern = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed chars
    "]+", 
    flags=re.UNICODE
)

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+|https\S+", " ", text)  # remove urls
    text = re.sub(r"@\w+", " ", text)  # remove mentions
    text = re.sub(r"#\w+", " ", text)  # remove hashtags
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"\d+", " ", text)  # remove numbers
    text = emoji_pattern.sub(r'', text)  # remove emojis
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text

# ----------- STEP 3: Apply Cleaning ------------
if "text" in df.columns:
    df["text"] = df["text"].progress_apply(clean_text)
if "title" in df.columns:
    df["title"] = df["title"].progress_apply(clean_text)

# Drop duplicates
df.drop_duplicates(subset=["text", "title"], inplace=True)

# ----------- STEP 4: Save ------------
output_cols = [col for col in ["text", "subreddit", "score", "category", "title","created_utc"] if col in df.columns]
df.to_csv("hermes_cleaned.csv", columns=output_cols, index=False)

print("✅ Cleaning complete! Saved as hermes_cleaned.csv")
