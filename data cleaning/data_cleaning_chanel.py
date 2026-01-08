import pandas as pd
import re
from tqdm import tqdm

tqdm.pandas(desc="Cleaning", mininterval=0.1, ascii=True)

cols_needed = ["text", "subreddit", "score", "category", "title","created_utc"]
df = pd.read_json(r"E:\Brand sentiment analysis\Brand-Sentiment-Analysis\chanel_matches.ndjson", lines=True)
df = df[[col for col in cols_needed if col in df.columns]].copy()


emoji_pattern = re.compile(
    "[" 
    "\U0001F600-\U0001F64F" 
    "\U0001F300-\U0001F5FF" 
    "\U0001F680-\U0001F6FF" 
    "\U0001F1E0-\U0001F1FF" 
    "\U00002700-\U000027BF"  
    "\U000024C2-\U0001F251"
    "]+", 
    flags=re.UNICODE
)

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+|https\S+", " ", text) 
    text = re.sub(r"@\w+", " ", text)  
    text = re.sub(r"#\w+", " ", text) 
    text = re.sub(r"[^\w\s]", " ", text)  
    text = re.sub(r"\d+", " ", text) 
    text = emoji_pattern.sub(r'', text)  
    text = re.sub(r"\s+", " ", text).strip()  
    return text

if "text" in df.columns:
    df["text"] = df["text"].progress_apply(clean_text)
if "title" in df.columns:
    df["title"] = df["title"].progress_apply(clean_text)

df.drop_duplicates(subset=["text", "title"], inplace=True)

output_cols = [col for col in ["text", "subreddit", "score", "category", "title","created_utc"] if col in df.columns]
df.to_csv("chanel_cleaned.csv", columns=output_cols, index=False)

print("✅ Cleaning complete! Saved as chanel_cleaned.csv")
