# process_ndjson_and_features.py
import os, json, re
import pandas as pd
from tqdm import tqdm

INPUT_DIR = '.'          # change if NDJSON in another folder
OUT_DIR = './processed'  # output folder for processed CSVs
SAMPLE_LIMIT = None      # set to integer to test (e.g., 5000)

os.makedirs(OUT_DIR, exist_ok=True)

# --- feature helpers ---
def count_exclam(s): return str(s).count('!')
def count_question(s): return str(s).count('?')
def count_ellipsis(s): return str(s).count('...')
def uppercase_ratio(s):
    s = str(s)
    total = max(1, len(s))
    return sum(1 for ch in s if ch.isupper()) / total
def elongated_words_count(s):
    return len(re.findall(r'([a-zA-Z])\1{2,}', str(s)))
def has_hashtag(s): return int(bool(re.search(r'#\w+', str(s))))
def has_mention(s): return int(bool(re.search(r'@\w+', str(s))))
def quote_count(s): return int(bool(re.search(r'["“”\']', str(s))))
def emoji_count(s):
    return len(re.findall(r'[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U0001F1E0-\U0001F1FF]', str(s)))

# weak sarcasm heuristic patterns
sarcasm_patterns = [
    r'#sarcasm', r'#sarcastic',
    r'\byeah,? right\b', r'\bas if\b',
    r'\b(oh|ohh|aw) (great|super|fantastic)\b',
    r'\bbest day ever\b', r'\blove( this)? (traffic|waiting|being)\b'
]
sarcasm_regex = re.compile('|'.join(sarcasm_patterns), flags=re.IGNORECASE)

def weak_sarcasm_label(text):
    t = str(text)
    if sarcasm_regex.search(t):
        return 1
    if count_exclam(t) + count_question(t) >= 3 and ('!' in t and '?' in t):
        return 1
    if re.search(r'\b(that was|yeah right|as if|i love|sure,|totally)\b', t, flags=re.IGNORECASE):
        return 1
    return 0

def make_clean(text):
    t = str(text)
    t = re.sub(r'http\S+|www\.\S+', ' ', t)   # remove urls
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def guess_text_col(record):
    for candidate in ('text','body','comment','content','post','title'):
        if candidate in record:
            return candidate
    # fallback: first string field
    for k,v in record.items():
        if isinstance(v, str):
            return k
    return None

# ---- process files ----
for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith('.ndjson'):
        continue
    inpath = os.path.join(INPUT_DIR, fname)
    rows_out = []
    print("Processing", inpath)
    with open(inpath, 'r', encoding='utf-8') as fh:
        for i, line in enumerate(tqdm(fh, desc=fname)):
            if SAMPLE_LIMIT and i >= SAMPLE_LIMIT:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            text_col = guess_text_col(rec)
            raw = str(rec.get(text_col, '')) if text_col else ''
            clean = make_clean(raw)
            out = {
                'text_raw': raw,
                'text_clean': clean,
                'exclam_count': count_exclam(raw),
                'question_count': count_question(raw),
                'ellipsis_count': count_ellipsis(raw),
                'uppercase_ratio': uppercase_ratio(raw),
                'elongated_count': elongated_words_count(raw),
                'has_hashtag': has_hashtag(raw),
                'has_mention': has_mention(raw),
                'quote_flag': quote_count(raw),
                'emoji_count': emoji_count(raw),
            }
            out['weak_sarcasm'] = weak_sarcasm_label(raw)
            # copy a few common metadata items if present
            for k in ('title','created_utc','subreddit','score'):
                if k in rec:
                    out[k] = rec[k]
            rows_out.append(out)
    df = pd.DataFrame(rows_out)
    outname = os.path.splitext(fname)[0] + '.processed.csv'
    outpath = os.path.join(OUT_DIR, outname)
    df.to_csv(outpath, index=False)
    print(f"Saved {len(df)} rows to {outpath}")
    print("Weak-sarcasm count:", int(df['weak_sarcasm'].sum()))
