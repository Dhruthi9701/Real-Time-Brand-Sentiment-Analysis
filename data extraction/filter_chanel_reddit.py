import os, io, json, re
from tqdm import tqdm
import zstandard as zstd

# ---------- CONFIG ----------
DOWNLOAD_DIR = r"c:\Users\Hp\OneDrive\Desktop\Project\Brand-Sentiment-Analysis\reddit"       # folder with your .zst files
OUTPUT_FILE = "chanel_matches.ndjson"

# Post 2020 Setptember 7 
AFTER_UNIX = 1599436800

# Process only these subs
PROCESS_ONLY = ("chanel", "handbags", "luxury", "femalefashionadvice", "fragranceswap")

# ---------- KEYWORDS (tiered for Chanel) ----------
CHANEL_KEYWORDS = {
    "bags": [
        r"\bclassic flap\b", r"\b2\.55\b", r"\bgabrielle\b", r"\bboy bag\b",
        r"\bcoco handle\b", r"\bwallet on chain\b", r"\bwoc\b", r"\bchanel tote\b",
        r"\b19 bag\b", r"\bdrawstring backpack\b"
    ],
    "shoes": [
        r"\btwo-tone\b", r"\bballerina flats?\b", r"\bloafers?\b",
        r"\bchanel sneakers?\b", r"\bespadrilles?\b", r"\bslides?\b", r"\bboots?\b"
    ],
    "rtw": [
        r"\btweed jacket\b", r"\btweed suit\b", r"\btweed dress\b",
        r"\bchanel blouse\b", r"\bchanel skirt\b"
    ],
    "fragrance_beauty": [
        r"\bno\.?\s*5\b", r"\bcoco mademoiselle\b", r"\bgabrielle essence\b",
        r"\ballure\b", r"\bchance\b", r"\bbleu de chanel\b", r"\bego[iî]ste\b",
        r"\brouge coco\b", r"\brouge allure\b", r"\bles exclusifs\b"
    ],
    "jewelry_watches": [
        r"\bcoco crush\b", r"\bcamellia\b", r"\bj12\b", r"\bpremière watch\b"
    ],
    "general": [
        r"\bchanel\b"
    ]
}

# Compile regex per category
CATEGORY_RES = {cat: re.compile("|".join(pats), flags=re.IGNORECASE | re.UNICODE)
                for cat, pats in CHANEL_KEYWORDS.items()}

# Exclude noise (finance, crypto, etc.)
EXCLUDE_RE = re.compile(r"\b(stocks?|crypto|broker|service|perfume dupes?)\b", flags=re.IGNORECASE)


# ---------- FUNCTIONS ----------
def extract_text(obj):
    parts = []
    for k in ("title", "selftext", "body"):
        v = obj.get(k)
        if v:
            parts.append(str(v))
    return " ".join(parts).strip()


def match_category(text):
    for cat, regex in CATEGORY_RES.items():
        if regex.search(text):
            return cat
    return None


def process_file(path, out_fh):
    count, matches = 0, 0
    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            for line in io.TextIOWrapper(reader, encoding="utf-8", errors="replace"):
                count += 1
                try:
                    obj = json.loads(line)
                except:
                    continue
                created = obj.get("created_utc") or obj.get("created")
                try:
                    created = int(created)
                except:
                    continue
                if created < AFTER_UNIX:
                    continue
                text = extract_text(obj)
                if not text:
                    continue
                if EXCLUDE_RE.search(text):
                    continue
                cat = match_category(text)
                if cat:
                    out_obj = {
                        "id": obj.get("id"),
                        "created_utc": created,
                        "subreddit": obj.get("subreddit"),
                        "author": obj.get("author"),
                        "score": obj.get("score"),
                        "permalink": obj.get("permalink") or obj.get("url"),
                        "title": obj.get("title"),
                        "text": text,
                        "category": cat,
                        "source_file": os.path.basename(path)
                    }
                    out_fh.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    matches += 1
    return count, matches


def main():
    zst_files = [os.path.join(DOWNLOAD_DIR, f) for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".zst")]
    if PROCESS_ONLY:
        zst_files = [f for f in zst_files if any(tag in os.path.basename(f).lower() for tag in PROCESS_ONLY)]
    if not zst_files:
        print("No files found in", DOWNLOAD_DIR)
        return
    print("Found", len(zst_files), "files:", zst_files)
    total_lines, total_matches = 0, 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_fh:
        for path in tqdm(zst_files):
            c, m = process_file(path, out_fh)
            total_lines += c
            total_matches += m
            print(f"Processed {os.path.basename(path)} → {m} matches")
    print("DONE. Total lines:", total_lines, "Total matches:", total_matches)
    print("Output written to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
