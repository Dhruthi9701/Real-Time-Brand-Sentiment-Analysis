import os, io, json, re
from tqdm import tqdm
import zstandard as zstd

# ---------- CONFIG ----------
DOWNLOAD_DIR = r"c:\Users\Hp\OneDrive\Desktop\Project\Brand-Sentiment-Analysis\reddit"          # folder with your .zst files
OUTPUT_FILE = "hermes_hits.ndjson"

# Post 2020 Setptember 7 
AFTER_UNIX = 1599436800

# If you only want to process some sub files, put their substrings here:
PROCESS_ONLY = ("handbags", "thehermesgame", "luxury", "femalefashionadvice")

# ---------- KEYWORDS (tiered) ----------
HERMES_KEYWORDS = {
    "bags": [
        r"\bbirkin\b", r"\bkelly\b", r"\bconstance\b", r"\bevelyne\b",
        r"\bpicotin\b", r"\blindy\b", r"\bbolide\b", r"\bherbag\b",
        r"\bgarden party\b", r"\broulis\b", r"\bhalzan\b", r"\bjypsi[eè]re\b",
        r"\bverrou\b", r"\b24/24\b", r"\bgeta\b"
    ],
    "slgs": [
        r"\bbearn\b", r"\bkelly wallet\b", r"\bconstance wallet\b",
        r"\bsilk.?in wallet\b", r"\bcalvi\b", r"\bbastia\b",
        r"\brodeo charm\b", r"\bcarmen\b", r"\bulysse\b"
    ],
    "shoes": [
        r"\boran\b", r"\boasis\b", r"\bchypre\b", r"\bizmir\b",
        r"\blegend\b", r"\bparis loafers\b", r"\bjumping boots\b"
    ],
    "scarves": [
        r"\bcarr[eé]\b", r"\btwilly\b", r"\bgavroche\b", r"\bmaxi.?twilly\b",
        r"\bcashmere shawl\b", r"\bstole\b"
    ],
    "jewelry": [
        r"\bclic(h| clac)\b", r"\bcollier de chien\b", r"\bkelly bracelet\b",
        r"\bcha[iî]ne d[’']ancre\b", r"\brivale\b"
    ],
    "watches": [
        r"\bheure h\b", r"\bcape cod\b", r"\barceau\b",
        r"\bnantucket\b", r"\bslim d[’']hermes\b"
    ],
    "fragrance": [
        r"\bterre d[’']hermes\b", r"\beau des merveilles\b",
        r"\btwilly d[’']hermes\b", r"\bjour d[’']hermes\b",
        r"\bun jardin sur le nil\b", r"\bun jardin\b", r"\brouge hermes\b"
    ],
    "home": [
        r"\bavalon blanket\b", r"\bavalon pillow\b", r"\bmosaique\b",
        r"\bpaddock\b", r"\bsellier\b", r"\bporcelain\b"
    ],
    "general": [
        r"herm[eè]s", r"\bhermes\b"
    ]
}

# Compile regex per category
CATEGORY_RES = {cat: re.compile("|".join(pats), flags=re.IGNORECASE | re.UNICODE)
                for cat, pats in HERMES_KEYWORDS.items()}

# Exclude tech/crypto/etc.
EXCLUDE_RE = re.compile(r"\b(protocol|crypto|wallet|graphql|sdk|rust|broker|service)\b",
                        flags=re.IGNORECASE)


# ---------- FUNCTIONS ----------
def extract_text(obj):
    """Combine title/selftext/body into one string"""
    parts = []
    for k in ("title", "selftext", "body"):
        v = obj.get(k)
        if v:
            parts.append(str(v))
    return " ".join(parts).strip()


def match_category(text):
    """Return first matching category for given text"""
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

