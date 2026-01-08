import os, io, json, re
from tqdm import tqdm
import zstandard as zstd

DOWNLOAD_DIR = r"c:\Users\Hp\OneDrive\Desktop\Project\Brand-Sentiment-Analysis\reddit"
OUTPUT_FILE = "gucci_matches.ndjson"

AFTER_UNIX = 1599436800

PROCESS_ONLY = ("fashionreps", "fragranceswap", "handbags", "luxury", "ogrepladies")

GUCCI_KEYWORDS = {
    "bags": [
        r"\bdionysus\b", r"\bjackie\b", r"\bmarmont\b",
        r"\bsoho disco\b", r"\bhorsetbit\b", r"\bophidia\b",
        r"\bsylvie\b", r"\bgg supreme\b", r"\bgucci tote\b"
    ],
    "shoes": [
        r"\bace sneakers?\b", r"\brython\b", r"\bscreener\b",
        r"\bprincetown\b", r"\bbrixton loafers?\b",
        r"\bhorsebit loafer\b", r"\btennis 1977\b"
    ],
    "rtw": [
        r"\bflora print\b", r"\bgg monogram\b", r"\bgucci suit\b",
        r"\bgucci hoodie\b", r"\bgucci jacket\b"
    ],
    "fragrance": [
        r"\bbloom\b", r"\bguilty\b", r"\bflora\b",
        r"\bmem[oó]ire d[’']une odeur\b", r"\bthe alchemist'?s garden\b",
        r"\benvy\b", r"\brush\b", r"\bpour homme\b"
    ],
    "jewelry_watches": [
        r"\bgg running\b", r"\binterlocking g\b", r"\bgucci link\b",
        r"\bgucci dive\b", r"\bgucci g-timeless\b"
    ],
    "general": [
        r"\bgucci\b"
    ]
}

CATEGORY_RES = {cat: re.compile("|".join(pats), flags=re.IGNORECASE | re.UNICODE)
                for cat, pats in GUCCI_KEYWORDS.items()}

EXCLUDE_RE = re.compile(r"\b(protocol|crypto|sdk|rust|broker|game|rapper)\b", flags=re.IGNORECASE)


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
