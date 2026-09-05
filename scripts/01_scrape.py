#!/usr/bin/env python3
"""
Nepal Crisis-Text Scraper
=========================
Collects raw Nepali / Romanized-Nepali / code-mixed disaster posts from:
  1. YouTube comments (official Data API v3)
  2. Reddit posts + comments (PRAW)
  3. A manual CSV of pasted Facebook comments (you copy, script normalizes)

Every record gets provenance (platform, source_url, timestamp, query),
PII stripping (phones, emails, @handles, URLs), a hashed author id, and
text-based dedup. Output: data/raw/corpus.jsonl + corpus.csv

Setup
-----
    pip install google-api-python-client praw pandas python-dotenv

Create a .env file next to this script:
    YOUTUBE_API_KEY=...          # console.cloud.google.com -> YouTube Data API v3
    REDDIT_CLIENT_ID=...         # reddit.com/prefs/apps -> "script" app
    REDDIT_CLIENT_SECRET=...
    REDDIT_USER_AGENT=nepal-crisis-nlp/0.1 by u/yourname

Run
---
    python nepal_crisis_scraper.py --youtube --reddit
    python nepal_crisis_scraper.py --manual data/manual/facebook.csv
    python nepal_crisis_scraper.py --youtube --max-videos 15 --max-comments 400

Manual CSV format (header required): text,source_url,timestamp
timestamp may be blank or any ISO-ish date string.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — edit freely
# ---------------------------------------------------------------------------

QUERIES = [
    # Devanagari
    "बाढी", "पहिरो", "बाढी पहिरो उद्धार", "रसुवा बाढी", "भोटेकोशी बाढी",
    "काठमाडौं बाढी", "राहत उद्धार",
    # Romanized / English
    "Rasuwa flood", "Bhotekoshi flood", "Nepal flood 2026", "Nepal landslide",
    "Kathmandu flood 2024", "Nepal badhi pahiro", "Nepal flood rescue",
    "Melamchi flood", "Nepal monsoon flood",
]

YT_QUERY_SUFFIX = ""          # e.g. " news" to bias toward news channels
YT_PUBLISHED_AFTER = "2024-08-01T00:00:00Z"   # ISO 8601 or None
REDDIT_SUBREDDITS = ["Nepal", "nepal", "kathmandu"]
REDDIT_TIME_FILTER = "all"    # hour, day, week, month, year, all

OUT_DIR = Path("data/raw")
OUT_JSONL = OUT_DIR / "corpus.jsonl"
OUT_CSV = OUT_DIR / "corpus.csv"
SALT = "nepal-crisis-nlp"     # change once; never publish

MIN_CHARS = 8                 # drop tiny comments ("😢", "👍")

# ---------------------------------------------------------------------------
# PII stripping + normalization
# ---------------------------------------------------------------------------

# Nepali mobiles: 98xxxxxxxx / 97xxxxxxxx / 96x, with optional +977 / 977 / 0
# Also landlines like 01-4xxxxxx and generic 7+ digit runs with separators.
PHONE_RE = re.compile(
    r"(?:\+?977[\s\-]?)?(?:0?9[678]\d[\s\-]?\d{3}[\s\-]?\d{4}|0?1[\s\-]?\d{7}|\d[\d\s\-]{7,13}\d)"
)
# Devanagari digit phone numbers (९८...)
DEV_DIGITS = "०१२३४५६७८९"
DEV_PHONE_RE = re.compile(r"[०-९][०-९\s\-]{7,13}[०-९]")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
HANDLE_RE = re.compile(r"(?<!\w)@[\w.]{2,}")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"\s+")


def strip_pii(text: str) -> str:
    text = URL_RE.sub("<URL>", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = HANDLE_RE.sub("<USER>", text)
    text = PHONE_RE.sub("<PHONE>", text)
    text = DEV_PHONE_RE.sub("<PHONE>", text)
    return WS_RE.sub(" ", text).strip()


def dedup_key(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"[^\w\u0900-\u097F]+", "", t)   # keep letters/digits + Devanagari
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


# Common Nepali function words / particles / verb endings, written in Latin
# script. Not exhaustive — just enough to catch Romanized Nepali reliably.
# Matched as whole words, case-insensitive.
ROMANIZED_NEPALI_MARKERS = {
    "ma", "ko", "ki", "ho", "hun", "cha", "xa", "chha", "chhaina", "xaina",
    "bhayo", "bhako", "bhaeko", "garne", "garnu", "garcha", "garxa",
    "garyo", "lai", "haru", "harule", "harulai", "ne", "ta", "ra", "sanga",
    "sangai", "vayo", "vako", "vaeko", "yo", "tyo", "yesto", "testo",
    "hunuparcha", "hunuparxa", "vaneko", "bhaneko", "malai", "hamro",
    "hamilai", "timro", "tapai", "vitra", "bhitra", "bata", "samma",
    "aaba", "aba", "kina", "kasari", "kati", "kaha", "kahan", "sarkar",
    "desh", "des", "nepal", "nepali", "janta", "jasto", "jastai", "matra",
    "lagi", "paryo", "parcha", "parxa", "vayeko", "le", "sabai", "chahiyo",
    "chahincha", "chaieko", "raheko", "bhako", "dherai", "thulo", "sano",
    "ekdam", "sakcha", "sakincha", "diyo", "dinu", "bata", "gharma",
    "gaun", "sahar", "manche", "manxe", "manisharu",
}


def script_hint(text: str) -> str:
    dev = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    lat = sum(1 for c in text if c.isascii() and c.isalpha())
    if dev and lat:
        return "mixed"
    if dev:
        return "devanagari"
    if lat:
        words = set(re.findall(r"[a-zA-Z]+", text.lower()))
        hits = words & ROMANIZED_NEPALI_MARKERS
        # 2+ marker words is a strong signal of Romanized Nepali, not English
        if len(hits) >= 2:
            return "romanized_nepali"
        return "english"
    return "other"


def hash_author(author: str | None) -> str:
    if not author:
        return ""
    return hashlib.sha256((SALT + author).encode("utf-8")).hexdigest()[:16]


def make_record(platform, text, source_url, timestamp, query, author=None,
                parent_title=None, parent_url=None, kind="comment"):
    clean = strip_pii(text)
    if len(clean) < MIN_CHARS:
        return None
    return {
        "id": hashlib.sha1(f"{platform}|{source_url}|{clean}".encode()).hexdigest()[:12],
        "platform": platform,
        "kind": kind,                      # comment | post | reply
        "text": clean,
        "script": script_hint(clean),
        "timestamp": timestamp or "",
        "source_url": source_url,
        "parent_title": parent_title or "",
        "parent_url": parent_url or "",
        "query": query,
        "author_hash": hash_author(author),
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "_dedup": dedup_key(clean),
    }


# ---------------------------------------------------------------------------
# Storage (append-safe, dedups against what's already on disk)
# ---------------------------------------------------------------------------

class Store:
    def __init__(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.seen = set()
        self.new = 0
        if OUT_JSONL.exists():
            with OUT_JSONL.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        self.seen.add(json.loads(line)["_dedup"])
                    except Exception:
                        pass
        self.fh = OUT_JSONL.open("a", encoding="utf-8")

    def add(self, rec):
        if rec is None or rec["_dedup"] in self.seen:
            return False
        self.seen.add(rec["_dedup"])
        self.fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.new += 1
        return True

    def close(self):
        self.fh.close()
        # rewrite CSV from JSONL each run
        rows = [json.loads(l) for l in OUT_JSONL.open(encoding="utf-8")]
        if not rows:
            return
        fields = [k for k in rows[0] if k != "_dedup"]
        with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------

def scrape_youtube(store: Store, max_videos: int, max_comments: int):
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        sys.exit("YOUTUBE_API_KEY missing in .env")
    yt = build("youtube", "v3", developerKey=key)
    seen_videos = set()

    for q in QUERIES:
        print(f"[yt] search: {q}")
        try:
            kwargs = dict(q=q + YT_QUERY_SUFFIX, part="snippet", type="video",
                          maxResults=min(max_videos, 50), relevanceLanguage="ne",
                          order="relevance")
            if YT_PUBLISHED_AFTER:
                kwargs["publishedAfter"] = YT_PUBLISHED_AFTER
            res = yt.search().list(**kwargs).execute()
        except HttpError as e:
            print(f"  search failed: {e}")
            continue

        for item in res.get("items", []):
            vid = item["id"]["videoId"]
            if vid in seen_videos:
                continue
            seen_videos.add(vid)
            title = item["snippet"]["title"]
            vurl = f"https://www.youtube.com/watch?v={vid}"
            got = 0
            token = None
            while got < max_comments:
                try:
                    cres = yt.commentThreads().list(
                        part="snippet,replies", videoId=vid, maxResults=100,
                        textFormat="plainText", pageToken=token, order="relevance"
                    ).execute()
                except HttpError as e:
                    # comments disabled (403) or quota (403) — skip video
                    print(f"  {vid}: {getattr(e, 'status_code', '')} skipping")
                    break
                for th in cres.get("items", []):
                    top = th["snippet"]["topLevelComment"]["snippet"]
                    cid = th["snippet"]["topLevelComment"]["id"]
                    store.add(make_record(
                        "youtube", top["textDisplay"],
                        f"{vurl}&lc={cid}", top["publishedAt"], q,
                        author=top.get("authorChannelId", {}).get("value"),
                        parent_title=title, parent_url=vurl))
                    got += 1
                    for rep in th.get("replies", {}).get("comments", []):
                        rs = rep["snippet"]
                        store.add(make_record(
                            "youtube", rs["textDisplay"],
                            f"{vurl}&lc={rep['id']}", rs["publishedAt"], q,
                            author=rs.get("authorChannelId", {}).get("value"),
                            parent_title=title, parent_url=vurl, kind="reply"))
                token = cres.get("nextPageToken")
                if not token:
                    break
                time.sleep(0.2)
            print(f"  {title[:60]!r}: {got} threads")
    print(f"[yt] done — {len(seen_videos)} videos")


# ---------------------------------------------------------------------------
# Reddit
# ---------------------------------------------------------------------------

def scrape_reddit(store: Store, max_posts: int):
    import praw

    cid, sec, ua = (os.getenv("REDDIT_CLIENT_ID"), os.getenv("REDDIT_CLIENT_SECRET"),
                    os.getenv("REDDIT_USER_AGENT"))
    if not all([cid, sec, ua]):
        sys.exit("REDDIT_* vars missing in .env")
    reddit = praw.Reddit(client_id=cid, client_secret=sec, user_agent=ua)
    seen_posts = set()

    for sub in REDDIT_SUBREDDITS:
        for q in QUERIES:
            print(f"[reddit] r/{sub} search: {q}")
            try:
                results = reddit.subreddit(sub).search(q, limit=max_posts,
                                                       time_filter=REDDIT_TIME_FILTER)
                for post in results:
                    if post.id in seen_posts:
                        continue
                    seen_posts.add(post.id)
                    purl = f"https://www.reddit.com{post.permalink}"
                    ts = datetime.fromtimestamp(post.created_utc, timezone.utc).isoformat()
                    body = f"{post.title}\n{post.selftext or ''}"
                    store.add(make_record("reddit", body, purl, ts, q,
                                          author=str(post.author) if post.author else None,
                                          parent_title=post.title, parent_url=purl,
                                          kind="post"))
                    post.comments.replace_more(limit=0)
                    for c in post.comments.list():
                        store.add(make_record(
                            "reddit", c.body,
                            f"https://www.reddit.com{c.permalink}",
                            datetime.fromtimestamp(c.created_utc, timezone.utc).isoformat(),
                            q, author=str(c.author) if c.author else None,
                            parent_title=post.title, parent_url=purl))
                    time.sleep(0.5)
            except Exception as e:
                print(f"  failed: {e}")
    print(f"[reddit] done — {len(seen_posts)} posts")


# ---------------------------------------------------------------------------
# Manual CSV (Facebook copy-paste etc.)
# ---------------------------------------------------------------------------

def import_manual(store: Store, path: str):
    n = 0
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if store.add(make_record("facebook_manual", row.get("text", ""),
                                     row.get("source_url", ""), row.get("timestamp", ""),
                                     "manual")):
                n += 1
    print(f"[manual] imported {n} new rows from {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--youtube", action="store_true")
    ap.add_argument("--reddit", action="store_true")
    ap.add_argument("--manual", metavar="CSV", help="import a manual CSV")
    ap.add_argument("--max-videos", type=int, default=10, help="videos per query (YouTube)")
    ap.add_argument("--max-comments", type=int, default=300, help="top-level comment threads per video")
    ap.add_argument("--max-posts", type=int, default=25, help="posts per query per subreddit (Reddit)")
    args = ap.parse_args()

    if not (args.youtube or args.reddit or args.manual):
        ap.error("pick at least one of --youtube, --reddit, --manual")

    store = Store()
    print(f"existing records: {len(store.seen)}")
    try:
        if args.youtube:
            scrape_youtube(store, args.max_videos, args.max_comments)
        if args.reddit:
            scrape_reddit(store, args.max_posts)
        if args.manual:
            import_manual(store, args.manual)
    finally:
        store.close()

    # quick summary
    rows = [json.loads(l) for l in OUT_JSONL.open(encoding="utf-8")]
    from collections import Counter
    print(f"\nnew this run: {store.new} | total: {len(rows)}")
    print("by platform:", dict(Counter(r["platform"] for r in rows)))
    print("by script:  ", dict(Counter(r["script"] for r in rows)))
    print(f"-> {OUT_JSONL}\n-> {OUT_CSV}")


if __name__ == "__main__":
    main()