import feedparser
import pandas as pd
from datetime import datetime
from pathlib import Path
STORAGE_PATH = Path(__file__).resolve().parents[1] / "storage" / "raw_news.csv"
# I am writing these comments for Future as I will not remember everything
RSS_FEEDS = [  # from AI
    "https://www.livemint.com/rss/markets",
    "https://www.livemint.com/rss/economy",
    "https://www.livemint.com/rss/commodities"
]
KEYWORDS = [   # from AI
    "silver", "xag", "gold silver ratio",
    "inflation", "federal reserve", "dollar",
    "precious metals"
]
# is relevant is only meant for shortlisting the relevant topics from many topics
def is_relevant(text: str) -> bool:  # Checks for match from KEYWORDS if match exist
    text = text.lower()
    return any(txt in text for txt in KEYWORDS)
def fetch_livemint():
    rows = []
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            text = f"{entry.title} {entry.summary}"
            if is_relevant(text):
                rows.append({
                    "date": datetime(*entry.published_parsed[:6]),
                    "title": entry.title,
                    "summary": entry.summary,
                    "link": entry.link,
                    "text": text
                })
    df = pd.DataFrame(rows)
    df.to_csv(STORAGE_PATH, index=False)
    print(f"Saved {len(df)} LiveMint articles → {STORAGE_PATH}")