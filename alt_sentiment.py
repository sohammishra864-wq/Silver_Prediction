import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

BASE = Path(__file__).resolve().parent
STORAGE = BASE / "storage"
OUTPUT = STORAGE / "past_sentiment_daily.csv"

START = "2025-01-01"
END = "2026-01-01"


def fetch_data():
    vix = yf.download("^VIX", start=START, end=END)["Close"]
    dxy = yf.download("DX-Y.NYB", start=START, end=END)["Close"]
    gold = yf.download("GLD", start=START, end=END)["Close"]
    silver = yf.download("SI=F", start=START, end=END)["Close"]

    df = pd.concat([vix, dxy, gold, silver], axis=1)
    df.columns = ["vix", "dxy", "gold", "silver"]
    df.dropna(inplace=True)

    return df


def build_sentiment(df):
    # Convert to daily returns (captures "mood")
    df["vix_ret"] = df["vix"].pct_change()
    df["dxy_ret"] = df["dxy"].pct_change()
    df["gold_ret"] = df["gold"].pct_change()
    df["silver_ret"] = df["silver"].pct_change()

    df.dropna(inplace=True)

    # Scale values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["vix_ret", "dxy_ret", "gold_ret", "silver_ret"]])

    scaled_df = pd.DataFrame(
        scaled,
        index=df.index,
        columns=["vix", "dxy", "gold", "silver"]
    )

    # Build sentiment formula
    # Fear (VIX) negative, Dollar negative, Gold positive, Silver momentum positive
    sentiment = (
        -scaled_df["vix"]
        -scaled_df["dxy"]
        +scaled_df["gold"]
        +scaled_df["silver"]
    )

    sentiment_df = sentiment.reset_index()
    sentiment_df.columns = ["date", "sentiment"]

    sentiment_df.to_csv(OUTPUT, index=False)
    print(f" 1 year proxy sentiment saved → {OUTPUT}")


if __name__ == "__main__":
    data = fetch_data()
    build_sentiment(data)
