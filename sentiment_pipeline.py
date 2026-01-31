import pandas as pd
from pathlib import Path
from tqdm import tqdm
from processes.sentiment import FinBERT_Sentiment
BASE_PATH = Path(__file__).resolve().parent
STORAGE = BASE_PATH / "storage"

news_path = STORAGE / "raw_news.csv"
out_path = STORAGE / "sentiment_daily.csv"
def load_data():
    df = pd.read_csv(news_path)
  #  reditt = pd.read_csv(news_path)
   # df = pd.concat([
        #news[["date", "text"]],  # concatenate(merge) all the existing data into one
        #reditt[["date", "text"]])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df
def run_sentiment():
    df = load_data()
    finbert = FinBERT_Sentiment()
    print('finbert running..')
    sentiment = []
    for _, row in tqdm(df.iterrows(), total=len(df)): # for loop by AI
        score = finbert.score(row["text"]) # didnt understand finbert so AI
        sentiment.append({
            "date": row["date"],
            "sentiment": score
        })

    final = pd.DataFrame(sentiment)
    final.to_csv(out_path, index=False)
    print('saved')
if __name__ == "__main__":
    run_sentiment()