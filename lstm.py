import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import torch
# Storing the data
BASE = Path(__file__).resolve().parent
STORAGE = BASE / "storage"
SENTIMENT_PATH = STORAGE / "past_sentiment_daily.csv"
DATASET_PATH = STORAGE / "lstm_dataset.csv"
MODEL_PATH = STORAGE / "lstm_model.pth"
LOOKBACK = 90
# Had some error due to lookback = 30 didnt had 30 day data so gave to AI
# rest it wrote many if statement i didnt write them by my own
# so this code is very much AI
# Original code is not there as there were errors due to 30 days lookback
def silver_price():
    sentiment_df = pd.read_csv(SENTIMENT_PATH)
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
    start_date = sentiment_df["date"].min().strftime("%Y-%m-%d")
    df = yf.download("SI=F", start=start_date) # SI=F is for silver chart in yf
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    df.rename(columns={ # converts to lowercase
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }, inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[["date", "open", "high", "low", "close", "volume"]]

def senti_plus_OHLCV():
    price_df = silver_price()
    sentiment_df = pd.read_csv(SENTIMENT_PATH)
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.date
    df = price_df.merge(sentiment_df, on="date", how="left")
    df["sentiment"] = df["sentiment"].fillna(0)
    df["sentiment"] = df["sentiment"] * 50
    df.to_csv(DATASET_PATH, index=False)
    return df

def lookback(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    x, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        x.append(scaled[i - LOOKBACK:i])
        y.append(scaled[i][3])
    if len(x) == 0:
        return None, None, None
    return np.array(x), np.array(y), scaler

class Lstm(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
def train_lstm():
    df = senti_plus_OHLCV()
    features = df[["open", "high", "low", "close", "volume", "sentiment"]].values
    x, y, scaler = lookback(features)

    if x is None:
        print(f"\n Not enough data for LSTM.")
        print(f"You need at least {LOOKBACK + 1} days of sentiment data.")
        print(f"Current days available: {len(df)}")
        return

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    model = Lstm()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training LSTM...")

    for epoch in range(20):
        optimizer.zero_grad()
        output = model(x).squeeze()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/20, Loss: {loss.item():.6f}")

    torch.save({
        "model_state": model.state_dict(),
        "scaler": scaler
    }, MODEL_PATH)

    print(f"\n LSTM model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train_lstm()
