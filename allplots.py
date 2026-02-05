import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from montecarlo import predict_next_close, monte_carlo_sim
from backtest import load_model, LOOKBACK

BASE = Path(__file__).resolve().parent
STORAGE = BASE / "storage"
DATASET_PATH = STORAGE / "lstm_dataset.csv"

df = pd.read_csv(DATASET_PATH)
df["date"] = pd.to_datetime(df["date"])


# ----------------------------
# 1. Price vs Sentiment
# ----------------------------
plt.figure()
plt.plot(df["date"], df["close"], label="Silver Close")
plt.plot(df["date"], df["sentiment"], label="Sentiment")
plt.legend()
plt.title("Silver Price vs Sentiment")
plt.show()


# ----------------------------
# 2. Sentiment Over Time
# ----------------------------
plt.figure()
plt.plot(df["date"], df["sentiment"])
plt.title("Sentiment Over Time")
plt.show()


# ----------------------------
# 3. Sentiment vs Next-Day Returns
# ----------------------------
df["next_return"] = df["close"].pct_change().shift(-1)

plt.figure()
plt.scatter(df["sentiment"], df["next_return"])
plt.title("Sentiment vs Next Day Return")
plt.show()

corr = df["sentiment"].corr(df["next_return"])
print(f"Correlation (Sentiment vs Next-Day Return): {corr:.4f}")


# ----------------------------
# 4. Correlation Heatmap (no seaborn, pure matplotlib)
# ----------------------------
corr_matrix = df[["open", "high", "low", "close", "volume", "sentiment"]].corr()

plt.figure()
plt.imshow(corr_matrix)
plt.colorbar()
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
plt.title("Feature Correlation Heatmap")
plt.show()


# ----------------------------
# 5. Monte Carlo Distribution
# ----------------------------
current, predicted = predict_next_close()
p_up, exp_price, worst, best = monte_carlo_sim(current, predicted)

returns = df["close"].pct_change().dropna()
sigma = returns.std()
mu = (predicted - current) / current

simulations = []
for _ in range(10000):
    price = current * (1 + np.random.normal(mu, sigma))
    simulations.append(price)

plt.figure()
plt.hist(simulations, bins=50)
plt.title("Monte Carlo Price Distribution")
plt.show()


# ----------------------------
# 6. Equity Curve from Backtest Logic
# ----------------------------
from backtest import run_backtest

print("\nRun backtest.py separately to view metrics.")
print("This plot shows conceptual equity curve if you log it in backtest.")
