import torch
import numpy as np
import pandas as pd
from pathlib import Path
from lstm import Lstm, LOOKBACK

BASE = Path(__file__).resolve().parent
STORAGE = BASE / "storage"
MODEL_PATH = STORAGE / "lstm_model.pth"
DATASET_PATH = STORAGE / "lstm_dataset.csv"

def load_model():
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    model = Lstm()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    scaler = checkpoint["scaler"]
    return model, scaler

def predict_next_close():
    model, scaler = load_model()

    df = pd.read_csv(DATASET_PATH)
    features = df[["open", "high", "low", "close", "volume", "sentiment"]].values

    last_30 = features[-LOOKBACK:]
    scaled = scaler.transform(last_30)

    x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = model(x).item()

    dummy = np.zeros((1, 6))
    dummy[0][3] = pred_scaled
    pred_price = scaler.inverse_transform(dummy)[0][3]

    current_price = df["close"].iloc[-1]

    return current_price, pred_price
# written by me but help using AI
def monte_carlo_sim(current_price, pred_price, days=1, paths=10000):
    returns = pd.read_csv(DATASET_PATH)["close"].pct_change().dropna()
    sigma = returns.std()

    mu = (pred_price - current_price) / current_price

    simulations = []

    for _ in range(paths):
        price = current_price
        for _ in range(days):
            shock = np.random.normal(mu, sigma)
            price = price * (1 + shock)
        simulations.append(price)

    simulations = np.array(simulations)

    prob_up = np.mean(simulations > current_price)
    expected_price = simulations.mean()
    worst_case = np.percentile(simulations, 5)
    best_case = np.percentile(simulations, 95)

    return prob_up, expected_price, worst_case, best_case


if __name__ == "__main__":
    current, predicted = predict_next_close()

    print(f"Current Price: {current:.2f}")
    print(f"LSTM Predicted Next Close: {predicted:.2f}")

    p_up, exp_price, worst, best = monte_carlo_sim(current, predicted)

    print("\nMonte Carlo Results")
    print(f"Probability Up: {p_up*100:.2f}%")
    print(f"Expected Price: {exp_price:.2f}")
    print(f"Worst Case (5%): {worst:.2f}")
    print(f"Best Case (95%): {best:.2f}")
