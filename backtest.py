import pandas as pd
import numpy as np
import torch
from lstm import Lstm, LOOKBACK
from montecarlo import monte_carlo_sim
from pathlib import Path

BASE = Path(__file__).resolve().parent
STORAGE = BASE / "storage"
MODEL_PATH = STORAGE / "lstm_model.pth"
DATASET_PATH = STORAGE / "lstm_dataset.csv"

CAPITAL = 100000
RISK_PER_TRADE = 0.02


# ----------------------------
# Load model
# ----------------------------
def load_model():
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    model = Lstm()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    scaler = checkpoint["scaler"]
    return model, scaler


# ----------------------------
# Decision rule
# ----------------------------
def trade_decision(prob_up):
    if prob_up > 0.65:
        return 1      # long
    elif prob_up < 0.35:
        return -1     # short
    else:
        return 0      # hold


# ----------------------------
# Backtest
# ----------------------------
def run_backtest():
    model, scaler = load_model()
    df = pd.read_csv(DATASET_PATH)

    equity = CAPITAL
    equity_curve = []
    wins = 0
    trades = 0

    for i in range(LOOKBACK, len(df) - 1):
        window = df.iloc[i - LOOKBACK:i]
        next_day = df.iloc[i + 1]

        features = window[["open", "high", "low", "close", "volume", "sentiment"]].values
        scaled = scaler.transform(features)

        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_scaled = model(x).item()

        dummy = np.zeros((1, 6))
        dummy[0][3] = pred_scaled
        predicted_price = scaler.inverse_transform(dummy)[0][3]

        current_price = window["close"].iloc[-1]

        prob_up, _, worst, _ = monte_carlo_sim(current_price, predicted_price)

        signal = trade_decision(prob_up)

        if signal == 0:
            equity_curve.append(equity)
            continue

        risk_amount = equity * RISK_PER_TRADE
        risk_per_unit = abs(current_price - worst)
        units = risk_amount / risk_per_unit if risk_per_unit != 0 else 0

        price_change = next_day["close"] - current_price
        pnl = signal * units * price_change

        equity += pnl
        equity_curve.append(equity)

        trades += 1
        if pnl > 0:
            wins += 1

    # ----------------------------
    # Metrics
    # ----------------------------
    equity_curve = np.array(equity_curve)
    returns = np.diff(equity_curve) / equity_curve[:-1]

    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
    drawdown = np.max(np.maximum.accumulate(equity_curve) - equity_curve)

    win_rate = (wins / trades) * 100 if trades > 0 else 0
    total_return = (equity - CAPITAL) / CAPITAL * 100

    print("\n===== BACKTEST RESULTS =====")
    print(f"Final Equity: {equity:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {drawdown:.2f}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Trades: {trades}")


if __name__ == "__main__":
    run_backtest()
