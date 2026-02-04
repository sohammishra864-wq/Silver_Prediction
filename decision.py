from montecarlo import predict_next_close, monte_carlo_sim

CAPITAL = 100000          # You can change later
RISK_PER_TRADE = 0.02    # 2% rule


# ----------------------------
# Decision Rules
# ----------------------------
def trade_decision(prob_up):
    if prob_up > 0.65:
        return "BUY"
    elif prob_up < 0.35:
        return "SELL"
    else:
        return "HOLD"


# ----------------------------
# Position Sizing using worst case
# ----------------------------
def position_size(current_price, worst_case):
    risk_amount = CAPITAL * RISK_PER_TRADE
    risk_per_unit = abs(current_price - worst_case)

    if risk_per_unit == 0:
        return 0

    units = risk_amount / risk_per_unit
    return int(units)


# ----------------------------
# Run Decision Engine
# ----------------------------
if __name__ == "__main__":
    current, predicted = predict_next_close()
    prob_up, exp_price, worst, best = monte_carlo_sim(current, predicted)

    decision = trade_decision(prob_up)
    size = position_size(current, worst)

    print("\n===== TRADE DECISION =====")
    print(f"Current Price: {current:.2f}")
    print(f"Expected Price: {exp_price:.2f}")
    print(f"Probability Up: {prob_up*100:.2f}%")
    print(f"Worst Case: {worst:.2f}")
    print(f"Decision: {decision}")
    print(f"Position Size: {size} units")
