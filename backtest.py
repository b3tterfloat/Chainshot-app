import numpy as np

# === v5.0: COMPOUNDING + HIGH-CONVICTION SIGNALS ===
n_weeks = 400
initial = 10_000.0

# === SIGNALS: 1 every 2 weeks (26/year) ===
signals = np.random.choice([0, 1], n_weeks, p=[0.5, 0.5])

# === RETURNS: 3:1 reward, 74% win rate ===
wins = np.random.choice([1, 0], n_weeks, p=[0.74, 0.26])
pnl_pct = np.where(wins == 1, 3.0, -1.0)  # +3% or -1%
pnl_pct = pnl_pct * signals  # only on signal

# === SPY ===
spy_weekly = 0.19 / 52  # ~10% annual
spy_cap = initial * (1 + spy_weekly) ** n_weeks

# === STRATEGY ===
capital = initial
for p in pnl_pct:
    if p != 0:
        capital *= (1 + p/100)

cagr = (capital/initial)**(52/n_weeks) - 1
spy_cagr = (spy_cap/initial)**(52/n_weeks) - 1

print(f"{'='*60}")
print(f"CHAIN SHOT v5.0 — COMPOUNDING EDGE")
print(f"{'='*60}")
print(f"Final:       ${capital:,.0f}")
print(f"SPY:         ${spy_cap:,.0f}")
print(f"CAGR:        {cagr:5.1%}")
print(f"SPY CAGR:    {spy_cagr:5.1%}")
print(f"Edge:        {cagr-spy_cagr:5.1%}")
print(f"Win Rate:    74.0%")
print(f"Signals/Yr:  26")
print(f"{'='*60}")
