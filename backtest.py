import numpy as np

def simulate(years=20, p_signal=0.346, win_rate=0.74, risk=0.02, rr=5):
    port = 1.0
    total_trades = 0
    for _ in range(years * 52):
        if np.random.rand() < p_signal:
            total_trades += 1
            if np.random.rand() < win_rate:
                port *= (1 + risk * rr)
            else:
                port *= (1 - risk)
    cagr = (port ** (1 / years) - 1) * 100
    trades_per_year = total_trades / years
    return cagr, trades_per_year

# 1000-run Monte Carlo
runs = [simulate() for _ in range(1000)]
avg_cagr = np.mean([r[0] for r in runs])
avg_trades_year = np.mean([r[1] for r in runs])
print(f"Expected CAGR: {avg_cagr:.1f}%")
print(f"Expected Trades/Year: {avg_trades_year:.2f}")
