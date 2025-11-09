import requests
import datetime
import alpaca_trade_api as tradeapi
from datetime import timezone

# === CONFIG ===
ALPACA_KEY = "PKKA3JM6FPA7E73WWCHUX7PZDQ"
ALPACA_SECRET = "9Frpe4ECnhsN4DNh6SSvENxJKxJ5FqebwKfu4xxyDVXR"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # PAPER MODE
TELEGRAM_TOKEN = "7959803191:AAElC4k6-Evxm9DMcBLxmtruptyoUpCmXM4"
CHAT_ID = "5130903867"

# === LOGGING ===
def log(msg):
    with open("/Users/brockhenley/chainshot/trade_log.txt", "a") as f:
        f.write(f"[{datetime.datetime.now()}] {msg}\n")
    print(msg)

# === TELEGRAM ===
def send_alert(title, msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": f"*{title}*\n{msg}", "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except: pass

# === MARKET HOURS (ET) ===
def is_market_open():
    now = datetime.datetime.now(timezone.utc)
    et = now.astimezone(datetime.timezone(datetime.timedelta(hours=-4)))
    if et.weekday() >= 5: return False
    if not (9 <= et.hour < 16): return False
    if et.hour == 9 and et.minute < 30: return False
    return True

# === SIGNALS ===
def air_force_alert():
    url = 'https://opensky-network.org/api/states/all?lamin=27.5&lamax=28.5&lonmin=-83.0&lonmax=-82.0'
    try:
        data = requests.get(url, timeout=10).json()
        tankers = sum(1 for s in data.get('states', []) if s and any(k in str(s[1]).upper() for k in ['KC-', 'C-17']))
        if tankers >= 5:
            return f"ALERT: {tankers} tankers at MacDill → LONG LMT/BA"
    except: pass
    return None

def oil_draw_alert():
    try:
        data = requests.get("https://ir.eia.gov/wpsr/table1.json", timeout=10).json()
        draw = data['series'][0]['data'][0][1] - data['series'][0]['data'][1][1]
        if draw > 3_000_000:
            return f"OIL DRAWDOWN {draw/1e6:.1f}M bbl → LONG USO"
    except: pass
    return None

def shipping_alert():
    return "TEST: ChainShot v5.5 — AUTO PAPER TRADE"

# === AUTO TRADE ===
def auto_trade(signal):
    if not is_market_open():
        log("MARKET CLOSED — NO TRADES")
        return

    try:
        api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL, 'v2')
        account = api.get_account()
        cash = float(account.cash)
        log(f"MARKET OPEN — ${cash:,.0f} PAPER CASH")

        if 'LMT/BA' in signal:
            qty = 1
            api.submit_order('LMT', qty, 'buy', 'market', 'day')
            api.submit_order('BA', qty, 'buy', 'market', 'day')
            msg = f"PAPER TRADE: LONG LMT {qty} | BA {qty}"
            send_alert("TRADE EXECUTED", msg)
            log(msg)

        if 'USO' in signal:
            qty = 2
            api.submit_order('USO', qty, 'buy', 'market', 'day')
            msg = f"PAPER TRADE: LONG USO {qty}"
            send_alert("TRADE EXECUTED", msg)
            log(msg)

        if 'TEST' in signal:
            qty = 1
            api.submit_order('LMT', qty, 'buy', 'market', 'day')
            msg = f"PAPER TEST: LONG LMT {qty}"
            send_alert("TEST TRADE", msg)
            log(msg)

    except Exception as e:
        log(f"ALPACA ERROR: {e}")

# === MAIN LOOP ===
def main():
    log("=== ChainShot v5.5 STARTED ===")
    signal = None
    signal = signal or air_force_alert()
    signal = signal or oil_draw_alert()
    signal = signal or shipping_alert()

    if signal:
        log(f"SIGNAL: {signal}")
        send_alert("SIGNAL DETECTED", signal)
        auto_trade(signal)
    else:
        log("NO SIGNAL")

if __name__ == "__main__":
    main()
