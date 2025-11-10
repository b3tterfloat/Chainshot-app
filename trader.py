import requests
import datetime
import alpaca_trade_api as tradeapi
from datetime import timezone

# === CONFIG ===
ALPACA_KEY = "YOUR_ALPACA_KEY"
ALPACA_SECRET = "YOUR_ALPACA_SECRET"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

# === LOGGING ===
def log(msg):
    with open("trade_log.txt", "a") as f:
        f.write(f"[{datetime.datetime.now()}] {msg}\n")
    print(msg)

# === TELEGRAM ===
def send_alert(title, msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": f"*{title}*\n{msg}", "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=10)
    except: pass

# === MARKET HOURS (ET) ===
def is_market_open():
    now = datetime.datetime.now(timezone.utc)
    et = now.astimezone(datetime.timezone(datetime.timedelta(hours=-4)))
    if et.weekday() >= 5: return False
    if not (9 <= et.hour < 16): return False
    if et.hour == 9 and et.minute < 30: return False
    return True

# === HIGH-CONVICTION SIGNALS ONLY ===
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
        eia_url = "https://ir.eia.gov/wpsr/table1.json"
        data = requests.get(eia_url, timeout=10).json()
        current = data['series'][0]['data'][0][1]
        previous = data['series'][0]['data'][1][1]
        draw = previous - current
        if draw > 3_000_000:
            return f"OIL DRAWDOWN {draw/1e6:.1f}M bbl → LONG USO"
    except: pass
    return None

# === NOAA GULF STORM ===
def noaa_storm_alert():
    try:
        url = "https://api.weather.gov/alerts/active?area=GM"
        data = requests.get(url, headers={'User-Agent': 'ChainShot/6.0'}).json()
        for alert in data.get('features', []):
            if 'Hurricane' in alert['properties']['headline'] or 'Tropical Storm' in alert['properties']['headline']:
                return f"NOAA: GULF STORM → SHORT HRL"
    except: pass
    return None

# === HRL EARNINGS MISS ===
def hrl_earnings_miss():
    try:
        url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol=HRL&apikey=YOUR_KEY"
        data = requests.get(url).json()
        if 'quarterlyEarnings' in data:
            latest = data['quarterlyEarnings'][0]
            if float(latest['reportedEPS']) < float(latest['estimatedEPS']):
                return f"HRL MISS: {latest['reportedEPS']} vs {latest['estimatedEPS']} → SHORT HRL"
    except: pass
    return None

# === BRENT FUTURES SPIKE ===
def brent_spike():
    try:
        url = "https://api.cmegroup.com/v1/quotes?symbol=BZ"
        data = requests.get(url).json()
        change = float(data['quotes'][0]['change'])
        return change > 2.0
    except: return False

# === VIX CHANGE ===
def get_vix_change():
    try:
        url = "https://api.nasdaq.com/api/quote/VIX/info?assetclass=indices"
        data = requests.get(url).json()
        return float(data['data']['percentageChange'])
    except: return 0.0

# === DoD CONTRACT ===
def dod_contract_alert():
    try:
        url = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
        params = {"keyword": "Lockheed", "award_type_codes": "P", "page": 1}
        data = requests.get(url, params=params).json()
        if data['results']:
            value = data['results'][0]['total_obligated_amount']
            if value > 1_000_000_000:
                return True
    except: pass
    return False

def auto_trade(action, risk_pct=0.02):
    if not is_market_open(): return
    try:
        api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL, 'v2')
        cash = float(api.get_account().cash)
        risk = cash * risk_pct

        if 'LMT/BA + USO' in action:
            qty = int(risk / 50)
            api.submit_order('LMT', qty, 'buy', 'market', 'day')
            api.submit_order('BA', qty, 'buy', 'market', 'day')
            api.submit_order('USO', int(qty*1.5), 'buy', 'market', 'day')
            log(f"TRADE: LONG LMT {qty} | BA {qty} | USO {int(qty*1.5)}")

        elif 'SHORT HRL' in action:
            qty = int(risk / 30)
            api.submit_order('HRL', qty, 'sell', 'market', 'day')
            log(f"TRADE: SHORT HRL {qty}")

        elif 'LONG GLD' in action:
            qty = int(risk / 30)
            api.submit_order('GLD', qty, 'buy', 'market', 'day')
            log(f"HEDGE: LONG GLD {qty}")

        elif 'LONG LMT' in action:
            qty = int(risk / 50)
            api.submit_order('LMT', qty, 'buy', 'market', 'day')
            log(f"TRADE: LONG LMT {qty}")

    except Exception as e:
        log(f"ERROR: {e}")

# === v6.0: 5 NEW AGENTS + LINKED LOGIC ===
def main():
    log("=== ChainShot v6.0 — INTELLIGENCE UPGRADE ===")
    
    signals = []
    context = {}

    # 1. MacDill Air Force
    if air_force_alert():
        signals.append("MACDILL")
        context['macdill'] = True

    # 2. EIA Oil Drawdown
    if oil_draw_alert():
        signals.append("OIL")
        context['oil'] = True

    # 3. NOAA Gulf Storm
    if noaa_storm_alert():
        signals.append("STORM")
        context['storm'] = True

    # 4. USAspending DoD Contract
    if dod_contract_alert():
        signals.append("DOD")
        context['dod'] = True

    # 5. VIX Spike (>3%)
    vix_change = get_vix_change()
    if vix_change > 3.0:
        signals.append("VIX")
        context['vix'] = vix_change

    # 6. Brent Futures Spike (>2%)
    if brent_spike():
        signals.append("BRENT")
        context['brent'] = True

    # 7. HRL Earnings Miss
    if hrl_earnings_miss():
        signals.append("HRL_MISS")
        context['hrl'] = True

    # === LINKED DECISION ENGINE ===
    linked = linked_triggers(signals, context)
    if linked:
        log(f"v6.0: {linked['signal']} → {linked['action']}")
        send_alert("LINKED SIGNAL", linked['signal'])
        auto_trade(linked['action'], risk_pct=linked['risk_pct'])
        return

    # === v5.5 Fallback ===
    if len(signals) == 1:
        log(f"v5.5 Fallback: {signals[0]}")
        send_alert("SINGLE SIGNAL", signals[0])
        auto_trade(signals[0])
    else:
        log("NO SIGNAL — WAITING")

# === LINKED TRIGGER LOGIC ===
def linked_triggers(signals, context):
    if len(signals) >= 2:
        if "MACDILL" in signals and "OIL" in signals:
            return {"signal": "MACDILL + OIL", "action": "LONG LMT/BA + USO", "risk_pct": 0.015}
        if "STORM" in signals and "HRL_MISS" in signals:
            return {"signal": "STORM + HRL_MISS", "action": "SHORT HRL", "risk_pct": 0.015}
        if "DOD" in signals and "BRENT" in signals:
            return {"signal": "DOD + BRENT", "action": "LONG LMT", "risk_pct": 0.015}
        if "VIX" in signals and len(signals) == 2:
            return {"signal": f"HEDGE: {signals[0]} + VIX", "action": "LONG GLD", "risk_pct": 0.025}
    return None
