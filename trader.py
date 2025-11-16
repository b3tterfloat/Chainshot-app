# trader.py
import os
import logging
import requests
import re
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from telegram import Bot
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
from xml.etree import ElementTree as ET

# ---------- LOAD .env ----------
PROJECT_ROOT = Path(__file__).parent.resolve()
ENV_PATH = PROJECT_ROOT / ".env"
print(f"Looking for .env at: {ENV_PATH}")

if not ENV_PATH.exists():
    raise FileNotFoundError(f".env not found at {ENV_PATH}")

load_dotenv(dotenv_path=ENV_PATH)

# ---------- CONFIG ----------
APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID')
APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

OPENSKY_USERNAME = os.getenv('OPENSKY_USERNAME', '')
OPENSKY_PASSWORD = os.getenv('OPENSKY_PASSWORD', '')
EIA_API_KEY = os.getenv('EIA_API_KEY', '')
POLYGON_KEY = os.getenv('POLYGON_KEY', '')

HEDGE_ENABLED = False
AI_OPTIONS_ENABLED = True

# ---------- DEBUG ----------
print(f"APCA_API_KEY_ID = {APCA_API_KEY_ID[:6] + '...' if APCA_API_KEY_ID else 'MISSING'}")
print(f"APCA_API_SECRET_KEY = {'SET' if APCA_API_SECRET_KEY else 'MISSING'}")
print(f"TELEGRAM_TOKEN = {'SET' if TELEGRAM_TOKEN else 'MISSING'}")
print(f"OpenSky: {'SET' if OPENSKY_USERNAME else 'MISSING'}")
print(f"EIA: {'SET' if EIA_API_KEY else 'MISSING'}")
print(f"Polygon: {'SET' if POLYGON_KEY else 'MISSING'}")
print(f"AI_OPTIONS_ENABLED = {AI_OPTIONS_ENABLED}")

# ---------- VALIDATION ----------
if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY:
    raise ValueError("Alpaca keys missing!")

# ---------- TELEGRAM ----------
if not TELEGRAM_TOKEN:
    print("Warning: Telegram token missing — alerts disabled")
    TELEGRAM_READY = False
else:
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        TELEGRAM_READY = True
        print("Telegram bot initialized")
    except Exception as e:
        print(f"Telegram init failed: {e}")
        TELEGRAM_READY = False

# ---------- ALPACA ----------
api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# ---------- AI ENGINE ----------
class OptionsAIEngine:
    MODEL_PATH = "models/spy_options_model.pkl"
    SCALER_PATH = "models/scaler.pkl"

    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_or_train()

    def load_or_train(self):
        os.makedirs("models", exist_ok=True)
        if os.path.exists(self.MODEL_PATH) and os.path.exists(self.SCALER_PATH):
            self.model = joblib.load(self.MODEL_PATH)
            self.scaler = joblib.load(self.SCALER_PATH)
            print("AI model loaded.")
        else:
            print("Training AI on 20 years of SPY data...")
            self.train_model()
            print("AI training complete.")

    def extract_features(self, df):
        close = df['Close']
        volume = df['Volume']
        f = pd.DataFrame(index=df.index)

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        f['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_fast = close.ewm(span=12).mean()
        ema_slow = close.ewm(span=26).mean()
        f['macd'] = ema_fast - ema_slow
        f['macd_sig'] = f['macd'].ewm(span=9).mean()

        # Bollinger
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        f['bb_upper'] = sma + 2 * std
        f['bb_lower'] = sma - 2 * std

        # Volume
        volume_sma = volume.rolling(20).mean()
        f['volume_ratio'] = volume / volume_sma

        # Gap
        f['gap'] = (df['Open'] - close.shift(1)) / close.shift(1)

        # ATR
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - close.shift()).abs(),
            (df['Low'] - close.shift()).abs()
        ], axis=1).max(axis=1)
        f['atr'] = tr.rolling(14).mean()

        # Volatility
        f['volatility'] = close.pct_change().rolling(5).std()

        return f.dropna()

    def train_model(self):
        spy = yf.download("SPY", period="20y", interval="1d", auto_adjust=True)
        features = self.extract_features(spy)

        # Future return: 5 days ahead
        future_close = spy['Close'].shift(-5)
        labels = ((future_close - spy['Close']) / spy['Close'] > 0.02).astype(int)

        # Align: Drop last 5 rows of features
        features_aligned = features.iloc[:-5]
        labels_aligned = labels.iloc[:-5]

        # Final alignment
        combined = pd.concat([features_aligned, labels_aligned], axis=1).dropna()
        X = combined.iloc[:, :-1]
        y = combined.iloc[:, -1]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
        self.model.fit(X_scaled, y)

        joblib.dump(self.model, self.MODEL_PATH)
        joblib.dump(self.scaler, self.SCALER_PATH)

    def predict_signal(self):
        try:
            df = yf.download("SPY", period="60d", interval="1d", auto_adjust=True)
            features = self.extract_features(df).iloc[-1:]
            if features.empty:
                return False, False, 0.0
            X = self.scaler.transform(features)
            prob_up = self.model.predict_proba(X)[0][1]
            prob_down = 1 - prob_up
            signal_up = prob_up > 0.70
            signal_down = prob_down > 0.70
            return signal_up, signal_down, prob_up
        except Exception as e:
            logging.error(f"AI prediction error: {e}")
            return False, False, 0.0

# ---------- OPTIONS TRADER ----------
class OptionsTrader:
    def __init__(self, api):
        self.api = api

    def place_call(self, confidence, equity):
        try:
            spy_price = float(self.api.get_last_trade('SPY').price)
            strike = round(spy_price * 1.01 / 5) * 5
            expiry = datetime.now() + timedelta(days=3)
            if expiry.weekday() != 2:
                expiry += timedelta(days=(2 - expiry.weekday()) % 7)
            symbol = f"SPY{expiry.strftime('%y%m%d')}C{int(strike)}000"
            contracts = int((equity * 0.01 * confidence) // (spy_price * 100))
            if contracts > 0:
                self.api.submit_order(symbol=symbol, qty=contracts, side='buy', type='market', time_in_force='day')
                logging.info(f"AI CALL: {contracts} {symbol}")
                return True
        except Exception as e:
            logging.error(f"Call trade failed: {e}")
        return False

class ShortPutTrader:
    def __init__(self, api):
        self.api = api

    def place_put(self, confidence, equity):
        try:
            spy_price = float(self.api.get_last_trade('SPY').price)
            strike = round(spy_price * 0.99 / 5) * 5
            expiry = datetime.now() + timedelta(days=3)
            if expiry.weekday() != 2:
                expiry += timedelta(days=(2 - expiry.weekday()) % 7)
            symbol = f"SPY{expiry.strftime('%y%m%d')}P{int(strike)}000"
            contracts = int((equity * 0.01 * confidence) // (spy_price * 100))
            if contracts > 0:
                self.api.submit_order(symbol=symbol, qty=contracts, side='sell', type='market', time_in_force='day')
                logging.info(f"AI SHORT PUT: {contracts} {symbol}")
                return True
        except Exception as e:
            logging.error(f"Short put failed: {e}")
        return False

# ---------- REAL SIGNALS ----------
def air_force_alert():
    try:
        url = "https://opensky-network.org/api/states/all"
        params = {'lamin': 27.8, 'lamax': 27.9, 'lomin': -82.6, 'lomax': -82.4, 'time': int(datetime.now().timestamp()) - 3600}
        auth = (OPENSKY_USERNAME, OPENSKY_PASSWORD) if OPENSKY_USERNAME else None
        r = requests.get(url, params=params, auth=auth, timeout=15)
        if r.status_code != 200: return False
        tankers = [s for s in r.json().get('states', []) if s[1] and any(m in s[1] for m in ['KC-135', 'KC-46'])]
        if len(tankers) >= 5:
            logging.info(f"AIR FORCE: {len(tankers)} tankers")
            return True
        return False
    except Exception as e:
        logging.error(f"OpenSky error: {e}")
        return False

def oil_draw_alert():
    try:
        url = "https://api.eia.gov/v2/petroleum/stk/wcrst/data/"
        params = {'frequency': 'weekly', 'data[0]': 'value', 'api_key': EIA_API_KEY, 'sort[0][column]': 'period', 'sort[0][direction]': 'desc', 'offset': 0, 'length': 2}
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200: return False
        data = r.json().get('response', {}).get('data', [])
        if len(data) < 2: return False
        draw = float(data[1]['value']) - float(data[0]['value'])
        if draw > 3_000_000:
            logging.info(f"OIL DRAW: {draw:,.0f} bbl")
            return True
        return False
    except Exception as e:
        logging.error(f"EIA error: {e}")
        return False

def dod_contract_alert():
    try:
        r = requests.get("https://www.defense.gov/rss/Contracts.xml", timeout=10)
        root = ET.fromstring(r.content)
        today = datetime.now().strftime('%Y-%m-%d')
        for item in root.findall('.//item'):
            pub = item.find('pubDate').text or ''
            title = item.find('title').text or ''
            desc = item.find('description').text or ''
            if today in pub and ('Lockheed' in title or 'Boeing' in title):
                m = re.search(r'\$([\d,]+)\s*m', desc.lower())
                if m and float(m.group(1).replace(',', '')) >= 50:
                    logging.info(f"DoD ≥$50M: {title}")
                    return True
        return False
    except Exception as e:
        logging.error(f"DoD RSS error: {e}")
        return False

def vix_crush_alert():
    try:
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        today = datetime.now().strftime('%Y-%m-%d')
        url = f"https://api.polygon.io/v2/aggs/ticker/X:VIX/range/1/day/{yesterday}/{today}"
        r = requests.get(url, params={'adjusted': 'true', 'apiKey': POLYGON_KEY}, timeout=10)
        data = r.json()
        if data.get('status') != 'OK' or not data.get('results'): return False
        closes = [bar['c'] for bar in data['results']]
        if len(closes) < 2: return False
        change = (closes[-1] - closes[0]) / closes[0] * 100
        if change < -5:
            logging.info(f"VIX CRUSH: {change:.2f}%")
            return True
        return False
    except Exception as e:
        logging.error(f"Polygon error: {e}")
        return False

# ---------- SIGNAL LOGIC ----------
def check_signal():
    af = air_force_alert()
    oil = oil_draw_alert()
    dod = dod_contract_alert()
    vix = vix_crush_alert()

    if (af and oil) or (dod and oil) or (vix and oil):
        source = "AirForce+Oil" if af and oil else "DoD+Oil" if dod and oil else "VIX+Oil"
        logging.info(f"LONG SIGNAL: {source}")
        return True, source
    return False, None

# ---------- EXECUTE LONG ----------
def execute_long(source):
    try:
        account = api.get_account()
        equity = float(account.cash)
        risk = equity * 0.02
        symbols = ['LMT', 'BA', 'USO']
        per_sym = risk / len(symbols)
        for sym in symbols:
            price = float(api.get_last_trade(sym).price)
            qty = int(per_sym // price)
            if qty > 0:
                api.submit_order(symbol=sym, qty=qty, side='buy', type='market', time_in_force='day')
                logging.info(f"BOUGHT {qty} {sym}")
        if TELEGRAM_READY:
            bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=f"ChainShot v7.3 LONG\nLMT/BA/USO\nTrigger: {source}\nRisk: 2%"
            )
    except Exception as e:
        logging.error(f"Long trade failed: {e}")

# ---------- MAIN LOOP ----------
if __name__ == "__main__":
    logging.info("ChainShot v7.3 LIVE — Monitoring 9:30–16:00 ET")

    ai_engine = OptionsAIEngine() if AI_OPTIONS_ENABLED else None
    options_trader = OptionsTrader(api) if AI_OPTIONS_ENABLED else None
    short_put_trader = ShortPutTrader(api) if AI_OPTIONS_ENABLED else None

    while True:
        now = datetime.now()
        if now.weekday() < 5 and 9 <= now.hour < 16:
            triggered, src = check_signal()
            if triggered:
                execute_long(src)
                time.sleep(3600)

            if AI_OPTIONS_ENABLED and ai_engine:
                signal_up, signal_down, prob_up = ai_engine.predict_signal()
                account = api.get_account()
                equity = float(account.cash)

                if signal_up:
                    options_trader.place_call(prob_up, equity)
                    if TELEGRAM_READY:
                        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"AI CALL\nConfidence: {prob_up:.1%}\nRisk: 1%")
                    time.sleep(300)

                if signal_down:
                    short_put_trader.place_put(1 - prob_up, equity)
                    if TELEGRAM_READY:
                        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"AI SHORT PUT\nConfidence: {1-prob_up:.1%}\nRisk: 1%")
                    time.sleep(300)

        time.sleep(300)
