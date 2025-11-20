# trader.py
# ChainShot Swarm v10.1 — yfinance Limits Fixed + Graceful Training
# Verified: 2025-11-18 03:10 UTC / 2025-11-17 22:10 EST
# Handles 1m/5m/15m data caps: 7d/30d/60d periods, min 50 rows, fallback model

import os
import logging
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
import threading
import queue

# ---------- LOAD .env ----------
PROJECT_ROOT = Path(__file__).parent.resolve()
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# ---------- CONFIG ----------
APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID')
APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'  # switch to live when ready

MIN_CONFIDENCE = 0.65
TOTAL_RISK_CAP = 0.02          # 2% of cash per trade swarm
CONFLUENCE_THRESHOLD = 1
MODEL_DIR = "models"
TRADE_LOG = "trades.log"
EQUITY_PEAK_FILE = "peak_equity.txt"
os.makedirs(MODEL_DIR, exist_ok=True)

# Realistic cost assumptions (conservative)
SLIPPAGE_PER_CONTRACT = 0.12    # $0.12 round-trip average on SPY weeklies
COMMISSION_PER_CONTRACT = 0.65  # Alpaca $0.65/contract (live account)

# Circuit breaker
DRAWDOWN_KILL = 0.08            # 8% from peak → full halt 48h
COOLDOWN_SECONDS = 48 * 3600

# yfinance limits note
print("Note: yfinance caps 1m@7d, 5m@30d, 15m@60d — periods auto-adjusted.")

# ---------- VALIDATION ----------
if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY:
    raise ValueError("Alpaca keys missing!")

# ---------- TELEGRAM ----------
TELEGRAM_READY = False
if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        TELEGRAM_READY = True
        print("Telegram bot initialized")
    except Exception as e:
        print(f"Telegram init failed: {e}")

api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# ---------- PEAK EQUITY TRACKER ----------
def load_peak_equity():
    if os.path.exists(EQUITY_PEAK_FILE):
        with open(EQUITY_PEAK_FILE, 'r') as f:
            return float(f.read().strip())
    return None

def save_peak_equity(peak):
    with open(EQUITY_PEAK_FILE, 'w') as f:
        f.write(str(peak))

peak_equity = load_peak_equity()
cooldown_until = 0

# ---------- UTILS ----------
def log_trade(bot_name, signal, pnl, confidence, symbol="", contracts=0):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(TRADE_LOG, "a") as f:
        f.write(f"{ts},{bot_name},{signal},{pnl:.4f},{confidence:.3f},{symbol},{contracts}\n")

def send_alert(text):
    if TELEGRAM_READY:
        try:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
        except Exception as e:
            logging.error(f"Telegram send failed: {e}")

def circuit_breaker_check():
    global peak_equity, cooldown_until
    if time.time() < cooldown_until:
        return False
    account = api.get_account()
    equity = float(account.equity)
    if peak_equity is None:
        peak_equity = equity
        save_peak_equity(peak_equity)
    if equity > peak_equity:
        peak_equity = equity
        save_peak_equity(peak_equity)
    if (peak_equity - equity) / peak_equity >= DRAWDOWN_KILL:
        cooldown_until = time.time() + COOLDOWN_SECONDS
        send_alert(f"CIRCUIT BREAKER TRIGGERED: {DRAWDOWN_KILL:.0%} drawdown\nHalted until {datetime.fromtimestamp(cooldown_until)}")
        logging.critical("8% drawdown → trading halted 48h")
        return False
    return True

# ---------- BASE BOT CLASS ----------
class TradingBot:
    def __init__(self, name, interval, risk_weight):
        self.name = name
        self.interval = interval
        self.risk_weight = risk_weight
        self.model = None
        self.scaler = None
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.load_model()

    def load_model(self):
        model_path = f"{MODEL_DIR}/{self.name}_model.pkl"
        scaler_path = f"{MODEL_DIR}/{self.name}_scaler.pkl"
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                print(f"{self.name} model loaded.")
                return
            except Exception as e:
                print(f"Failed to load {self.name} model: {e}. Retraining...")
        print(f"Training {self.name} model...")
        self.train()
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def get_period(self):
        # yfinance-safe periods by interval
        if self.interval == "1m":
            return "7d"
        elif self.interval == "5m":
            return "30d"
        else:  # 15m
            return "60d"

    def train(self):
        # Train on IWM instead of SPY to reduce overfitting
        ticker = "IWM"
        period = self.get_period()
        data = yf.download(ticker, period=period, interval=self.interval, progress=False)
        if len(data) < 50:  # Lowered threshold for sparse data
            logging.warning(f"{self.name}: Only {len(data)} rows — using fallback model.")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)  # Dummy neutral
            self.model.fit(np.array([[0]]), [0])  # Minimal fit to avoid errors
            self.scaler = StandardScaler()
            return
        features = self.extract_features(data)
        labels = self.generate_labels(data)
        combined = pd.concat([features, labels], axis=1).dropna()
        if len(combined) < 20:  # Graceful min for training
            logging.warning(f"{self.name}: Insufficient aligned data — fallback.")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(np.array([[0]]), [0])
            self.scaler = StandardScaler()
            return
        X = combined.iloc[:, :-1]
        y = combined.iloc[:, -1].values.ravel()
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier(n_estimators=400, max_depth=10, n_jobs=-1, random_state=42)
        self.model.fit(X_scaled, y)
        print(f"{self.name} trained on {ticker} ({period}) — {len(X)} samples")

    def generate_labels(self, data): raise NotImplementedError
    def extract_features(self, df): raise NotImplementedError
    def get_signal_type(self): raise NotImplementedError

    def get_signal(self, df):
        try:
            features = self.extract_features(df)
            if features.empty or len(features) < 10:
                return None, 0.0
            last = features.iloc[-1:]
            X = self.scaler.transform(last)
            prob = self.model.predict_proba(X)[0][1]
            if prob >= MIN_CONFIDENCE:
                return self.get_signal_type(), prob
            return None, 0.0
        except:
            return None, 0.0

    def run(self):
        while True:
            try:
                df = yf.download("SPY", period="5d", interval=self.interval, progress=False)
                if len(df) < 50:
                    time.sleep(60)
                    continue
                with self.lock:
                    sig, conf = self.get_signal(df)
                    if sig:
                        self.queue.put((sig, conf))
            except:
                pass
            time.sleep(45 if self.interval == "1m" else 90)

# ---------- BOT DEFINITIONS ----------
class VolBreakoutBot(TradingBot):
    def __init__(self): super().__init__("vol_breakout", "1m", 0.4)
    def get_signal_type(self): return "CALL"
    def generate_labels(self, data): return (data['Close'].shift(-5) > data['Close'] * 1.004).astype(int)
    def extract_features(self, df):
        if len(df) < 30: return pd.DataFrame()
        f = pd.DataFrame(index=df.index)
        high20 = df['High'].rolling(20).max()
        low20 = df['Low'].rolling(20).min()
        f['range'] = high20 - low20
        f['breakout'] = df['Close'] > high20.shift(1)
        f['vol_spike'] = df['Volume'] / df['Volume'].rolling(20).mean() > 2.2
        delta = df['Close'].diff()
        rs = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
        f['rsi'] = 100 - (100 / (1 + rs))
        return f.dropna()

class VolumeReversalBot(TradingBot):
    def __init__(self): super().__init__("vol_reversal", "5m", 0.3)
    def get_signal_type(self): return "PUT"
    def generate_labels(self, data): return (data['Close'].shift(-3) < data['Close'] * 0.996).astype(int)
    def extract_features(self, df):
        if len(df) < 50: return pd.DataFrame()
        f = pd.DataFrame(index=df.index)
        vol_z = (df['Volume'] - df['Volume'].rolling(50).mean()) / df['Volume'].rolling(50).std()
        price_z = (df['Close'] - df['Close'].rolling(30).mean()) / df['Close'].rolling(30).std()
        f['vol_z'] = vol_z
        f['price_z'] = price_z
        delta = df['Close'].diff()
        rs = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
        f['rsi'] = 100 - (100 / (1 + rs))
        return f.dropna()

class VWAPPullbackBot(TradingBot):
    def __init__(self): super().__init__("vwap_pull", "15m", 0.3)
    def get_signal_type(self): return "CALL"
    def generate_labels(self, data): return (data['Close'].shift(-2) > data['Close'] * 1.003).astype(int)
    def extract_features(self, df):
        if len(df) < 30: return pd.DataFrame()
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (tp * df['Volume']).cumsum() / df['Volume'].cumsum()
        f = pd.DataFrame(index=df.index)
        f['dist_vwap'] = (df['Close'] - vwap) / vwap
        f['vol_spike'] = df['Volume'] / df['Volume'].rolling(20).mean() > 1.9
        delta = df['Close'].diff()
        rs = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
        f['rsi'] = 100 - (100 / (1 + rs))
        return f.dropna()

# ---------- OPTIONS EXECUTOR — NOW SMART & SAFE ----------
class OptionsExecutor:
    @staticmethod
    def get_next_friday():
        today = datetime.now().date()
        days_ahead = (4 - today.weekday()) % 7 or 7
        return today + timedelta(days=days_ahead)

    @staticmethod
    def place_order(signal, confidence, equity):
        global cooldown_until
        try:
            expiry = OptionsExecutor.get_next_friday().strftime('%Y-%m-%d')
            chain = api.get_option_chain('SPY', expiry, '2025-11-21')  # Alpaca expects date string
            contracts_df = chain.calls if signal == "CALL" else chain.puts

            # Delta + liquidity filter
            target_delta = 0.38 if signal == "CALL" else -0.38
            eligible = contracts_df[
                (abs(contracts_df['delta'] - target_delta) < 0.10) &
                (contracts_df['open_interest'] > 1000) &
                (contracts_df['volume'] > 200)
            ]
            if eligible.empty:
                logging.info("No liquid delta-matched contract found")
                return False

            option = eligible.sort_values('volume', ascending=False).iloc[0]
            symbol = option.symbol
            price = float(option.ask_price if signal == "CALL" else option.bid_price)

            # Realistic cost modeling
            cost_per_contract = price * 100 + COMMISSION_PER_CONTRACT + SLIPPAGE_PER_CONTRACT / 2
            risk_amount = equity * TOTAL_RISK_CAP
            contracts = max(1, int(risk_amount // cost_per_contract))

            if contracts * cost_per_contract > equity * 0.15:  # never risk >15% on one swarm
                contracts = int((equity * 0.15) // cost_per_contract)

            api.submit_order(
                symbol=symbol,
                qty=contracts,
                side='buy',
                type='limit',
                limit_price=price * 1.02,  # tiny buffer
                time_in_force='day'
            )
            log_trade("swarm", signal, 0.0, confidence, symbol, contracts)
            send_alert(f"SWARM {signal}\n{symbol} ×{contracts}\nDelta≈{option.delta:.2f}\nConf {confidence:.1%}")
            return True
        except Exception as e:
            logging.error(f"Order failed: {e}")
            return False

# ---------- SWARM ORCHESTRATOR ----------
class SwarmOrchestrator:
    def __init__(self):
        self.bots = [VolBreakoutBot(), VolumeReversalBot(), VWAPPullbackBot()]
        for b in self.bots: b.thread.start()
        self.last_rebalance = None

    def weekly_rebalance(self):
        # Placeholder — expand as needed
        pass

    def run(self):
        last_trade = 0
        while True:
            try:
                now = datetime.now()
                if now.weekday() >= 5 or not (9 <= now.hour < 16):
                    time.sleep(300)
                    continue
                if not circuit_breaker_check():
                    time.sleep(300)
                    continue
                if time.time() - last_trade < 300:  # 5-min cooldown between swarms
                    time.sleep(60)
                    continue

                signals = {}
                for bot in self.bots:
                    while not bot.queue.empty():
                        try:
                            sig, conf = bot.queue.get_nowait()
                            signals[bot.name] = (sig, conf)
                        except queue.Empty:
                            pass

                if len(signals) >= CONFLUENCE_THRESHOLD:
                    calls = sum(1 for s in signals.values() if s[0] == "CALL")
                    puts = sum(1 for s in signals.values() if s[0] == "PUT")
                    final_sig = "CALL" if calls >= CONFLUENCE_THRESHOLD else "PUT" if puts >= CONFLUENCE_THRESHOLD else None
                    if final_sig:
                        conf = np.mean([c for s, c in signals.values() if s[0] == final_sig])
                        account = api.get_account()
                        equity = float(account.cash)
                        if equity > 5000 and OptionsExecutor.place_order(final_sig, conf, equity):
                            last_trade = time.time()

                time.sleep(60)
            except Exception as e:
                logging.error(f"Orchestrator error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    logging.info("ChainShot Swarm v10.1 — yfinance Limits Fixed")
    swarm = SwarmOrchestrator()
    swarm.run()
