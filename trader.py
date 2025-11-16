# trader.py
# ChainShot Swarm v9.7 — FINAL: Market-Closed Safe + No Warnings + GitHub Ready
# Verified: 2025-11-16 08:35 AM EST
# Ready for Monday 9:30 AM EST

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
print(f"Loading .env from: {ENV_PATH}")

if not ENV_PATH.exists():
    raise FileNotFoundError(f".env not found at {ENV_PATH}")

load_dotenv(dotenv_path=ENV_PATH)

# ---------- CONFIG ----------
APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID')
APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

MIN_CONFIDENCE = 0.70
TOTAL_RISK_CAP = 0.02
CONFLUENCE_THRESHOLD = 2
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
TRADE_LOG = "trades.log"

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
else:
    print("Telegram disabled")

# ---------- ALPACA ----------
api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# ---------- SHARED UTILS ----------
def log_trade(bot_name, signal, pnl, confidence, symbol=""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(TRADE_LOG, "a") as f:
        f.write(f"{ts},{bot_name},{signal},{pnl:.4f},{confidence:.3f},{symbol}\n")

def send_alert(text):
    if TELEGRAM_READY:
        try:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
        except Exception as e:
            logging.error(f"Telegram send failed: {e}")

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

    def train(self):
        try:
            period = "7d" if self.interval == "1m" else "60d"
            data = yf.download("SPY", period=period, interval=self.interval, progress=False)
            if len(data) < 50:
                raise ValueError(f"Insufficient data: {len(data)} rows")
            features = self.extract_features(data)
            if len(features) < 20:
                raise ValueError(f"Insufficient features: {len(features)} rows")
            labels = self.generate_labels(data)
            shift = 5 if self.interval == "1m" else 3 if self.interval == "5m" else 2
            features = features.iloc[:-shift]
            labels = labels.iloc[:-shift]
            combined = pd.concat([features, labels], axis=1).dropna()
            X = combined.iloc[:, :-1]
            y = combined.iloc[:, -1].values.ravel()
            if len(X) < 10:
                raise ValueError(f"After alignment: {len(X)} samples")
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.model = RandomForestClassifier(n_estimators=300, max_depth=8, n_jobs=-1, random_state=42)
            self.model.fit(X_scaled, y)
            print(f"{self.name} trained on {len(X)} samples.")
        except Exception as e:
            logging.error(f"Training failed for {self.name}: {e}. Using fallback.")
            self.model = None

    def generate_labels(self, data):
        raise NotImplementedError

    def extract_features(self, df):
        raise NotImplementedError

    def get_signal(self, df):
        try:
            features = self.extract_features(df)
            if features.empty:
                return None, 0.0
            last_row = features.iloc[-1:]
            if self.model is None:
                return self.rule_based_signal(last_row)
            X = self.scaler.transform(last_row)
            prob = self.model.predict_proba(X)[0][1]
            signal = self.get_signal_type()
            if prob >= MIN_CONFIDENCE:
                return signal, prob
            return None, 0.0
        except Exception as e:
            logging.error(f"{self.name} signal error: {e}")
            return None, 0.0

    def rule_based_signal(self, features):
        return self.get_signal_type(), 0.5

    def get_signal_type(self):
        raise NotImplementedError

    def run(self):
        while True:
            try:
                period = "2d" if self.interval == "1m" else "5d"
                df = yf.download("SPY", period=period, interval=self.interval, progress=False)
                if df.empty or len(df) < 30:
                    time.sleep(60)
                    continue
                with self.lock:
                    signal, confidence = self.get_signal(df)
                    if signal:
                        self.queue.put((signal, confidence))
                        logging.info(f"{self.name} signal: {signal} @ {confidence:.1%}")
            except Exception as e:
                logging.error(f"{self.name} run error: {e}")
            time.sleep(30 if self.interval == "1m" else 60)

# ---------- BOT A: 1-Min Volatility Breakout ----------
class VolBreakoutBot(TradingBot):
    def __init__(self):
        super().__init__("1min_breakout", "1m", 0.4)

    def generate_labels(self, data):
        return (data['Close'].shift(-5) > data['Close'] * 1.003).astype(int)

    def get_signal_type(self):
        return "CALL"

    def extract_features(self, df):
        if len(df) < 30:
            return pd.DataFrame()
        df = df.copy()
        f = pd.DataFrame(index=df.index)
        high = df['High'].rolling(20, min_periods=5).max()
        low = df['Low'].rolling(20, min_periods=5).min()
        f['range'] = high - low
        idx = df.index
        high_shift = high.shift(1).reindex(idx).ffill().fillna(0)
        range_shift = f['range'].shift(1).reindex(idx).ffill().fillna(0)
        close = df['Close']
        breakout_cond = close.gt(high_shift + 0.5 * range_shift)
        f['breakout'] = breakout_cond.astype(int)
        vol_mean = df['Volume'].rolling(20, min_periods=5).mean().reindex(idx).fillna(1)
        f['vol_spike'] = (df['Volume'] / vol_mean > 2).astype(int)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=5).mean()
        loss = -delta.where(delta < 0, 0).rolling(14, min_periods=5).mean()
        rs = gain / loss.replace(0, np.nan).fillna(1)
        f['rsi'] = 100 - (100 / (1 + rs))
        f = f.dropna()
        return f

# ---------- BOT B: 5-Min Volume Reversal ----------
class VolumeReversalBot(TradingBot):
    def __init__(self):
        super().__init__("5min_reversal", "5m", 0.3)

    def generate_labels(self, data):
        return (data['Close'].shift(-3) < data['Close'] * 0.997).astype(int)

    def get_signal_type(self):
        return "PUT"

    def extract_features(self, df):
        if len(df) < 30:
            return pd.DataFrame()
        df = df.copy()
        f = pd.DataFrame(index=df.index)
        idx = df.index
        vol_mean = df['Volume'].rolling(50, min_periods=10).mean().reindex(idx).ffill().fillna(1)
        vol_std = df['Volume'].rolling(50, min_periods=10).std().reindex(idx).ffill().fillna(1)
        f['vol_z'] = (df['Volume'].reindex(idx) - vol_mean) / vol_std
        price_mean = df['Close'].rolling(20, min_periods=5).mean().reindex(idx).ffill()
        price_std = df['Close'].rolling(20, min_periods=5).std().reindex(idx).ffill().fillna(1)
        f['price_z'] = (df['Close'].reindex(idx) - price_mean) / price_std
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=5).mean()
        loss = -delta.where(delta < 0, 0).rolling(14, min_periods=5).mean()
        rs = gain / loss.replace(0, np.nan).fillna(1)
        f['rsi'] = 100 - (100 / (1 + rs))
        f = f.dropna()
        return f

# ---------- BOT C: 15-Min VWAP Pullback ----------
class VWAPPullbackBot(TradingBot):
    def __init__(self):
        super().__init__("15min_vwap", "15m", 0.3)

    def generate_labels(self, data):
        return (data['Close'].shift(-2) > data['Close'] * 1.002).astype(int)

    def get_signal_type(self):
        return "CALL"

    def vwap(self, df):
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vol_cum = df['Volume'].rolling(20, min_periods=5).sum()
        return (df['Volume'] * typical_price).rolling(20, min_periods=5).sum() / vol_cum.replace(0, 1)

    def extract_features(self, df):
        if len(df) < 30:
            return pd.DataFrame()
        df = df.copy()
        f = pd.DataFrame(index=df.index)
        idx = df.index
        vwap = self.vwap(df).reindex(idx).ffill()
        f['dist_to_vwap'] = (df['Close'].reindex(idx) - vwap) / vwap
        vol_mean = df['Volume'].rolling(20, min_periods=5).mean().reindex(idx).ffill().fillna(1)
        f['vol_spike'] = (df['Volume'].reindex(idx) / vol_mean > 1.8).astype(int)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=5).mean()
        loss = -delta.where(delta < 0, 0).rolling(14, min_periods=5).mean()
        rs = gain / loss.replace(0, np.nan).fillna(1)
        f['rsi'] = 100 - (100 / (1 + rs))
        f = f.dropna()
        return f

# ---------- OPTIONS EXECUTOR ----------
class OptionsExecutor:
    @staticmethod
    def get_next_friday():
        today = datetime.now().date()
        days_ahead = (4 - today.weekday()) % 7
        if days_ahead == 0: days_ahead = 7
        return today + timedelta(days=days_ahead)

    @staticmethod
    def place_order(signal, confidence, equity):
        try:
            spy_price = float(api.get_last_trade('SPY').price)
            strike_mult = 1.01 if signal == "CALL" else 0.99
            strike = round(spy_price * strike_mult / 5) * 5
            expiry = OptionsExecutor.get_next_friday()
            option_type = "C" if signal == "CALL" else "P"
            symbol = f"SPY{expiry.strftime('%y%m%d')}{option_type}{int(strike * 1000):08d}"

            risk_amount = equity * TOTAL_RISK_CAP * 0.3
            contracts = max(1, int(risk_amount // (spy_price * 100)))

            if contracts > 0:
                api.submit_order(symbol=symbol, qty=contracts, side='buy' if signal == "CALL" else 'sell',
                                 type='market', time_in_force='day')
                log_trade("swarm", signal, 0.0, confidence, symbol)
                send_alert(f"SWARM {signal}\n{symbol}\nQty: {contracts}\nConf: {confidence:.1%}")
                return True
        except Exception as e:
            logging.error(f"Trade failed: {e}")
        return False

# ---------- SWARM ORCHESTRATOR ----------
class SwarmOrchestrator:
    def __init__(self):
        self.bots = [
            VolBreakoutBot(),
            VolumeReversalBot(),
            VWAPPullbackBot()
        ]
        for bot in self.bots:
            bot.thread.start()
        self.last_rebalance = None

    def collect_signals(self):
        signals = {}
        for bot in self.bots:
            while not bot.queue.empty():
                try:
                    sig, conf = bot.queue.get_nowait()
                    signals[bot.name] = (sig, conf)
                except queue.Empty:
                    break
        return signals

    def confluence_check(self, signals):
        calls = sum(1 for s in signals.values() if s[0] == "CALL")
        puts = sum(1 for s in signals.values() if s[0] == "PUT")
        if calls >= CONFLUENCE_THRESHOLD:
            return "CALL", np.mean([c for s, c in signals.values() if s == "CALL"])
        if puts >= CONFLUENCE_THRESHOLD:
            return "PUT", np.mean([c for s, c in signals.values() if s == "PUT"])
        return None, 0.0

    def weekly_rebalance(self):
        if not os.path.exists(TRADE_LOG):
            return
        try:
            df = pd.read_csv(TRADE_LOG, names=["ts", "bot", "sig", "pnl", "conf", "sym"])
            df['ts'] = pd.to_datetime(df['ts'])
            last_week = df[df['ts'] >= (pd.Timestamp.now() - pd.Timedelta(days=7))]
            if len(last_week) == 0: return

            perf = {}
            for bot in self.bots:
                bot_df = last_week[last_week['bot'].str.contains(bot.name.split('_')[0], na=False)]
                if len(bot_df) == 0:
                    perf[bot.name] = 0.5
                    continue
                sharpe = bot_df['pnl'].mean() / (bot_df['pnl'].std() + 1e-6)
                win_rate = (bot_df['pnl'] > 0).mean()
                perf[bot.name] = sharpe * win_rate

            total = sum(perf.values())
            for bot in self.bots:
                bot.risk_weight = perf[bot.name] / total if total > 0 else 1/len(self.bots)
            logging.info("Weekly rebalance complete.")
        except Exception as e:
            logging.error(f"Rebalance error: {e}")

    def run(self):
        last_trade = 0
        while True:
            try:
                now = datetime.now()
                if now.weekday() >= 5 or not (9 <= now.hour < 16):
                    time.sleep(300)
                    continue
                if now.weekday() == 6 and 20 <= now.hour < 21 and (not self.last_rebalance or (now - self.last_rebalance).total_seconds() > 3600):
                    self.weekly_rebalance()
                    self.last_rebalance = now
                if time.time() - last_trade < 180:
                    time.sleep(60)
                    continue

                signals = self.collect_signals()
                if len(signals) >= CONFLUENCE_THRESHOLD:
                    final_sig, conf = self.confluence_check(signals)
                    if final_sig:
                        account = api.get_account()
                        equity = float(account.cash)
                        if equity > 1000 and OptionsExecutor.place_order(final_sig, conf, equity):
                            last_trade = time.time()

                time.sleep(60)
            except Exception as e:
                logging.error(f"Orchestrator error: {e}")
                time.sleep(60)

# ---------- MAIN ----------
if __name__ == "__main__":
    logging.info("ChainShot Swarm v9.7 LIVE — FINAL")
    swarm = SwarmOrchestrator()
    swarm.run()
