# ai_engine.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import yfinance as yf

MODEL_PATH = "models/spy_options_model.pkl"
SCALER_PATH = "models/scaler.pkl"

class OptionsAIEngine:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_or_train()

    def load_or_train(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            print("AI model loaded.")
        else:
            print("Training AI on 20 years of SPY data...")
            self.train_model()
            print("AI training complete.")

    def extract_features(self, df):
        """Engineer 50+ features from OHLCV"""
        f = pd.DataFrame()
        f['rsi'] = self.rsi(df['Close'])
        f['macd'], f['macd_sig'] = self.macd(df['Close'])
        f['bb_upper'], f['bb_lower'] = self.bollinger(df['Close'])
        f['volatility'] = df['Close'].pct_change().rolling(5).std()
        f['volume_sma'] = df['Volume'].rolling(20).mean()
        f['volume_ratio'] = df['Volume'] / f['volume_sma']
        f['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        f['atr'] = self.atr(df)
        f = f.dropna()
        return f

    def generate_labels(self, df, horizon=5, threshold=0.02):
        """Label: 1 if SPY up >2% in 5 days, else 0"""
        future = df['Close'].shift(-horizon)
        label = (future - df['Close']) / df['Close'] > threshold
        return label.astype(int)

    def train_model(self):
        os.makedirs("models", exist_ok=True)
        spy = yf.download("SPY", period="20y", interval="1d")
        features = self.extract_features(spy)
        labels = self.generate_labels(spy)

        X = features.iloc[:-5]
        y = labels.iloc[:-5]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
        self.model.fit(X_scaled, y)

        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)

    def predict_signal(self):
        """Real-time prediction"""
        try:
            df = yf.download("SPY", period="60d", interval="1d")
            features = self.extract_features(df).iloc[-1:]
            X = self.scaler.transform(features)
            prob = self.model.predict_proba(X)[0][1]
            signal = prob > 0.70  # 70%+ confidence
            return signal, prob
        except:
            return False, 0.0

    # Technical indicators
    def rsi(self, series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def macd(self, series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_sig = macd.ewm(span=signal).mean()
        return macd, macd_sig

    def bollinger(self, series, window=20, std=2):
        sma = series.rolling(window).mean()
        std_dev = series.rolling(window).std()
        return sma + std_dev * std, sma - std_dev * std

    def atr(self, df, window=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window).mean()
