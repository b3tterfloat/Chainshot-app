# backtest_ai.py
from ai_engine import OptionsAIEngine
import yfinance as yf

engine = OptionsAIEngine()
df = yf.download("SPY", period="2y", interval="1d")
features = engine.extract_features(df)
labels = engine.generate_labels(df)

# Accuracy
preds = engine.model.predict(engine.scaler.transform(features.iloc[:-5]))
acc = (preds == labels.iloc[:-5]).mean()
print(f"AI Accuracy: {acc:.1%}")
