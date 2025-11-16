# options_trader.py
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import logging

class OptionsTrader:
    def __init__(self, api):
        self.api = api

    def get_otm_call(self, strike_price, expiry):
        """Get next weekly OTM call"""
        symbol = f"SPY{expiry.strftime('%y%m%d')}C{int(strike_price)}"
        return symbol

    def place_call(self, signal_strength, equity):
        try:
            spy_price = float(self.api.get_last_trade('SPY').price)
            strike = round(spy_price * 1.01 / 5) * 5  # 1% OTM
            expiry = datetime.now() + timedelta(days=3)  # next Wednesday
            if expiry.weekday() != 2:
                expiry += timedelta(days=(2 - expiry.weekday()) % 7)

            symbol = f"SPY{expiry.strftime('%y%m%d')}C{int(strike)}000"
            contracts = int((equity * 0.01 * signal_strength) // (spy_price * 100))

            if contracts > 0:
                self.api.submit_order(
                    symbol=symbol,
                    qty=contracts,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                logging.info(f"AI CALL: {contracts} {symbol}")
                return True
        except Exception as e:
            logging.error(f"Options trade failed: {e}")
        return False
