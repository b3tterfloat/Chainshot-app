from flask import Flask, render_template
import os
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def dashboard():
    try:
        with open('chainshot.log', 'r') as f:
            lines = f.readlines()[-50:]
        log = ''.join(lines)
        
        # Real equity from Alpaca
        from trader import api
        account = api.get_account()
        equity = f"${float(account.cash):,.0f}"
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return render_template('dashboard.html', log=log, equity=equity, now=now)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
