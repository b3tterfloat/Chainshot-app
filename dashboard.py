from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def dashboard():
    log_path = "trade_log.txt"
    recent = "<p>No trades yet — waiting for signal</p>"
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()[-10:]
                recent = "<pre style='background:#111;color:#0f0;padding:10px;border-radius:5px;font-family:monospace;'>" + "".join(lines) + "</pre>"
        except Exception as e:
            recent = f"<p>Log error: {e}</p>"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ChainShot v5.5</title>
        <meta http-equiv="refresh" content="300">
        <style>
            body {{ font-family: 'Courier New', monospace; background: #000; color: #0f0; padding: 20px; }}
            pre {{ background: #111; padding: 15px; border-radius: 8px; overflow-x: auto; }}
            h1 {{ color: #00ff00; }}
            h2 {{ color: gold; }}
        </style>
    </head>
    <body>
        <h1>CHAIN SHOT v5.5 — CLOUD LIVE</h1>
        <h2>Status: <span style="color:gold;">AUTO PAPER TRADING</span></h2>
        <p><strong>Market Hours:</strong> 9:30 AM – 4:00 PM ET</p>
        <p><strong>Trader:</strong> Render Cron Job (every 15 mins)</p>
        <p><strong>Alpaca:</strong> Paper Mode</p>
        <hr>
        <h3>Recent Activity (trade_log.txt):</h3>
        {recent}
        <p><i>Page auto-refreshes every 5 minutes</i></p>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
