kfrom flask import Flask
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
                recent = "<pre style='background:#111;color:#0f0;padding:10px;border-radius:5px;'>" + "".join(lines) + "</pre>"
        except:
            recent = "<p>Log read error</p>"

    return f"""
    <meta http-equiv="refresh" content="300">
    <h1 style="color:#00ff00; font-family:monospace;">CHAIN SHOT v5.5 — CLOUD LIVE</h1>
    <h2>Status: <span style="color:gold;">AUTO PAPER TRADING</span></h2>
    <p>Market Hours: 9:30 AM – 4:00 PM ET</p>
    <p>Trader: <b>Cron Job (every 15 mins)</b></p>
    <p>Alpaca: Paper Mode</p>
    <h3>Recent Trades:</h3>
    {recent}
    <p><i>Auto-refresh in 5 min</i></p>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
