from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def dashboard():
    log_path = "/Users/brockhenley/chainshot/trade_log.txt"
    recent = ""
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            lines = f.readlines()[-5:]
            recent = "<pre>" + "".join(lines) + "</pre>"
    
    return f"""
    <h1 style="color:#00ff00; font-family:monospace;">CHAIN SHOT v5.5 — CLOUD LIVE</h1>
    <h2>Status: <span style="color:gold;">AUTO PAPER TRADING</span></h2>
    <p>Market Hours: 9:30 AM – 4:00 PM ET</p>
    <p>Cron: Every 15 mins</p>
    <p>Alpaca: Paper Mode</p>
    <h3>Recent Activity:</h3>
    {recent or "<p>No trades yet — waiting for signal</p>"}
    <p><i>Next check in <span id="c">15:00</span></i></p>
    <script>
        let s=900; setInterval(()=>{s--; let m=Math.floor(s/60); let sec=s%60;
        document.getElementById('c').innerText=m+':'+(sec<10?'0':'')+sec;
        if(s<=0) location.reload();},1000);
    </script>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
