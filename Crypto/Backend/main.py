import os
import base64
from io import BytesIO

from flask import Flask, render_template, request
import mysql.connector as ms
import pandas as pd
import matplotlib.pyplot as plt

# 🔹 Import loader + trainer
from utils.loader import download_dataset, load_coin
from utils.model_trainer import run_all_predictions

# -----------------------------------
# Initialize MySQL connection
# -----------------------------------
conn = ms.connect(
    host="localhost",
    port=3306,
    user="root",
    passwd="Rudrag11!",
    database="crypto"
)

if conn.is_connected():
    print("✅ MySQL connection established.")
else:
    print("❌ Failed to connect to MySQL.")

mc = conn.cursor()

# -----------------------------------
# Initialize Flask app
# -----------------------------------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_PATH, "../frontend/templates"),
    static_folder=os.path.join(BASE_PATH, "../frontend/static")
)


# Supported crypto symbols
crypto_symbols = ["ADA", "BTC", "BNB", "DOGE", "ETH"]

# Global cache for predictions
PREDICTIONS_CACHE = {}

# -----------------------------------
# RSI calculation
# -----------------------------------
def calculate_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# -----------------------------------
# Flask routes
# -----------------------------------
@app.route("/")
def main_page():
    return render_template("landing.html")

@app.route("/signup")
def signup_page():
    return render_template("get_started.html")

@app.route("/dashboard", methods=["POST"])
def enter_details():
    uname = request.form.get("username")
    passwd = request.form.get("password")
    email = request.form.get("email")

    if not uname or not passwd or not email:
        return render_template("get_started.html", err="All fields are required.")

    mc.execute("SELECT uname FROM users WHERE uname=%s", (uname,))
    result = mc.fetchall()
    conn.commit()

    if result:
        return render_template("get_started.html", err="Username already exists")
    else:
        mc.execute(
            "INSERT INTO users (uname, passwd, email) VALUES (%s, %s, %s)",
            (uname, passwd, email),
        )
        conn.commit()
        return render_template("option.html", result=result)

@app.route("/start/login")
def login_page():
    return render_template("login.html")

@app.route("/start/dashboard", methods=["POST"])
def dashboard_page():
    global uname
    uname = request.form["username"]
    passwd = request.form["password"]

    mc.execute("SELECT * FROM users WHERE uname=%s AND passwd=%s", (uname, passwd))
    result = mc.fetchall()
    conn.commit()

    if result:
        return render_template("option.html", result=result)
    else:
        return render_template("login.html", err="Invalid username or password!")

# -------------------------------
# Prediction route
# -------------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction, error = None, ""

    if request.method == "POST":
        symbol = request.form.get("symbol", "").replace("-USD", "").upper()

        if symbol not in crypto_symbols:
            error = f"Symbol not supported. Use: {', '.join(crypto_symbols)}"
        else:
            prediction = PREDICTIONS_CACHE.get(symbol)
            if not prediction:
                error = "Prediction not available. Try again later."

    return render_template("predict.html", error=error, prediction=prediction)

# -------------------------------
# View crypto data + RSI + chart
# -------------------------------
@app.route("/view-crypto", methods=["GET", "POST"])
def view():
    if request.method == "POST":
        coin = request.form.get("coin", "").upper()

        if coin not in crypto_symbols:
            return render_template(
                "view.html", error=f"Coin not supported. Use: {', '.join(crypto_symbols)}"
            )

        try:
            df = load_coin(coin)
        except FileNotFoundError:
            return render_template("view.html", error="Data for this coin is not available.")

        # Ensure correct columns
        if "date" not in df.columns and "Date" in df.columns:
            df.rename(columns={"Date": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        if "close" in df.columns:
            df.rename(columns={"close": "Close"}, inplace=True)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

        df.dropna(subset=["date", "Close"], inplace=True)

        # RSI
        df["RSI"] = calculate_rsi(df)
        latest_rsi = df["RSI"].iloc[-1]

        # Closing price chart
        plt.figure(figsize=(10, 5))
        plt.plot(df["date"], df["Close"], label=f"{coin} Closing Price", color="blue")
        plt.title(f"{coin} Closing Prices Over Time")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid()

        img = BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        chart_base64 = base64.b64encode(img.getvalue()).decode("utf-8")
        plt.close()

        return render_template(
            "view.html", chart=chart_base64, rsi=round(latest_rsi, 2), coin=coin
        )

    return render_template("view.html")

@app.route("/home")
def home():
    return render_template("option.html")

# -----------------------------------
# Run app
# -----------------------------------
if __name__ == "__main__":
    # Only run once, avoid double training with Flask reloader
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print("🔄 Downloading datasets & training models...")
        download_dataset()
        PREDICTIONS_CACHE = run_all_predictions()   # save results in memory
        print(PREDICTIONS_CACHE)
        print("✅ Predictions ready.")

    app.run(host="0.0.0.0", port=5001, debug=True)
