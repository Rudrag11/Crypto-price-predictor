from flask import Flask, render_template, request
import mysql.connector as ms
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import timedelta

# 🔹 Import your loader package
from crypto_loader import download_dataset, load_coin

# -----------------------------------
# Initialize MySQL connection
# -----------------------------------
conn = ms.connect(host="localhost", port=3306, user="root", passwd="Rudrag11!", database="crypto")

if conn.is_connected():
    print("✅ MySQL connection established.")
else:
    print("❌ Failed to connect to MySQL.")

mc = conn.cursor()

# -----------------------------------
# Initialize Flask app
# -----------------------------------
app = Flask(__name__)

# Supported crypto symbols
crypto_symbols = ['ADA', 'BTC', 'BNB', 'DOGE', 'ETC']

# -----------------------------------
# Prediction function (reads from CSV)
# -----------------------------------
def predict_next_close(symbol):
    try:
        df = load_coin(f"{symbol}_predictions")   # loader handles file paths
        print(f"Prediction data for {symbol} loaded successfully.")
        
        # Convert and sort dates
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        latest_prediction = df.sort_values(by='Date').iloc[-1]
        
        return {
            "symbol": symbol,
            "date": latest_prediction['Date'].strftime('%Y-%m-%d'),
            "predicted_price": round(latest_prediction['Predicted_Close'], 2)
        }
    except Exception as e:
        print(f"Error fetching prediction for {symbol}: {e}")
        return {"error": str(e)}

# -----------------------------------
# RSI calculation
# -----------------------------------
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# -----------------------------------
# Flask routes
# -----------------------------------
@app.route('/')
def main_page():
    return render_template("landing.html")

@app.route('/signup')
def signup_page():
    return render_template("get_started.html")

@app.route('/dashboard', methods=['POST'])
def enter_details():
    uname = request.form.get('username')
    passwd = request.form.get('password')
    email = request.form.get('email')
    
    if not uname or not passwd or not email:
        return render_template("get_started.html", err="All fields are required.")
    
    mc.execute("SELECT uname FROM users WHERE uname=%s", (uname,))
    result = mc.fetchall()
    conn.commit()
    
    if result:
        return render_template("get_started.html", err="Username already exists")
    else:
        mc.execute("INSERT INTO users (uname, passwd, email) VALUES (%s, %s, %s)", (uname, passwd, email))
        conn.commit()
        return render_template("option.html", result=result)

@app.route('/start/login')
def login_page():
    return render_template("login.html")

@app.route('/start/dashboard', methods=['POST'])
def dashboard_page():
    global uname
    uname = request.form['username']
    passwd = request.form['password']
    
    mc.execute("SELECT * FROM users WHERE uname=%s AND passwd=%s", (uname, passwd))
    result = mc.fetchall()
    conn.commit()
    
    if result:
        return render_template("option.html", result=result)
    else:
        return render_template("login.html", err="Invalid username or password!")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction, error = None, ""
    
    if request.method == 'POST':
        symbol = request.form.get('symbol', '').replace("-USD", "").upper()
        
        if symbol not in crypto_symbols:
            error = f"Symbol not supported. Use: {', '.join(crypto_symbols)}"
        else:
            prediction = predict_next_close(symbol)
            if "error" in prediction:
                error = prediction["error"]
    
    return render_template("predict.html", error=error, prediction=prediction)

@app.route('/view-crypto', methods=['GET', 'POST'])
def view():
    if request.method == 'POST':
        coin = request.form.get('coin', '').upper()
        
        if coin not in crypto_symbols:
            return render_template("view.html", error=f"Coin not supported. Use: {', '.join(crypto_symbols)}")
        
        try:
            df = load_coin(coin)   # 🔹 use loader instead of manual CSV path
        except FileNotFoundError:
            return render_template("view.html", error="Data for this coin is not available.")
        
        # Ensure correct column handling
        if 'date' not in df.columns and 'Date' in df.columns:
            df.rename(columns={'Date': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        if 'close' in df.columns:
            df.rename(columns={'close': 'Close'}, inplace=True)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        
        df.dropna(subset=['date', 'Close'], inplace=True)
        
        # RSI
        df['RSI'] = calculate_rsi(df)
        latest_rsi = df['RSI'].iloc[-1]
        
        # Closing price chart
        plt.figure(figsize=(10, 5))
        plt.plot(df['date'], df['Close'], label=f'{coin} Closing Price', color='blue')
        plt.title(f'{coin} Closing Prices Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid()
        
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        chart_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()
        
        return render_template("view.html", chart=chart_base64, rsi=round(latest_rsi, 2), coin=coin)
    
    return render_template("view.html")

@app.route('/home')
def home():
    return render_template('option.html')

# -----------------------------------
# Run app
# -----------------------------------
if __name__ == '__main__':
    # Download dataset once when app starts (skips if already exists)
    download_dataset()
    app.run(host='0.0.0.0', port=5001, debug=True)
