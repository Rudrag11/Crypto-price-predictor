from flask import Flask, render_template, request, redirect, session, url_for, jsonify
import mysql.connector as ms
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import yfinance as yf

# Initialize MySQL connection
conn = ms.connect(host="localhost", port=3306, user="root", passwd="Rudrag11!", database="crypto")

# Check if MySQL connection is successful
if conn.is_connected():
    print("MySQL connection established.")
else:
    print("Failed to connect to MySQL.")

mc = conn.cursor()

# Initialize Flask app
app = Flask(__name__)

# Constants
crypto_symbols = ['ADA-USD', 'BTC-USD', 'BNB-USD', 'DOGE-USD', 'ETC-USD']
start_date = '2020-01-01'
end_date = '2024-11-14'
folder_path = 'C:/AI/datareq/'  
os.makedirs(folder_path, exist_ok=True)

# Prediction function to read from the CSV prediction files
def predict_next_close(symbol):
    try:
        # Load the prediction data from the CSV file
        data = pd.read_csv(f'C:/AI/datareq/{symbol}_predictions.csv')
        print(f"Prediction data for {symbol} loaded successfully.")
        
        # Convert the Date column to datetime
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        
        # Sort by Date and get the latest prediction
        latest_prediction = data.sort_values(by='Date').iloc[-1]
        
        # Extract the relevant fields
        predicted_price = latest_prediction['Predicted_Close']
        prediction_date = latest_prediction['Date']
        
        return {
            "symbol": symbol,
            "date": prediction_date.strftime('%Y-%m-%d'),
            "predicted_price": round(predicted_price, 2)
        }
    except Exception as e:
        print(f"Error in fetching prediction for {symbol}: {e}")
        return {"error": str(e)}

# RSI calculation function
# delta=the difference between each closing price and the previous one for the rest.
# gain: Filters positive changes from delta.
# loss: Filters negative changes from delta
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Flask routes
@app.route('/')
def main_page():
    return render_template("landing.html")

@app.route('/signup')
def signup_page():
    return render_template("get_started.html")

@app.route('/dashboard', methods=['POST'])
def enter_details():
    if request.method == 'POST':
        uname = request.form.get('username')
        passwd = request.form.get('password')
        email = request.form.get('email')
        
        if not uname or not passwd or not email:
            err = "All fields are required."
            return render_template("get_started.html", err=err)
        
        mc.execute("SELECT uname FROM users WHERE uname=%s", (uname,))
        result = mc.fetchall()
        conn.commit()
        
        if result:
            err = 'Username already exists'
            return render_template("get_started.html", err=err)
        else:
            mc.execute("INSERT INTO users (uname, passwd, email) VALUES (%s, %s, %s)", (uname, passwd, email))
            conn.commit()
            return render_template("option.html", result=result)

@app.route('/start/login')
def login_page():
    return render_template("login.html")

@app.route('/start/dashboard', methods=['POST'])
def dashboard_page():
    if request.method == 'POST':
        global uname
        uname = request.form['username']
        passwd = request.form['password']
        mc.execute("SELECT * FROM users WHERE uname=%s AND passwd=%s", (uname, passwd))
        result = mc.fetchall()
        conn.commit()
        
        if result != []:
            return render_template("option.html", result=result)
        else:
            err = "Invalid username or password!"
            return render_template("login.html", err=err)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    error = ""
    prediction = None
    
    if request.method == 'POST':
        symbol = request.form.get('symbol')
        
        if symbol not in crypto_symbols:
            error = "Symbol not supported"
        else:
            prediction = predict_next_close(symbol)
            if 'error' in prediction:
                error = prediction['error']
            else:
                prediction = {
                    "symbol": prediction["symbol"],
                    "date": prediction["date"],
                    "predicted_price": prediction["predicted_price"]
                }
    
    return render_template("predict.html", error=error, prediction=prediction)


@app.route('/view-crypto', methods=['GET', 'POST'])
def view():
    if request.method == 'POST':
        # Get the coin name from the form
        coin = request.form.get('coin', '').upper()  # Convert to uppercase for consistency
        
        # Check if the coin is in the list of supported symbols
        if coin not in crypto_symbols:
            return render_template("view.html", error="Coin name is not supported. Please use one of the following: " + ", ".join(crypto_symbols))
        
        file_path = os.path.join('C:/AI/datareq', f'{coin}.csv')

        # Check if the file exists for the given coin
        if not os.path.isfile(file_path):
            return render_template("view.html", error="Data for this coin is not available")

        # Load the data for the coin
        df = pd.read_csv(file_path)

        # Ensure correct column names and parse dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert to datetime, coerce errors
        elif 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'], errors='coerce')  # Convert to datetime, coerce errors
            df.drop(columns=['Date'], inplace=True)

        # Rename 'close' to 'Close' if necessary
        if 'close' in df.columns:
            df.rename(columns={'close': 'Close'}, inplace=True)

        # Convert 'Close' column to numeric, forcing errors to NaN
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

        # Check for NaN values in 'Close' after conversion
        if df['Close'].isnull().all():
            return render_template("view.html", error="No valid closing prices found for this coin.")

        # Drop rows with NaT in 'date' or NaN in 'Close'
        df.dropna(subset=['date', 'Close'], inplace=True)

        # Calculate RSI and get the latest RSI value
        df['RSI'] = calculate_rsi(df)
        latest_rsi = df['RSI'].iloc[-1]


        # Generate Closing Price Chart
        plt.figure(figsize=(10, 5))
        plt.plot(df['date'], df['Close'], label=f'{coin} Closing Price', color='blue')
        plt.title(f'{coin} Closing Prices Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid()

        # Save plot to a Base64 image
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        chart_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()

        return render_template("view.html", chart=chart_base64, rsi=round(latest_rsi, 2), coin=coin)
    else:
        return render_template("view.html")
# Home route to render view.html
@app.route('/home')
def home():
    return render_template('option.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
