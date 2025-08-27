import pandas as pd
import os
import matplotlib.pyplot as plt

folder_path = 'C:/AI/dataset'  

coins = ['ada', 'btc', 'bnb', 'doge', 'eth']

# Dictionary to store dataframes
coin_data = {}

# Load data from CSV files into pandas DataFrames
for coin in coins:
    file_path = os.path.join(folder_path, f'{coin}.csv')
    coin_data[coin] = pd.read_csv(file_path)
    
    print(f"Columns in {coin.upper()} dataset: {coin_data[coin].columns}")

# Plot data for each coin in a separate graph
for coin in coins:
    df = coin_data[coin]
    
    # Check if the 'Date' column exists in lowercase and convert to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        df['Date'] = pd.to_datetime(df['Date'])  
    
    plt.figure(figsize=(10, 5))  
    if 'close' in df.columns:
        plt.plot(df['date'], df['close'], label=coin.upper())
    elif 'Close' in df.columns:
        plt.plot(df['date'], df['Close'], label=coin.upper())

    plt.title(f'Closing Prices of {coin.upper()} Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    plt.show()
