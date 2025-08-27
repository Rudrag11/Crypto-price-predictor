import pandas as pd
import os

folder_path = 'C:/AI/dataset'  

coins = ['ada', 'btc', 'bnb', 'doge', 'eth']

# Dictionary to store dataframes for each coin
coin_data = {}

for coin in coins:
    file_path = os.path.join(folder_path, f'{coin}.csv')
    coin_data[coin] = pd.read_csv(file_path)

# Calculate RSI
# delta=the difference between each closing price and the previous one for the rest.
# gain: Filters positive changes from delta.
# loss: Filters negative changes from delta
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi