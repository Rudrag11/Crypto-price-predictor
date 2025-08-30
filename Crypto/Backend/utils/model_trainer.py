# model_trainer.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import timedelta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_PATH, "datareq")
os.makedirs(DATA_FOLDER, exist_ok=True)

crypto_symbols = ["ADA", "BNB", "BTC", "DOGE", "ETH"]

# -------------------------------
# Train + predict
# -------------------------------
def train_and_predict(symbol):
    file_path = os.path.join(DATA_FOLDER, f"{symbol}.csv")
    try:
        df = pd.read_csv(file_path)

        # Preprocess
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["close"], inplace=True)

        # Moving averages
        df["MA5"] = df["close"].rolling(5).mean()
        df["MA10"] = df["close"].rolling(10).mean()
        df["MA20"] = df["close"].rolling(20).mean()
        df["Next_Close"] = df["close"].shift(-1)
        df.dropna(inplace=True)

        features = df[["open", "low", "high", "close", "MA5", "MA10", "MA20"]]
        target = df["Next_Close"]

        scaler_features = MinMaxScaler()
        X_scaled = scaler_features.fit_transform(features)

        scaler_target = MinMaxScaler()
        y_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1))

        # Sequences
        def create_sequences(X, y, n_steps=30):
            Xs, ys = [], []
            for i in range(len(X) - n_steps):
                Xs.append(X[i:i+n_steps])
                ys.append(y[i+n_steps])
            return np.array(Xs), np.array(ys)

        X, y = create_sequences(X_scaled, y_scaled)
        if len(X) == 0:
            return None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.1),
            LSTM(64),
            Dropout(0.1),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=0)

        # Evaluate
        y_pred = model.predict(X_test, verbose=0)
        r2 = r2_score(scaler_target.inverse_transform(y_test), scaler_target.inverse_transform(y_pred))
        print(f"✅ {symbol} R²: {r2:.4f}")

        # Predict next day
        last_seq = X_scaled[-30:]
        last_seq = np.expand_dims(last_seq, axis=0)
        next_scaled = model.predict(last_seq, verbose=0)
        next_price = scaler_target.inverse_transform(next_scaled)[0][0]

        next_date = df["date"].max() + timedelta(days=2)

        # Save predictions
        pred_file = os.path.join(DATA_FOLDER, f"{symbol}_predictions.csv")
        df = pd.DataFrame([{"Date": next_date, "Predicted_Close": next_price}])
        df.to_csv(pred_file, index=False)

        return {"symbol": symbol, "date": next_date.strftime("%Y-%m-%d"), "predicted_price": round(next_price, 2)}

    except Exception as e:
        print(f"❌ Error training {symbol}: {e}")
        return None


def run_all_predictions():
    results = {}
    for sym in crypto_symbols:
        res = train_and_predict(sym)
        if res:
            results[sym] = res
    return results


# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import r2_score, mean_absolute_error
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout
# from datetime import timedelta

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# DATA_FOLDER = os.path.join(BASE_PATH, "datareq")
# os.makedirs(DATA_FOLDER, exist_ok=True)

# crypto_symbols = ["ADA", "BNB", "BTC", "DOGE", "ETH"]

# # -------------------------------
# # Train + predict (backtest mode)
# # -------------------------------
# def train_and_predict(symbol, backtest_yesterday=False):
#     file_path = os.path.join(DATA_FOLDER, f"{symbol}.csv")
#     try:
#         df = pd.read_csv(file_path)

#         # Preprocess
#         df["date"] = pd.to_datetime(df["date"], errors="coerce")
#         for col in ["open", "high", "low", "close"]:
#             df[col] = pd.to_numeric(df[col], errors="coerce")
#         df.dropna(subset=["close"], inplace=True)

#         # Moving averages
#         df["MA5"] = df["close"].rolling(5).mean()
#         df["MA10"] = df["close"].rolling(10).mean()
#         df["MA20"] = df["close"].rolling(20).mean()
#         df["Next_Close"] = df["close"].shift(-1)
#         df.dropna(inplace=True)

#         # ---------------- Backtest mode ----------------
#         if backtest_yesterday:
#             # Remove last row for training
#             test_row = df.iloc[-1].copy()
#             df = df.iloc[:-1]
#             print(f"🔍 Backtesting {symbol}: hiding {test_row['date'].date()} close={test_row['close']}")

#         # Features + Target
#         features = df[["open", "low", "high", "close", "MA5", "MA10", "MA20"]]
#         target = df["Next_Close"]

#         scaler_features = MinMaxScaler()
#         X_scaled = scaler_features.fit_transform(features)

#         scaler_target = MinMaxScaler()
#         y_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1))

#         # Sequences
#         def create_sequences(X, y, n_steps=30):
#             Xs, ys = [], []
#             for i in range(len(X) - n_steps):
#                 Xs.append(X[i:i+n_steps])
#                 ys.append(y[i+n_steps])
#             return np.array(Xs), np.array(ys)

#         X, y = create_sequences(X_scaled, y_scaled)
#         if len(X) == 0:
#             return None

#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )

#         # Model
#         model = Sequential([
#             LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
#             Dropout(0.1),
#             LSTM(64),
#             Dropout(0.1),
#             Dense(32, activation="relu"),
#             Dense(1)
#         ])
#         model.compile(optimizer="adam", loss="mse")
#         model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=0)

#         # Evaluate
#         y_pred = model.predict(X_test, verbose=0)
#         r2 = r2_score(scaler_target.inverse_transform(y_test), scaler_target.inverse_transform(y_pred))
#         print(f"✅ {symbol} R²: {r2:.4f}")

#         # ---------------- Prediction ----------------
#         last_seq = X_scaled[-30:]
#         last_seq = np.expand_dims(last_seq, axis=0)
#         next_scaled = model.predict(last_seq, verbose=0)
#         next_price = scaler_target.inverse_transform(next_scaled)[0][0]

#         if backtest_yesterday:
#             # Predict for yesterday (held-out row)
#             next_date = test_row["date"]
#             actual_price = test_row["close"]
#             mae = mean_absolute_error([actual_price], [next_price])
#             print(f"📊 {symbol} Predicted={next_price:.2f}, Actual={actual_price:.2f}, MAE={mae:.2f}")
#         else:
#             # Normal mode → predict tomorrow
#             next_date = df["date"].max() + timedelta(days=1)

#         # Save predictions
#         pred_file = os.path.join(DATA_FOLDER, f"{symbol}_predictions.csv")
#         pd.DataFrame([{"Date": next_date, "Predicted_Close": next_price}]).to_csv(pred_file, index=False)

#         return {"symbol": symbol, "date": next_date.strftime("%Y-%m-%d"), "predicted_price": round(next_price, 2)}

#     except Exception as e:
#         print(f"❌ Error training {symbol}: {e}")
#         return None


# def run_all_predictions(backtest_yesterday=False):
#     results = {}
#     for sym in crypto_symbols:
#         res = train_and_predict(sym, backtest_yesterday=backtest_yesterday)
#         if res:
#             results[sym] = res
#     return results
