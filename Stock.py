import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import streamlit as st

# Function to download stock data
def download_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        print(f"Downloaded data for {ticker}")
        if stock_data.empty:
            print(f"No data found for {ticker}")
            return None
        return stock_data
    except Exception as e:
        print(f"Failed to download data for {ticker}: {str(e)}")
        return None

# Function to preprocess data
def preprocess_data(stock_data):
    if stock_data is None:
        return None, None
    data = stock_data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Function to prepare training data
def prepare_training_data(scaled_data, lookback=40):
    if scaled_data is None:
        return None, None
    x_train, y_train = [], []
    for i in range(lookback, len(scaled_data)):
        x_train.append(scaled_data[i-lookback:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train

# Function to create LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train LSTM model
def train_lstm_model(x_train, y_train):
    if x_train is None or y_train is None:
        return None
    model = create_lstm_model((x_train.shape[1], 1))
    early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=20, batch_size=32, callbacks=[early_stopping])
    return model

# Function to predict stock price for the next 30 days
def predict_stock_price(model, scaled_data, scaler, lookback=40, prediction_days=30):
    if model is None or scaled_data is None or scaler is None:
        return None, None
    x_data = scaled_data[-lookback:].reshape(1, -1, 1)
    predictions = []
    dates = []
    for i in range(prediction_days):
        next_day_close = model.predict(x_data)
        predictions.append(next_day_close[0, 0])
        x_data = np.append(x_data[:, 1:], next_day_close).reshape(1, -1, 1)
        next_date = pd.Timestamp.now() + pd.Timedelta(days=i+1)
        dates.append(next_date)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions, dates

# Streamlit app
def main():
    st.title('Stock Price Prediction App')
    ticker = st.text_input("Enter the company's ticker name:")
    
    if st.button('Predict Stock Prices'):
        start_date = pd.Timestamp.now() - pd.Timedelta(days=365)  # 1 year of historical data
        end_date = pd.Timestamp.now() - pd.Timedelta(days=1)  # Yesterday's date

        # Download stock data
        stock_data = download_stock_data(ticker, start_date, end_date)
        if stock_data is None:
            st.error(f"Failed to download data for {ticker}. Please enter a valid company ticker.")
            return

        # Preprocess data
        scaled_data, scaler = preprocess_data(stock_data)
        if scaled_data is None or scaler is None:
            st.error("Failed to preprocess data. Please try again.")
            return

        # Prepare training data
        x_train, y_train = prepare_training_data(scaled_data)
        if x_train is None or y_train is None:
            st.error("Failed to prepare training data. Please try again.")
            return

        # Train LSTM model
        model = train_lstm_model(x_train, y_train)
        if model is None:
            st.error("Failed to train the model. Please try again.")
            return

        # Predict stock price for the next 30 days
        predictions, dates = predict_stock_price(model, scaled_data, scaler)

        # Visualize stock prices
        st.subheader("Stock Prices and Predictions")
        plt.figure(figsize=(12, 6))
        plt.plot(stock_data.index, stock_data['Close'], label='Actual Price', color='blue')
        plt.plot(dates, predictions, label='Predicted Price', color='red', linestyle='--', marker='o')
        plt.title("Stock Prices and Predictions")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        # Print predicted closing prices for the next 30 days
        st.subheader("Predicted closing prices for the next 30 days:")
        for i, (price, date) in enumerate(zip(predictions, dates)):
            st.write(f"Day {i+1}: {price[0]} - {date.date()}")

if __name__ == "__main__":
    main()
