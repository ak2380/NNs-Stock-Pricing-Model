import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

class StockPredictor:
    def __init__(self, stock, start='2000-01-01', end='2020-12-31'):
        self.stock = stock
        self.start = start
        self.end = end
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.new_data = None
        self.scaled_data = None

    def download_data(self):
        data = yf.download(self.stock, self.start, self.end)
        data = data.sort_index(ascending=True, axis=0)
        self.new_data = pd.DataFrame(index=range(0, len(data)), columns=['Date', 'Close'])
        
        for i in range(0, len(data)):
            self.new_data["Date"][i] = data.index[i]
            self.new_data["Close"][i] = data["Close"][i]

        self.new_data.index = self.new_data.Date
        self.new_data.drop("Date", axis=1, inplace=True)

        dataset = self.new_data.values

        train = dataset[0:int(len(dataset)*0.8), :]
        test = dataset[int(len(dataset)*0.8):, :]

        self.scaled_data = self.scaler.fit_transform(dataset)

        x_train, y_train = [], []

        for i in range(60, len(train)):
            x_train.append(self.scaled_data[i-60:i, 0])
            y_train.append(self.scaled_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        return x_train, y_train, test

    def create_model(self, x_train, y_train):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=2)

        self.model = model

    def predict_prices(self, test):
        inputs = self.new_data[len(self.new_data) - len(test) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = self.scaler.transform(inputs)

        X_test = []
        for i in range(60, inputs.shape[0]):
            X_test.append(inputs[i-60:i, 0])
        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        closing_price = self.model.predict(X_test)
        closing_price = self.scaler.inverse_transform(closing_price)

        return closing_price[len(closing_price)-len(test):]

    def plot_predictions(self, actual_prices, predicted_prices):
        actual_prices = actual_prices[-len(predicted_prices):]  # Consider only overlapping data points

        test_dates = self.new_data[-len(actual_prices):].index

        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, actual_prices, color='blue', label='Actual Prices')
        plt.plot(test_dates, predicted_prices, color='red', label='Predicted Prices')
        plt.title(f'{self.stock} Share Prices')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def run_prediction(self):
        x_train, y_train, test = self.download_data()
        self.create_model(x_train, y_train)
        predicted_prices = self.predict_prices(test)
        actual_prices = self.scaler.inverse_transform(self.scaled_data[len(x_train):])

        self.plot_predictions(actual_prices, predicted_prices)
