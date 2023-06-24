# NNs-Stock-Pricing-Model

This project is intended to be an introduction into using Neural Networks to predict stock prices. This pricing model uses a neural network comprised of hidden LSTM (long short-term memory) layers in order to accurately predict the stock prices of any underlying stock based on the price of the stock over the last 60 days. It utilises the Yahoo Finance API to download historical stock data alongside a deep learning model, built using TensorFlow.

This project uses a specific form of Recurrent Neural Networks, comprising of LSTM layers, in order to predict the stock price with greater accuracy. This suitability of the model is determined as a result of the nature of the underlying data being fed in since LSTMs can handle time-series data with ease and aid sequential predicting, as shown in the model in `main.py`.

Neural networks offer a powerful substitute to conventional methods including the ability to model non-linear relationships, extract and learn features from the raw data and flexibility, allowing certain NNs (neural networks) to predict stock prices based on non-numerical inputs e.g. news surrounding the company's financials (text data).

##  Workflow

The overall worflow of this project is highlighted here:

1. Data is obtained using the Yahoo Finance API for a specified stock ticker
2. Data is processed and formatted
3. Data is split into training data and testing data, and scaled to avoid slow convergence/sub-optimal solutions.
4. Model is created and trained based on the training dataset.
5. Predictions are made on the testing data using the trained model.
6. Actual Prices and Predicted Prices are plotted together using `matplotlib`

## Installation

To run the code in this project, you need to have the following dependencies installed:

- Python 3.x
- yfinance
- pandas
- numpy
- scikit-learn
- TensorFlow
- matplotlib

You can install the dependencies using pip: 
`pip install -r requirements.txt`

## Usage

1. Clone the repository or download the source code.

2. Open a terminal or command prompt and navigate to the project directory.

3. Modify the `StockPredictor` class parameters in the `main.py` file as needed. Set the `stock` variable to the desired stock symbol and adjust the `start` and `end` dates for the historical data.

4. Run the code: `python main.py`

## Improvements

Improvements can be made to the model through a number of methods, including the following:

1. include a regularliser to prevent overfitting
2. tweak the hyperparameters of the model e.g. the number of nodes in the LSTM layers, number of epochs, batch size (number of sequences trained at one time - usually 32) etc. to capture greater complexity in the patterms
