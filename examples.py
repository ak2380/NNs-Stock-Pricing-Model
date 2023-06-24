from main import StockPredictor

tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

# Iterate over each ticker and run the prediction
for ticker in tickers:
    stock_predictor = StockPredictor(stock=ticker)
    stock_predictor.run_prediction()
