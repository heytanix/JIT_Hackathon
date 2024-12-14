from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from longterm import create_intraday_model, prepare_intraday_data
import requests
from SentimentANDrisks import determine_sentiment
import tensorflow as tf  # Import TensorFlow for GPU acceleration

app = Flask(__name__)

# Enable GPU acceleration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(e)

def fetch_news(ticker):
    # Placeholder function to fetch news headlines related to the stock ticker.
    return [
        "Company X reports strong earnings.",
        "Analysts predict growth for Company X.",
        "Company X faces challenges in market expansion."
    ]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker'].upper()
    exchange = request.form['exchange']
    
    ticker += '.NS' if exchange == 'NSE' else '.BO' if exchange == 'BSE' else ''
    
    start_date = request.form['start_date']
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    if stock_data.empty:
        return render_template('index.html', error="No data found for the given ticker.")
    
    try:
        x_train, y_train, scaler = prepare_intraday_data(stock_data)
    except ValueError as e:
        return render_template('index.html', error=str(e))
    
    model = create_intraday_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, epochs=5, batch_size=16)  # Reduced epochs and batch size for faster training

    last_60_days = stock_data['Close'][-60:].values
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))

    x_test = np.array([last_60_days_scaled]).reshape(1, 60, 1)

    predictions = []
    prediction_dates = []
    
    current_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    
    num_days = 60  # Number of days to predict

    # Use a more efficient prediction loop
    for _ in range(num_days):
        predicted_price = model.predict(x_test, verbose=0)  # Suppress output for faster execution
        predictions.append(predicted_price[0][0])
        
        x_test = np.append(x_test[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)

        if current_date.weekday() < 5:
            prediction_dates.append(current_date)
        
        current_date += pd.Timedelta(days=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # Fetch and analyze news headlines related to the ticker
    news_headlines = fetch_news(ticker)
    sentiment_data = {
        "nifty_change": 0.7,  # Example value, replace with actual data
        "india_vix": 12,      # Example value, replace with actual data
        "advances": 1200,     # Example value, replace with actual data
        "declines": 800,      # Example value, replace with actual data
        "inflation_rate": 3.8, # Example value, replace with actual data
        "interest_rate": 5.0,  # Example value, replace with actual data
        "usd_inr": 78,        # Example value, replace with actual data
        "earnings_growth": 6, # Example value, replace with actual data
        "fii_net_investment": 1000, # Example value, replace with actual data
        "crude_oil_price": 65, # Example value, replace with actual data
        "geopolitical_risk_index": 25, # Example value, replace with actual data
        "budget_deficit": 3.2, # Example value, replace with actual data
    }
    sentiment_score = determine_sentiment(sentiment_data)

    # Prepare predictions and sentiments for display
    prediction_results = predictions.tolist()
    
    prediction_days = [date.strftime('%A') for date in prediction_dates]

    combined_results = [{'date': date.strftime('%Y-%m-%d'), 'day': day, 'price': price} 
                        for date, day, price in zip(prediction_dates, prediction_days, prediction_results)]

    return render_template('index.html', 
                           predictions=combined_results,
                           ticker=ticker,
                           sentiment_score=sentiment_score)

if __name__ == '__main__':
    app.run(debug=True)
