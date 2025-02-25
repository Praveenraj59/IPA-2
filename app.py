import joblib
from flask import Flask, render_template, request
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from lstm_model import train_lstm_model, predict_future_prices  # Ensure this import is correct
from sentiment_analysis import get_sentiment_score
from quantum_optimization import optimize_portfolio
import numpy as np
import os

app = Flask(__name__)

# Risk tolerance mapping
risk_mapping = {
    'Conservative': 0.2,
    'Moderate': 0.5,
    'Aggressive': 0.8
}

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/beginner', methods=['GET', 'POST'])
def beginner():
    if request.method == 'POST':
        # Get form data
        ticker = request.form['ticker']
        risk_tolerance_text = request.form['risk_tolerance']
        risk_tolerance = risk_mapping[risk_tolerance_text]
        investment_amount = float(request.form['investment_amount'])
        review_frequency = request.form['review_frequency']
        
        # Define date range
        start_date = "2020-01-01"
        end_date = "2023-10-01"
        
        # Check if a pre-trained model exists
        model_path = f'models/lstm_{ticker}.h5'
        scaler_path = f'models/scaler_{ticker}.pkl'
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)  # Load the scaler
            print(f"Loaded pre-trained model and scaler for {ticker}.")
        else:
            # Train a new model
            model, scaler = train_lstm_model(ticker, start_date, end_date)
            if model is None or scaler is None:
                return render_template('error.html', message=f"Failed to train LSTM model for {ticker}. Please check the ticker symbol and try again.")
        
        # Fetch data and predict future prices
        data = yf.download(ticker, start=start_date, end=end_date)['Close'].values.reshape(-1, 1)
        predicted_price = predict_future_prices(model, scaler, data)
        
        if predicted_price is None:
            return render_template('error.html', message=f"Failed to predict future prices for {ticker}. Please try again.")
        
        # Sentiment analysis
        sentiment_score = get_sentiment_score(ticker)
        
        # Portfolio optimization
        assets = [ticker, 'AAPL', 'MSFT']
        predicted_returns = [predicted_price, 0.05, 0.03]
        weights = optimize_portfolio(assets, predicted_returns, risk_tolerance)
        
        # Calculate investment amounts
        investment_amounts = [investment_amount * weight for weight in weights]
        
        # Insights for beginners
        insights = f"This portfolio is diversified to balance risk and return. {ticker} offers high growth potential, Apple provides stability, and Gold acts as a hedge against market volatility."
        
        return render_template('result.html', 
                               ticker=ticker, 
                               predicted_price=predicted_price, 
                               sentiment_score=sentiment_score, 
                               weights=weights,
                               investment_amounts=investment_amounts,
                               insights=insights)
    return render_template('beginner.html')


@app.route('/investor', methods=['GET', 'POST'])
def investor():
    if request.method == 'POST':
        # Get form data
        tickers = request.form.getlist('tickers')
        risk_tolerance_text = request.form['risk_tolerance']
        risk_tolerance = risk_mapping[risk_tolerance_text]
        current_allocation = request.form['current_allocation']
        investment_horizon = request.form['investment_horizon']
        rebalance_frequency = request.form['rebalance_frequency']
        
        # Fetch data and train LSTM models
        start_date = "2020-01-01"
        end_date = "2023-10-01"
        predicted_prices = []
        sentiment_scores = []
        for ticker in tickers:
            model, scaler = train_lstm_model(ticker, start_date, end_date)
            if model is None or scaler is None:
                return render_template('error.html', message=f"Failed to train LSTM model for {ticker}. Please check the ticker symbol and try again.")
            
            data = yf.download(ticker, start=start_date, end=end_date)['Close'].values.reshape(-1, 1)
            predicted_price = predict_future_prices(model, scaler, data)
            
            if predicted_price is None:
                return render_template('error.html', message=f"Failed to predict future prices for {ticker}. Please try again.")
            
            predicted_prices.append(predicted_price)
            
            # Sentiment analysis
            sentiment_score = get_sentiment_score(ticker)
            sentiment_scores.append(sentiment_score)
        
        # Portfolio optimization
        weights = optimize_portfolio(tickers, predicted_prices, risk_tolerance)
        
        # Risk metrics
        var = 10  # Example value
        cvar = 15  # Example value
        
        # Rebalancing recommendations
        rebalancing_recommendations = f"Rebalance your portfolio to reduce {tickers[0]}'s weight from 50% to 10%."
        
        # Risk management strategies
        risk_management = "Set a stop-loss order for Tesla at $700 to limit losses."
        
        return render_template('result.html', 
                               tickers=tickers, 
                               predicted_prices=predicted_prices, 
                               sentiment_scores=sentiment_scores, 
                               weights=weights,
                               var=var,
                               cvar=cvar,
                               rebalancing_recommendations=rebalancing_recommendations,
                               risk_management=risk_management)
    return render_template('investor.html')

if __name__ == '__main__':
    app.run(debug=True)