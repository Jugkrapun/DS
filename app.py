import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from data_collection import get_stock_data
from data_preprocessing import preprocess_data
from feature_engineering import add_features
from labeling import label_risk
from save_load_model import load_model_from_file
from test_model import calculate_distance, load_scaler_from_file, recommend_stocks  # Assuming this provides the risk labeling logic

app = Flask(__name__)

# Load the pre-trained model and scaler
model = load_model_from_file('suitability_model_nn.h5')
scaler = load_scaler_from_file('scaler.pkl')

# Sample stock tickers (replace with the full list if needed)
tickers =  [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'V', 'NFLX', 'DIS',
    'INTC', 'BA', 'JPM', 'WMT', 'PG', 'KO', 'IBM', 'PEP', 'CVX', 'XOM', 
    'SPY', 'VTI', 'QQQ', 'IWM', 'GLD', 'XLF', 'XLE', 'VOO', 'EFA', 'VWO',
    '^GSPC', '^DJI', '^IXIC', '^RUT', '^FTSE', '^N225', '^STOXX50E', '^HSI', '^AORD',
    'GC=F', 'CL=F', 'SI=F', 'NG=F', 'ZC=F', 'ZW=F', 'KC=F', 'C=F', 'PL=F', 'PA=F'
]

@app.route('/')
def index():
    return render_template('index.html')  # Render the frontend HTML page

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get user input from the request
        data = request.get_json()
        user_return = data['return']
        user_risk = data['risk']

        # Initialize a list to store the data of all stocks
        all_data = []
        for ticker in tickers:
            try:
                print(f"Fetching data for {ticker}...")
                # Fetch the stock data
                data = get_stock_data(ticker, period='max')
                # Preprocess the data
                data = preprocess_data(data)
                # Add features to the data
                data = add_features(data, ticker)
                # Label the risk for the stock
                data = label_risk(data)
                all_data.append(data)
            except Exception as e:
                print(f"[Warning] Skipping {ticker}: {e}")

        # Concatenate all the data into a single dataframe
        df = pd.concat(all_data)
        # Remove rows with missing values
        df = df.dropna()

        # Ensure the dataframe has only one row per stock (use the last available data for each ticker)
        df = df.groupby('Ticker').last().reset_index()

        # Recommend stocks based on user input
        recommended_stocks = recommend_stocks(user_return, user_risk, df, model, scaler)

        # Print the recommended stocks for debugging
        print("\nTop 5 recommended stocks based on your input:")
        print(recommended_stocks)

        # Return the results as JSON in the correct format
        return jsonify({'stocks': recommended_stocks.to_dict(orient='records')})

    except Exception as e:
        print(f"Error in analyze: {e}")
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
