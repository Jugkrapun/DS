import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from data_collection import get_stock_data
from data_preprocessing import preprocess_data
from feature_engineering import add_features

def load_model_and_scaler(model_path='models/suitability_model_nn.h5', scaler_path='models/scaler.pkl'):
    """
    Load the trained model and the scaler.
    """
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def recommend_stocks(user_1yr_return, user_risk_level, ticker_list, model, scaler):
    """
    Recommend the top 5 stocks based on user input and the trained model.
    """
    all_data = []

    for ticker in ticker_list:
        try:
            # Fetch stock data for each ticker and add features
            data = get_stock_data(ticker, period='max')
            data = preprocess_data(data)
            data = add_features(data, ticker)
            all_data.append(data)
        except Exception as e:
            print(f"[Warning] Skipping {ticker}: {e}")
            continue

    if not all_data:
        print("No valid stock data available for recommendation.")
        return None

    # Concatenate all stock data into a single DataFrame
    df = pd.concat(all_data)
    df = df.dropna()

    # Extract features needed for prediction
    X = df[['1-Year Return', 'Risk Level']]

    # Normalize the features with the scaler
    X_scaled = scaler.transform(X)

    # Predict the suitability score using the trained model
    predicted_scores = model.predict(X_scaled)

    # Add predicted scores to the dataframe
    df['Predicted Suitability Score'] = predicted_scores

    # Sort by predicted suitability score (higher score is better)
    recommended_stocks = df[['Ticker', '1-Year Return', 'Risk Level', 'Predicted Suitability Score']]
    recommended_stocks = recommended_stocks.sort_values(by='Predicted Suitability Score', ascending=False).head(5)

    return recommended_stocks

def main():
    # User input (Example: user provides their desired 1-Year Return and Risk Level)
    user_1yr_return = float(input("Enter desired 1-Year Return (e.g., 0.15 for 15%): "))
    user_risk_level = int(input("Enter desired Risk Level (1 to 10): "))

    # List of stock tickers to choose from
    ticker_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'GOLD']

    # Load the trained model and scaler
    model, scaler = load_model_and_scaler()

    # Recommend the top 5 stocks based on user input
    recommended_stocks = recommend_stocks(user_1yr_return, user_risk_level, ticker_list, model, scaler)

    if recommended_stocks is not None:
        print("\nTop 5 recommended stocks based on your input:")
        print(recommended_stocks)

if __name__ == "__main__":
    main()
