import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
# from labeling import label_risk
from tensorflow.keras.models import load_model
import joblib

from feature_engineering import add_features
from data_collection import get_stock_data
from data_preprocessing import preprocess_data

# Load the trained model and scaler
def load_model_from_file(model_filename='suitability_model_nn.h5'):
    """Load the trained neural network model."""
    return load_model(model_filename)

def load_scaler_from_file(scaler_filename='scaler.pkl'):
    """Load the scaler from file."""
    return joblib.load(scaler_filename)

# Calculate Euclidean Distance
def calculate_distance(user_input, stock_data, scaler):
    """
    Calculate Euclidean distance between user input and stock data
    user_input: [desired 1-Year Return, desired Risk Score]
    stock_data: DataFrame with stock features (1-Year Return, Risk Score)
    scaler: fitted scaler used to normalize the data
    """
    # Normalize the user input
    user_data_scaled = scaler.transform([user_input])

    # Normalize the stock data
    stock_data_scaled = scaler.transform(stock_data[['1-Year Return', 'Risk Score']])

    # Calculate Euclidean distances between the user input and all stock data
    distances = euclidean_distances(user_data_scaled, stock_data_scaled)
    return distances[0]

# Recommend stocks based on user input
def recommend_stocks(user_1yr_return, user_risk_level, df, model, scaler):
    """
    Recommend the top 5 stocks based on user input and predicted suitability score.
    """
    # Prepare user input
    user_input = np.array([[user_1yr_return, user_risk_level]])

    # Calculate distances from user input to each stock
    df['Distance'] = calculate_distance(user_input[0], df, scaler)

    # Calculate Suitability Score where distance = 0 gives a score of 10, and higher distances result in lower scores
    min_distance = df['Distance'].min()
    max_distance = df['Distance'].max()

    # Normalize the distances to calculate Suitability Score (0 to 10)
    df['Suitability Score'] = 10 * (1 - (df['Distance'] / max_distance))

    # Sort by Suitability Score and get the top 5 most suitable stocks
    recommended_stocks = df.sort_values(by='Suitability Score', ascending=False).head(5)

    return recommended_stocks[['Ticker', '1-Year Return', 'Risk Score', 'Volatility', 'Suitability Score', 'Distance']]

# Main function for using the model
def main():
    # Load the trained model and scaler
    model = load_model_from_file('suitability_model_nn.h5')
    scaler = load_scaler_from_file('scaler.pkl')

    # List of stock tickers
    ticker_list = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'V', 'NFLX', 'DIS',
    'INTC', 'BA', 'JPM', 'WMT', 'PG', 'KO', 'IBM', 'PEP', 'CVX', 'XOM', 
    'SPY', 'VTI', 'QQQ', 'IWM', 'GLD', 'XLF', 'XLE', 'VOO', 'EFA', 'VWO',
    '^GSPC', '^DJI', '^IXIC', '^RUT', '^FTSE', '^N225', '^STOXX50E', '^HSI', '^AORD',
    'GC=F', 'CL=F', 'SI=F', 'NG=F', 'ZC=F', 'ZW=F', 'KC=F', 'C=F', 'PL=F', 'PA=F'
]

    # Load stock data and create features
    all_data = []
    for ticker in ticker_list:
        try:
            print(f"Fetching data for {ticker}...")
            data = get_stock_data(ticker, period='max')
            data = preprocess_data(data)
            data = add_features(data, ticker)
            # data = label_risk(data)
            all_data.append(data)
        except Exception as e:
            print(f"[Warning] Skipping {ticker}: {e}")

    # Concatenate all data into a single dataframe
    df = pd.concat(all_data)
    df = df.dropna()

    # Ensure the dataframe has only 1 row per stock (if it has more, aggregate or select last data)
    df = df.groupby('Ticker').last().reset_index()

    # Get user input for desired 1-Year Return and Risk Score
    user_1yr_return = float(input("Enter desired 1-Year Return (e.g., 0.15 for 15%): "))
    user_risk_level = int(input("Enter desired Risk Score (1 to 10): "))

    # Recommend stocks based on user input
    recommended_stocks = recommend_stocks(user_1yr_return, user_risk_level, df, model, scaler)

    print("\nTop 5 recommended stocks based on your input:")
    print(recommended_stocks)

if __name__ == "__main__":
    main()
