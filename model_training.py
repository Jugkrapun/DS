# model_training.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

from data_collection import get_stock_data
from data_preprocessing import preprocess_data
from feature_engineering import add_features  # Import add_features

def save_model(model, filename='suitability_model_nn.h5'):
    """
    Save the trained neural network model to disk.
    """
    model.save(filename)

def save_scaler(scaler, filename='scaler.pkl'):
    """
    Save the scaler for later use in normalization.
    """
    joblib.dump(scaler, filename)

def load_scaler(filename='scaler.pkl'):
    """
    Load the scaler from disk.
    """
    return joblib.load(filename)

# def recommend_stocks(user_1yr_return, user_risk_level, stock_data):
#     """
#     Recommend the top 5 stocks based on Euclidean distance between user input and stock features.
#     """
#     if stock_data is None or stock_data.empty:
#         print("Stock data is not available for recommendation.")
#         return None

#     # Extract the features we need
#     stock_data = stock_data[['Ticker', '1-Year Return', 'Risk Level']].dropna()

#     # Normalize the features
#     scaler = StandardScaler()
#     stock_data[['1-Year Return', 'Risk Level']] = scaler.fit_transform(stock_data[['1-Year Return', 'Risk Level']])

#     # Normalize the user input
#     user_data = np.array([[user_1yr_return, user_risk_level]])
#     user_data = scaler.transform(user_data)

#     # Calculate Euclidean distances between the user input and all stocks
#     distances = euclidean_distances(user_data, stock_data[['1-Year Return', 'Risk Level']])

#     # Sort by distance and get the top 5 stocks
#     recommended_stocks = stock_data.sort_values(by='Distance').head(5)

#     return recommended_stocks[['Ticker', '1-Year Return', 'Risk Level']]

def train_model(ticker_list):
    """
    Train the neural network model to predict the suitability score based on 1-Year Return and Risk Level.
    """
    all_data = []

    for ticker in ticker_list:
        try:
            print(f"\nFetching data for {ticker}...")
            data = get_stock_data(ticker, period='max')
            data = preprocess_data(data)
            data = add_features(data, ticker)  # Use add_features from feature_engineering.py
            all_data.append(data)
        except Exception as e:
            print(f"[Warning] Skipping {ticker}: {e}")
            continue

    if not all_data:
        print("No valid stock data available. Aborting.")
        return

    df = pd.concat(all_data)
    df = df.dropna()

    # Only use '1-Year Return' and 'Risk Level' as input features
    features = ['1-Year Return', 'Risk Level']
    
    X = df[features]

    # We will use the same 'Suitability Score' heuristic for simplicity
    df['Suitability Score'] = df['1-Year Return'] / (df['Volatility'] + 0.1)  # Heuristic for suitability score
    y = df['Suitability Score']

    # Handle non-numeric data
    X = X.apply(pd.to_numeric, errors='coerce')

    # Handle NaN values (replace with 0 or drop rows with NaN)
    X = X.fillna(0)

    # Normalize the input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler for later use in inference
    save_scaler(scaler)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    # Build the model
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))  # Use Input layer instead of input_dim
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))  # Output layer for regression

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test))

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f"\nModel Loss: {loss}")

    # Save the trained model
    save_model(model)

    print("Neural network model training completed and saved as 'suitability_model_nn.h5'.")

if __name__ == "__main__":
    ticker_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'GOLD']
    # Train the model first
    train_model(ticker_list)

    # Example: recommend stocks based on user input
    user_1yr_return = 0.15  # Example 1-Year Return: 15%
    user_risk_level = 5  # Example Risk Level (0-10)

    # Load stock data and features
    df = pd.concat([add_features(get_stock_data(ticker, period='max'), ticker) for ticker in ticker_list])

    # Recommend stocks
    recommended_stocks = recommend_stocks(user_1yr_return, user_risk_level, df)
    print("Top 5 recommended stocks based on user input:")
    print(recommended_stocks)
