import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

from data_collection import get_stock_data
from data_preprocessing import preprocess_data
from feature_engineering import add_features

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

def create_features(ticker_list):
    """
    Create features for the given list of stock tickers.
    This will return a dataframe with 1-Year Return, Volatility, and Risk Score for each stock.
    The dataframe will have only 6 rows, one for each ticker.
    """
    all_data = []

    for ticker in ticker_list:
        try:
            print(f"\nFetching data for {ticker}...")
            data = get_stock_data(ticker, period='max')  # Fetch stock data
            data = preprocess_data(data)  # Preprocess the data
            data = add_features(data, ticker)  # Add features like '1-Year Return', 'Volatility'
            all_data.append(data)
        except Exception as e:
            print(f"[Warning] Skipping {ticker}: {e}")
            continue

    if not all_data:
        print("No valid stock data available. Aborting.")
        return None

    # Concatenate all data into a single dataframe
    df = pd.concat(all_data)

    # Ensure we only keep the necessary columns
    df = df[['Ticker', '1-Year Return', 'Risk Score', 'Volatility']]

    # Group by 'Ticker' and keep only one row for each stock (either the last or mean)
    # Use 'last' to get the most recent row for each ticker, or 'mean' for averaging
    df = df.groupby('Ticker').last().reset_index()  # Alternatively, use .mean() instead of .last()

    # Ensure we have only 6 rows corresponding to the 6 tickers
    if len(df) != len(ticker_list):
        print(f"Warning: Expected {len(ticker_list)} rows, but got {len(df)} rows.")
        return None

    # Drop any rows with NaN values
    df = df.dropna()

    # Print the columns to check if everything is correctly included
    print("Columns in the DataFrame after feature engineering:", df.columns)

    return df

def train_model(ticker_list):
    """
    Train the neural network model to predict the suitability score based on 1-Year Return and Risk Score.
    """
    # Create features
    df = create_features(ticker_list)
    if df is None:
        return

    #Use '1-Year Return' and 'Risk Score' as input features
    features = ['1-Year Return', 'Risk Score']

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
    ticker_list = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'V', 'NFLX', 'DIS',
    'INTC', 'BA', 'JPM', 'WMT', 'PG', 'KO', 'IBM', 'PEP', 'CVX', 'XOM', 
    'SPY', 'VTI', 'QQQ', 'IWM', 'GLD', 'XLF', 'XLE', 'VOO', 'EFA', 'VWO',
    '^GSPC', '^DJI', '^IXIC', '^RUT', '^FTSE', '^N225', '^STOXX50E', '^HSI', '^AORD',
    'GC=F', 'CL=F', 'SI=F', 'NG=F', 'ZC=F', 'ZW=F', 'KC=F', 'C=F', 'PL=F', 'PA=F'
]

    # Train the model
    train_model(ticker_list)