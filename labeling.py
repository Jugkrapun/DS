import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def label_risk(df):
    """
    Label the risk of a stock based on a combination of factors like volatility, beta, and market conditions.
    Here, we use a combination of Volatility and Beta, and we normalize to a scale from 1 (low risk) to 10 (high risk).
    """
    # Assign Risk Level based on Volatility (Standard Deviation of Returns)
    df['Volatility'] = df['Daily Return'].rolling(window=21).std() * np.sqrt(252)  # Annualized volatility

    # Example: Assume we have Beta data or calculate it based on historical market data.
    # For this example, assume Beta values (you can replace this with real data or API call to get Beta)
    df['Beta'] = np.random.uniform(0.5, 2, size=len(df))  # Random Beta values for illustration

    # Calculate a basic risk score based on Volatility and Beta
    df['Risk Level'] = (df['Volatility'] * 0.7 + df['Beta'] * 0.3)  # 70% Volatility, 30% Beta

    # Normalize Risk Level to a scale from 1 to 10
    min_risk = df['Risk Level'].min()
    max_risk = df['Risk Level'].max()

    # Normalize the Risk Level to be between 1 and 10
    df['Risk Level'] = 1 + (df['Risk Level'] - min_risk) / (max_risk - min_risk) * 9

    # Clip values to ensure they are within the range of 1 to 10
    df['Risk Level'] = df['Risk Level'].clip(1, 10)

    return df

