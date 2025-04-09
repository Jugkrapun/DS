from flask import Flask, render_template, request, jsonify, url_for
import yfinance as yf
import pandas as pd
import tensorflow as tf
from save_load_model import load_model
from data_collection import get_stock_data
from data_preprocessing import preprocess_data
from feature_engineering import add_features
from labeling import label_risk

app = Flask(__name__)

def recommend_stocks(user_return, user_risk, model, scaler, df):
    """
    Recommend the top 5 stocks based on user preferences for 1-year return and risk level.
    """
    # User input: 1-year return and risk level
    user_input = np.array([[user_return, user_risk]])

    # Scale the user input (same as how the model was trained)
    user_input_scaled = scaler.transform(user_input)

    # Predict suitability for all stocks
    suitability_scores = model.predict(user_input_scaled)

    # Add the predicted suitability scores to the DataFrame
    df['Suitability Score'] = suitability_scores

    # Rank the stocks based on the predicted suitability score
    recommended_stocks = df.sort_values(by='Suitability Score', ascending=False).head(5)

    return recommended_stocks[['Stock', 'Suitability Score']]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    ticker = request.form['ticker'].upper()
    user_return = float(request.form['user_return'])  # User input for desired 1-year return
    user_risk = float(request.form['user_risk'])      # User input for desired risk level

    try:
        # Get stock data and analyze
        data = get_stock_data(ticker)
        data = preprocess_data(data)
        data = add_features(data)
        data = label_risk(data)

        # Load the model
        model = load_model()
        
        # Use a scaler that was used during model training (ensure you save and load it)
        scaler = joblib.load('scaler.pkl')

        # Get the top 5 recommended stocks
        recommended_stocks = recommend_stocks(user_return, user_risk, model, scaler, data)

        # Prepare response with the recommended stocks
        recommendations = recommended_stocks.to_dict(orient='records')
        
        return jsonify({
            'recommended_stocks': recommendations
        })
        
    except Exception as e:
        import traceback
        import logging
        logging.error(traceback.format_exc())  # Log the full error on the server
        return jsonify({'error': 'An internal error has occurred.'}), 400

@app.route('/api/analyze/<ticker>', methods=['GET'])
def analyze_api(ticker):
    try:
        # Get stock data and analyze
        data = get_stock_data(ticker.upper())
        data = preprocess_data(data)
        data = add_features(data)
        data = label_risk(data)

        # Load the model
        model = load_model()
        
        # Get features for prediction
        features = ['Daily Return', 'Volatility', 'MA50', 'MA200']
        latest_data = data[features].iloc[-1]

        # Make prediction
        risk_level = model.predict(latest_data.values.reshape(1, -1))[0]

        # Prepare API response
        response = {
            'ticker': ticker.upper(),
            'analysis': {
                'risk_level': int(risk_level),
                'current_price': float(data['Close'].iloc[-1]),
                'volatility': float(data['Volatility'].iloc[-1]),
                'daily_return': float(data['Daily Return'].iloc[-1]),
                'last_updated': data.index[-1].isoformat()
            },
            'historical_data': {
                'dates': [d.isoformat() for d in data.index[-30:]],
                'prices': [float(p) for p in data['Close'].tail(30)]
            }
        }
        
        return jsonify(response)
    except Exception as e:
        import traceback
        import logging
        logging.error(traceback.format_exc())  # Log the full error on the server
        return jsonify({
            'error': 'An internal error has occurred.',
            'ticker': ticker.upper()
        }), 400

if __name__ == '__main__':
    app.run(debug=True)
