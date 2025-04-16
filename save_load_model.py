import joblib
import os
from tensorflow.keras.models import load_model

# Create a 'models' directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def save_model(model, filename='risk_model.pkl'):
    """
    Save the trained model to disk in the models directory.
    """
    filepath = os.path.join('models', filename)
    joblib.dump(model, filepath)

def load_model_from_file(model_filename='suitability_model_nn.h5'):
    """Load the trained neural network model."""
    return load_model(model_filename)
