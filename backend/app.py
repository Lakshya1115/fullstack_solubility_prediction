import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template  # Import render_template

app = Flask(__name__)

# Define the column names for the model features
column_names = ['feature1', 'feature2', 'feature3', 'feature4']  # Replace with actual feature names used during training

# Load models at the start of the application
lr_model = None
rf_model = None

def load_models():
    global lr_model, rf_model
    with open('linear_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('random_forest.pkl', 'rb') as f:
        rf_model = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")  # Render the index.html template

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from frontend
        data = request.get_json(force=True)
        print("Received data:", data)  # Debug: print received data
        
        # Extract and ensure the features are in the correct format
        features = np.array([data['features']], dtype=float)  # Ensure features are float
        print("Received features (as numpy array):", features)  # Debug: print the features array
        
        # Convert input features to DataFrame to match model's training format
        features_df = pd.DataFrame(features, columns=column_names)
        print("Converted features to DataFrame:\n", features_df)  # Debug: print DataFrame

        # Make predictions using the models
        lr_pred = lr_model.predict(features_df)[0]  # Linear Regression Prediction
        rf_pred = rf_model.predict(features_df)[0]  # Random Forest Prediction
        print(f"Predictions - LR: {lr_pred}, RF: {rf_pred}")  # Debug: print predictions

        # Return predictions as a JSON response
        return jsonify({
            'linear_regression': lr_pred,
            'random_forest': rf_pred
        })

    except KeyError as e:
        return jsonify({'error': f'Missing key: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid value: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': 'An unexpected error occurred. Please try again later.'}), 500

if __name__ == '__main__':
    load_models()
    app.run(debug=True)