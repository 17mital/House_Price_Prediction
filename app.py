from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input and match with trained features
        feature_names = {
            'longitude': "How far the location is to the west or east on the map",
            'latitude': "How far the location is to the north or south on the map",
            'housing_median_age': "Average age of houses in the area",
            'total_rooms': "Total rooms available in all houses in the area",
            'total_bedrooms': "Total number of bedrooms in all houses in the area",
            'population': "Total number of people living in the area",
            'households': "Total number of families or residential groups in the area",
            'median_income': "Average income of residents in the area",
            'rooms_per_household': "Average number of rooms per house",
            'bedrooms_per_room': "Ratio of bedrooms to total rooms in a house",
            'population_per_household': "Average number of people living in a house",
            'ocean_proximity_INLAND': "Is the house far from the coast? (Yes = 1, No = 0)",
            'ocean_proximity_ISLAND': "Is the house located on an island? (Yes = 1, No = 0)",
            'ocean_proximity_NEAR BAY': "Is the house near a bay area? (Yes = 1, No = 0)",
            'ocean_proximity_NEAR OCEAN': "Is the house near the ocean? (Yes = 1, No = 0)"
        }
        
        data = [float(request.form.get(feature, 0)) for feature in feature_names.keys()]
        
        # Convert to array and scale
        data_array = np.array(data).reshape(1, -1)
        scaled_data = scaler.transform(data_array)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        
        return jsonify({'predicted_price': round(prediction[0], 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
