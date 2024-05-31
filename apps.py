from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the models and scalers
smoker_model = joblib.load('linear_regression_smoker_model.pkl')
non_smoker_model = joblib.load('linear_regression_non_smoker_model.pkl')
smoker_scaler = joblib.load('scaler_smoker.pkl')
non_smoker_scaler = joblib.load('scaler_non_smoker.pkl')

# Define the route to render the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    
    # Extract the input features from the form data
    age = float(data.get('age'))
    bmi = float(data.get('bmi'))
    children = int(data.get('children'))
    smoker_code = int(data.get('smoker_code'))
    sex_code = int(data.get('sex_code'))
    region = data.get('region')
    
    # Create a DataFrame to hold the input data
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'smoker_code': [smoker_code],
        'sex_code': [sex_code],
        'northeast': [1 if region == 'northeast' else 0],
        'northwest': [1 if region == 'northwest' else 0],
        'southeast': [1 if region == 'southeast' else 0],
        'southwest': [1 if region == 'southwest' else 0]
    })
    
    # Select the appropriate model and scaler based on smoker_code
    if smoker_code == 1:
        scaler = smoker_scaler
        model = smoker_model
    else:
        scaler = non_smoker_scaler
        model = non_smoker_model

    # Standardize the numeric columns
    numeric_cols = ['age', 'bmi', 'children']
    scaled_inputs = scaler.transform(input_data[numeric_cols])
    
    # Combine scaled inputs with categorical data
    cat_cols = ['sex_code', 'northeast', 'northwest', 'southeast', 'southwest']
    categorical_data = input_data[cat_cols].values
    inputs = np.concatenate((scaled_inputs, categorical_data), axis=1)
    
    # Make predictions
    prediction = model.predict(inputs)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, port=5001)

