# Import necessary libraries and modules
import pandas as pd
from flask import Flask, request
import json
import Preprocess  # Import a custom module for data preprocessing
import Predict  # Import a custom module for making predictions
import Utils  # Import a custom utility module

app = Flask(__name__)

# Define the path to the pre-trained machine learning model
model_path = '../output/deep-ae-model'

# Load the pre-trained model and associated columns using a custom utility function
ml_model, columns = Utils.load_model(model_path)

# Define an endpoint for handling POST requests to get fraud scores
@app.post("/get_fraud_score")
async def get_license_status():
    # Parse the JSON data from the POST request
    items = json.loads(request.data)

    # Create a Pandas DataFrame using the received data
    test_df = pd.DataFrame([items], columns=items.keys())

    # Apply data preprocessing to the input data (is_train=False indicates it's for prediction)
    processed_df = Preprocess.apply(test_df, is_train=False)

    # Use the pre-trained model to make predictions
    prediction = Predict.init(processed_df, ml_model, columns)

    # Create a JSON response with the prediction result
    output = {"status": prediction}
    
    return output

# Run the Flask application if this script is executed
if __name__ == '__main__':
    # Run the Flask app on host '0.0.0.0' to make it accessible from other devices and on port 5001
    app.run(host='0.0.0.0', port=5001)
