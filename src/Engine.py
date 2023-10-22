# Import necessary modules and functions from custom Python packages
from ML_Pipeline import Predict, Train_Model
from ML_Pipeline.Preprocess import apply
from ML_Pipeline.Utils import load_model, save_model
import pandas as pd
import subprocess

# Prompt the user for their choice: 0 for training, 1 for prediction, 2 for deployment
val = int(input("Train - 0\nPredict - 1\nDeploy - 2\nEnter your value: "))

# If the user chooses to train a model
if val == 0:
    # Load data from a CSV file and perform data preprocessing
    data = pd\
        .read_csv("../input/final_cred_data.csv", low_memory=False, index_col=0)\
        .drop_duplicates()\
        .reset_index(drop=True)

    print("Data loaded into pandas dataframe")

    processed_df = apply(data, is_train=True)
    
    # Train a machine learning model and save it
    ml_model, columns = Train_Model.fit(processed_df)
    model_path = save_model(ml_model, columns)
    print("Model saved in: ", "output/deep-ae-model")

# If the user chooses to make predictions
elif val == 1:
    # Specify the path to the pre-trained model
    model_path = "../output/deep-ae-model"
    # Alternative: Allow the user to input the model path
    # model_path = input("Enter full model path: ")

    # Load the model and test data from CSV, preprocess the test data
    ml_model, columns = load_model(model_path)
    test_data = pd \
        .read_csv("../input/test_data.csv", low_memory=False, index_col=0) \
        .drop_duplicates() \
        .reset_index(drop=True)

    processed_df = apply(test_data, is_train=False)

    # Use the model to make predictions
    prediction = Predict.init(processed_df, ml_model, columns)
    print(prediction)

# For deployment
else:
    # For production deployment, run 'wsgi.sh' script
    # For development deployment, run 'deploy.py' script
    process = subprocess.Popen(['python', 'ML_Pipeline/deploy.py'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True
                               )

    # Capture and print the output of the deployment process
    for stdout_line in process.stdout:
        print(stdout_line)

    # Get any remaining output and errors, if applicable
    stdout, stderr = process.communicate()
    print(stdout, stderr)
