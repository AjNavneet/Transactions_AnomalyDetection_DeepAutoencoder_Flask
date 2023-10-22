import keras
import pickle
import pandas as pd

PREDICTORS = ['Value', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12']
TARGET = ["Class"]

# Function to save a Keras model and associated columns mapping
def save_model(model, columns, output_dir="../output"):
    # Save the Keras model to the specified output directory
    model.save(f"{output_dir}/deep-ae-model")

    # Open a file to save the columns mapping
    file = open(f"{output_dir}/columns.mapping", "wb")

    # Serialize and save the columns using pickle
    pickle.dump(columns, file)
    file.close()

    return True  # Return True to indicate successful saving

# Function to load a Keras model and associated columns mapping
def load_model(model_path, output_dir="../output"):
    model = None
    try:
        # Attempt to load the Keras model from the specified path
        model = keras.models.load_model(model_path)
    except:
        print("Please enter correct path")
        exit(0)

    # Open the file containing the columns mapping
    file = open(f"{output_dir}/columns.mapping", "rb")

    # Load and deserialize the columns mapping using pickle
    columns = pickle.load(file)
    file.close()

    return model, columns  # Return the loaded model and columns mapping
