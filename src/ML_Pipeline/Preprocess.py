import pandas as pd

# Function to cleanup data, convert to numerical variables, and impute missing values with the mean
def cleanup(data):
    data = data.fillna(data.mean())  # Fill missing values with column means
    data = data.drop("Timestamp", axis=1)  # Drop the "Timestamp" column
    return data

# Function to normalize data between 0 and 1
def normalize(data, is_train, output_dir='../output'):
    min_val = data.min()  # Calculate the minimum values for each column
    max_val = data.max()  # Calculate the maximum values for each column

    if is_train:
        # If in training mode, save the min and max values for later use
        min_val.to_pickle(f"{output_dir}/min_val.pkl")
        max_val.to_pickle(f"{output_dir}/max_val.pkl")
    else:
        # If in non-training mode, load the previously saved min and max values
        min_val = pd.read_pickle(f"{output_dir}/min_val.pkl")
        max_val = pd.read_pickle(f"{output_dir}/max_val.pkl")

    # Normalize the data using the min and max values
    normalized_df = (data - min_val) / (max_val - min_val)
    return normalized_df

# Function to call dependent preprocessing functions
def apply(data, is_train):
    print("Preprocessing started....")

    # Clean up the data by filling missing values and dropping the "Timestamp" column
    data = cleanup(data)
    print("Data cleanup completed....")

    # Normalize the data to the range [0, 1]
    data = normalize(data, is_train)
    print("Normalization completed....")

    data = data.loc[:, ~data.columns.duplicated()]  # Remove duplicated columns, if any

    print("Preprocessing completed....")
    return data
