import numpy as np

def init(test_data, model, columns):
    # Preprocess test data to match columns expected by the model
    columns = list(columns)
    columns.remove("Class")  # Remove the "Class" column, if present in the columns list
    test_data = test_data[columns]
    test_data = test_data.loc[:, ~test_data.columns.duplicated()]  # Remove any duplicated columns

    # Convert the test data to a NumPy array
    x_test = test_data.values

    # Make predictions using the provided model
    predict = model.predict(x_test)

    # Calculate the difference between predictions and test data
    difference_array = np.subtract(predict, x_test)

    # Square the differences
    squared_array = np.square(difference_array)

    # Calculate the mean squared error (MSE) as the average of squared differences
    mse = squared_array.mean()

    return mse  # Return the mean squared error
