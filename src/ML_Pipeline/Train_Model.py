from . import Utils  # Import utility functions from the current package
from tensorflow import keras
from tensorflow.keras import layers

# Function to train the ML model
def train(model, x_train, y_train, batch_size=128, epochs=20):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    return model  # Return the trained model

# Function to initiate the model and training data
def fit(data, loss="mse", learning_rate=0.01):
    columns = data.columns

    # Prepare the training data and target variable
    x_train = data.drop(Utils.TARGET, axis=1).values  # Features (input)
    y_train = data[Utils.TARGET].values  # Target variable

    # Define a neural network model using Keras
    model = keras.Sequential(
        [
            keras.Input(shape=(13,)),  # Input layer with 13 features
            layers.Dense(13, activation="relu"),  # Hidden layer with ReLU activation
            layers.Dense(6, activation="relu"),   # Hidden layer with ReLU activation
            layers.Dense(6, activation="relu"),   # Hidden layer with ReLU activation
            layers.Dense(13, activation="linear"),  # Output layer with linear activation
        ]
    )

    # Configure the optimizer with the specified learning rate
    optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

    # Compile the model with the specified loss function and optimizer
    model.compile(loss=loss, optimizer=optimizer)

    # Print a summary of the model architecture
    print(model.summary())

    # Train the model using the training data
    model = train(model, x_train, x_train)  # Autoencoder, x_train is used as both input and target

    return model, columns  # Return the trained model and column names
