# Deep Autoencoders Model for Anomaly Detection in Transcational data

## Project Overview

### Objective

This project focuses on the development of deep learning models based on autoencoders for the purpose of anomaly detection. Autoencoders are neural networks used to learn compressed representations of raw data, making them effective tools for detecting anomalies in datasets. The project also involves deploying the trained model as an API using Flask.

---

### Aim

The primary objectives of this project include:

- For normal transactions developing a deep learning model based on autoencoders for anomaly detection. 
- Deploying the model as an API using Flask for real-time anomaly detection.

---

### Data Overview

The dataset used in this project is a transaction dataset containing information on more than 100,000 transactions, each characterized by several features. This data serves as the foundation for training and testing the deep autoencoder model.

---

### Tech Stack

- Language: `Python`
- Packages: `Pandas`, `Numpy`, `Matplotlib`, `Keras`, `Tensorflow`
- API Service: `Flask`, `Gunicorn`

---

### Approach

The project follows a structured approach:

1. Understand the business objective and the importance of anomaly detection.
2. Perform exploratory data analysis (EDA) to gain insights into the dataset.
3. Normalize and clean the data, addressing any missing values through imputation.
4. Delve into the theory behind autoencoders and their architecture.
5. Build a base autoencoder model using the Keras library.
6. Fine-tune the model to extract the best performance for anomaly detection.
7. Make predictions using the trained model to identify anomalies.
8. Serve the model as an API endpoint using Flask, enabling real-time anomaly detection.

---

### Modular Code

1. **input**: Contains the dataset files used for analysis (e.g., `final_cred_data.csv`, `Test-data.csv`).

2. **src**: The heart of the project, this folder contains modularized code for various steps, including data preprocessing, model building, and deployment. It consists of the `ML_pipeline` and `engine.py` files, each containing functions for different functionalities.

3. **output**: Contains pre-trained models saved as .pkl files. These models can be conveniently loaded and used without the need for retraining.

4. **lib**: A reference folder with the original IPython notebook used in the videos.

5. **requirements.txt**: Lists all required libraries and their versions for easy installation using `pip`.

---

### Key Concepts Explored

Upon completing this project, participants will gain insights and skills related to:

1. Understanding the concept of autoencoders and their applications in anomaly detection.
2. Building autoencoder models using Keras for anomaly detection.
3. Tuning autoencoder models to achieve optimal performance.
4. Deploying deep learning models as API endpoints using Flask.
5. Performing real-time anomaly detection using the deployed model.

---



