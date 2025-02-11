import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample data
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [5, np.nan, 3, 2, 1],
    'C': [2, 3, 4, np.nan, 1],
    'D': [np.nan, 1, 2, 3, 4]
}

# Create DataFrame
df = pd.DataFrame(data)

# Function to predict missing values using SVM
def predict_missing_values(df):
    for column in df.columns:
        # Separate the data into training and prediction sets
        train_data = df[df[column].notna()]
        predict_data = df[df[column].isna()]

        if not predict_data.empty:
            X_train = train_data.drop(columns=[column])
            y_train = train_data[column]
            X_predict = predict_data.drop(columns=[column])

            # Train the SVM model
            model = SVR()
            model.fit(X_train, y_train)

            # Predict the missing values
            predicted_values = model.predict(X_predict)

            # Fill the missing values with the predictions
            df.loc[df[column].isna(), column] = predicted_values

    return df

# Predict and fill missing values
df_filled = predict_missing_values(df)
print(df_filled)




