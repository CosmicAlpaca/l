import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load data from URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
           "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(url, names=columns)

# Replace zeros with NaN in columns where zeros are invalid
cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

# Split into features (X) and target (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
    ('scaler', StandardScaler())                  # Scale features
])

# Fit the pipeline on training data
preprocessing_pipeline.fit(X_train)

# Transform training and testing data
X_train_transformed = preprocessing_pipeline.transform(X_train)
X_test_transformed = preprocessing_pipeline.transform(X_test)

# Define the deep learning model
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))  # First hidden layer
model.add(Dense(8, activation='relu'))                # Second hidden layer
model.add(Dense(1, activation='sigmoid'))             # Output layer for binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_transformed, y_train, epochs=100, batch_size=32, validation_data=(X_test_transformed, y_test))

# Save the model and preprocessing pipeline
model.save("model/diabetes_model.h5")
joblib.dump(preprocessing_pipeline, "preprocessing_pipeline.joblib")