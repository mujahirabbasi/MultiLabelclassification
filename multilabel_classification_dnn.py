# Import necessary libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import hamming_loss, f1_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset with a relative path
dataset_path = os.path.join(os.path.dirname(__file__), 'datasets', 'final_labeled_df.csv')
df = pd.read_csv(dataset_path)


# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to the multilabel target columns
for col in ['Growth Label', 'Volatility Label', 'Momentum Label']:
    df[col] = label_encoder.fit_transform(df[col])

# Separate features and multilabel target columns
X = df.drop(columns=['Ticker', 'Growth Label', 'Volatility Label', 'Momentum Label'])  # Adjust if needed
y = df[['Growth Label', 'Volatility Label', 'Momentum Label']].values

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the deep neural network architecture
def create_model(input_dim, output_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(output_dim, activation='sigmoid')  # Sigmoid for multilabel classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Get input and output dimensions
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

# Create and compile the model
model = create_model(input_dim, output_dim)

# Define early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# Predict on the validation set
y_val_pred_prob = model.predict(X_val)
y_val_pred = (y_val_pred_prob > 0.5).astype(int)  # Apply threshold to get binary predictions

print (y_val)
print (y_val_pred)

# Evaluation metrics for multilabel classification
hamming = hamming_loss(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred, average='micro')
subset_accuracy = accuracy_score(y_val, y_val_pred)  # Exact match ratio

# Display performance
print("\nDeep Neural Network Performance:")
print(f"Hamming Loss: {hamming:.4f}")
print(f"F1 Score (Micro-averaged): {f1:.4f}")
print(f"Subset Accuracy: {subset_accuracy:.4f}")
