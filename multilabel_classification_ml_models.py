# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import hamming_loss, f1_score, accuracy_score
import xgboost as xgb
import os

# Load the dataset with a relative path
dataset_path = os.path.join(os.path.dirname(__file__), 'datasets', 'final_labeled_df.csv')
df = pd.read_csv(dataset_path)

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to the multilabel target columns
for col in ['Growth Label', 'Volatility Label', 'Momentum Label']:
    df[col] = label_encoder.fit_transform(df[col])

# Separate features and multilabel target columns
X = df.drop(columns=['Ticker', 'Growth Label', 'Volatility Label', 'Momentum Label'])  
y = df[['Growth Label', 'Volatility Label', 'Momentum Label']]

# Calculate class weights for each target label
class_weights = {}
for col in y.columns:
    value_counts = y[col].value_counts(normalize=True)
    # If imbalance > 50%
    if abs(value_counts[1] - value_counts[0]) > 0.5:  
        class_weights[col] = {0: value_counts[1], 1: value_counts[0]}
print("\nClass Weights:\n", class_weights)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define probabilistic multilabel classifiers with class weights where applicable
classifiers = {
    "Logistic Regression": MultiOutputClassifier(
        LogisticRegression(max_iter=200, class_weight='balanced')
    ),
    "Naive Bayes": MultiOutputClassifier(GaussianNB()), 
    "Random Forest": MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    ),
    "XGBoost": MultiOutputClassifier(
        xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=2, random_state=42)
    )
}

# Custom evaluation function for multilabel classification
def evaluate_multilabel_model(name, model, X_train, y_train, X_val, y_val, threshold=0.5):
    model.fit(X_train, y_train)
    
    # Predict probabilities for each label and apply threshold to convert to binary predictions
    y_val_pred_prob = model.predict_proba(X_val)
    
    # Ensure y_val_pred is in a multilabel binary format
    y_val_pred = np.array([(prob[:, 1] > threshold).astype(int) for prob in y_val_pred_prob]).T

    # Calculate evaluation metrics
    hamming = hamming_loss(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred, average='micro')
    subset_accuracy = accuracy_score(y_val, y_val_pred)  # Exact match ratio
    
    print(f"\n{name} Performance:")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"F1 Score (Micro-averaged): {f1:.4f}")
    print(f"Subset Accuracy: {subset_accuracy:.4f}")

# Train and evaluate each classifier
for name, clf in classifiers.items():
    evaluate_multilabel_model(name, clf, X_train, y_train, X_val, y_val)
