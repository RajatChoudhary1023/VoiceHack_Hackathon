from src.data_loader import load_data
from src.feature_engineering import advanced_features
from src.model import train_model
from src.evaluate import evaluate_model

# Load data
train_df, val_df = load_data("data/hackathon_train.csv", "data/hackathon_val.csv")

# Features
X_train = advanced_features(train_df)
y_train = train_df['has_ticket']

X_val = advanced_features(val_df)
y_val = val_df['has_ticket']

# Train
model = train_model(X_train, y_train)

# Evaluate
# evaluate_model(model, X_val, y_val)
print("\n--- TRAIN PERFORMANCE ---")
evaluate_model(model, X_train, y_train)

print("\n--- VALIDATION PERFORMANCE ---")
evaluate_model(model, X_val, y_val)