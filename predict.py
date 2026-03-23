import pandas as pd
from src.feature_engineering import advanced_features
from src.model import train_model

# Load data
train_df = pd.read_csv("data/hackathon_train.csv")
test_df = pd.read_csv("data/hackathon_test.csv")

# Prepare training data
X_train = advanced_features(train_df)
y_train = train_df['has_ticket']

# Train model
model = train_model(X_train, y_train)

# Prepare test data
X_test = advanced_features(test_df)

# Predict
probs = model.predict_proba(X_test)[:, 1]
preds = (probs > 0.7).astype(int)

# Create submission
submission = pd.DataFrame({
    "call_id": test_df["call_id"],
    "predicted_ticket": preds
})

# Save
submission.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")

print(len(test_df))
print(len(submission))

print(submission['predicted_ticket'].unique())
print(submission['predicted_ticket'].value_counts())