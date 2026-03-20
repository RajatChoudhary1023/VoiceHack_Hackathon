from src.data_loader import load_data

# Load data
train_df, val_df = load_data("data/hackathon_train.csv", "data/hackathon_val.csv")

print(train_df.shape)