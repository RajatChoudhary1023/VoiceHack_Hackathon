from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate_model(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    
    threshold = 0.7
    preds = (probs > threshold).astype(int)

    f1 = f1_score(y_val, preds)
    precision = precision_score(y_val, preds)
    recall = recall_score(y_val, preds)
    
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")