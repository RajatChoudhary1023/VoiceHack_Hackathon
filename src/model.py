from xgboost import XGBClassifier

def train_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=10  # IMPORTANT (imbalanced data)
    )
    
    model.fit(X_train, y_train)
    
    return model