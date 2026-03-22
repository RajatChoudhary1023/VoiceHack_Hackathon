def explain_prediction(row):
    reasons = []

    if row['negative_intent_score'] > 0:
        reasons.append("Negative user intent detected")

    if row['mismatch_flag'] == 1:
        reasons.append("Outcome mismatch")

    if row['incomplete_signal'] == 1:
        reasons.append("Call seems incomplete")

    if row['medical_violation'] == 1:
        reasons.append("Agent gave medical advice")

    if row['response_completeness'] < 0.7:
        reasons.append("Low response completeness")

    return reasons