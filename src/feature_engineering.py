def advanced_features(df):
    df = df.copy()
    
    # -----------------------------
    # Basic cleaning
    # -----------------------------
    df['transcript_text'] = df['transcript_text'].fillna("").str.lower()
    
    df['response_completeness'] = df['response_completeness'].fillna(0)
    df['whisper_mismatch_count'] = df['whisper_mismatch_count'].fillna(0)

    # -----------------------------
    # 1. Negative intent
    # -----------------------------
    negative_keywords = [
        "not interested", "don't want", "stop calling",
        "prefer you not to call", "rather not",
        "i'm busy", "maybe later"
    ]

    df['negative_intent_score'] = df['transcript_text'].apply(
        lambda x: sum(word in x for word in negative_keywords)
    )

    # -----------------------------
    # 2. Incomplete signal
    # -----------------------------
    df['incomplete_signal'] = df['transcript_text'].apply(
        lambda x: int("hello?" in x or "are you there" in x)
    )

    # -----------------------------
    # 3. Conversation length
    # -----------------------------
    df['conversation_length'] = df['transcript_text'].apply(len)

    # -----------------------------
    # 4. Question count
    # -----------------------------
    df['num_questions'] = df['transcript_text'].apply(lambda x: x.count('?'))

    # -----------------------------
    # 5. User response count
    # -----------------------------
    df['user_turns'] = df['transcript_text'].apply(lambda x: x.count('[user]'))

    # -----------------------------
    # 6. Response ratio
    # -----------------------------
    df['response_ratio'] = df.apply(
        lambda row: row['user_turns'] / row['num_questions']
        if row['num_questions'] > 0 else 0,
        axis=1
    )

    # -----------------------------
    # 7. Medical violation (STRICT)
    # -----------------------------
    medical_keywords = [
        "you should take",
        "increase your dose",
        "decrease your dose"
    ]

    df['medical_violation'] = df['transcript_text'].apply(
        lambda x: int(any(word in x for word in medical_keywords))
    )

    # -----------------------------
    # FINAL FEATURES
    # -----------------------------
    features = df[
        [
            'response_completeness',
            'whisper_mismatch_count',
            'negative_intent_score',
            'incomplete_signal',
            'conversation_length',
            'num_questions',
            'user_turns',
            'response_ratio',
            'medical_violation'
        ]
    ]

    return features, df