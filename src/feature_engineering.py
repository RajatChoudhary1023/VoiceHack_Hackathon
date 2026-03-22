def advanced_features(df):
    df = df.copy()
    
    # Fill missing values
    df['call_duration'] = df['call_duration'].fillna(0)
    df['response_completeness'] = df['response_completeness'].fillna(0)
    df['whisper_mismatch_count'] = df['whisper_mismatch_count'].fillna(0)

    # Negative intent
    negative_keywords = [
        "not interested", "don't want", "stop calling",
        "prefer you not to call", "rather not",
        "i'm busy", "maybe later"
    ]

    df['negative_intent_score'] = df['transcript_text'].apply(
        lambda x: sum(word in x.lower() for word in negative_keywords)
    )

    # Outcome mismatch
    df['is_completed'] = (df['outcome'] == 'completed').astype(int)

    df['mismatch_flag'] = (
        (df['negative_intent_score'] > 0) & (df['is_completed'] == 1)
    ).astype(int)

    # Incomplete signal
    df['incomplete_signal'] = df['transcript_text'].apply(
        lambda x: int("hello?" in x.lower() or "are you there" in x.lower())
    )

    # Medical violation
    medical_keywords = [
        "dosage adjustment", "increase dose",
        "reduce dose", "consult doctor"
    ]

    df['medical_violation'] = df['transcript_text'].apply(
        lambda x: int(any(word in x.lower() for word in medical_keywords))
    )

    # Conversation length
    df['conversation_length'] = df['transcript_text'].apply(lambda x: len(x))

    # Number of questions
    df['num_questions_asked'] = df['transcript_text'].apply(
        lambda x: x.count('?')
    )

    features = df[
        [
            'call_duration',
            'response_completeness',
            'whisper_mismatch_count',
            'negative_intent_score',
            'mismatch_flag',
            'incomplete_signal',
            'medical_violation',
            'conversation_length',
            'num_questions_asked'
        ]
    ]

    return features