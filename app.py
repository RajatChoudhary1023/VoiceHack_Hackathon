import streamlit as st
import pandas as pd

from src.feature_engineering import advanced_features
from src.model import train_model
from src.explain import explain_prediction

# -----------------------------
# Load and train model (once)
# -----------------------------
@st.cache_resource
def load_model():
    train_df = pd.read_csv("data/hackathon_train.csv")
    
    X_train, _ = advanced_features(train_df)
    y_train = train_df['has_ticket']
    
    model = train_model(X_train, y_train)
    return model


def estimate_features(transcript):
    text = transcript.lower()
    
    question_count = text.count('?')
    answered_count = text.count('[user]')
    
    response_completeness = (
        answered_count / question_count if question_count > 0 else 0
    )
    
    user_word_count = text.count('[user]')
    agent_word_count = text.count('[agent]')
    
    call_duration = len(text) // 5  # rough proxy
    
    return {
        "call_duration": call_duration,
        "response_completeness": response_completeness,
        "whisper_mismatch_count": 0,
        "outcome": "completed"
    }


model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("📞 AI Call Quality Analyzer")
st.write("Detect failed healthcare calls with explanation")

transcript = st.text_area("Paste Call Transcript Here")

if st.button("Analyze Call"):
    if transcript.strip() == "":
        st.warning("Please enter a transcript")
    else:
        features = estimate_features(transcript)

        input_df = pd.DataFrame([{
            "transcript_text": transcript,
            **features
        }])
        
        # Generate features
        X_input, processed_df = advanced_features(input_df)
        
        # Predict
        prob = model.predict_proba(X_input)[0][1]
        pred = int(prob > 0.7)
        
        # Display result
        st.subheader("Prediction")
        
        if pred == 1:
            st.error(f"⚠️ Ticket Detected (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ No Issue (Confidence: {1-prob:.2f})")
        
        # Explanation
        st.subheader("Explanation")
        reasons = explain_prediction(processed_df.iloc[0])
        
        if reasons:
            for r in reasons:
                st.write(f"- {r}")
        else:
            st.write("No major issues detected")