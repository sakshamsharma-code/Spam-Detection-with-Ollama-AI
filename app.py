import streamlit as st
import joblib
import ollama 
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class KeywordFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, keywords=None):
        self.keywords = keywords if keywords else []
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rows = []
        for msg in X:
            s = str(msg).lower()
        
            rows.append([0 if kw in s else 1 for kw in self.keywords])
        import numpy as np
        return np.array(rows)

pipeline = joblib.load(r"C:\Training Project\models\pipeline.pkl")
label_map = joblib.load(r"C:\Training Project\models\label_map.pkl")

def get_ai_explanation(message, pred_numeric):
    label = label_map[pred_numeric]   
    prompt = f"You are an AI spam detector. The message was classified as: {label}.\n\nMessage: {message}\n\nExplain briefly why and also suggest steps to be safe from these kinds of messages."
    try:
        response = ollama.chat(model="gemma3", messages=[{"role":"user","content":prompt}])
        # handle Ollama response variants:
        if isinstance(response, dict) and "message" in response and "content" in response["message"]:
            return response["message"]["content"]
        if isinstance(response, dict) and "messages" in response and len(response["messages"])>0:
            return response["messages"][-1]["content"]
        return str(response)
    except Exception as e:
        return f"AI explanation unavailable: {e}"

st.set_page_config(page_title="Spam Detector", layout="centered")
st.title("📧 Spam Detector with Ollama AI")
st.write("Paste message below, click Analyze. ")

message = st.text_area("✍️ Paste your message here:")

if st.button("Analyze"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        # prediction
        pred_numeric = int(pipeline.predict([message])[0])
        proba = pipeline.predict_proba([message])[0]
        confidence = round(proba[pipeline.classes_.tolist().index(pred_numeric)] * 100, 2)
        pred_label = label_map[pred_numeric]

        
        if pred_label == "SPAM":
            st.error(f"🚨 This looks like **SPAM**")
        else:
            st.success(f"✅ This looks like **HAM** (Safe message)")

        st.progress(confidence/100)
        st.caption(f"Confidence: {confidence}%")

        with st.expander("🔎 Show detailed probabilities"):
            probs_df = pd.DataFrame([
                {"Class": label_map[int(cls)], "Probability (%)": round(float(p)*100,2)}
                for cls, p in zip(pipeline.classes_, proba)
            ])
            st.table(probs_df)

   
        with st.spinner("🧠 Generating AI explanation..."):
            response = ollama.chat(model="gemma3", messages=[
                {"role": "user", "content": f"Explain why this message is classified as {pred_label}: {message}"}
            ])
            explanation = response['message']['content']
            st.markdown("### 🧠 AI Explanation")
            st.info(explanation)


