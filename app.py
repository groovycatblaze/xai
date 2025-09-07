import streamlit as st
from transformers import pipeline
from lime.lime_text import LimeTextExplainer
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Text XAI (LIME)")
st.title("Explainable AI for Text — LIME")

@st.cache_resource
def load_pipeline():
    return pipeline("sentiment-analysis")

clf = load_pipeline()
explainer = LimeTextExplainer(class_names=["NEGATIVE", "POSITIVE"])

text = st.text_area("Enter text", "I love AI but training deep models is expensive.")
num_feats = st.slider("Num features", 4, 10, 6)

def predict_proba(texts):
    preds = clf(texts)
    probs = []
    for p in preds:
        if p["label"] == "POSITIVE":
            probs.append([1-p["score"], p["score"]])
        else:
            probs.append([p["score"], 1-p["score"]])
    return np.array(probs)

if st.button("Explain"):
    pred = clf(text)[0]
    st.write(f"**Prediction:** {pred['label']} (score={pred['score']:.3f})")
    exp = explainer.explain_instance(text, predict_proba, num_features=num_feats)
    fig = exp.as_pyplot_figure()
    st.pyplot(fig)
