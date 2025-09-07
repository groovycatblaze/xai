from transformers import pipeline
from lime.lime_text import LimeTextExplainer
import numpy as np

clf = pipeline("sentiment-analysis")  # downloads small default model

text = "I love AI but training deep models is expensive."

pred = clf(text)[0]
print("Prediction:", pred)

class_names = ["NEGATIVE", "POSITIVE"]
explainer = LimeTextExplainer(class_names=class_names)

def predict_proba(texts):
    preds = clf(texts)
    probs = []
    for p in preds:
        if p["label"] == "POSITIVE":
            probs.append([1-p["score"], p["score"]])
        else:
            probs.append([p["score"], 1-p["score"]])
    return np.array(probs)

exp = explainer.explain_instance(text, predict_proba, num_features=6)
print("Top features:", exp.as_list())
