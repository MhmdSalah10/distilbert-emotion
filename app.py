import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from emotion_inference import EMOTIONS, load_predictor

st.set_page_config(page_title="Emotion Classifier with Attention", page_icon="😊", layout="centered")

EMOJI_MAP = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
    "disgust": "🤢",
}

DEFAULT_TEXT = "I am so grateful and thrilled that everything worked out today!"
EXAMPLES = [
    "I am so grateful and thrilled that everything worked out today!",
    "It breaks my heart how much I miss my old friends, I feel so lonely.",
    "Wait what?! I had no idea they were planning a surprise announcement!",
]

ROOT_DIR = Path(__file__).resolve().parent
CFG_PATH = ROOT_DIR / "deployment_config.json"

if not CFG_PATH.exists():
    st.error("Missing deployment_config.json in the application directory.")
    st.stop()

with open(CFG_PATH, "r", encoding="utf-8") as f:
    deploy_cfg = json.load(f)

best_model_name = deploy_cfg.get("best_model", "Unknown")
model_type = deploy_cfg.get("model_type", "rnn")
hub_repo = deploy_cfg.get("hub_repo")
model_dir = ROOT_DIR / "model"

if not (model_type == "transformer" and hub_repo):
    if not model_dir.exists():
        st.error("Local 'model/' folder is missing and no 'hub_repo' was configured in deployment_config.json.")
        st.stop()

@st.cache_resource
def get_cached_predictor():
    return load_predictor(ROOT_DIR)

predict_fn, cfg = get_cached_predictor()

# Sidebar with model info and metrics
with st.sidebar:
    st.header("Model Details")
    st.write(f"**Best Model:** `{best_model_name}`")
    st.write(f"**Architecture Type:** `{model_type}`")
    if hub_repo and model_type == "transformer":
        st.write(f"**HF Hub Repo:** `{hub_repo}`")
    metrics = cfg.get("metrics", {})
    if metrics:
        st.subheader("Evaluation Metrics")
        if "accuracy" in metrics:
            st.metric("Test Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        if "macro_f1" in metrics:
            st.metric("Test Macro F1", f"{metrics['macro_f1'] * 100:.2f}%")
        if "val_macro_f1" in metrics:
            st.metric("Validation Macro F1", f"{metrics['val_macro_f1'] * 100:.2f}%")

st.title("Emotion Classification with Attention")
st.caption(f"Powered by **{best_model_name}** (GoEmotions 6-Class Ekman Taxonomy)")

if "input_text" not in st.session_state:
    st.session_state["input_text"] = DEFAULT_TEXT

def set_example_text(text_val: str):
    st.session_state["input_text"] = text_val

st.write("**Try an example sentence:**")
cols = st.columns(3)
for idx, ex_text in enumerate(EXAMPLES):
    cols[idx].button(
        f"Example {idx + 1}",
        key=f"ex_btn_{idx}",
        on_click=set_example_text,
        args=(ex_text,),
    )

user_text = st.text_area(
    "Enter text to classify (press Ctrl+Enter to analyze):",
    key="input_text",
    max_chars=500,
    height=110,
)

if user_text and user_text.strip():
    result = predict_fn(user_text.strip())
    pred_label = result["label"]
    probs = result["probs"]
    conf = probs.get(pred_label, 0.0)
    emoji = EMOJI_MAP.get(pred_label, "✨")

    st.subheader("Prediction Result")
    st.markdown(f"### {emoji} **{pred_label.upper()}** — Confidence: `{conf * 100:.1f}%`")

    prob_df = pd.DataFrame(
        {"Probability": [probs.get(e, 0.0) for e in EMOTIONS]},
        index=EMOTIONS,
    )
    st.bar_chart(prob_df)

    st.subheader("Token Attention Heatmap")
    tokens = result.get("tokens") or []
    attn = result.get("attention")
    if tokens and attn:
        max_a = max(max(attn), 1e-6)
        spans = []
        for tok, weight in zip(tokens, attn):
            alpha = min(max((weight / max_a) * 0.85, 0.08), 0.90)
            safe_tok = html.escape(tok)
            spans.append(
                f'<span title="weight: {weight:.3f}" style="display:inline-block;margin:3px 4px;padding:4px 8px;'
                f'border-radius:6px;background-color:rgba(239,68,68,{alpha:.3f});color:#111827;'
                f'font-family:monospace;font-size:0.95rem;">{safe_tok} <small>({weight:.2f})</small></span>'
            )
        html_block = "<div style='line-height:2.0;padding:8px;border:1px solid #e5e7eb;border-radius:8px;'>" + "".join(spans) + "</div>"
        st.markdown(html_block, unsafe_allow_html=True)
    else:
        st.info("Attention weights are unavailable for this input.")
    st.caption("Note: Token-level attention weights are computed by the packaged BiLSTM+Attention model.")
