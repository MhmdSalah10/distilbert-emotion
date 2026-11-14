# Emotion Classification with Attention — Deployment Package

- **Selected Best Model**: `DistilBERT` (`model_type=transformer`)
- **Validation Macro F1**: `0.6931`
- **Test Macro F1**: `0.6891`
- **Test Accuracy**: `0.7875`

## Quickstart (Local Deployment)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Package Contents
- `app.py`: Interactive Streamlit web application
- `emotion_inference.py`: Shared tokenizer, `SeqClassifier`, and `load_predictor`
- `deployment_config.json`: Model metadata and test evaluation metrics
- `model/`: Best model weights and tokenizer/vocabulary
- `attention_model/`: Packaged BiLSTM+Attention checkpoint for token attention visualization
- `results/`: Evaluation CSV tables and comparison/attention PNG plots
