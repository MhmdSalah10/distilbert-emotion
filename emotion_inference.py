import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

EMOTIONS: List[str] = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?|[!?.,]")

def clean_tokenize(text: str) -> List[str]:
    """Lowercase, normalize GoEmotions placeholders, and keep apostrophe words + !?.,"""
    text = text.lower()
    text = re.sub(r"\[name\]", "name", text)
    text = re.sub(r"\[religion\]", "religion", text)
    return TOKEN_PATTERN.findall(text)


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention: score_t = v^T tanh(W h_t) with masked softmax."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: (B, L, H), mask: (B, L) boolean where True = valid token
        energy = torch.tanh(self.W(hidden_states))          # (B, L, H)
        scores = self.v(energy).squeeze(-1)                 # (B, L)
        scores = scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)        # (B, L)
        context = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)  # (B, H)
        return context, attn_weights


class SeqClassifier(nn.Module):
    """Unified LSTM / GRU sequence classifier with optional bidirectionality and additive attention."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 100,
        hidden_size: int = 128,
        num_classes: int = 6,
        dropout: float = 0.3,
        rnn_type: str = "lstm",
        bidirectional: bool = False,
        use_attention: bool = False,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = True,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.pad_idx = pad_idx

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings.clone().detach().float(),
                freeze=freeze_embeddings,
                padding_idx=pad_idx,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)

        self.dropout = nn.Dropout(dropout)
        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=emb_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )

        out_dim = hidden_size * (2 if bidirectional else 1)
        self.attention = AdditiveAttention(out_dim) if use_attention else None
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(
        self, words: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # words: (B, L), lengths: (B,)
        B, L = words.shape
        emb = self.dropout(self.embedding(words))  # (B, L, E)

        clamped_lengths = lengths.clamp(min=1, max=L).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, clamped_lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, hidden = self.rnn(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=L
        )  # (B, L, H_out)

        attn_weights = None
        if self.use_attention and self.attention is not None:
            device = words.device
            arange = torch.arange(L, device=device).unsqueeze(0)
            mask = arange < lengths.to(device).unsqueeze(1)  # (B, L)
            rep, attn_weights = self.attention(rnn_out, mask)
        else:
            h_n = hidden[0] if self.rnn_type == "lstm" else hidden
            if self.bidirectional:
                rep = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                rep = h_n[-1]

        logits = self.fc(self.dropout(rep))
        return logits, attn_weights


def _encode_tokens(tokens: List[str], vocab: Dict[str, int], max_len: int) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    truncated = tokens[:max_len] if tokens else ["<UNK>"]
    length = len(truncated)
    unk_idx = vocab.get("<UNK>", 1)
    pad_idx = vocab.get("<PAD>", 0)
    ids = [vocab.get(tok, unk_idx) for tok in truncated]
    if len(ids) < max_len:
        ids = ids + [pad_idx] * (max_len - len(ids))
    words_t = torch.tensor([ids], dtype=torch.long)
    lengths_t = torch.tensor([length], dtype=torch.long)
    return words_t, lengths_t, truncated


def _load_rnn_bundle(folder: Path, rnn_cfg: Dict, num_labels: int, device: torch.device):
    vocab_path = folder / "vocab.json"
    weights_path = folder / "model.pt"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    model = SeqClassifier(
        vocab_size=len(vocab),
        emb_dim=rnn_cfg.get("emb_dim", 100),
        hidden_size=rnn_cfg.get("hidden_size", 128),
        num_classes=num_labels,
        dropout=rnn_cfg.get("dropout", 0.3),
        rnn_type=rnn_cfg.get("rnn_type", "lstm"),
        bidirectional=rnn_cfg.get("bidirectional", True),
        use_attention=rnn_cfg.get("use_attention", True),
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, vocab


def load_predictor(root: Union[str, Path] = "."):
    """
    Loads the best classifier and the BiLSTM+Attention explainer from root.
    Returns (predict_fn, deployment_config).
    """
    root_path = Path(root)
    cfg_path = root_path / "deployment_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing deployment_config.json in {root_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    labels: List[str] = cfg.get("labels", EMOTIONS)
    model_type: str = cfg.get("model_type", "rnn")
    rnn_cfg: Dict = cfg.get("rnn", {})
    max_len: int = int(rnn_cfg.get("max_len", 64))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load primary prediction model (either Transformer or RNN)
    primary_rnn = None
    primary_vocab = None
    hf_model = None
    hf_tokenizer = None

    if model_type == "transformer":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        hub_repo = cfg.get("hub_repo")
        model_source = hub_repo if hub_repo else str(root_path / "model")
        hf_tokenizer = AutoTokenizer.from_pretrained(model_source)
        hf_model = AutoModelForSequenceClassification.from_pretrained(model_source)
        hf_model.to(device)
        hf_model.eval()
    else:
        model_folder = root_path / "model" if (root_path / "model").exists() else root_path
        primary_rnn, primary_vocab = _load_rnn_bundle(
            model_folder, rnn_cfg, len(labels), device
        )

    # 2. Load BiLSTM+Attention model for token-level attention weights (checks both subfolder and root)
    attn_rnn = None
    attn_vocab = None
    attn_dir = root_path / "attention_model" if (root_path / "attention_model").exists() else root_path
    attn_cfg = {
        "emb_dim": rnn_cfg.get("emb_dim", 100),
        "hidden_size": rnn_cfg.get("hidden_size", 128),
        "dropout": rnn_cfg.get("dropout", 0.3),
        "rnn_type": "lstm",
        "bidirectional": True,
        "use_attention": True,
    }
    if primary_rnn is not None and rnn_cfg.get("use_attention", False):
        attn_rnn, attn_vocab = primary_rnn, primary_vocab
    elif (attn_dir / "model.pt").exists() and (attn_dir / "vocab.json").exists():
        attn_rnn, attn_vocab = _load_rnn_bundle(attn_dir, attn_cfg, len(labels), device)

    def predict(text: str) -> Dict:
        raw_tokens = clean_tokenize(text)
        if not raw_tokens:
            raw_tokens = ["<UNK>"]

        # Compute attention weights from BiLSTM+Attention
        tokens_out = raw_tokens[:max_len]
        attn_list: Optional[List[float]] = None
        if attn_rnn is not None and attn_vocab is not None:
            w_t, l_t, tokens_out = _encode_tokens(raw_tokens, attn_vocab, max_len)
            with torch.no_grad():
                logits_attn, attn_w = attn_rnn(w_t.to(device), l_t.to(device))
                if attn_w is not None:
                    valid_len = int(l_t.item())
                    attn_list = [float(x) for x in attn_w[0, :valid_len].cpu().tolist()]

        # Compute class probabilities from best model
        if model_type == "transformer" and hf_model is not None and hf_tokenizer is not None:
            inputs = hf_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
                padding=True,
            )
            inputs_dev = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = hf_model(**inputs_dev, output_attentions=True)
                logits = outputs.logits
                probs_t = torch.softmax(logits, dim=-1)[0].cpu().numpy()

                # Fallback: Extract attention from DistilBERT if BiLSTM weights aren't loaded
                if attn_list is None and getattr(outputs, "attentions", None) is not None:
                    last_layer_attn = outputs.attentions[-1][0]
                    cls_attn = last_layer_attn.mean(dim=0)[0]
                    input_ids = inputs["input_ids"][0].tolist()
                    raw_hf_tokens = hf_tokenizer.convert_ids_to_tokens(input_ids)

                    valid_tokens, valid_weights = [], []
                    for tok, w in zip(raw_hf_tokens, cls_attn.cpu().tolist()):
                        if tok not in hf_tokenizer.all_special_tokens:
                            clean_tok = tok.replace("##", "")
                            if clean_tok:
                                valid_tokens.append(clean_tok)
                                valid_weights.append(float(w))

                    if valid_weights:
                        total_w = sum(valid_weights) or 1.0
                        tokens_out = valid_tokens
                        attn_list = [w / total_w for w in valid_weights]
        else:
            w_t, l_t, tokens_out = _encode_tokens(raw_tokens, primary_vocab, max_len)
            with torch.no_grad():
                logits, _ = primary_rnn(w_t.to(device), l_t.to(device))
                probs_t = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        probs_dict = {labels[i]: float(probs_t[i]) for i in range(len(labels))}
        best_idx = int(np.argmax(probs_t))
        return {
            "label": labels[best_idx],
            "probs": probs_dict,
            "tokens": tokens_out,
            "attention": attn_list,
        }

    return predict, cfg
