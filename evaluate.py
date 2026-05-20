"""
Evaluation of the trained HantaBERT model on the held-out test set.

Two evaluation paths:
  A) Neural heads  — logit argmax predictions from the fine-tuned model
  B) Sklearn SVM   — SVM trained on frozen 768-dim embeddings (comparison baseline)

Reuses calculate_metric_with_sklearn from DNABERT_2/finetune/train.py:189-207.

Usage:
    python evaluate.py
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config
from dataset import HantavirusDataset
from model import MultiTaskHantaBERT


# ── Metric helper (reused from DNABERT_2/finetune/train.py:189-207) ───────────

def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    import sklearn.metrics as skm
    valid = labels != -100
    p, l  = predictions[valid], labels[valid]
    return {
        "accuracy":              skm.accuracy_score(l, p),
        "f1_macro":              skm.f1_score(l, p, average="macro",  zero_division=0),
        "f1_weighted":           skm.f1_score(l, p, average="weighted", zero_division=0),
        "matthews_correlation":  skm.matthews_corrcoef(l, p),
        "precision_macro":       skm.precision_score(l, p, average="macro",  zero_division=0),
        "recall_macro":          skm.recall_score(l, p, average="macro",  zero_division=0),
    }


# ── Embedding + prediction extraction ─────────────────────────────────────────

def extract(model, loader, device):
    """Returns (embeddings, sp_true, host_true, geo_true, sp_pred, host_pred, geo_pred)."""
    model.eval()
    embs = []
    sp_true, host_true, geo_true = [], [], []
    sp_pred, host_pred, geo_pred = [], [], []

    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model(ids, mask)

            embs.append(out["embedding"].cpu().numpy())
            sp_true.extend(batch["species_id"].numpy())
            host_true.extend(batch["host_id"].numpy())
            geo_true.extend(batch["geo_id"].numpy())
            sp_pred.extend(out["species_logits"].argmax(1).cpu().numpy())
            host_pred.extend(out["host_logits"].argmax(1).cpu().numpy())
            geo_pred.extend(out["geo_logits"].argmax(1).cpu().numpy())

    return (
        np.vstack(embs),
        np.array(sp_true), np.array(host_true), np.array(geo_true),
        np.array(sp_pred), np.array(host_pred), np.array(geo_pred),
    )


# ── Confusion matrix plot ──────────────────────────────────────────────────────

def plot_cm(cm, labels, title, path):
    fig_size = max(8, len(labels))
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size - 2))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=labels, yticklabels=labels,
        cmap="Blues", ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.xticks(rotation=45, ha="right", fontsize=max(6, 10 - len(labels) // 4))
    plt.yticks(rotation=0,  fontsize=max(6, 10 - len(labels) // 4))
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maps   = json.load(open(f"{config.OUTPUT_DIR}/label_maps.json"))
    sp_cls = maps["species"]
    ho_cls = maps["host"]
    ge_cls = maps["geo"]

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH, trust_remote_code=True)
    model = MultiTaskHantaBERT(len(sp_cls), len(ho_cls), len(ge_cls)).to(device)
    model.load_state_dict(
        torch.load(f"{config.OUTPUT_DIR}/best_model.pt", map_location=device)
    )

    # ── Loaders ──────────────────────────────────────────────────────────────
    train_df = pd.read_csv(f"{config.OUTPUT_DIR}/train.csv")
    test_df  = pd.read_csv(f"{config.OUTPUT_DIR}/test.csv")

    te_loader = DataLoader(HantavirusDataset(test_df,  tokenizer), batch_size=64,
                           shuffle=False, num_workers=4)
    tr_loader = DataLoader(HantavirusDataset(train_df, tokenizer), batch_size=64,
                           shuffle=False, num_workers=4)

    # ── Path A: Neural head predictions ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("PATH A — NEURAL HEAD PREDICTIONS (test set)")
    print("=" * 60)

    te_emb, sp_t, ho_t, ge_t, sp_p, ho_p, ge_p = extract(model, te_loader, device)

    for name, true, pred, labels, fname in [
        ("Species / Lineage", sp_t, sp_p, sp_cls, "cm_species.png"),
        ("Host",              ho_t, ho_p, ho_cls, "cm_host.png"),
        ("Geography",         ge_t, ge_p, ge_cls, "cm_geo.png"),
    ]:
        print(f"\n--- {name} ---")
        print(classification_report(true, pred, target_names=labels, digits=3, zero_division=0))
        metrics = calculate_metric_with_sklearn(pred, true)
        print("Summary:", {k: f"{v:.3f}" for k, v in metrics.items()})
        plot_cm(
            confusion_matrix(true, pred),
            labels, f"{name} Confusion Matrix",
            f"{config.OUTPUT_DIR}/{fname}",
        )

    # ── Path B: Sklearn SVM on frozen embeddings ──────────────────────────────
    print("\n" + "=" * 60)
    print("PATH B — SVM ON FROZEN EMBEDDINGS (comparison baseline)")
    print("=" * 60)

    tr_emb, tr_sp, tr_ho, tr_ge, _, _, _ = extract(model, tr_loader, device)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(tr_emb)
    X_te   = scaler.transform(te_emb)

    for name, y_tr, y_te, labels in [
        ("Species", tr_sp, sp_t, sp_cls),
        ("Host",    tr_ho, ho_t, ho_cls),
        ("Geo",     tr_ge, ge_t, ge_cls),
    ]:
        print(f"\n--- {name} (SVM) ---")
        clf   = SVC(kernel="rbf", C=10, class_weight="balanced", random_state=config.SEED)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        print(classification_report(y_te, preds, target_names=labels, digits=3, zero_division=0))

    # ── Training history plot ─────────────────────────────────────────────────
    hist_path = f"{config.OUTPUT_DIR}/training_history.csv"
    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(hist["epoch"], hist["train_loss"], label="train")
        axes[0].plot(hist["epoch"], hist["val_loss"],   label="val")
        axes[0].set_title("Loss"); axes[0].legend()
        for col, label in [("val_sp_acc","species"), ("val_host_acc","host"), ("val_geo_acc","geo")]:
            axes[1].plot(hist["epoch"], hist[col], label=label)
        axes[1].set_title("Val accuracy"); axes[1].legend()
        plt.tight_layout()
        plt.savefig(f"{config.OUTPUT_DIR}/training_curves.png", dpi=120)
        plt.close()
        print(f"\nSaved training_curves.png")


if __name__ == "__main__":
    evaluate()
