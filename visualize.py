"""
UMAP visualizations of HantaBERT embeddings.

Plot 1 — All species overview:   full dataset colored by species_group
Plot 2 — Within-species strain clustering:
         Seoul virus (1,391 seqs) and Puumala virus (2,709 seqs)
         3 panels each: by RNA segment / host / derived geography

Mean-pool embedding reused from DNABERT_S/train/pretrain/models/dnabert_s.py:55-59.

Usage:
    python visualize.py
"""

import json

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config
from dataset import HantavirusDataset
from model import MultiTaskHantaBERT


# ── Embedding extraction ───────────────────────────────────────────────────────

def get_embeddings(model, df, tokenizer, device, batch_size=64):
    model.eval()
    loader = DataLoader(
        HantavirusDataset(df, tokenizer),
        batch_size=batch_size, shuffle=False, num_workers=4,
    )
    embs = []
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            embs.append(out["embedding"].cpu().numpy())
    return np.vstack(embs)


# ── Plot helpers ───────────────────────────────────────────────────────────────

def scatter_colored(ax, xy, color_series, palette, title, legend_loc="best"):
    for label, color in palette.items():
        mask = color_series == label
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            c=color, label=label, s=6, alpha=0.65, linewidths=0,
        )
    ax.set_title(title, fontsize=10)
    ax.legend(loc=legend_loc, fontsize=6, markerscale=3, framealpha=0.6)
    ax.set_xticks([]); ax.set_yticks([])


# ── Plot 1: all-species overview ───────────────────────────────────────────────

def plot_all_species(model, df, tokenizer, device, maps):
    print("Extracting embeddings for all-species UMAP...")
    emb = get_embeddings(model, df, tokenizer, device)

    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, metric="cosine", random_state=config.SEED)
    xy      = reducer.fit_transform(emb)

    sp_classes = maps["species"]
    n_sp       = len(sp_classes)
    palette    = {sp: cm.tab20(i / max(n_sp - 1, 1)) for i, sp in enumerate(sp_classes)}

    fig, ax = plt.subplots(figsize=(14, 10))
    for sp, color in palette.items():
        mask = df["species_group"].values == sp
        ax.scatter(xy[mask, 0], xy[mask, 1], c=[color], label=sp, s=5, alpha=0.6, linewidths=0)
    ax.legend(loc="upper right", fontsize=5, markerscale=2, framealpha=0.7,
              ncol=max(1, n_sp // 12))
    ax.set_title("UMAP — All species (colored by lineage)", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()

    out_path = f"{config.OUTPUT_DIR}/umap_all_species.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ── Plot 2: within-species strain clustering ───────────────────────────────────

SEG_PALETTE  = {"S": "steelblue",    "M": "darkorange",   "L": "forestgreen"}
HOST_PALETTE = {"Rodent": "saddlebrown", "Human": "crimson", "Others": "slategray",
                "Unknown": "lightgray"}
GEO_PALETTE  = {"Europe":  "royalblue", "Americas": "gold",  "Asia":    "tomato",
                "Africa":  "seagreen",  "Oceania":  "purple","Other":   "lightgray",
                "Unknown": "gainsboro"}


def plot_within_species(model, df_full, tokenizer, device, species_name):
    df_f = df_full[df_full["species_label"] == species_name].copy().reset_index(drop=True)
    if len(df_f) < 20:
        print(f"Skipping {species_name}: only {len(df_f)} rows")
        return

    print(f"Within-species UMAP for {species_name} (n={len(df_f)})...")
    emb = get_embeddings(model, df_f, tokenizer, device)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.05, metric="cosine", random_state=config.SEED)
    xy      = reducer.fit_transform(emb)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Within-species strain clustering: {species_name}", fontsize=12)

    scatter_colored(axes[0], xy, df_f["segment_type"],
                    SEG_PALETTE,  "By RNA segment (S / M / L)")
    scatter_colored(axes[1], xy, df_f["host_label"],
                    HOST_PALETTE, "By host")
    scatter_colored(axes[2], xy, df_f.get("geo_derived", df_f.get("geo_label_broad")),
                    GEO_PALETTE,  "By geography (derived)")

    plt.tight_layout()
    safe_name = species_name.replace(" ", "_").replace("/", "-")
    out_path  = f"{config.OUTPUT_DIR}/umap_{safe_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maps   = json.load(open(f"{config.OUTPUT_DIR}/label_maps.json"))

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH, trust_remote_code=True)
    model = MultiTaskHantaBERT(len(maps["species"]), len(maps["host"]), len(maps["geo"])).to(device)
    model.load_state_dict(
        torch.load(f"{config.OUTPUT_DIR}/best_model.pt", map_location=device)
    )

    # Combine train + test for richer visualization
    df_full = pd.concat([
        pd.read_csv(f"{config.OUTPUT_DIR}/train.csv"),
        pd.read_csv(f"{config.OUTPUT_DIR}/test.csv"),
    ]).reset_index(drop=True)

    # ── Plot 1 ──────────────────────────────────────────────────────────────
    plot_all_species(model, df_full, tokenizer, device, maps)

    # ── Plot 2: two best-sampled species ────────────────────────────────────
    for species in ["Orthohantavirus seoulense", "Orthohantavirus puumalaense"]:
        plot_within_species(model, df_full, tokenizer, device, species)


if __name__ == "__main__":
    visualize()
