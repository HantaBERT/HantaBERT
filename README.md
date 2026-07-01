---
license: apache-2.0
base_model: zhihan1996/DNABERT-2-117M
datasets:
- HantaBERT/Orthohantavirus-Genome-Atlas
library_name: transformers
pipeline_tag: text-classification
tags:
- biology
- genomics
- virology
- bioinformatics
- dna
- multi-task-learning
- hantavirus
- sequence-classification
- feature-extraction
metrics:
- accuracy
model-index:
- name: HantaBERT
  results:
  - task:
      type: text-classification
      name: Species Classification (23 classes)
    dataset:
      name: Hantavirus NCBI GenBank
      type: custom
      split: test
    metrics:
    - type: accuracy
      value: 0.967
      name: Test Accuracy
  - task:
      type: text-classification
      name: Host Classification (3 classes)
    dataset:
      name: Hantavirus NCBI GenBank
      type: custom
      split: test
    metrics:
    - type: accuracy
      value: 0.914
      name: Test Accuracy
  - task:
      type: text-classification
      name: Geographic Origin Classification (7 classes)
    dataset:
      name: Hantavirus NCBI GenBank
      type: custom
      split: test
    metrics:
    - type: accuracy
      value: 0.805
      name: Test Accuracy
---

# HantaBERT: Multi-Task Hantavirus Classification with DNABERT-2

HantaBERT fine-tunes [DNABERT-2](https://github.com/Zhihan1996/DNABERT_2) on hantavirus RNA sequences for three simultaneous classification tasks: **species/lineage**, **host**, and **geographic origin**. A single forward pass produces predictions for all three tasks along with a 768-dimensional embedding suitable for phylogenetic visualization.

## Contents

- [1. Introduction](#1-introduction)
- [2. Dataset](#2-dataset)
- [3. Model Architecture](#3-model-architecture)
- [4. Setup Environment](#4-setup-environment)
- [5. Quick Start](#5-quick-start)
- [6. Training](#6-training)
- [7. Results](#7-results)
- [8. Visualization](#8-visualization)
- [9. File Structure](#9-file-structure)
- [10. Citation](#10-citation)

---

## 1. Introduction

Hantaviruses are zoonotic RNA viruses carried primarily by rodents and capable of causing severe hemorrhagic fever in humans. Rapid genomic classification of a new sequence — identifying the viral species, likely reservoir host, and geographic origin — is critical for outbreak response.

HantaBERT adapts the DNABERT-2 multi-species genome foundation model to this task through multi-task fine-tuning. Rather than training three independent classifiers, a shared bottleneck layer forces the encoder to learn a single representation useful for all three tasks simultaneously, which regularizes training and yields richer embeddings for downstream analysis.

**Key features:**

- Simultaneous prediction of species (23 classes), host (3 classes), and continent (7 classes) from a raw nucleotide sequence
- Mean-pool embeddings over BPE tokens — no [CLS] token dependence
- Weighted cross-entropy to handle severe class imbalance across hantavirus lineages
- Gradient accumulation + AMP (fp16) for training on consumer-grade GPUs
- UMAP visualizations of embedding space for phylogenetic exploration

---

## 2. Dataset

The dataset (`final_hantavirus_dataset.csv`) was assembled from NCBI GenBank hantavirus submissions and processed through a dedicated data pipeline. The preprocessed dataset splits, class weights, and label maps are hosted on Hugging Face at [HantaBERT/HantaBERT](https://huggingface.co/HantaBERT/HantaBERT).

| Split | Sequences |
|-------|-----------|
| Train | 7,057 |
| Val   | 882 |
| Test  | 883 |
| **Total** | **8,822** |

**Label summary after preprocessing:**

| Task | Classes | Notes |
|------|---------|-------|
| Species | 23 named + "Other" | Species with < 30 samples grouped as "Other"; `Orthohantavirus sp.` dropped (unlabeled) |
| Host | Rodent / Human / Others | 579 "Unknown" rows dropped from training |
| Geography | Americas / Europe / Asia / Africa / Oceania / Other / Unknown | Derived from GPS coordinates via reverse geocoding; fixes broken `geo_label_broad` in raw data |

**RNA segments:** S, M, and L segments are all included. Sequences already use T (not U) — no substitution needed.

---

## 3. Model Architecture

```
Input: nucleotide sequence (up to 512 BPE tokens)
        ↓
DNABERT-2-117M encoder  (fine-tuned, LR = 3e-5)
        ↓  mean pool over non-padding token positions
Shared bottleneck
  Linear(768 → 768) → LayerNorm → GELU → Dropout(0.1)  (LR = 3e-4)
        ↓
  ┌──────────────┬─────────────┬────────────┐
  species_head   host_head     geo_head
  Dropout(0.1)   Dropout(0.1)  Dropout(0.1)
  Linear(768→23) Linear(768→3) Linear(768→7)
```

**Multi-task loss:**

```
L = 1.0 × L_species  +  0.5 × L_host  +  0.3 × L_geo
```

Each per-task loss is a weighted `CrossEntropyLoss` with class weights computed from training-set frequencies (sklearn `compute_class_weight('balanced', ...)`).

---

## 4. Setup Environment

```bash
# Create a virtual environment (Python 3.9+ recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note on Triton / Flash Attention:** DNABERT-2's cached `flash_attn_triton.py` uses the deprecated Triton 1.x API (`trans_b` / `trans_a` kwargs to `tl.dot`). If you encounter `TypeError: dot() got an unexpected keyword argument 'trans_b'`, the cached file must be patched — replace `tl.dot(A, B, trans_b=True)` with `tl.dot(A, tl.trans(B))` and `tl.dot(A, B, trans_a=True)` with `tl.dot(tl.trans(A), B)`.
>
> Alternatively, `model.py` sidesteps the flash-attention path entirely by setting `attention_probs_dropout_prob = 0.1` in the loaded `BertConfig`, which forces PyTorch's standard attention kernel on all hardware.

---

## 5. Quick Start

### 5.1 Local Execution

Use this snippet if you have already run `preprocess.py` and trained or downloaded the model weights to the local `output/` directory:

```python
import json
import torch
from transformers import AutoTokenizer
from model import MultiTaskHantaBERT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load label maps produced by preprocess.py
maps = json.load(open("output/label_maps.json"))

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = MultiTaskHantaBERT(
    n_species=len(maps["species"]),
    n_host=len(maps["host"]),
    n_geo=len(maps["geo"]),
).to(device)
model.load_state_dict(torch.load("output/best_model.pt", map_location=device))
model.eval()

# Classify a hantavirus sequence (DNA encoding, T not U)
sequence = "ATGAAAGACCTTCTGAAGAAATTTGAGACCAGCAAATTCAACAAGGCCCAGGCCATGATT..."

enc = tokenizer(
    sequence,
    return_tensors="pt",
    max_length=512,
    padding="max_length",
    truncation=True,
)
with torch.no_grad():
    out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))

id2species = {v: k for k, v in maps["species"].items()}
id2host    = {v: k for k, v in maps["host"].items()}
id2geo     = {v: k for k, v in maps["geo"].items()}

print(f"Species  : {id2species[out['species_logits'].argmax(-1).item()]}")
print(f"Host     : {id2host[out['host_logits'].argmax(-1).item()]}")
print(f"Geography: {id2geo[out['geo_logits'].argmax(-1).item()]}")
print(f"Embedding: {out['embedding'].shape}")  # [1, 768]
```

### 5.2 Loading from Hugging Face

You can load the model weights and label maps directly from Hugging Face using the model/dataset repository identifier `HantaBERT/HantaBERT`:

```python
import json
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from model import MultiTaskHantaBERT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download label maps and best model weights from Hugging Face Hub
maps_path = hf_hub_download(repo_id="HantaBERT/HantaBERT", filename="output/label_maps.json")
weights_path = hf_hub_download(repo_id="HantaBERT/HantaBERT", filename="output/best_model.pt")

# Load label maps
with open(maps_path) as f:
    maps = json.load(f)

# Initialize model structure and load weights
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = MultiTaskHantaBERT(
    n_species=len(maps["species"]),
    n_host=len(maps["host"]),
    n_geo=len(maps["geo"]),
).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# Classify a hantavirus sequence (DNA encoding, T not U)
sequence = "ATGAAAGACCTTCTGAAGAAATTTGAGACCAGCAAATTCAACAAGGCCCAGGCCATGATT..."

enc = tokenizer(
    sequence,
    return_tensors="pt",
    max_length=512,
    padding="max_length",
    truncation=True,
)
with torch.no_grad():
    out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))

id2species = {v: k for k, v in maps["species"].items()}
id2host    = {v: k for k, v in maps["host"].items()}
id2geo     = {v: k for k, v in maps["geo"].items()}

print(f"Species  : {id2species[out['species_logits'].argmax(-1).item()]}")
print(f"Host     : {id2host[out['host_logits'].argmax(-1).item()]}")
print(f"Geography: {id2geo[out['geo_logits'].argmax(-1).item()]}")
print(f"Embedding: {out['embedding'].shape}")  # [1, 768]
```

---

## 6. Training

### 6.1 Preprocess the dataset

```bash
python preprocess.py
```

This will:
1. Normalize RNA segment aliases (`"s"` → `"S"`, `"S-segment"` → `"S"`, etc.)
2. Fix the broken `geo_label_broad` column via batch reverse geocoding of GPS coordinates
3. Group rare species (< 30 samples) into the "Other" class; drop `Orthohantavirus sp.`
4. Drop rows where `host_label == "Unknown"` (579 rows)
5. Stratified 80 / 10 / 10 split by species group
6. Save `output/train.csv`, `output/val.csv`, `output/test.csv`, `output/label_maps.json`, `output/class_weights.json`

### 6.2 Fine-tune

```bash
python train.py
```

Key hyperparameters (configured in `config.py`):

| Parameter | Value |
|-----------|-------|
| Backbone | `zhihan1996/DNABERT-2-117M` |
| Max sequence length | 512 BPE tokens |
| Batch size | 4 (× 4 gradient accumulation = effective 16) |
| Encoder LR | 3e-5 |
| Head / bottleneck LR | 3e-4 |
| Epochs | 10 |
| Warmup steps | 200 |
| Mixed precision | AMP (fp16) |
| λ species / host / geo | 1.0 / 0.5 / 0.3 |

The best checkpoint (lowest val loss) is saved to `output/best_model.pt`. Training resumes automatically: if `output/latest_checkpoint.pt` exists, the next run continues from the following epoch with optimizer and scaler state fully restored. Training history is written to `output/training_history.csv` after each epoch.

### 6.3 Evaluate

```bash
python evaluate.py
```

Two evaluation paths:

- **Neural heads** — forward pass on test set, `classification_report` per task, confusion matrix PNGs saved to `output/cm_*.png`
- **SVM on frozen embeddings** — extracts 768-dim bottleneck embeddings, fits `SVC(kernel='rbf', C=10, class_weight='balanced')` on train embeddings, reports on test embeddings

### 6.4 Visualize

```bash
python visualize.py
```

Generates UMAP plots saved to `output/`:

- `umap_all_species.png` — full dataset colored by species lineage
- `umap_Orthohantavirus_seoulense.png` — within-species strain clustering: 3 panels by segment / host / geography
- `umap_Orthohantavirus_puumalaense.png` — same layout for Puumala virus

---

## 7. Results

Results on the held-out **test set** (883 sequences) using the neural classification heads. Trained for 10 epochs on a single GPU with AMP.

| Task | Accuracy (test) |
|------|----------------|
| Species (23 classes) | **96.7%** |
| Host (3 classes) | **91.4%** |
| Geography (7 classes) | **80.5%** |

Training progression:

| Epoch | Train Loss | Val Loss | Val Species | Val Host | Val Geo |
|-------|-----------|----------|-------------|----------|---------|
| 1  | 3.024 | 2.443 | 78.9% | 72.8% | 72.3% |
| 2  | 1.415 | 1.828 | 87.6% | 78.3% | 66.4% |
| 3  | 1.034 | 1.195 | 92.9% | 83.1% | 74.1% |
| 4  | 0.779 | 1.121 | 94.1% | 88.5% | 72.8% |
| 5  | 0.634 | 1.150 | 94.0% | 87.1% | 77.6% |
| 6  | 0.533 | 1.035 | 95.4% | 87.5% | 79.5% |
| 7  | 0.452 | 0.869 | 96.5% | 90.4% | 78.1% |
| 8  | 0.388 | 0.802 | 96.6% | 90.7% | 79.5% |
| 9  | 0.336 | 0.818 | 96.3% | 91.4% | 80.3% |
| **10** | **0.307** | **0.797** | **96.7%** | **91.4%** | **80.5%** |

---

## 8. Visualization

UMAP of the shared bottleneck embeddings shows clear clustering by viral lineage with no explicit clustering objective — the multi-task classification signal alone drives geometrically meaningful separation.

| Plot | Description |
|------|-------------|
| `umap_all_species.png` | All 8,822 sequences colored by species lineage |
| `umap_Orthohantavirus_seoulense.png` | 1,391 Seoul virus sequences — by segment / host / geography |
| `umap_Orthohantavirus_puumalaense.png` | 2,709 Puumala virus sequences — same layout |

---

## 9. File Structure

```
HantaBERT/
├── config.py          # All hyperparameters and paths
├── preprocess.py      # Data cleaning, geo-fix, stratified split
├── dataset.py         # HantavirusDataset (PyTorch Dataset)
├── model.py           # MultiTaskHantaBERT (backbone + bottleneck + 3 heads)
├── train.py           # Training loop with AMP, grad-accum, resume support
├── evaluate.py        # Neural head evaluation + SVM-on-embeddings baseline
├── visualize.py       # UMAP embedding visualizations
├── requirements.txt   # Python dependencies
└── output/            # Generated by the scripts (not committed)
    ├── train.csv / val.csv / test.csv
    ├── label_maps.json
    ├── class_weights.json
    ├── best_model.pt
    ├── latest_checkpoint.pt
    ├── training_history.csv
    ├── training_curves.png
    ├── cm_species.png / cm_host.png / cm_geo.png
    └── umap_*.png
```

---

## 10. Citation

If you use HantaBERT in your work, please also cite the underlying foundation models:

**DNABERT-2**

```bibtex
@misc{zhou2023dnabert2,
      title={DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome},
      author={Zhihan Zhou and Yanrong Ji and Weijian Li and Pratik Dutta and Ramana Davuluri and Han Liu},
      year={2023},
      eprint={2306.15006},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN}
}
```

**DNABERT-S**

```bibtex
@misc{zhou2024dnaberts,
      title={DNABERT-S: Learning Species-Aware DNA Embedding with Genome Foundation Models},
      author={Zhihan Zhou and Winmin Wu and Harrison Ho and Jiayi Wang and Lizhen Shi and Ramana V Davuluri and Zhong Wang and Han Liu},
      year={2024},
      eprint={2402.08777},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN}
}
```

**DNABERT (original)**

```bibtex
@article{ji2021dnabert,
    author  = {Ji, Yanrong and Zhou, Zhihan and Liu, Han and Davuluri, Ramana V},
    title   = "{DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome}",
    journal = {Bioinformatics},
    volume  = {37},
    number  = {15},
    pages   = {2112-2120},
    year    = {2021},
    doi     = {10.1093/bioinformatics/btab083},
}
```

---

## License

HantaBERT is released under the **Apache License 2.0**, consistent with the DNABERT-2 backbone it is derived from. See [LICENSE](../DNABERT_2/LICENSE) for the full terms. When redistributing, include a copy of the license and attribution to the DNABERT-2 authors.