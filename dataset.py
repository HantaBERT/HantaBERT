"""
PyTorch Dataset for multi-task hantavirus classification.
Tokenization pattern reused from DNABERT_2/finetune/train.py (DataCollatorForSupervisedDataset).
"""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

import config


class HantavirusDataset(Dataset):
    def __init__(self, df, tokenizer: PreTrainedTokenizer):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row["sequence"]).upper().strip()

        # Reuses get_batch_token pattern from DNABERT_S/train/pretrain/training.py:29-38
        enc = self.tokenizer(
            seq,
            max_length=config.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      enc["input_ids"].squeeze(0),      # [L]
            "attention_mask": enc["attention_mask"].squeeze(0),  # [L]
            "species_id": torch.tensor(int(row["species_id"]), dtype=torch.long),
            "host_id":    torch.tensor(int(row["host_id"]),    dtype=torch.long),
            "geo_id":     torch.tensor(int(row["geo_id"]),     dtype=torch.long),
        }
