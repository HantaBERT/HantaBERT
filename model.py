"""
Multi-task HantaBERT model.

Architecture:
  DNABERT-S backbone (zhihan1996/DNABERT-S)
      ↓  mean pool over non-padding tokens          ← pattern from dnabert_s.py:55-59
  Shared bottleneck  (Linear → LayerNorm → GELU → Dropout)
      ↓
  ┌─────────────┬──────────────┬──────────────┐
  species_head   host_head      geo_head
"""

import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig

import config


class MultiTaskHantaBERT(nn.Module):
    def __init__(self, n_species: int, n_host: int, n_geo: int):
        super().__init__()

        bert_config  = BertConfig.from_pretrained(config.MODEL_PATH)
        # DNABERT-2 uses a Triton flash-attention path when attention dropout is
        # zero. On smaller GPUs that kernel can exceed shared-memory limits, so
        # use the model's PyTorch attention fallback during fine-tuning.
        bert_config.attention_probs_dropout_prob = 0.1
        self.encoder = AutoModel.from_pretrained(
            config.MODEL_PATH,
            trust_remote_code=True,
            config=bert_config,
        )
        D = 768  # DNABERT-S hidden dimension

        # Shared bottleneck — forces a representation useful across all three tasks
        self.bottleneck = nn.Sequential(
            nn.Linear(D, D),
            nn.LayerNorm(D),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Per-task heads with independent dropout to prevent cross-task gradient interference
        self.species_head = nn.Sequential(nn.Dropout(0.1), nn.Linear(D, n_species))
        self.host_head    = nn.Sequential(nn.Dropout(0.1), nn.Linear(D, n_host))
        self.geo_head     = nn.Sequential(nn.Dropout(0.1), nn.Linear(D, n_geo))

    def _mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Reuses get_mean_embeddings pattern from DNABERT_S/train/pretrain/models/dnabert_s.py:55-59
        mask   = attention_mask.unsqueeze(-1).float()          # [B, L, 1]
        summed = (hidden_states * mask).sum(dim=1)             # [B, D]
        count  = mask.sum(dim=1).clamp(min=1e-9)              # [B, 1]
        return summed / count                                  # [B, D]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # custom bert_layers.py returns a tuple, not BaseModelOutput
        hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
        pooled = self._mean_pool(hidden, attention_mask)  # [B, 768]
        shared = self.bottleneck(pooled)                                  # [B, 768]

        return {
            "species_logits": self.species_head(shared),   # [B, n_species]
            "host_logits":    self.host_head(shared),      # [B, n_host]
            "geo_logits":     self.geo_head(shared),       # [B, n_geo]
            "embedding":      shared,                      # [B, 768] — for visualize.py
        }
