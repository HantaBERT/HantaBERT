"""
Multi-task fine-tuning of DNABERT-2 on hantavirus sequences.

Run preprocess.py first to generate output/train.csv, val.csv, label_maps.json,
and class_weights.json.

Usage:
    python train.py
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import config
from dataset import HantavirusDataset
from model import MultiTaskHantaBERT


CHECKPOINT_PATH = os.path.join(config.OUTPUT_DIR, "latest_checkpoint.pt")
BEST_MODEL_PATH = os.path.join(config.OUTPUT_DIR, "best_model.pt")


def _steps_per_epoch(loader):
    return int(np.ceil(len(loader) / config.GRAD_ACCUM_STEPS))


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, best_val_loss, history):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": best_val_loss,
            "history": history,
            "config": {
                "model_path": config.MODEL_PATH,
                "max_length": config.MAX_LENGTH,
                "batch_size": config.BATCH_SIZE,
                "grad_accum_steps": config.GRAD_ACCUM_STEPS,
                "learning_rate": config.LEARNING_RATE,
                "epochs": config.EPOCHS,
                "warmup_steps": config.WARMUP_STEPS,
                "weight_decay": config.WEIGHT_DECAY,
            },
        },
        path,
    )


def load_resume_state(model, optimizer, scheduler, scaler, device, train_loader):
    """
    Returns (start_epoch, best_val_loss, history).

    Exact resume uses latest_checkpoint.pt. If only best_model.pt exists, warm-start
    from epoch 1 weights with fresh optimizer state so the current saved model can
    continue at epoch 2. Future resumes from latest_checkpoint.pt are exact.
    """
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state") and scaler is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])

        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        history = checkpoint.get("history", [])
        print(f"Resuming exact checkpoint from epoch {checkpoint['epoch']} → starting epoch {start_epoch}")
        return start_epoch, best_val_loss, history

    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        for _ in range(_steps_per_epoch(train_loader)):
            scheduler.step()
        save_checkpoint(
            CHECKPOINT_PATH,
            1,
            model,
            optimizer,
            scheduler,
            scaler,
            float("inf"),
            [],
        )
        print("No latest checkpoint found; warm-starting from best_model.pt at epoch 2 with fresh optimizer state")
        print(f"Created warm-start checkpoint: {CHECKPOINT_PATH}")
        return 2, float("inf"), []

    return 1, float("inf"), []


def run_epoch(
    model, loader, device, criteria,
    optimizer=None, scheduler=None, grad_accum_steps=1,
    scaler=None, use_amp=False,
):
    """Single train or eval pass. Returns dict of loss + per-task accuracy."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    sp_crit, host_crit, geo_crit = criteria
    totals = {"loss": 0.0, "sp_ok": 0, "host_ok": 0, "geo_ok": 0, "n": 0}

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(loader, leave=False, desc="train" if is_train else "val"), start=1):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            sp_y = batch["species_id"].to(device)
            ho_y = batch["host_id"].to(device)
            ge_y = batch["geo_id"].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(ids, mask)

                loss = (
                    config.LAMBDA_SPECIES * sp_crit(out["species_logits"], sp_y)
                    + config.LAMBDA_HOST  * host_crit(out["host_logits"],   ho_y)
                    + config.LAMBDA_GEO   * geo_crit(out["geo_logits"],    ge_y)
                )

            if is_train:
                scaled_loss = loss / grad_accum_steps
                if scaler is not None and use_amp:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                if step % grad_accum_steps == 0 or step == len(loader):
                    if scaler is not None and use_amp:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if scaler is not None and use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            B = sp_y.size(0)
            totals["loss"]    += loss.item() * B
            totals["sp_ok"]   += (out["species_logits"].argmax(1) == sp_y).sum().item()
            totals["host_ok"] += (out["host_logits"].argmax(1)    == ho_y).sum().item()
            totals["geo_ok"]  += (out["geo_logits"].argmax(1)     == ge_y).sum().item()
            totals["n"]       += B

    N = totals["n"]
    return {
        "loss":     totals["loss"]    / N,
        "sp_acc":   totals["sp_ok"]   / N,
        "host_acc": totals["host_ok"] / N,
        "geo_acc":  totals["geo_ok"]  / N,
    }


def train():
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load preprocessed data ──────────────────────────────────────────────
    train_df = pd.read_csv(f"{config.OUTPUT_DIR}/train.csv")
    val_df   = pd.read_csv(f"{config.OUTPUT_DIR}/val.csv")
    maps     = json.load(open(f"{config.OUTPUT_DIR}/label_maps.json"))
    weights  = json.load(open(f"{config.OUTPUT_DIR}/class_weights.json"))

    n_sp   = len(maps["species"])
    n_host = len(maps["host"])
    n_geo  = len(maps["geo"])

    print(f"Classes → species:{n_sp}, host:{n_host}, geo:{n_geo}")

    # ── Tokenizer & datasets ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH, trust_remote_code=True)

    train_ds = HantavirusDataset(train_df, tokenizer)
    val_ds   = HantavirusDataset(val_df,   tokenizer)

    use_cuda = device.type == "cuda"
    num_workers = 4 if use_cuda else 0
    use_amp = use_cuda

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False,
        num_workers=num_workers, pin_memory=use_cuda
    )

    # ── Weighted losses for class imbalance ─────────────────────────────────
    def w(key, n):
        return torch.tensor(weights[key], dtype=torch.float32).to(device)

    criteria = (
        nn.CrossEntropyLoss(weight=w("species", n_sp)),
        nn.CrossEntropyLoss(weight=w("host",    n_host)),
        nn.CrossEntropyLoss(weight=w("geo",     n_geo)),
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model = MultiTaskHantaBERT(n_sp, n_host, n_geo).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Differential LR: pre-trained encoder gets gentler updates than new heads
    optimizer = AdamW(
        [
            {"params": model.encoder.parameters(),     "lr": config.LEARNING_RATE},
            {"params": model.bottleneck.parameters(),  "lr": config.LEARNING_RATE * 10},
            {"params": model.species_head.parameters(),"lr": config.LEARNING_RATE * 10},
            {"params": model.host_head.parameters(),   "lr": config.LEARNING_RATE * 10},
            {"params": model.geo_head.parameters(),    "lr": config.LEARNING_RATE * 10},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )

    total_steps = _steps_per_epoch(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    # ── Training loop ───────────────────────────────────────────────────────
    start_epoch, best_val_loss, history = load_resume_state(
        model, optimizer, scheduler, scaler, device, train_loader
    )

    for epoch in range(start_epoch, config.EPOCHS + 1):
        tr = run_epoch(
            model, train_loader, device, criteria,
            optimizer, scheduler, config.GRAD_ACCUM_STEPS,
            scaler=scaler, use_amp=use_amp,
        )
        va = run_epoch(model, val_loader, device, criteria, use_amp=use_amp)

        print(
            f"Ep {epoch:02d} | "
            f"train_loss={tr['loss']:.4f} | "
            f"val_loss={va['loss']:.4f}  "
            f"sp={va['sp_acc']:.3f}  host={va['host_acc']:.3f}  geo={va['geo_acc']:.3f}"
        )

        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}":   v for k, v in va.items()},
        })

        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("  → Saved best model")

        save_checkpoint(
            CHECKPOINT_PATH,
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            best_val_loss,
            history,
        )
        pd.DataFrame(history).to_csv(f"{config.OUTPUT_DIR}/training_history.csv", index=False)
        print(f"  → Saved resumable checkpoint: {CHECKPOINT_PATH}")

    pd.DataFrame(history).to_csv(f"{config.OUTPUT_DIR}/training_history.csv", index=False)
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
