"""
Preprocess the hantavirus dataset:
  1. Normalize segment type labels
  2. Fix geo_label_broad (broken in source — "Others" mislabel) by reverse-geocoding coordinates
  3. Group rare species (< MIN_SPECIES_COUNT) into "Other"
  4. Drop Unknown host rows
  5. Stratified train/val/test split
  6. Save splits, label maps, and balanced class weights
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import reverse_geocoder as rg

import config


# ── Segment normalization ──────────────────────────────────────────────────────

SEG_ALIAS = {
    "s": "S", "S segment": "S", "S-segment": "S",
    "L-segment": "L", "L segment": "L", "L-Segment": "L",
    "M-segment": "M", "M segment": "M", "M-Segment": "M",
}


def normalize_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["segment_type"] = df["segment_type"].replace(SEG_ALIAS)
    df = df[df["segment_type"].isin(["S", "M", "L"])].copy()
    return df


# ── Geography fix ──────────────────────────────────────────────────────────────

def derive_geo_from_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the broken geo_label_broad with continent derived from coordinates.
    Rows with valid lokasi_geografis_koordinat get a continent via reverse geocoding.
    Rows with no coordinates fall back to "Unknown".
    """
    df = df.copy()
    coords_raw = df["lokasi_geografis_koordinat"].fillna("").str.strip()
    has_coord  = coords_raw.astype(bool)

    coord_rows = df[has_coord].index.tolist()
    coord_pairs = []
    for idx in coord_rows:
        try:
            lat, lon = map(float, df.at[idx, "lokasi_geografis_koordinat"].split(","))
            coord_pairs.append((lat, lon))
        except Exception:
            coord_pairs.append((0.0, 0.0))

    print(f"Reverse geocoding {len(coord_pairs)} coordinates...")
    results    = rg.search(coord_pairs, verbose=False)
    continents = [config.CC_TO_CONTINENT.get(r["cc"], "Other") for r in results]

    geo_col = pd.Series("Unknown", index=df.index)
    for i, idx in enumerate(coord_rows):
        geo_col[idx] = continents[i]

    df["geo_derived"] = geo_col
    print(f"\nGeo distribution after fix:\n{df['geo_derived'].value_counts()}\n")
    return df


# ── Species grouping ───────────────────────────────────────────────────────────

def group_species(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Remove genuinely unlabeled entries
    df = df[~df["species_label"].isin(config.DROP_SPECIES)].copy()
    counts = df["species_label"].value_counts()
    keep   = set(counts[counts >= config.MIN_SPECIES_COUNT].index)
    df["species_group"] = df["species_label"].apply(
        lambda x: x if x in keep else "Other"
    )
    print(f"Species classes ({df['species_group'].nunique()}): "
          f"{sorted(df['species_group'].unique())}\n")
    return df


# ── Label encoding ─────────────────────────────────────────────────────────────

def encode_labels(df: pd.DataFrame):
    species_classes = sorted(df["species_group"].unique())
    host_classes    = config.HOST_CLASSES
    geo_classes     = sorted(df["geo_derived"].unique())

    species2id = {s: i for i, s in enumerate(species_classes)}
    host2id    = {h: i for i, h in enumerate(host_classes)}
    geo2id     = {g: i for i, g in enumerate(geo_classes)}

    df = df.copy()
    df["species_id"] = df["species_group"].map(species2id)
    df["host_id"]    = df["host_label"].map(host2id)
    df["geo_id"]     = df["geo_derived"].map(geo2id)

    label_maps = {
        "species": species_classes,
        "host":    host_classes,
        "geo":     geo_classes,
    }
    return df, label_maps


# ── Class weights (balanced) ───────────────────────────────────────────────────

def _weights(df, col, n):
    return compute_class_weight(
        "balanced",
        classes=np.arange(n),
        y=df[col].values,
    ).tolist()


# ── Main ───────────────────────────────────────────────────────────────────────

def preprocess():
    np.random.seed(config.SEED)
    df = pd.read_csv(config.DATA_CSV)
    print(f"Raw rows: {len(df)}")

    df = normalize_segments(df)
    print(f"After segment filter: {len(df)}")

    df = derive_geo_from_coords(df)
    df = group_species(df)

    # Drop Unknown host rows — can't train on unlabeled host
    df = df[df["host_label"] != "Unknown"].copy()
    print(f"After dropping Unknown host: {len(df)}")

    df, label_maps = encode_labels(df)

    # Stratified split on species_group (finest label)
    train_df, tmp = train_test_split(
        df, test_size=0.2, stratify=df["species_group"], random_state=config.SEED
    )
    val_df, test_df = train_test_split(
        tmp, test_size=0.5, stratify=tmp["species_group"], random_state=config.SEED
    )
    print(f"Split → train:{len(train_df)}, val:{len(val_df)}, test:{len(test_df)}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    train_df.reset_index(drop=True).to_csv(f"{config.OUTPUT_DIR}/train.csv", index=False)
    val_df.reset_index(drop=True).to_csv(f"{config.OUTPUT_DIR}/val.csv",   index=False)
    test_df.reset_index(drop=True).to_csv(f"{config.OUTPUT_DIR}/test.csv",  index=False)

    json.dump(label_maps, open(f"{config.OUTPUT_DIR}/label_maps.json", "w"), indent=2)

    n_sp  = len(label_maps["species"])
    n_host = len(label_maps["host"])
    n_geo  = len(label_maps["geo"])

    class_weights = {
        "species": _weights(train_df, "species_id", n_sp),
        "host":    _weights(train_df, "host_id",    n_host),
        "geo":     _weights(train_df, "geo_id",     n_geo),
    }
    json.dump(class_weights, open(f"{config.OUTPUT_DIR}/class_weights.json", "w"), indent=2)

    print("\nSaved: train.csv, val.csv, test.csv, label_maps.json, class_weights.json")
    print(f"\nFinal label counts — species: {n_sp}, host: {n_host}, geo: {n_geo}")
    return label_maps


if __name__ == "__main__":
    preprocess()
