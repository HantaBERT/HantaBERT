import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(
    BASE_DIR,
    "../data-pipeline/data/processed/final_hantavirus_dataset.csv"
)
MODEL_PATH = "zhihan1996/DNABERT-2-117M"   # DNABERT-S base; avoids flash-attn/triton incompatibility on non-A100 GPUs
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ── Preprocessing ──────────────────────────────────────────────────────────────
MIN_SPECIES_COUNT = 30                   # species with fewer samples → "Other"
DROP_SPECIES      = {"Orthohantavirus sp."}   # genuinely unlabeled, not a useful class

# ── Label definitions ──────────────────────────────────────────────────────────
HOST_CLASSES = ["Rodent", "Human", "Others"]

# Country code → continent mapping used by preprocess.py
CC_TO_CONTINENT = {
    # Europe
    "GB":"Europe","DE":"Europe","FI":"Europe","SE":"Europe","FR":"Europe",
    "ES":"Europe","NO":"Europe","PL":"Europe","CZ":"Europe","SK":"Europe",
    "SI":"Europe","HR":"Europe","RU":"Europe","UA":"Europe","BY":"Europe",
    "EE":"Europe","LV":"Europe","LT":"Europe","RO":"Europe","BG":"Europe",
    "HU":"Europe","AT":"Europe","CH":"Europe","BE":"Europe","NL":"Europe",
    "DK":"Europe","PT":"Europe","GR":"Europe","IT":"Europe","RS":"Europe",
    "ME":"Europe","MK":"Europe","BA":"Europe","AL":"Europe","MD":"Europe",
    "IS":"Europe","IE":"Europe","LU":"Europe","MT":"Europe","CY":"Europe",
    # Asia
    "TR":"Asia","CN":"Asia","KR":"Asia","JP":"Asia","KZ":"Asia","MN":"Asia",
    "TH":"Asia","VN":"Asia","PH":"Asia","SG":"Asia","IN":"Asia","ID":"Asia",
    "MM":"Asia","KH":"Asia","LA":"Asia","MY":"Asia","BD":"Asia","PK":"Asia",
    "IR":"Asia","IQ":"Asia","SA":"Asia","UZ":"Asia","TJ":"Asia","KG":"Asia",
    "TM":"Asia","AM":"Asia","AZ":"Asia","GE":"Asia","TW":"Asia","HK":"Asia",
    "MO":"Asia","NP":"Asia","LK":"Asia","AF":"Asia","SY":"Asia","JO":"Asia",
    # Americas
    "US":"Americas","CA":"Americas","MX":"Americas","CR":"Americas",
    "PA":"Americas","CO":"Americas","VE":"Americas","PE":"Americas",
    "BR":"Americas","AR":"Americas","CL":"Americas","BO":"Americas",
    "PY":"Americas","UY":"Americas","EC":"Americas","GY":"Americas",
    "SR":"Americas","GT":"Americas","HN":"Americas","SV":"Americas",
    "NI":"Americas","HT":"Americas","DO":"Americas","CU":"Americas",
    "JM":"Americas","TT":"Americas","BB":"Americas","PR":"Americas",
    # Africa
    "MG":"Africa","TZ":"Africa","KE":"Africa","UG":"Africa","MW":"Africa",
    "ZM":"Africa","ZW":"Africa","MZ":"Africa","AO":"Africa","ZA":"Africa",
    "CM":"Africa","NG":"Africa","GH":"Africa","SN":"Africa","CI":"Africa",
    "ET":"Africa","SD":"Africa","ML":"Africa","BF":"Africa","NE":"Africa",
    # Oceania
    "AU":"Oceania","NZ":"Oceania","PG":"Oceania","FJ":"Oceania",
}

# ── Model / Training ───────────────────────────────────────────────────────────
MAX_LENGTH  = 512    # BPE tokens; S fits fully, M mostly, L gets 3'-end truncated
BATCH_SIZE  = 4
GRAD_ACCUM_STEPS = 4  # effective batch size remains 16
LEARNING_RATE  = 3e-5
EPOCHS         = 10
WARMUP_STEPS   = 200
WEIGHT_DECAY   = 0.01

# Multi-task loss weights — species is primary; geo is noisiest so lowest weight
LAMBDA_SPECIES = 1.0
LAMBDA_HOST    = 0.5
LAMBDA_GEO     = 0.3

SEED = 42
