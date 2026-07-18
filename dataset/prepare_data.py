
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset, get_dataset_config_names
from sklearn.model_selection import train_test_split

OUTPUT_DIR = Path(__file__).parent / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

LANG_PAIRS = ["hi", "fr", "es", "de", "ru"]
SUBSET_SIZE = 100_000

MIN_LEN_BY_LANG = {"en": 3, "hi": 3, "fr": 3, "es": 3, "de": 3, "ru": 3, "ja": 2}
MAX_LEN_BY_LANG = {"en": 200, "hi": 200, "fr": 200, "es": 200, "de": 200, "ru": 200, "ja": 100}
MAX_RATIO_BY_LANG = {"hi": 3.0, "fr": 3.0, "es": 3.0, "de": 3.0, "ru": 3.0, "ja": 4.5}

JUNK_PATTERN = re.compile(r'[\{\}]|"Id":|"Index":|"Count":')

def is_json_junk(text):
    return bool(JUNK_PATTERN.search(text))

def resolve_config_name(lang, available_configs):
    for candidate in (f"en-{lang}", f"{lang}-en"):
        if candidate in available_configs:
            return candidate
    raise ValueError(f"No config found for en-{lang}")

def is_valid_pair(src, tgt, src_lang, tgt_lang):
    src, tgt = src.strip(), tgt.strip()
    if not src or not tgt or src == tgt:
        return False
    if is_json_junk(src) or is_json_junk(tgt):
        return False
    if len(src) < MIN_LEN_BY_LANG.get(src_lang, 3) or len(tgt) < MIN_LEN_BY_LANG.get(tgt_lang, 3):
        return False
    if len(src) > MAX_LEN_BY_LANG.get(src_lang, 200) or len(tgt) > MAX_LEN_BY_LANG.get(tgt_lang, 200):
        return False
    ratio_threshold = MAX_RATIO_BY_LANG.get(tgt_lang, 3.0)
    longer, shorter = max(len(src), len(tgt)), max(min(len(src), len(tgt)), 1)
    return (longer / shorter) <= ratio_threshold

def clean_split(ds, lang, src_key="en", tgt_key=None):
    tgt_key = tgt_key or lang
    def _extract_and_validate(row):
        src, tgt = row["translation"][src_key], row["translation"][tgt_key]
        return {"en": src, lang: tgt, "valid": is_valid_pair(src, tgt, src_key, lang)}
    ds = ds.map(_extract_and_validate, remove_columns=ds.column_names)
    n_before = len(ds)
    ds = ds.filter(lambda row: row["valid"]).remove_columns("valid")
    df = ds.to_pandas().drop_duplicates(subset=["en", lang])
    print(f"  {lang}: {n_before} -> {len(df)} clean pairs ({(n_before - len(df)) / n_before:.1%} removed)")
    return df

def build_dataset():
    available_configs = get_dataset_config_names("Helsinki-NLP/opus-100")
    cleaned = {}
    print("Cleaning report:")
    for lang in LANG_PAIRS:
        config_name = resolve_config_name(lang, available_configs)
        raw = load_dataset("Helsinki-NLP/opus-100", config_name, split="train")
        raw = raw.shuffle(seed=42).select(range(min(SUBSET_SIZE, len(raw))))
        cleaned[lang] = clean_split(raw, lang, src_key="en", tgt_key=lang)

    rows = []
    for lang, df in cleaned.items():
        for _, row in df.iterrows():
            en_text, tgt_text = row["en"], row[lang]
            rows.append({"source_tagged": f"<2{lang}> {en_text}", "target": tgt_text,
                         "src_lang": "en", "tgt_lang": lang})
            rows.append({"source_tagged": f"<2en> {tgt_text}", "target": en_text,
                         "src_lang": lang, "tgt_lang": "en"})

    combined_df = pd.DataFrame(rows)
    combined_df["direction"] = combined_df["src_lang"] + "->" + combined_df["tgt_lang"]

    min_count = combined_df["direction"].value_counts().min()
    balanced_df = (
        combined_df.groupby("direction", group_keys=False)[combined_df.columns]
        .apply(lambda g: g.sample(min_count, random_state=42))
        .reset_index(drop=True)
    )
    print("\nFinal per-direction counts (balanced):")
    print(balanced_df["direction"].value_counts())

    train_df, temp_df = train_test_split(balanced_df, test_size=0.02, random_state=42,
                                           stratify=balanced_df["direction"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42,
                                         stratify=temp_df["direction"])
    print(f"\nTrain: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    train_df.to_parquet(OUTPUT_DIR / "tinylingo_train.parquet")
    val_df.to_parquet(OUTPUT_DIR / "tinylingo_val.parquet")
    test_df.to_parquet(OUTPUT_DIR / "tinylingo_test.parquet")
    print(f"\nSaved to {OUTPUT_DIR}")

if __name__ == "__main__":
    build_dataset()