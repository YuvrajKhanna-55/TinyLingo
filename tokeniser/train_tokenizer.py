
import re
import random
from pathlib import Path
import sentencepiece as spm
import pandas as pd

DATASET_DIR = Path(__file__).parent.parent / "dataset" / "processed"
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

LANGUAGES = ["en", "hi", "fr", "es", "de", "ru"]
DIRECTION_TAGS = [f"<2{lang}>" for lang in LANGUAGES]
VOCAB_SIZE = 32000
CORPUS_PATH = ARTIFACT_DIR / "tinylingo_corpus.txt"
MODEL_PREFIX = str(ARTIFACT_DIR / "tinylingo_spm")

TAG_PREFIX_RE = re.compile(r"^<2\w+>\s*")

def build_corpus(train_df, corpus_path):
    lines = set()
    for src_tagged in train_df["source_tagged"]:
        clean = TAG_PREFIX_RE.sub("", src_tagged).strip()
        if clean:
            lines.add(clean)
    for tgt in train_df["target"]:
        if tgt.strip():
            lines.add(tgt.strip())
    lines = list(lines)
    random.seed(42)
    random.shuffle(lines)
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Corpus written: {len(lines)} unique lines -> {corpus_path}")

def train_tokenizer(corpus_path, model_prefix, vocab_size, user_defined_symbols):
    spm.SentencePieceTrainer.train(
        input=str(corpus_path), model_prefix=model_prefix, vocab_size=vocab_size,
        model_type="unigram", character_coverage=1.0,
        user_defined_symbols=user_defined_symbols,
        pad_id=0, pad_piece="<pad>", unk_id=1, unk_piece="<unk>",
        bos_id=2, bos_piece="<s>", eos_id=3, eos_piece="</s>",
        shuffle_input_sentence=True,
    )
    print(f"Trained model saved as {model_prefix}.model / {model_prefix}.vocab")

if __name__ == "__main__":
    train_df = pd.read_parquet(DATASET_DIR / "tinylingo_train.parquet")
    build_corpus(train_df, CORPUS_PATH)
    train_tokenizer(CORPUS_PATH, MODEL_PREFIX, VOCAB_SIZE, DIRECTION_TAGS)