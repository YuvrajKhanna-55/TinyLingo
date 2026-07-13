import pandas as pd
from torch.utils.data import Dataset, DataLoader


class TranslationDataset(Dataset):
    """Returns raw text pairs -- tokenization happens in the collate_fn,
    so batch_encode_source/target (which handle padding) run once per batch,
    not once per sample."""
    def __init__(self, parquet_path):
        self.df = pd.read_parquet(parquet_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row["source_tagged"], row["target"]


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        sources, targets = zip(*batch)
        src_tensor = tokenizer.batch_encode_source(list(sources))
        tgt_tensor = tokenizer.batch_encode_target(list(targets))
        return src_tensor, tgt_tensor
    return collate_fn


def make_dataloader(parquet_path, tokenizer, batch_size=32, shuffle=True):
    dataset = TranslationDataset(parquet_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,collate_fn=make_collate_fn(tokenizer))