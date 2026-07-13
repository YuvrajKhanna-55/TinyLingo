import torch
import sentencepiece as spm

class TranslationTokenizer:
    """
    Runtime wrapper around a trained SentencePiece model.
    Converts raw sentences into padded integer tensors ready for the model.
    """

    def __init__(self, model_path, max_seq_len=128):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.vocab_size = self.sp.get_piece_size()
        self.max_seq_len = max_seq_len

    def _encode_raw(self, text):
        return self.sp.encode(text, out_type=int)

    def encode_source(self, text):
        """Encoder input: content + EOS. No BOS -- encoder isn't autoregressive."""
        ids = self._encode_raw(text)
        truncated = len(ids) > (self.max_seq_len - 1)
        ids = ids[: self.max_seq_len - 1] + [self.eos_id]
        return ids, truncated

    def encode_target(self, text):
        """
        Decoder sequence: BOS + content + EOS.
        Training loop later slices [:-1] as decoder input, [1:] as the label.
        """
        ids = self._encode_raw(text)
        truncated = len(ids) > (self.max_seq_len - 2)
        ids = [self.bos_id] + ids[: self.max_seq_len - 2] + [self.eos_id]
        return ids, truncated

    def _pad(self, ids):
        return ids + [self.pad_id] * (self.max_seq_len - len(ids))

    def batch_encode_source(self, texts):
        return self._batch_encode(texts, self.encode_source)

    def batch_encode_target(self, texts):
        return self._batch_encode(texts, self.encode_target)

    def _batch_encode(self, texts, encode_fn):
        all_ids, n_truncated = [], 0
        for text in texts:
            ids, truncated = encode_fn(text)
            if truncated:
                n_truncated += 1
            all_ids.append(self._pad(ids))

        if n_truncated > 0:
            print(f"[TranslationTokenizer] {n_truncated}/{len(texts)} "
                  f"sequences truncated to max_seq_len={self.max_seq_len}")

        return torch.tensor(all_ids, dtype=torch.long)

    def create_padding_mask(self, token_id_tensor):
        return token_id_tensor == self.pad_id

    def decode(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        ids = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.sp.decode(ids)