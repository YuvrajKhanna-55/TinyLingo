import sys
sys.path.append("tokeniser")
sys.path.append("tinylingo")

from tokenizer_wrapper import TranslationTokenizer
from embeddings import TokenEmbedding
import pandas as pd

tok = TranslationTokenizer("tokeniser/artifacts/tinylingo_spm.model", max_seq_len=64)
train_df = pd.read_parquet("dataset/processed/tinylingo_train.parquet")

batch = train_df.sample(8, random_state=42)
src_tensor = tok.batch_encode_source(batch["source_tagged"].tolist())

print("src_tensor shape:", src_tensor.shape)   
print("src_tensor dtype:", src_tensor.dtype)   

d_model = 256 
embed_layer = TokenEmbedding(vocab_size=tok.vocab_size, d_model=d_model, max_seq_len=64)
out = embed_layer(src_tensor)

print("Embedding output shape:", out.shape)   # expect (8, 64, 256)
print("Embedding output dtype:", out.dtype)   # expect torch.float32
print("Sample values:", out[0, 0, :8])


print("\nPE is a registered buffer, not a parameter:")
print("  in named_parameters?", any("pe" in n for n, _ in embed_layer.named_parameters()))
print("  in named_buffers?   ", any("pe" in n for n, _ in embed_layer.named_buffers()))