from datasets import load_dataset
import numpy as np
import tiktoken
import os

print("1. Downloading Elite Python/C++ Data (The Stack Dedup)...")
# We take a 2% slice. This will be a few Gigabytes, perfect for Kaggle disk limits.
dataset = load_dataset("bigcode/the-stack-dedup", data_dir="data/python", split="train[:2%]")

print("2. Initializing Translator...")
enc = tiktoken.get_encoding("gpt2")
all_tokens = []

print(f"3. Packing {len(dataset)} files...")
for i, item in enumerate(dataset):
    tokens = enc.encode_ordinary(item['content'])
    all_tokens.extend(tokens)
    all_tokens.append(enc.eot_token)
    if i % 5000 == 0:
        print(f"   Packed {i} files...")

remainder = len(all_tokens) % 4
if remainder != 0:
    all_tokens.extend([enc.eot_token] * (4 - remainder))

tokens_np = np.array(all_tokens, dtype=np.uint64)
t1, t2 = tokens_np[0::4] << 48, tokens_np[1::4] << 32
t3, t4 = tokens_np[2::4] << 16, tokens_np[3::4]
packed_array = t1 | t2 | t3 | t4

print("4. Saving high-octane fuel to Kaggle Working Directory...")
packed_array.tofile("/kaggle/working/packed_data.bin")
print(f"Dataset ready! Size: {packed_array.nbytes / (1024*1024):.2f} MB")