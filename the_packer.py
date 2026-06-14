import numpy as np
from datasets import load_dataset
import tiktoken

print("1. Loading the raw Python files from your hard drive...")
# This loads the exact files you just downloaded
dataset = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train")

print("2. Initializing the Tokenizer (The Translator)...")
enc = tiktoken.get_encoding("gpt2") # Uses the exact 50,257 vocab size your model expects

all_tokens = []
print("3. Converting code into numbers... (This might take a minute)")

# Go through all 10,000 files and translate text to numbers
for i, item in enumerate(dataset):
    text = item['content']
    # Turn the text string into a list of numbers
    tokens = enc.encode_ordinary(text) 
    all_tokens.extend(tokens)
    all_tokens.append(enc.eot_token) # Add a stop sign at the end of each file
    
    if i > 0 and i % 2000 == 0:
        print(f"   Translated {i} out of 10,000 files...")

print(f"\nTotal words/symbols translated: {len(all_tokens):,}")

print("4. Crushing 4 numbers into 1 64-bit block (The Bitwise Math)...")
# Ensure we have an exact multiple of 4 before we pack
remainder = len(all_tokens) % 4
if remainder != 0:
    all_tokens.extend([enc.eot_token] * (4 - remainder))

# Convert to a massive array of numbers
tokens_np = np.array(all_tokens, dtype=np.uint64)

# THE CRUSHER: Shift and combine [A, B, C, D] -> 1 solid block
t1 = tokens_np[0::4] << 48
t2 = tokens_np[1::4] << 32
t3 = tokens_np[2::4] << 16
t4 = tokens_np[3::4]

packed_array = t1 | t2 | t3 | t4

print("5. Saving the high-octane fuel to packed_data.bin...")
packed_array.tofile("packed_data.bin")

mb_size = packed_array.nbytes / (1024 * 1024)
print(f"\n--- SUCCESS ---")
print(f"packed_data.bin created! File size: {mb_size:.2f} MB")
print("Your fuel is ready. The engine is waiting.")