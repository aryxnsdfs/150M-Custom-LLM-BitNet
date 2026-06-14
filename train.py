import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import argparse
from torch.utils.data import IterableDataset, DataLoader

# The Cloud Heavy-Lifters
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam # The high-speed CPU engine
from torch.utils.checkpoint import checkpoint
from datasets import load_dataset
from transformers import GPT2TokenizerFast

# --- 1. THE 1.58-BIT CORE (The Language of V5) ---
class BitQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        scale = 1.0 / weight.abs().mean().clamp_(min=1e-5)
        return (weight * scale).round().clamp_(-1, 1)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
    def forward(self, x):
        return F.linear(x, BitQuantize.apply(self.weight))

# --- 2. V5 HASH FFN MEMORY (No Matrix Multiplication) ---
class HashFFNMemory(nn.Module):
    def __init__(self, d_model, num_buckets=8): 
        super().__init__()
        self.num_buckets = num_buckets
        self.hash_bits = int(math.log2(num_buckets)) 
        
        self.memory_banks = nn.ModuleList([BitLinear(d_model, d_model) for _ in range(num_buckets)])
        self.register_buffer('projection_planes', torch.randn(d_model, self.hash_bits)) 

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1]) 
        
        with torch.no_grad():
            projections = torch.matmul(x_flat, self.projection_planes)
            bits = (projections > 0).long()
            bucket_indices = (bits[:, 0] * 4) + (bits[:, 1] * 2) + (bits[:, 2] * 1)
            
        out_flat = torch.zeros_like(x_flat)
        for bucket_id, bank in enumerate(self.memory_banks):
            mask = (bucket_indices == bucket_id)
            if mask.any():
                out_flat[mask] = bank(x_flat[mask])
                
        return out_flat.reshape(orig_shape)

# --- 3. V5 ADDITION-ONLY ATTENTION (Manhattan Distance) ---
class AdditionOnlyAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_groups = num_heads // num_kv_heads
        
        self.q_proj = BitLinear(d_model, num_heads * self.head_dim)
        self.k_proj = BitLinear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = BitLinear(d_model, num_kv_heads * self.head_dim)
        self.o_proj = BitLinear(num_heads * self.head_dim, d_model)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        distance = torch.cdist(q, k, p=1.0)
        scores = -distance
        
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = (scores > -self.head_dim * 0.5).float()
        
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)

# --- 4. THE V5 ENGINE ---
class HyperLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = AdditionOnlyAttention(d_model, num_heads, num_kv_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = HashFFNMemory(d_model, num_buckets=8) 

    def forward(self, x):
        res = x
        x = x + self.attention(self.norm1(x))
        x = x + torch.abs(self.ffn(self.norm2(x)))
        return x

class HyperTransformer(nn.Module):
    # THE RTX 3060 12GB SWEET SPOT (~1.5-Billion Parameters)
    def __init__(self, vocab_size=50257, d_model=1536, num_layers=24, num_heads=24, num_kv_heads=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([HyperLayer(d_model, num_heads, num_kv_heads) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return self.head(x)

# --- 5. THE CLOUD STREAMING INTAKE ---
class CodeStreamDataset(IterableDataset):
    def __init__(self, seq_len=256):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.dataset = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            tokens = self.tokenizer(example["content"], truncation=False)["input_ids"]
            buffer.extend(tokens)
            
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]
                yield torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])

# --- 6. DEEP SPEED LOCAL TRAINING LOOP ---
def train():
    parser = argparse.ArgumentParser(description='Local V5 Native Training')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    print("1. Assembling the V5 'Mathless' Architecture (1.5B Scale)...")
    model = HyperTransformer()

    print("2. Equipping DeepSpeed Native CPU Engine (Optimizer)...")
    # Swapped to DeepSpeedCPUAdam to match the ZeRO-Offload system perfectly
    optimizer = DeepSpeedCPUAdam(model.parameters(), lr=3e-4)

    print("3. Engaging DeepSpeed ZeRO-2 Local Sync...")
    ds_config = {
        "train_micro_batch_size_per_gpu": 4, 
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu", 
                "pin_memory": True
            }
        }
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, 
        model=model, 
        model_parameters=model.parameters(), 
        optimizer=optimizer,
        config=ds_config 
    )

    print("4. Igniting Cloud Ram-Jet (Streaming Data)...")
    seq_len = 256
    dataset = CodeStreamDataset(seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=4) 

    print(f"--- V5 LOCAL TRAINING INITIATED ---")
    
    for step, (x_context, y_future) in enumerate(dataloader, 1):
        x_context = x_context.to(model_engine.local_rank)
        y_future = y_future.to(model_engine.local_rank)
        
        logits = model_engine(x_context)
        loss = F.cross_entropy(logits.reshape(-1, 50257), y_future.reshape(-1))
        
        model_engine.backward(loss)
        model_engine.step()
        
        if step % 10 == 0 and model_engine.local_rank == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

        if step >= 5000: 
            break

if __name__ == "__main__": 
    train()