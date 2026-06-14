import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken
import time
import os

print("1. Rebuilding the V2 50-Million Parameter Architecture...")

# --- 1. THE EXACT V2 ARCHITECTURE ---
class BitQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        scale = weight.abs().mean().clamp_(min=1e-4)
        return (weight / scale).round().clamp_(-1, 1) * scale 
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
    def forward(self, x):
        return F.linear(x, BitQuantize.apply(self.weight))

class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts=4): 
        super().__init__()
        self.experts = nn.ModuleList([BitLinear(d_model, d_model) for _ in range(num_experts)])
        self.register_buffer('routing_matrix', torch.randn(d_model, num_experts))
    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1]) 
        routing_scores = torch.matmul(x_flat, self.routing_matrix)
        routing_probs = F.softmax(routing_scores, dim=-1)
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            out += expert(x_flat) * routing_probs[:, i].unsqueeze(-1)
        return out.reshape(orig_shape)

class GroupedQueryAttention(nn.Module):
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
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)

class HyperLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = MoELayer(d_model, num_experts=4) 

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + F.gelu(self.ffn(self.norm2(x)))
        return x

class HyperTransformer(nn.Module):
    def __init__(self, vocab_size=50257, d_model=512, num_layers=8, num_heads=8, num_kv_heads=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model) # The V2 expanded clock
        self.layers = nn.ModuleList([HyperLayer(d_model, num_heads, num_kv_heads) for _ in range(num_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(0, T, dtype=torch.long, device=x.device)
        x = self.embed(x) + self.pos_embed(positions)
        for layer in self.layers:
            # Checkpointing removed so it types lightning-fast
            x = layer(x) 
        x = self.norm_f(x)
        return self.head(x)

# --- 2. GENERATION LOGIC ---
def generate_code(model, prompt, max_new_tokens=100, temperature=0.6, top_k=40):
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    
    tokens = enc.encode(prompt)
    device = next(model.parameters()).device
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"\n--- PROMPT ---\n{prompt}")
    print("--- MODEL OUTPUT ---", end="")
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop context window to 256
            x_cond = x if x.size(1) <= 256 else x[:, -256:]
            
            logits = model(x_cond)
            logits = logits[:, -1, :] / temperature 
            
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            x = torch.cat((x, next_token), dim=1)
            
            word = enc.decode([next_token.item()])
            print(word, end="", flush=True)
            time.sleep(0.02) # Typing effect
            
    print("\n--------------------")

# --- 3. IGNITION ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HyperTransformer().to(device)

print("2. Loading 1.58-Bit Brain Memories...")
CHECKPOINT_PATH = "V2_50M_CHECKPOINT.pth" 

if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print("   Brain Successfully Loaded!")
else:
    print(f"ERROR: Could not find {CHECKPOINT_PATH}. Check your file path!")

# --- TALK TO IT ---
# We give it a slightly more structured prompt so it knows we want code
prompt = "def calculate_fibonacci(n):\n    \"\"\"\n    Calculate the nth Fibonacci number.\n    \"\"\"\n"
generate_code(model, prompt, max_new_tokens=100, temperature=0.6)