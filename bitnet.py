import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from datasets import load_dataset
import time

# ============================================================
# 1. THE BITNET ENGINE (1.58-bit Ternary Math)
# ============================================================
class BitLinear(nn.Linear):
    def forward(self, x):
        w = self.weight
        w = w - w.mean()
        scale = w.abs().mean()
        w_quantized = torch.sign(w)
        # Zero-out small weights -> the "0" state, giving ternary {-1, 0, +1}
        w_quantized = torch.where(w.abs() > 0.5 * scale,
                                  w_quantized, torch.zeros_like(w))
        # Straight-through estimator: quantize on the forward pass,
        # let gradients flow through the unquantized weights on the backward pass.
        w_bit = w + (w_quantized - w).detach()
        return F.linear(x, w_bit, self.bias)


# ============================================================
# 2. THE TRANSFORMER ARCHITECTURE (The Context Brain)
# ============================================================
class BitTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 512, d_model))
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_model, nhead, batch_first=True),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    BitLinear(d_model, d_model * 4),
                    nn.GELU(),
                    BitLinear(d_model * 4, d_model)
                ),
                'norm2': nn.LayerNorm(d_model)
            }) for _ in range(num_layers)
        ])
        self.output_head = BitLinear(d_model, vocab_size)

    def forward(self, x):
        b, t = x.size()
        x = self.embedding(x) + self.pos_encoding[:, :t, :]
        mask = torch.triu(torch.ones(t, t, device=x.device) * float('-inf'), diagonal=1)
        for layer in self.layers:
            attn_out, _ = layer['attn'](x, x, x, attn_mask=mask)
            x = layer['norm1'](x + attn_out)
            x = layer['norm2'](x + layer['ffn'](x))
        return self.output_head(x)


# ============================================================
# 3. THE TRAINING ENGINE (Optimized for RTX 3060)
# ============================================================
def train():
    device = torch.device("cuda")
    print(f"--- INITIALIZING ON: {torch.cuda.get_device_name(0)} ---")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = BitTransformer(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler()

    dataset = load_dataset("roneneldan/TinyStories", streaming=True, split="train")
    data_iter = iter(dataset)

    batch_size = 12
    seq_len = 64

    print("Training Started. Watch the Loss drop.")
    print("-" * 40)

    start_time = time.time()
    # TIP: Increase 10001 if you want it to train even longer.
    for step in range(1, 10001):
        inputs_list = []
        for _ in range(batch_size):
            try:
                text = next(data_iter)['text']
            except StopIteration:
                data_iter = iter(dataset)  # Restart data if it ends
                text = next(data_iter)['text']

            ids = tokenizer.encode(text, truncation=True,
                                   max_length=seq_len + 1, return_tensors="pt")[0]
            if len(ids) < seq_len + 1:
                ids = F.pad(ids, (0, (seq_len + 1) - len(ids)),
                            value=tokenizer.eos_token_id)
            inputs_list.append(ids)

        batch = torch.stack(inputs_list).to(device)
        x, y = batch[:, :-1], batch[:, 1:]

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(
                logits.contiguous().view(-1, tokenizer.vocab_size),
                y.contiguous().view(-1)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 20 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:4} | Loss: {loss.item():.4f} | Sec/Step: {elapsed/20:.2f}s")
            start_time = time.time()

        # Save a checkpoint every 1000 steps
        if step % 1000 == 0:
            torch.save(model.state_dict(), f"bitnet_checkpoint_{step}.pth")
            print(f"--- [PROGRESS SAVED AT STEP {step}] ---")

    # Save the final model
    torch.save(model.state_dict(), "bitnet_model.pth")
    print("\nTraining Complete! Final brain saved to bitnet_model.pth")


if __name__ == "__main__":
    train()
