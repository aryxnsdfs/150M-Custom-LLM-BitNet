# BitNet 1.58-bit Ternary Language Model (from scratch)

A from-scratch implementation and training of a **1.58-bit ternary Transformer**
language model on a single consumer GPU (NVIDIA RTX 3060). The goal of this
project is to test how far heavily quantized, low-compute language modeling can
be pushed on commodity hardware by constraining linear-layer weights to three
states (-1, 0, +1) instead of full-precision floats.

This repository contains the actual training code, the inference scripts, and
the real training telemetry produced during the runs. Nothing in this README is
aspirational marketing - the numbers below are taken directly from the training
logs.

## What this is (and is not)

- **It is** a working, trained-from-scratch ternary Transformer that produces
  coherent short text after training on the TinyStories dataset.
- **It is not** a production-grade or state-of-the-art model. It is a research /
  learning project exploring extreme quantization on limited hardware.

## Core idea: 1.58-bit (ternary) weights

Standard Transformers store linear-layer weights as 16- or 32-bit floats. This
project replaces every linear layer with a `BitLinear` layer whose weights are
quantized to **{-1, 0, +1}** at forward time, using a straight-through estimator
so gradients still flow during backpropagation:

```python
class BitLinear(nn.Linear):
    def forward(self, x):
        w = self.weight
        w = w - w.mean()
        scale = w.abs().mean()
        w_quantized = torch.sign(w)
        # zero-out weights below a threshold -> the "0" state (ternary)
        w_quantized = torch.where(w.abs() > 0.5 * scale,
                                  w_quantized, torch.zeros_like(w))
        # straight-through estimator: quantize forward, pass gradient through
        w_bit = w + (w_quantized - w).detach()
        return F.linear(x, w_bit, self.bias)
```

Because the weights collapse to three states, the heavy matrix multiplies of a
standard Transformer reduce to additions and subtractions in principle, which is
the property that makes ternary networks attractive on constrained hardware.

## Architecture

The trained model (`bitnet.py`) is a standard decoder-only Transformer with
ternary linear layers:

| Component        | Choice                                             |
|------------------|----------------------------------------------------|
| Token embedding  | `nn.Embedding(vocab_size, d_model)`                |
| Positional info  | Learned positional parameter (`nn.Parameter`)      |
| Attention        | Causal multi-head self-attention (`nn.MultiheadAttention`) |
| Normalization    | `LayerNorm`                                        |
| Feed-forward     | `BitLinear -> GELU -> BitLinear` (4x expansion)    |
| Output head      | `BitLinear(d_model, vocab_size)`                   |
| Tokenizer        | GPT-2 BPE (vocab size 50,257)                       |

Default hyperparameters: `d_model=256`, `nhead=8`, `num_layers=4`, which is
approximately **30M parameters**.

## Training telemetry (real logs)

Trained on the `roneneldan/TinyStories` dataset, streamed, with AdamW, mixed
precision (`float16` autocast + `GradScaler`), `batch_size=12`, `seq_len=64` on a
single RTX 3060. Selected lines taken verbatim from the run:

```
Step   20 | Loss: 22.0108 | Sec/Step: 0.34s
Step  100 | Loss:  6.9006 | Sec/Step: 0.03s
Step  500 | Loss:  4.4462 | Sec/Step: 0.03s
Step 1000 | Loss:  4.0698 | Sec/Step: 0.03s
Step 2000 | Loss:  4.0388 | Sec/Step: 0.03s
Step 2500 | Loss:  2.4677 | Sec/Step: 0.03s
Step 3000 | Loss:  4.1275 | Sec/Step: 0.03s
```

Notes on the numbers:
- Loss drops from ~22 to roughly the 3-4 range, with the best observed value
  around **2.47**. The curve is noisy and has not fully converged; this is a
  short run, not a long polished training schedule.
- After warmup/compilation, step time is steady at ~0.03s/step.
- Checkpoints are saved every 1,000 steps (`bitnet_checkpoint_<step>.pth`) and
  the final weights to `bitnet_model.pth`.

## Sample output

After training on TinyStories, the model generates coherent short narrative text
(e.g. a short story about a dog named Max). Output quality is consistent with a
~30M model trained for a few thousand steps - grammatical and on-topic for short
contexts, not factual or instruction-following.

## Repository contents

| File                | Description |
|---------------------|-------------|
| `bitnet.py`         | The 1.58-bit `BitLinear` + `BitTransformer` model and the TinyStories training loop that produced the logs above. |
| `verify.py`         | Inference script for a separate, larger variant (`d_model=512`, 8 layers, grouped-query attention + a mixture-of-experts FFN), loading the `V2_50M_CHECKPOINT.pth` checkpoint (~50M params). |
| `train.py`          | **Experimental / work-in-progress.** A larger research variant exploring addition-only (Manhattan-distance) attention, locality-sensitive-hash routed FFN, and DeepSpeed ZeRO offload. Not the model that produced the telemetry above; included as ongoing research. |
| `magic_textbook.py` | Tokenizes and bit-packs source data into `packed_data.bin`. |
| `the_packer.py`     | Variant of the data packer used for a larger data slice. |

## Running

```bash
pip install torch transformers datasets tiktoken numpy

# Train the ternary model from scratch on TinyStories
python bitnet.py

# Run inference on the 50M checkpoint variant (requires V2_50M_CHECKPOINT.pth)
python verify.py
```

Trained weight files (`*.pth`) are **not** committed to this repository because of
their size; they are produced locally by running the training scripts. The
streaming datasets are downloaded automatically on first run.

## Honest limitations

- Short training run; loss has not converged and is noisy.
- Parameter count is ~30M for the trained model, not larger.
- Positional encoding is a learned parameter, not RoPE; normalization is
  LayerNorm, not RMSNorm; the FFN uses GELU, not SwiGLU. These are deliberate
  simplifications, not claims of modern best-practice components.
- The "addition-only" research direction in `train.py` is experimental and has
  not been validated to the same degree as `bitnet.py`.

## License

Released under the MIT License. See [LICENSE](LICENSE).
