# Attention Study Note

This example focuses on understanding `attention` by calculating it directly.

## Core Flow

Given an input matrix `X`, each token is projected into three vectors:

- `Q(Query) = X @ W_q`
- `K(Key) = X @ W_k`
- `V(Value) = X @ W_v`

Then attention is computed in this order:

1. `Q @ K^T`
   This gives a score showing how much one token relates to another token.
2. `/ sqrt(d_k)`
   This keeps the scores from growing too large.
3. `softmax(...)`
   This turns the scores into normalized weights. Each row sums to 1.
4. `attention weights @ V`
   This creates a new representation by mixing in more information from important tokens.

The standard self-attention formula is:

`Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) V`

## What To Watch In The Code

- `attention_study.py` uses NumPy for a from-scratch implementation.
- `attention_torch_study.py` shows the same logic in PyTorch.
- Compare `Raw scores`, `Scaled scores`, `Attention weights`, and `Output` in that order.
- Look at which tokens receive larger weights to build intuition.

## Suggested Study Order

1. Run `attention_study.py` and inspect each printed matrix.
2. Change one value in `x` and see how the attention weights move.
3. Change `w_q`, `w_k`, and `w_v` to see how each matrix affects behavior.
4. Move on later to `multi-head attention` and `mask attention`.

## PyTorch Version

`attention_torch_study.py` implements the same self-attention flow with `torch.tensor`.

- It uses `torch.softmax(..., dim=-1)` instead of the custom NumPy softmax.
- The matrix operations are intentionally kept almost identical to the NumPy version.
- This makes it easier to connect the math to the style used in deep learning codebases.

If `torch` is not installed, the script exits with a short guidance message.

## Why `dim=-1` In `torch.softmax`

In attention, each query token produces one row of scores over all key tokens.

- If the score shape is `(seq_len, seq_len)`, each row should be normalized across the columns.
- In PyTorch, `dim=-1` means "apply softmax on the last axis."
- For attention scores, the last axis is the axis containing "which token to attend to."

So `torch.softmax(scaled_scores, dim=-1)` means:
"For each query token, turn its scores over all candidate key tokens into probabilities."

## New Study Files

- `attention_multihead_study.py`
  A NumPy implementation of multi-head attention that shows split heads, per-head scores, per-head outputs, concatenation, and final projection.
- `attention_masked_study.py`
  A NumPy implementation of Transformer-style masked self-attention using a causal mask, so future tokens are blocked.
