import math


try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed. Install it first, then run this file again."
    ) from exc


def self_attention(x: torch.Tensor, w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor):
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    d_k = k.shape[-1]
    raw_scores = q @ k.T
    scaled_scores = raw_scores / math.sqrt(d_k)
    attention_weights = torch.softmax(scaled_scores, dim=-1)
    output = attention_weights @ v

    return q, k, v, raw_scores, scaled_scores, attention_weights, output


def print_matrix(name: str, matrix: torch.Tensor) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(torch.round(matrix * 1000) / 1000)


def main() -> None:
    tokens = ["I", "love", "AI"]

    x = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],  # I
            [0.0, 2.0, 0.0, 1.0],  # love
            [1.0, 1.0, 0.0, 1.0],  # AI
        ],
        dtype=torch.float32,
    )

    w_q = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    w_k = torch.tensor(
        [
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    w_v = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    q, k, v, raw_scores, scaled_scores, attention_weights, output = self_attention(
        x, w_q, w_k, w_v
    )

    print("=== Self-Attention Study Example: PyTorch ===")
    print(f"Tokens: {tokens}")

    print_matrix("Input X", x)
    print_matrix("Query Q = X @ W_q", q)
    print_matrix("Key K = X @ W_k", k)
    print_matrix("Value V = X @ W_v", v)
    print_matrix("Raw scores = Q @ K^T", raw_scores)
    print_matrix("Scaled scores = (Q @ K^T) / sqrt(d_k)", scaled_scores)
    print_matrix("Attention weights = softmax(scaled scores)", attention_weights)
    print_matrix("Output = attention weights @ V", output)

    print("\n=== Per-token interpretation ===")
    for i, token in enumerate(tokens):
        print(f"\nToken '{token}' attends to:")
        for j, other in enumerate(tokens):
            print(f"  {other:>5}: {attention_weights[i, j].item():.3f}")
        print(f"  Output vector: {torch.round(output[i] * 1000) / 1000}")


if __name__ == "__main__":
    main()
