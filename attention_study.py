import math
from typing import Tuple

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def self_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    d_k = k.shape[-1]
    scores = (q @ k.T) / math.sqrt(d_k)
    attention_weights = softmax(scores)
    output = attention_weights @ v

    return q, k, v, attention_weights, output


def print_matrix(name: str, matrix: np.ndarray) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(np.round(matrix, 3))


def main() -> None:
    np.set_printoptions(suppress=True)

    tokens = ["I", "love", "AI"]

    # Example input: each token is represented as a 4D vector.
    x = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],  # I
            [0.0, 2.0, 0.0, 1.0],  # love
            [1.0, 1.0, 0.0, 1.0],  # AI
        ]
    )

    # Think of these as learned projection matrices.
    w_q = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    w_k = np.array(
        [
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    w_v = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )

    q, k, v, attention_weights, output = self_attention(x, w_q, w_k, w_v)

    print("=== Self-Attention Study Example ===")
    print(f"Tokens: {tokens}")

    print_matrix("Input X", x)
    print_matrix("Query Q = X @ W_q", q)
    print_matrix("Key K = X @ W_k", k)
    print_matrix("Value V = X @ W_v", v)

    raw_scores = q @ k.T
    print_matrix("Raw scores = Q @ K^T", raw_scores)

    scaled_scores = raw_scores / math.sqrt(k.shape[-1])
    print_matrix("Scaled scores = (Q @ K^T) / sqrt(d_k)", scaled_scores)

    print_matrix("Attention weights = softmax(scaled scores)", attention_weights)
    print_matrix("Output = attention weights @ V", output)

    print("\n=== Per-token interpretation ===")
    for i, token in enumerate(tokens):
        print(f"\nToken '{token}' attends to:")
        for j, other in enumerate(tokens):
            print(f"  {other:>5}: {attention_weights[i, j]:.3f}")
        print(f"  Output vector: {np.round(output[i], 3)}")


if __name__ == "__main__":
    main()
