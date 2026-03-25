import math

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def causal_mask(seq_len: int) -> np.ndarray:
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)


def masked_self_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
):
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    d_k = k.shape[-1]
    raw_scores = q @ k.T
    scaled_scores = raw_scores / math.sqrt(d_k)

    mask = causal_mask(x.shape[0])
    masked_scores = scaled_scores.copy()
    masked_scores[mask] = -1e9

    attention_weights = softmax(masked_scores)
    output = attention_weights @ v

    return {
        "q": q,
        "k": k,
        "v": v,
        "raw_scores": raw_scores,
        "scaled_scores": scaled_scores,
        "mask": mask,
        "masked_scores": masked_scores,
        "attention_weights": attention_weights,
        "output": output,
    }


def print_matrix(name: str, matrix: np.ndarray) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(np.round(matrix, 3))


def main() -> None:
    np.set_printoptions(suppress=True)

    tokens = ["I", "study", "deep", "learning"]

    x = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ]
    )

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

    result = masked_self_attention(x, w_q, w_k, w_v)

    print("=== Masked Transformer-Style Attention Study Example ===")
    print(f"Tokens: {tokens}")
    print("Mask type: causal mask")

    print_matrix("Input X", x)
    print_matrix("Q", result["q"])
    print_matrix("K", result["k"])
    print_matrix("V", result["v"])
    print_matrix("Raw scores", result["raw_scores"])
    print_matrix("Scaled scores", result["scaled_scores"])
    print_matrix("Causal mask (True means blocked)", result["mask"].astype(int))
    print_matrix("Masked scores", result["masked_scores"])
    print_matrix("Attention weights after mask", result["attention_weights"])
    print_matrix("Output", result["output"])

    print("\n=== Why masking matters ===")
    for i, token in enumerate(tokens):
        visible_tokens = tokens[: i + 1]
        print(f"Token '{token}' can attend only to: {visible_tokens}")


if __name__ == "__main__":
    main()
