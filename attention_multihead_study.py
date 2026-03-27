import math

import numpy as np

# Forward-only study example:
# the projection matrices below are manually chosen for inspection,
# so no learning or parameter updates happen in this file.


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    seq_len, d_model = x.shape
    head_dim = d_model // num_heads
    return x.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)


def combine_heads(x: np.ndarray) -> np.ndarray:
    num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 0, 2).reshape(seq_len, num_heads * head_dim)


def multi_head_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_heads: int,
):
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    q_heads = split_heads(q, num_heads)
    k_heads = split_heads(k, num_heads)
    v_heads = split_heads(v, num_heads)

    head_dim = q_heads.shape[-1]
    scores = (q_heads @ np.transpose(k_heads, (0, 2, 1))) / math.sqrt(head_dim)
    attention_weights = softmax(scores)
    head_outputs = attention_weights @ v_heads

    concatenated = combine_heads(head_outputs)
    # Output projection step in standard multi-head attention.
    # In this study example, W_o can be the identity matrix so the
    # concatenated result passes through unchanged and is easier to inspect.
    output = concatenated @ w_o

    return {
        "q": q,
        "k": k,
        "v": v,
        "q_heads": q_heads,
        "k_heads": k_heads,
        "v_heads": v_heads,
        "scores": scores,
        "attention_weights": attention_weights,
        "head_outputs": head_outputs,
        "concatenated": concatenated,
        "output": output,
    }


def print_matrix(name: str, matrix: np.ndarray) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(np.round(matrix, 3))


def main() -> None:
    np.set_printoptions(suppress=True)

    tokens = ["I", "study", "deep", "learning"]
    num_heads = 2

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
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )

    w_k = np.array(
        [
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )

    w_v = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 2.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )

    # Identity output projection: keeps the formal Transformer step,
    # but does not change values so we can study the concatenated heads directly.
    w_o = np.eye(4)

    result = multi_head_attention(x, w_q, w_k, w_v, w_o, num_heads)

    print("=== Multi-Head Attention Study Example ===")
    print(f"Tokens: {tokens}")
    print(f"num_heads = {num_heads}")

    print_matrix("Input X", x)
    print_matrix("Q", result["q"])
    print_matrix("K", result["k"])
    print_matrix("V", result["v"])

    for head_idx in range(num_heads):
        print_matrix(f"Head {head_idx} - Q", result["q_heads"][head_idx])
        print_matrix(f"Head {head_idx} - K", result["k_heads"][head_idx])
        print_matrix(f"Head {head_idx} - V", result["v_heads"][head_idx])
        print_matrix(f"Head {head_idx} - scores", result["scores"][head_idx])
        print_matrix(
            f"Head {head_idx} - attention weights",
            result["attention_weights"][head_idx],
        )
        print_matrix(f"Head {head_idx} - output", result["head_outputs"][head_idx])

    print_matrix("Concatenated heads", result["concatenated"])
    print_matrix("Final output = concatenated @ W_o", result["output"])


if __name__ == "__main__":
    main()
