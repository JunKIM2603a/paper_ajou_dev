import math


try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed. Install it first, then run this file again."
    ) from exc


class TinySelfAttentionClassifier(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.classifier = nn.Linear(d_model * 2, 2)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(
            batch_size, seq_len
        )

        hidden = self.token_embedding(x) + self.position_embedding(positions)

        q = self.w_q(hidden)
        k = self.w_k(hidden)
        v = self.w_v(hidden)

        d_k = q.shape[-1]
        scores = (q @ k.transpose(1, 2)) / math.sqrt(d_k)
        attention_weights = torch.softmax(scores, dim=-1)
        attended = attention_weights @ v
        output = self.w_o(attended)

        # For this toy task, combine the contextualized first and last tokens.
        summary = torch.cat([output[:, 0, :], output[:, -1, :]], dim=-1)
        logits = self.classifier(summary)
        return logits, attention_weights


def build_toy_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    # Label is 1 when the first and last tokens are the same, else 0.
    sequences = []
    labels = []

    for first in range(4):
        for middle in range(4):
            for last in range(4):
                sequences.append([first, middle, last])
                labels.append(1 if first == last else 0)

    x = torch.tensor(sequences, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def decode_sequence(sequence: torch.Tensor, vocab: list[str]) -> list[str]:
    return [vocab[idx] for idx in sequence.tolist()]


def main() -> None:
    torch.manual_seed(7)

    vocab = ["A", "B", "C", "D"]
    x_train, y_train = build_toy_dataset()

    model = TinySelfAttentionClassifier(vocab_size=len(vocab), seq_len=3, d_model=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    print("=== Tiny Attention Training Example ===")
    print("Task: predict whether the first token and last token are the same.")
    print("This file performs real training with loss, backward, and optimizer.step().")

    for epoch in range(1, 301):
        optimizer.zero_grad()
        logits, attention_weights = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0 or epoch == 1:
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == y_train).float().mean().item()
            print(
                f"Epoch {epoch:03d} | loss = {loss.item():.4f} | accuracy = {accuracy:.3f}"
            )

    with torch.no_grad():
        logits, attention_weights = model(x_train)
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == y_train).float().mean().item()

    print(f"\nFinal training accuracy: {accuracy:.3f}")

    example_indices = [0, 6, 18, 27]
    print("\n=== Example predictions after training ===")
    for idx in example_indices:
        sequence = x_train[idx]
        label = y_train[idx].item()
        prediction = predictions[idx].item()
        readable = decode_sequence(sequence, vocab)
        print(
            f"Sequence {readable} | label = {label} | prediction = {prediction}"
        )
        print(
            "First-token attention weights:",
            torch.round(attention_weights[idx, 0] * 1000) / 1000,
        )


if __name__ == "__main__":
    main()
