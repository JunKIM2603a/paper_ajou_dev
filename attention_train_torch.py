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
        # 0, 1, 2, 3 같은 토큰 id를 학습 가능한 밀집 벡터로 바꾼다.
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        print(f"token_embedding weight shape: {tuple(self.token_embedding.weight.shape)}")
        print(f"token_embedding weight example:\n{torch.round(self.token_embedding.weight[:4] * 1000) / 1000}")
        # 같은 토큰이라도 몇 번째 위치에 있는지 구분할 수 있도록
        # 위치별 임베딩을 따로 더해 준다.
        self.position_embedding = nn.Embedding(seq_len, d_model)
        print(f"position_embedding weight shape: {tuple(self.position_embedding.weight.shape)}")
        print(f"position_embedding weight example:\n{torch.round(self.position_embedding.weight[:4] * 1000) / 1000}")

        # 같은 hidden 표현에서 Query, Key, Value를 각각 만든다.
        # 이것이 self-attention의 기본 구성이다.
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # attention으로 섞인 정보를 한 번 더 선형 변환한다.
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # 첫 번째 토큰과 마지막 토큰의 문맥 반영 벡터를 이어 붙여 분류하므로
        # 입력 차원은 d_model * 2가 된다.
        self.classifier = nn.Linear(d_model * 2, 2)

    def forward(self, x: torch.Tensor):
        # x의 shape은 (batch_size, seq_len)이다.
        # 각 원소는 정수 토큰 id이다.
        batch_size, seq_len = x.shape

        # 배치의 각 샘플마다 [[0, 1, 2], [0, 1, 2], ...] 같은 위치 id를 만든다.
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(
            batch_size, seq_len
        )

        # hidden의 shape은 (batch_size, seq_len, d_model)이다.
        # 토큰 의미 벡터와 위치 벡터를 원소별로 더해 최종 입력 표현을 만든다.
        hidden = self.token_embedding(x) + self.position_embedding(positions)

        # q, k, v의 shape은 모두 (batch_size, seq_len, d_model)이다.
        q = self.w_q(hidden)
        k = self.w_k(hidden)
        v = self.w_v(hidden)

        # scores의 shape은 (batch_size, seq_len, seq_len)이다.
        # scores[b, i, j]는 b번째 샘플에서 i번째 토큰이
        # j번째 토큰을 얼마나 참고할지 나타내는 점수다.
        d_k = q.shape[-1]
        scores = (q @ k.transpose(1, 2)) / math.sqrt(d_k)

        # softmax를 적용해 각 토큰이 다른 토큰들을 보는 비율로 바꾼다.
        attention_weights = torch.softmax(scores, dim=-1)

        # 각 토큰은 모든 value 벡터의 가중합으로 새로운 문맥 표현을 얻는다.
        attended = attention_weights @ v

        # attention 결과를 한 번 더 선형 변환한다.
        output = self.w_o(attended)

        # 이 toy 문제에서는 첫 토큰과 마지막 토큰의 문맥 반영 벡터를 합쳐 요약한다.
        summary = torch.cat([output[:, 0, :], output[:, -1, :]], dim=-1)
        logits = self.classifier(summary)
        return logits, attention_weights


def build_toy_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    # 작은 vocabulary에서 가능한 모든 길이 3의 시퀀스를 만든다.
    # 라벨은 첫 토큰과 마지막 토큰이 같으면 1, 다르면 0이다.
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
    # 숫자 토큰 id를 사람이 읽기 쉬운 문자열로 바꾼다.
    return [vocab[idx] for idx in sequence.tolist()]


def print_embedding_debug(
    model: TinySelfAttentionClassifier, sample: torch.Tensor, vocab: list[str]
) -> None:
    # nn.Embedding이 "정수 id -> 해당 행 벡터 lookup"으로 동작하는 모습을
    # 직접 확인할 수 있도록 중간 결과를 출력한다.
    with torch.no_grad():
        sample_batch = sample.unsqueeze(0)
        seq_len = sample_batch.shape[1]
        positions = torch.arange(seq_len, device=sample.device).unsqueeze(0)

        token_vectors = model.token_embedding(sample_batch)
        position_vectors = model.position_embedding(positions)
        hidden = token_vectors + position_vectors

    print("\n=== Embedding lookup debug ===")
    print(f"sample token ids: {sample.tolist()}")
    print(f"sample tokens: {decode_sequence(sample, vocab)}")
    print(f"position ids: {positions.squeeze(0).tolist()}")
    print(f"token_embedding weight shape: {tuple(model.token_embedding.weight.shape)}")
    print(
        f"position_embedding weight shape: {tuple(model.position_embedding.weight.shape)}"
    )

    for pos, token_id in enumerate(sample.tolist()):
        token_name = vocab[token_id]
        token_vector = torch.round(token_vectors[0, pos] * 1000) / 1000
        position_vector = torch.round(position_vectors[0, pos] * 1000) / 1000
        hidden_vector = torch.round(hidden[0, pos] * 1000) / 1000

        print(f"\n위치 {pos} | 토큰 '{token_name}' (id={token_id})")
        print(
            f"  token_embedding[{token_id}] -> {token_vector.tolist()}"
        )
        print(
            f"  position_embedding[{pos}] -> {position_vector.tolist()}"
        )
        print(f"  둘을 더한 hidden[{pos}] -> {hidden_vector.tolist()}")


def main() -> None:
    # 같은 결과를 재현하기 쉽도록 랜덤 시드를 고정한다.
    torch.manual_seed(7)

    vocab = ["A", "B", "C", "D"]
    x_train, y_train = build_toy_dataset()
    print(
        f"x_train shape: {x_train.shape} | y_train shape: {y_train.shape} | vocab size: {len(vocab)}"
    )
    print(f"x_train example:\n{x_train[:5]}\ny_train example:\n{y_train[:5]}")

    # d_model은 임베딩과 attention 내부에서 사용하는 특징 벡터 차원이다.
    model = TinySelfAttentionClassifier(vocab_size=len(vocab), seq_len=3, d_model=8)

    # 첫 번째 학습 샘플을 예시로 사용해 임베딩 lookup 과정을 먼저 보여 준다.
    print_embedding_debug(model, x_train[0], vocab)

    # CrossEntropyLoss는 (N, 클래스 수) 형태의 logits와
    # (N,) 형태의 정수 라벨을 입력으로 받는다.
    criterion = nn.CrossEntropyLoss()

    # Adam은 역전파로 계산된 gradient를 이용해 모든 학습 파라미터를 업데이트한다.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    print("=== Tiny Attention Training Example ===")
    print("Task: predict whether the first token and last token are the same.")
    print("This file performs real training with loss, backward, and optimizer.step().")

    for epoch in range(1, 301):
        # PyTorch는 gradient를 누적하므로 이전 step의 gradient를 먼저 비운다.
        optimizer.zero_grad()

        # 순전파를 수행해 분류 점수와 attention 행렬을 얻는다.
        logits, attention_weights = model(x_train)

        # 예측 결과와 정답 라벨을 비교해 loss를 계산한다.
        loss = criterion(logits, y_train)

        # 전체 네트워크에 대해 역전파를 수행해 gradient를 계산한다.
        loss.backward()

        # 계산된 gradient를 사용해 모델 파라미터를 갱신한다.
        optimizer.step()

        if epoch % 25 == 0 or epoch == 1:
            # 가장 큰 logit의 인덱스를 예측 클래스로 사용한다.
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == y_train).float().mean().item()
            print(
                f"Epoch {epoch:03d} | loss = {loss.item():.4f} | accuracy = {accuracy:.3f}"
            )

    with torch.no_grad():
        # 평가 시에는 gradient 추적을 끄고 계산만 수행한다.
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
            # attention_weights[idx, 0]은 idx번째 샘플에서
            # 첫 번째 토큰이 각 위치를 얼마나 참고하는지 의미한다.
            "First-token attention weights:",
            torch.round(attention_weights[idx, 0] * 1000) / 1000,
        )


if __name__ == "__main__":
    main()
