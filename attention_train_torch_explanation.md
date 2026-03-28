# `attention_train_torch.py` 상세 설명

이 파일은 아주 작은 예제를 통해 PyTorch로 self-attention 분류 모델을 만들고, 실제로 학습시키고, 학습 후 attention 결과까지 출력하는 교육용 스크립트다.

핵심 목표는 다음과 같다.

- 길이 3인 토큰 시퀀스를 입력으로 받는다.
- 첫 번째 토큰과 마지막 토큰이 같은지 분류한다.
- attention이 어떤 위치를 참고하는지 함께 확인한다.

## 1. 전체 흐름

스크립트는 아래 순서로 동작한다.

1. PyTorch를 import한다.
2. `TinySelfAttentionClassifier` 모델을 정의한다.
3. 가능한 모든 3토큰 조합으로 toy dataset을 만든다.
4. 모델, loss, optimizer를 준비한다.
5. 300 epoch 동안 전체 데이터를 학습한다.
6. 학습 후 정확도를 계산한다.
7. 몇 개 예제에 대해 예측 결과와 attention weight를 출력한다.

## 2. 데이터셋 구조

`build_toy_dataset()`은 길이 3인 모든 시퀀스를 만든다.

- vocabulary: `["A", "B", "C", "D"]`
- 각 위치에는 4개 토큰 중 하나가 올 수 있다.
- 전체 경우의 수는 `4 x 4 x 4 = 64`

예를 들어 입력 하나는 다음처럼 생긴다.

```python
[0, 2, 0]
```

이 값은 vocabulary를 기준으로 보면:

```python
["A", "C", "A"]
```

라벨 규칙은 아주 단순하다.

- 첫 번째 토큰 == 마지막 토큰이면 `1`
- 아니면 `0`

예시:

- `["A", "C", "A"]` -> `1`
- `["A", "C", "B"]` -> `0`

최종 텐서 shape은 다음과 같다.

- `x_train`: `(64, 3)`
- `y_train`: `(64,)`

## 3. 모델 구조

모델 클래스는 `TinySelfAttentionClassifier`다.

생성자에서 준비하는 주요 레이어는 다음과 같다.

### 3-1. 토큰 임베딩

```python
self.token_embedding = nn.Embedding(vocab_size, d_model)
```

역할:

- 정수 토큰 id를 dense vector로 바꾼다.
- 예를 들어 토큰 id `2`를 길이 `d_model`인 벡터로 변환한다.

내부적으로는 `(vocab_size, d_model)` 크기의 학습 가능한 테이블이다.

### 3-2. 위치 임베딩

```python
self.position_embedding = nn.Embedding(seq_len, d_model)
```

역할:

- 토큰이 시퀀스에서 몇 번째 위치인지 벡터로 표현한다.
- Transformer 계열 구조는 순서를 자동으로 알지 못하므로 위치 정보를 따로 넣어야 한다.

이 예제에서는 시퀀스 길이가 3이므로 위치는 `0, 1, 2` 세 개다.

### 3-3. Query, Key, Value 투영

```python
self.w_q = nn.Linear(d_model, d_model, bias=False)
self.w_k = nn.Linear(d_model, d_model, bias=False)
self.w_v = nn.Linear(d_model, d_model, bias=False)
```

역할:

- 같은 입력 표현에서 Query, Key, Value를 각각 만든다.
- self-attention의 핵심 계산에 사용된다.

### 3-4. 출력 투영과 분류기

```python
self.w_o = nn.Linear(d_model, d_model, bias=False)
self.classifier = nn.Linear(d_model * 2, 2)
```

역할:

- attention 결과를 한 번 더 변환한다.
- 첫 토큰과 마지막 토큰의 출력 벡터를 이어붙인 뒤 2개 클래스 분류를 수행한다.

`classifier`의 입력 차원이 `d_model * 2`인 이유는 아래 코드 때문이다.

```python
summary = torch.cat([output[:, 0, :], output[:, -1, :]], dim=-1)
```

즉, 첫 토큰 벡터와 마지막 토큰 벡터를 붙여서 하나의 요약 벡터로 만든다.

## 4. `forward()` 동작

입력 `x`의 shape은 다음과 같다.

```python
(batch_size, seq_len)
```

이 예제에서는 `seq_len = 3`이다.

### 4-1. 위치 인덱스 만들기

```python
batch_size, seq_len = x.shape
positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(
    batch_size, seq_len
)
```

예를 들어 batch size가 4면:

```python
positions =
[[0, 1, 2],
 [0, 1, 2],
 [0, 1, 2],
 [0, 1, 2]]
```

각 예제에 같은 위치 정보가 들어간다.

### 4-2. 토큰 임베딩 + 위치 임베딩

```python
hidden = self.token_embedding(x) + self.position_embedding(positions)
```

shape:

```python
hidden: (batch_size, seq_len, d_model)
```

의미:

- `self.token_embedding(x)`는 "무슨 토큰인지"
- `self.position_embedding(positions)`는 "어느 위치인지"

를 나타내며, 둘을 더해서 각 위치의 최종 입력 표현을 만든다.

### 4-3. Q, K, V 만들기

```python
q = self.w_q(hidden)
k = self.w_k(hidden)
v = self.w_v(hidden)
```

shape:

```python
q, k, v: (batch_size, seq_len, d_model)
```

### 4-4. attention score 계산

```python
d_k = q.shape[-1]
scores = (q @ k.transpose(1, 2)) / math.sqrt(d_k)
```

shape:

```python
scores: (batch_size, seq_len, seq_len)
```

`scores[b, i, j]`의 뜻:

- `b`번째 샘플에서
- `i`번째 토큰이
- `j`번째 토큰을 얼마나 참고할지에 대한 점수

`sqrt(d_k)`로 나누는 이유는 dot product 값이 너무 커지는 것을 완화하기 위해서다. 이것이 scaled dot-product attention이다.

### 4-5. attention weight 계산

```python
attention_weights = torch.softmax(scores, dim=-1)
```

shape:

```python
attention_weights: (batch_size, seq_len, seq_len)
```

각 행의 합은 1이다.

예를 들어:

```python
attention_weights[5, 0] = [0.1, 0.7, 0.2]
```

의 의미는:

- 5번째 샘플의 첫 번째 토큰이
- 자기 자신을 10%
- 두 번째 토큰을 70%
- 세 번째 토큰을 20%

정도로 참고한다는 뜻이다.

### 4-6. value의 가중합

```python
attended = attention_weights @ v
```

shape:

```python
attended: (batch_size, seq_len, d_model)
```

각 토큰은 다른 위치들의 `v`를 attention weight로 가중합한 새로운 문맥 표현을 얻게 된다.

### 4-7. 출력 투영과 요약

```python
output = self.w_o(attended)
summary = torch.cat([output[:, 0, :], output[:, -1, :]], dim=-1)
logits = self.classifier(summary)
```

shape:

- `output`: `(batch_size, seq_len, d_model)`
- `summary`: `(batch_size, d_model * 2)`
- `logits`: `(batch_size, 2)`

이 예제의 분류 목표는 "첫 토큰과 마지막 토큰이 같은가?"이므로, 첫 토큰과 마지막 토큰의 문맥화된 표현을 합쳐서 분류에 사용한다.

## 5. 학습 루프 설명

학습 루프의 핵심 부분은 다음과 같다.

```python
for epoch in range(1, 301):
    optimizer.zero_grad()
    logits, attention_weights = model(x_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()
```

각 줄의 의미:

- `optimizer.zero_grad()`
  - 이전 step의 gradient를 초기화한다.
  - PyTorch는 gradient를 누적하므로 매 반복마다 보통 호출한다.

- `logits, attention_weights = model(x_train)`
  - 전체 학습 데이터를 모델에 통과시켜 예측 점수와 attention 결과를 얻는다.

- `loss = criterion(logits, y_train)`
  - 예측과 정답의 차이를 loss로 계산한다.
  - 여기서는 `nn.CrossEntropyLoss()`를 사용한다.

- `loss.backward()`
  - 역전파를 통해 각 파라미터에 대한 gradient를 계산한다.

- `optimizer.step()`
  - 계산된 gradient를 이용해 파라미터를 업데이트한다.

중간중간 정확도를 출력하는 부분도 있다.

```python
predictions = logits.argmax(dim=-1)
accuracy = (predictions == y_train).float().mean().item()
```

의미:

- 가장 큰 logit의 인덱스를 예측 클래스라고 본다.
- 정답과 비교해 맞은 비율을 계산한다.

## 6. `torch.no_grad()` 평가 블록

학습이 끝난 뒤에는 다음 코드로 다시 예측한다.

```python
with torch.no_grad():
    logits, attention_weights = model(x_train)
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == y_train).float().mean().item()
```

`torch.no_grad()`의 역할:

- gradient를 추적하지 않는다.
- 메모리를 절약한다.
- 평가나 추론 시에 일반적으로 사용한다.

여기서는 테스트셋이 아니라 학습셋에 대해 정확도를 다시 계산하므로, 출력되는 값은 training accuracy다.

## 7. 예제 출력 해석

마지막 부분에서는 몇 개 시퀀스를 골라서 사람이 보기 쉽게 출력한다.

```python
example_indices = [0, 6, 18, 27]
```

각 샘플에 대해 다음을 출력한다.

- 원래 시퀀스
- 정답 라벨
- 모델 예측
- 첫 번째 토큰의 attention weight

핵심 코드는 다음과 같다.

```python
readable = decode_sequence(sequence, vocab)
print(f"Sequence {readable} | label = {label} | prediction = {prediction}")
print("First-token attention weights:", torch.round(attention_weights[idx, 0] * 1000) / 1000)
```

여기서 `attention_weights[idx, 0]`는:

- `idx`번째 샘플에서
- 첫 번째 토큰이
- 각 위치를 얼마나 참고하는지

를 뜻한다.

예를 들어:

```python
tensor([0.800, 0.050, 0.150])
```

라면 첫 번째 토큰이:

- 첫 번째 위치를 0.800
- 두 번째 위치를 0.050
- 세 번째 위치를 0.150

비율로 참고했다는 의미다.

이 값이 중요한 이유는, 이 toy task에서 모델이 "첫 토큰과 마지막 토큰이 같은지"를 판단해야 하므로 attention이 첫/마지막 위치를 의미 있게 참고하는지 관찰할 수 있기 때문이다.

## 8. `decode_sequence()` 역할

모델 입력은 숫자 토큰 id이므로 사람이 읽기에는 불편하다.

```python
def decode_sequence(sequence: torch.Tensor, vocab: list[str]) -> list[str]:
    return [vocab[idx] for idx in sequence.tolist()]
```

예:

```python
sequence = tensor([0, 2, 3])
vocab = ["A", "B", "C", "D"]
```

결과:

```python
["A", "C", "D"]
```

즉, 출력 로그를 읽기 쉬운 문자열 형태로 바꿔주는 도우미 함수다.

## 9. 이 코드에서 배울 수 있는 핵심

- `nn.Embedding`으로 토큰과 위치를 벡터로 표현하는 방법
- Query, Key, Value를 만드는 기본 구조
- scaled dot-product attention 계산 방식
- attention 결과를 분류 문제에 연결하는 방법
- PyTorch에서 `loss.backward()`와 `optimizer.step()`로 실제 학습하는 흐름
- attention weight를 출력해 모델이 어디를 보는지 해석하는 방법

## 10. 한 줄 요약

`attention_train_torch.py`는 "첫 토큰과 마지막 토큰이 같은지"를 맞히는 아주 작은 분류 문제를 통해, self-attention의 입력 준비, attention 계산, 분류, 학습, 결과 해석까지 한 파일 안에서 보여주는 예제다.
