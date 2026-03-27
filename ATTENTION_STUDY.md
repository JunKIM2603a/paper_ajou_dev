# Attention Study Note

이 문서는 `attention`의 핵심 흐름을 직접 계산해 보면서 이해하기 위한 학습 노트입니다.
예제 코드는 모두 작은 행렬을 사용하므로, 출력 결과를 눈으로 따라가며 각 단계가 어떤 의미인지 확인하기 좋습니다.

## 1. 기본 Self-Attention 흐름

입력 행렬 `X`가 주어지면, 각 토큰은 세 가지 벡터로 투영됩니다.

- `Q(Query) = X @ W_q`
- `K(Key) = X @ W_k`
- `V(Value) = X @ W_v`

그 다음 attention은 아래 순서로 계산됩니다.

1. `Q @ K^T`
   각 토큰이 다른 토큰과 얼마나 관련 있는지 점수(score)를 계산합니다.
2. `/ sqrt(d_k)`
   score 값이 너무 커지는 것을 막기 위해 스케일링합니다.
3. `softmax(...)`
   score를 확률처럼 해석할 수 있는 attention weight로 바꿉니다. 각 행의 합은 1이 됩니다.
4. `attention weights @ V`
   중요한 토큰의 정보를 더 많이 반영해서 새로운 표현을 만듭니다.

표준 self-attention 식은 아래와 같습니다.

`Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) V`

## 2. 코드에서 볼 포인트

- `attention_study.py`
  NumPy로 self-attention을 처음부터 구현한 예제입니다.
- `attention_torch_study.py`
  같은 로직을 PyTorch 스타일로 옮긴 버전입니다.
- `attention_study_torch.py`
  `attention_study.py`를 거의 그대로 옮긴 PyTorch 대응 버전입니다.
- `attention_multihead_study.py`
  multi-head attention에서 head를 나누고 다시 합치는 과정을 보여줍니다.
- `attention_masked_study.py`
  causal mask를 적용해서 미래 토큰을 가리는 Transformer 스타일 masked attention 예제입니다.
- `attention_train_torch.py`
  아주 작은 toy 데이터로 실제 loss 계산, 역전파, optimizer step까지 수행하는 학습 예제입니다.

출력을 볼 때는 아래 순서로 따라가면 이해가 쉽습니다.

1. `Q`, `K`, `V`
2. `Raw scores`
3. `Scaled scores`
4. `Attention weights`
5. `Output`

## 3. 학습 순서 추천

1. `attention_study.py`를 먼저 실행해서 기본 self-attention 계산 흐름을 익힙니다.
2. 입력 `x`의 값을 하나 바꿔 보고, attention weight가 어떻게 달라지는지 봅니다.
3. `w_q`, `w_k`, `w_v`를 바꿔 보면서 각 투영 행렬의 역할을 구분합니다.
4. 그 다음 `attention_multihead_study.py`로 넘어가서 head를 왜 나누는지 확인합니다.
5. 마지막으로 `attention_masked_study.py`에서 미래 토큰 차단이 어떻게 구현되는지 봅니다.
6. 학습이 실제로 어떻게 들어가는지 보고 싶다면 `attention_train_torch.py`를 실행합니다.

## 4. PyTorch 버전에서 `dim=-1` 의미

`attention_torch_study.py`에서는 NumPy의 custom softmax 대신 `torch.softmax(..., dim=-1)`를 사용합니다.

- attention score의 shape가 `(seq_len, seq_len)`이면,
- 각 query 토큰은 모든 key 토큰에 대한 점수를 한 행(row)으로 가집니다.
- `dim=-1`은 마지막 축을 따라 softmax를 적용하라는 뜻입니다.

즉, 각 query 토큰이 "어느 토큰을 얼마나 볼지"를 마지막 축 기준으로 정규화하는 것입니다.

## 5. Multi-Head Attention 설명

multi-head attention은 하나의 큰 attention을 한 번 하는 대신,
여러 개의 작은 attention head로 나누어 서로 다른 관점의 관계를 동시에 보게 합니다.

### `split_heads`

```python
def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    seq_len, d_model = x.shape
    head_dim = d_model // num_heads
    return x.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
```

이 함수는 입력 shape를 `(seq_len, d_model)`에서 `(num_heads, seq_len, head_dim)`으로 바꿉니다.

- `seq_len, d_model = x.shape`
  입력 토큰 개수와 전체 feature 차원을 읽습니다.
- `head_dim = d_model // num_heads`
  전체 차원을 head 개수로 나눠, 각 head가 담당할 차원을 계산합니다.
- `reshape(seq_len, num_heads, head_dim)`
  각 토큰의 벡터를 여러 head용 작은 벡터로 분할합니다.
- `transpose(1, 0, 2)`
  축 순서를 바꿔서 head별 계산이 쉽도록 `(num_heads, seq_len, head_dim)` 형태로 만듭니다.

예를 들어 `x.shape = (4, 8)`, `num_heads = 2`이면:

- 원래 shape: `(4, 8)`
- reshape 후: `(4, 2, 4)`
- transpose 후: `(2, 4, 4)`

즉, 4개 토큰의 8차원 표현을 2개의 4차원 head로 나눈 것입니다.

### `head_dim = q_heads.shape[-1]`

```python
head_dim = q_heads.shape[-1]
```

`q_heads.shape`가 `(num_heads, seq_len, head_dim)`일 때,
`shape[-1]`은 마지막 차원 크기이므로 곧 각 head의 벡터 차원입니다.

예:

```python
q_heads.shape = (2, 4, 2)
q_heads.shape[-1] == 2
```

이 값은 scaled dot-product attention에서 아래처럼 사용됩니다.

```python
scores = (q_heads @ np.transpose(k_heads, (0, 2, 1))) / math.sqrt(head_dim)
```

즉, head마다 계산된 score를 `sqrt(head_dim)`으로 나누기 위한 기준 차원값입니다.

### Head별 score 계산

```python
scores = (q_heads @ np.transpose(k_heads, (0, 2, 1))) / math.sqrt(head_dim)
```

- `q_heads` shape: `(num_heads, seq_len, head_dim)`
- `k_heads` shape: `(num_heads, seq_len, head_dim)`
- `np.transpose(k_heads, (0, 2, 1))` shape: `(num_heads, head_dim, seq_len)`

따라서 head마다 `Q @ K^T`가 계산되어 최종 shape는 `(num_heads, seq_len, seq_len)`가 됩니다.

즉, 각 head가 독립적으로 attention score 행렬을 가지게 됩니다.

### `combine_heads`

head별 output을 다시 합칠 때는 다음 함수를 사용합니다.

```python
def combine_heads(x: np.ndarray) -> np.ndarray:
    num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 0, 2).reshape(seq_len, num_heads * head_dim)
```

이 함수는 `(num_heads, seq_len, head_dim)`를 다시 `(seq_len, d_model)`로 되돌립니다.

### `output = concatenated @ w_o`

```python
output = concatenated @ w_o
```

이 줄은 여러 head의 결과를 이어붙인 뒤, 마지막 선형 변환(output projection)을 적용하는 단계입니다.

- `concatenated`
  각 head의 출력을 이어붙인 값입니다. shape는 보통 `(seq_len, d_model)`입니다.
- `w_o`
  최종 출력용 가중치 행렬입니다. 보통 shape는 `(d_model, d_model)`입니다.
- `@`
  행렬 곱셈입니다.

즉, head별 정보를 단순히 붙여 둔 상태에서 끝내지 않고,
`w_o`를 곱해서 여러 head의 정보를 다시 섞고 정리해 최종 attention 출력을 만듭니다.

Transformer 식으로 쓰면 아래와 같습니다.

`MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O`

여기서 `w_o`가 바로 `W^O`에 해당합니다.

## 6. Masked Attention 설명

masked self-attention은 현재 위치가 미래 토큰을 보지 못하도록 막는 방식입니다.
주로 Transformer decoder에서 사용됩니다.

### `causal_mask`

```python
def causal_mask(seq_len: int) -> np.ndarray:
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
```

이 함수는 미래 위치를 `True`로 표시하는 상삼각 mask를 만듭니다.

먼저:

```python
np.ones((seq_len, seq_len), dtype=bool)
```

는 모든 값이 `True`인 `seq_len x seq_len` 배열을 만듭니다.

그 다음:

```python
np.triu(..., k=1)
```

은 주대각선 바로 위쪽부터 상삼각 부분만 남깁니다.

예를 들어 `seq_len = 4`이면 결과는 다음과 같습니다.

```python
[[False,  True,  True,  True],
 [False, False,  True,  True],
 [False, False, False,  True],
 [False, False, False, False]]
```

의미는 다음과 같습니다.

- `[i, j] == True`이면 `j > i`
- 즉 현재 토큰 `i`가 미래 토큰 `j`를 보는 것을 막아야 한다는 뜻입니다.

### `np.triu` 설명

`np.triu`는 triangular upper의 줄임말로, 배열에서 상삼각 부분만 남기는 함수입니다.

- `np.triu(A)`
  주대각선을 포함한 위쪽만 남깁니다.
- `np.triu(A, k=1)`
  주대각선을 제외하고 그 위쪽만 남깁니다.
- `np.triu(A, k=2)`
  대각선보다 두 칸 위부터 남깁니다.

masked attention에서는 `k=1`을 자주 쓰는데,
이렇게 해야 자기 자신은 볼 수 있고 미래 토큰만 막을 수 있기 때문입니다.

### Mask 적용

```python
masked_scores = scaled_scores.copy()
masked_scores[mask] = -1e9
```

mask가 `True`인 위치의 score를 아주 작은 값으로 바꿉니다.
그 후 softmax를 적용하면 그 위치의 확률은 거의 0이 됩니다.

즉, 미래 토큰에 대한 attention이 사실상 차단됩니다.

## 7. 직접 실험해 보면 좋은 것들

1. `num_heads`를 바꾸고 `split_heads` 결과 shape가 어떻게 달라지는지 확인해 보기
2. `w_o = np.eye(...)` 대신 다른 행렬을 넣어 output projection의 효과 보기
3. `causal_mask(4)`와 `causal_mask(5)`를 출력해서 mask 패턴 비교하기
4. masked attention에서 `masked_scores`를 직접 보고 어떤 위치가 막혔는지 확인하기
5. 토큰 하나의 입력값을 바꾸고 attention weight 변화 관찰하기
