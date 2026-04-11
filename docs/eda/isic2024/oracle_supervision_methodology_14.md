# 14. Auxiliary Oracle Supervision 방법론 정리

이 문서는 [`src/eda/isic2024_eda_20260411.ipynb`](/home/junkim2603a/proj/paper_ajou_dev/src/eda/isic2024_eda_20260411.ipynb) 의 `14. 선택적 Auxiliary Oracle Supervision 설계`를 별도 방법론 문서로 풀어쓴 것이다. 핵심은 `iddx_full`을 추론 시점 입력으로 쓰지 않고, `pathology-informative subset`에서만 `train-time auxiliary oracle supervision`의 원천으로 쓰는 것이다.

즉 이 장의 질문은 `iddx_full 전체를 범주형 feature로 넣을 것인가`가 아니다. 대신 아래 두 질문에 답한다.

1. `iddx_full` 안에서 실제 병리 의미가 살아 있는 샘플 범위를 어떻게 분리할 것인가
2. 그 텍스트를 어떤 저차원 cluster target으로 요약해 auxiliary head가 학습하게 할 것인가

현재 실행 결과는 아래 artifact에 저장돼 있다.

- subset 요약: [oracle_subset_summary.csv](/home/junkim2603a/proj/paper_ajou_dev/artifacts/eda/isic2024/thesis_plan/oracle_supervision/oracle_subset_summary.csv)
- subset label mix: [pathology_subset_label_mix.csv](/home/junkim2603a/proj/paper_ajou_dev/artifacts/eda/isic2024/thesis_plan/oracle_supervision/pathology_subset_label_mix.csv)
- 대표 진단 문자열: [pathology_subset_top_text.csv](/home/junkim2603a/proj/paper_ajou_dev/artifacts/eda/isic2024/thesis_plan/oracle_supervision/pathology_subset_top_text.csv)
- cluster 후보 평가: [oracle_cluster_eval.csv](/home/junkim2603a/proj/paper_ajou_dev/artifacts/eda/isic2024/thesis_plan/oracle_supervision/oracle_cluster_eval.csv)
- cluster 요약: [oracle_cluster_summary.csv](/home/junkim2603a/proj/paper_ajou_dev/artifacts/eda/isic2024/thesis_plan/oracle_supervision/oracle_cluster_summary.csv)
- train assignment: [oracle_cluster_assignment_train.csv](/home/junkim2603a/proj/paper_ajou_dev/artifacts/eda/isic2024/thesis_plan/oracle_supervision/oracle_cluster_assignment_train.csv)
- model spec: [oracle_cluster_model_spec.csv](/home/junkim2603a/proj/paper_ajou_dev/artifacts/eda/isic2024/thesis_plan/oracle_supervision/oracle_cluster_model_spec.csv)

## 1. 입력 계약과 목표

이번 EDA의 계약은 이미 [`x_strict_contract.csv`](/home/junkim2603a/proj/paper_ajou_dev/artifacts/eda/isic2024/thesis_plan/x_strict_contract.csv) 에 명시돼 있다.

- 메인 입력은 `X_strict = [strict raw preprocessed; strict FE]`
- `oracle_supervision_source = iddx_full only`
- `inference_time_oracle_input = not allowed`
- `train_time_oracle_usage = allowed only via auxiliary oracle supervision on pathology-informative subset`

따라서 `iddx_full`은 메인 tabular input이 아니고, `학습 중에만 사용하는 보조 타깃 생성 소스`다. 이 점이 14장의 모든 설계를 결정한다.

## 2. 표기법

아래 표기법을 사용한다.

- `D_train`: 현재 내부 `train split`
- `N = |D_train|`: train row 수
- `t_i`: 샘플 `i`의 원본 `iddx_full`
- `n_i`: 정규화된 `iddx_full`
- `y_i ∈ {0, 1}`: 메인 target
- `S_path`: pathology-informative subset
- `K`: oracle cluster 개수
- `c_i ∈ {0, ..., K-1}`: 샘플 `i`의 oracle cluster 라벨

현재 실행에서는 `N = 280,335` 이고, pathology-informative subset 크기는 `|S_path| = 740` 이다.

## 3. 14.1 pathology-informative subset 정의

### 3.1 텍스트 정규화

노트북은 먼저 `iddx_full`을 아래 함수로 정규화한다.

```text
normalize_iddx_text(text):
  1. strip + lowercase
  2. '::' -> ' '
  3. [a-z0-9] 이외 문자는 공백으로 치환
  4. 연속 공백 축소
```

수식으로 쓰면, 원본 문자열 `t_i`에 대해 정규화 함수 `g(·)`를 적용해서

`n_i = g(t_i)`

를 만든다. 여기서 `g(·)`는 lowercasing, separator normalization, non-alphanumeric 제거, whitespace collapse를 포함한다.

이 정규화가 필요한 이유는 간단하다.

- `Benign`, `benign`, `Benign::...` 같은 표기 흔들림을 한 축으로 맞추기 위해서다.
- 이후 TF-IDF vocabulary가 불필요한 punctuation 차이로 쪼개지지 않게 하기 위해서다.
- cluster가 진짜 병리 단어 차이로 나뉘도록 하고, 기호/대소문자 차이로 갈라지지 않게 하기 위해서다.

### 3.2 subset 정의

pathology-informative subset은 아래처럼 정의한다.

`S_path = { i ∈ D_train : n_i ≠ "" and n_i ≠ "benign" }`

즉 정규화된 진단 텍스트가 비어 있지 않고, literal benign 단일 문자열이 아닌 경우만 남긴다.

이 정의를 이렇게 잡은 이유는 다음과 같다.

- `iddx_full = "Benign"` 한 단어만 있는 행은 병리적으로 세분화된 정보가 거의 없다.
- 반면 `Benign::...`, `Indeterminate::...`, `Malignant::...`처럼 계층형 진단 문자열이 있는 행은 `조직학적 의미 축`이 더 풍부하다.
- 따라서 auxiliary oracle target은 `모든 샘플`이 아니라 `병리 정보가 실제로 서술된 샘플`에서만 만드는 편이 맞다.

중요한 점은 `S_path`가 `positive-only subset`이 아니라는 점이다. 이 subset에는 아래가 함께 들어간다.

- malignant 세부 진단
- indeterminate 세부 진단
- dysplastic nevus나 seborrheic keratosis처럼 병리적으로 기술된 benign 세부 진단

즉 `pathology-informative`는 `악성만 남긴다`는 뜻이 아니라, `병리 텍스트가 실제로 정보량을 가지는 샘플만 남긴다`는 뜻이다.

### 3.3 subset 요약 수식

노트북은 subset 규모를 아래처럼 정리한다.

`row_ratio_pct(S) = |S| / |D_train| × 100`

`positive_ratio_pct(S) = (Σ_{i ∈ S} y_i) / |S| × 100`

`unique_iddx_full(S) = | { n_i : i ∈ S } |`

현재 결과는 아래와 같다.

- 전체 train split: `280,335` rows, positive `270` rows, positive ratio `0.0963%`, unique normalized `iddx_full` `47`
- pathology-informative subset: `740` rows, positive `270` rows, positive ratio `36.4865%`, unique normalized `iddx_full` `46`

여기서 중요한 관찰은 두 가지다.

1. subset 크기는 전체의 `0.264%`로 매우 작다.
2. 그런데 positive `270`개가 모두 이 subset 안에 들어간다.

즉 oracle text는 `드물지만 병리 정보가 매우 농축된 부분집합`에 몰려 있다. 그래서 이를 전샘플 입력으로 넣기보다, `희소한 보조 감독 신호`로 다루는 게 더 자연스럽다.

### 3.4 실제 label mix 해석

현재 `S_path`의 `iddx_1` 분포는 아래와 같다.

- `Benign`: `389` rows (`52.5676%`)
- `Malignant`: `270` rows (`36.4865%`)
- `Indeterminate`: `81` rows (`10.9459%`)

이 분포는 왜 `S_path`를 단순 positive subset으로 정의하지 않았는지를 잘 보여준다. 이 subset은 `malignant vs benign` 보조 라벨이 아니라, `병리적 의미 축 자체`를 학습시키기 위한 영역이다. 따라서 benign/indeterminate 내부의 세부 분류도 auxiliary head가 알아야 할 구조에 포함된다.

## 4. 14.2 train-time only oracle cluster 설계

이제 `S_path` 안의 텍스트를 직접 모델 입력으로 넣지 않고, `cluster target`으로 압축한다. 절차는 아래 네 단계다.

1. normalized text 생성
2. TF-IDF embedding
3. Truncated SVD
4. K-means clustering

### 4.1 TF-IDF 표현

정규화된 텍스트 `n_i`에서 unigram과 bigram을 뽑아 TF-IDF 행렬 `X ∈ R^{|S_path| × |V|}` 를 만든다.

현재 vectorizer spec은 아래와 같다.

- `TfidfVectorizer`
- `ngram_range = (1, 2)`
- `min_df = 2`
- `max_df = 0.95`
- `sublinear_tf = True`

각 term `v_j ∈ V`에 대해, document `i`의 raw count를 `tf_{ij}`, document frequency를 `df_j`, subset document 수를 `M = |S_path|` 라고 두면 scikit-learn의 기본형에 맞춰 대략 아래처럼 쓸 수 있다.

`tf'_{ij} = 1 + log(tf_{ij})` if `tf_{ij} > 0`, else `0`

`idf_j = log((1 + M) / (1 + df_j)) + 1`

`x_{ij} = tf'_{ij} · idf_j`

그 뒤 각 row는 기본 설정대로 L2 normalize 된다.

`x_i <- x_i / ||x_i||_2`

이 표현을 쓰는 이유는 다음과 같다.

- 자주 나오는 일반 단어보다, 특정 진단 축을 구분하는 term에 더 높은 가중치를 주기 위해서다.
- unigram만 쓰면 `melanoma in situ`와 `basal cell carcinoma` 같은 multi-word 진단 패턴이 잘려나가므로 bigram을 함께 쓴다.
- `min_df = 2`는 train subset에서 한 번만 나온 오타/희귀 표현을 줄이기 위한 장치다.
- `max_df = 0.95`는 거의 모든 텍스트에 나오는 비구분적인 토큰을 약하게 만들기 위한 장치다.

### 4.2 Truncated SVD

TF-IDF는 sparse high-dimensional representation이므로, K-means를 바로 걸기 전에 저차원 dense embedding으로 바꾼다.

노트북은 아래 rank를 사용한다.

`r = max(2, min(32, M - 1, |V| - 1))`

그리고 TF-IDF 행렬 `X`를 rank-`r` 근사로 압축한다.

`X ≈ U_r Σ_r V_r^T`

각 샘플의 dense embedding은

`z_i = x_i V_r`

로 볼 수 있다. 현재 실행에서는 `r = 32`이고, 누적 설명분산 비율 합은 `0.994136`이다.

이 단계의 의미는 다음과 같다.

- TF-IDF 단어 차원의 잡음을 줄인다.
- 동의어/유사 표현을 몇 개의 잠재 semantic 축으로 모은다.
- 이후 K-means가 raw token sparsity보다 `진단 의미 방향`을 기준으로 군집화하게 돕는다.

### 4.3 K-means objective

SVD embedding `z_i` 위에서 candidate `k`별로 K-means를 학습한다.

`min_{μ_1, ..., μ_k} Σ_{i ∈ S_path} || z_i - μ_{c_i} ||_2^2`

여기서 `μ_c`는 cluster `c`의 centroid이고, `c_i`는 샘플 `i`의 군집 할당이다.

현재 탐색 범위는 아래와 같다.

`K_candidates = { 2, 3, ..., min(8, |S_path| - 1) }`

현재는 `|S_path| = 740`이므로 실제 candidate는 `2`부터 `8`까지다.

### 4.4 silhouette score

각 `k`에 대해 silhouette score를 계산한다.

샘플 `i`에 대해

- `a(i)`: 같은 cluster 안 평균 거리
- `b(i)`: 가장 가까운 다른 cluster와의 평균 거리

라 두면 silhouette는

`s(i) = (b(i) - a(i)) / max(a(i), b(i))`

이고, 전체 score는 평균

`Silhouette(k) = (1 / |S_path|) Σ_i s(i)`

이다.

이 값이 클수록 `같은 cluster 안에서는 가깝고, 다른 cluster와는 멀다`는 뜻이다.

### 4.5 추천 규칙

노트북은 silhouette만으로 `k`를 고르지 않고, cluster 최소 크기까지 함께 본다.

`recommended(k) = 1[ min_cluster_size(k) ≥ 5 ]`

즉 너무 작은 cluster가 생기면 silhouette가 좋아 보여도 추천하지 않는다. 그 이유는 이후 auxiliary head의 보조 target으로 쓰려면, 최소한 각 cluster에 학습 가능한 샘플 수가 있어야 하기 때문이다.

그 뒤 선택 규칙은 아래와 같다.

1. `recommended(k)=True`인 후보만 남긴다.
2. 그 안에서 `silhouette_score` 내림차순, `min_cluster_size` 내림차순으로 정렬한다.
3. 추천 후보가 하나도 없으면 `min_cluster_size`, `silhouette_score` 순으로 fallback 선택한다.

현재는 모든 `k=2..8`이 최소 cluster size `≥ 5`를 만족해서 모두 추천되었고, 최종 선택은 `k=8`이다.

현재 평가 결과는 아래와 같다.

| k | silhouette | min size | max size | ratio max/min | recommended |
|---:|---:|---:|---:|---:|---|
| 2 | 0.311050 | 313 | 427 | 1.364217 | True |
| 3 | 0.375385 | 156 | 313 | 2.006410 | True |
| 4 | 0.443795 | 81 | 389 | 4.802469 | True |
| 5 | 0.515170 | 81 | 234 | 2.888889 | True |
| 6 | 0.579367 | 53 | 176 | 3.320755 | True |
| 7 | 0.615425 | 39 | 180 | 4.615385 | True |
| 8 | 0.654941 | 26 | 180 | 6.923077 | True |

즉 현재 train split에서는 `k`를 늘릴수록 pathology text의 세부 구조가 더 분리됐고, `k=8`에서 가장 높은 silhouette를 얻었다.

## 5. 현재 `k=8` oracle cluster의 의미

현재 선택된 8개 cluster는 단순 숫자 라벨이 아니라, 꽤 명확한 병리 semantic 축으로 읽힌다.

| cluster | rows | positive ratio | 대표 의미 축 |
|---:|---:|---:|---|
| 0 | 180 | 0.0% | 일반 benign nevus 및 관련 benign melanocytic lesions |
| 1 | 119 | 99.1597% | basal cell carcinoma 계열 |
| 2 | 113 | 100.0% | melanoma in situ / invasive melanoma 계열 |
| 3 | 53 | 0.0% | seborrheic keratosis / solar lentigo / LPLK 계열 |
| 4 | 55 | 0.0% | atypical melanocytic neoplasm / AIMP 계열 |
| 5 | 155 | 0.0% | dysplastic / Clark nevus 계열 |
| 6 | 39 | 100.0% | squamous cell carcinoma 계열 |
| 7 | 26 | 0.0% | solar or actinic keratosis 계열 |

이 결과가 말해 주는 바는 분명하다.

- cluster는 단순 benign/malignant 이진 복제가 아니다.
- malignant 안에서도 `BCC`, `melanoma`, `SCC`가 서로 다른 축으로 분리된다.
- benign/indeterminate 안에서도 `nevus`, `dysplastic nevus`, `seborrheic keratosis`, `actinic keratosis`, `atypical melanocytic neoplasm`이 별도 축으로 나뉜다.

즉 auxiliary oracle target은 `메인 이진 분류를 다시 한번 말해 주는 신호`가 아니라, `병리 semantic axis를 더 미세하게 정리한 보조 구조`에 가깝다.

다만 cluster id 숫자 자체에는 고유 의미가 없다. `cluster=2`가 melanoma라는 사실은 현재 train split에 적합한 결과일 뿐이고, outer fold를 다시 적합하면 번호는 달라질 수 있다. 중요한 것은 `번호`가 아니라 `semantic grouping`이다.

## 6. downstream 학습으로 연결하는 수식

14장은 EDA이지만, 결국 이후 모델이 어떻게 이 target을 사용할지까지 연결돼야 의미가 있다. 가장 단순한 연결은 `masked auxiliary classification`이다.

### 6.1 oracle cluster label

outer-fold train에서만 적합한 변환기를 사용해, subset 내부 샘플에만 cluster label을 붙인다.

`c_i = h_train_only(t_i)` for `i ∈ S_path`

여기서 `h_train_only(·)`는 `normalize -> TF-IDF -> SVD -> K-means assignment` 전체를 묶은 함수다.

subset 바깥 샘플에는 oracle target이 없다.

`c_i = undefined` for `i ∉ S_path`

### 6.2 subset mask

보조 loss 적용 여부를 나타내는 mask를

`m_i = 1[i ∈ S_path]`

로 둔다.

### 6.3 auxiliary oracle loss

모델 representation `f_i`에서 oracle auxiliary head가 cluster posterior를 출력한다고 하면,

`p_i = softmax(W_oracle f_i + b_oracle)`

보조 loss는 mask된 cross-entropy로 쓸 수 있다.

`L_aux = (1 / Σ_i m_i) Σ_i m_i · CE(p_i, c_i)`

여기서 `CE(p_i, c_i) = -log p_i[c_i]` 이다.

즉 pathology-informative subset에 속한 샘플만 oracle loss에 참여한다. subset 밖의 benign 거대 클래스 샘플은 `메인 이진 분류 손실`에만 참여한다.

### 6.4 전체 손실

메인 분류 손실을 `L_cls`라고 하면 전체 목적함수는 예를 들어

`L_total = L_cls + λ_aux L_aux`

로 둘 수 있다.

여기서 `λ_aux`는 oracle 보조 목적의 세기를 조절하는 하이퍼파라미터다.

이 구조가 중요한 이유는 다음과 같다.

- `iddx_full` 텍스트를 추론 입력으로 넣지 않아도 된다.
- train-time only semantic supervision으로 representation을 더 구조화할 수 있다.
- oracle 정보가 있는 희소한 샘플만 보조적으로 활용하므로, 메인 regime의 strictness를 깨지 않는다.

## 7. leakage 방지 규칙

oracle supervision은 텍스트를 다루기 때문에 leakage에 특히 민감하다. 이번 노트북과 [`outer_fold_fit_protocol.csv`](/home/junkim2603a/proj/paper_ajou_dev/artifacts/eda/isic2024/thesis_plan/fold_design/outer_fold_fit_protocol.csv) 이 강조하는 규칙은 아래와 같다.

1. TF-IDF vocabulary는 `outer_fold_train_only`
2. SVD basis는 `outer_fold_train_only`
3. K-means centroid는 `outer_fold_train_only`
4. val/test는 오직 `frozen transform + frozen centroids`만 적용

수식으로 쓰면, fold `r`에서

`h^(r) = fit_oracle_transform(D_train^(r))`

`c_i^(r) = h^(r)(t_i)` for `i ∈ S_path^(r)`

여기서 `S_path^(r)`는 fold `r`에서 정규화 후 `n_i ≠ "" and n_i ≠ "benign"`을 만족하는 샘플 집합이다. 이때 `fit`은 오직 `D_train^(r)`에서만 수행되고, `D_val^(r), D_test^(r)`는 transform/assignment만 받는다. 이 원칙이 깨지면 oracle vocabulary와 centroid가 validation/test 정보를 미리 보게 된다.

## 8. 왜 이 설계가 계획서와 잘 맞는가

이번 설계는 논문계획서의 세 조건을 동시에 만족한다.

1. `strict input purity 유지`
   `iddx_full`은 `X_strict`에 들어가지 않는다.
2. `train-time only oracle usage`
   oracle text는 보조 감독 신호로만 쓰인다.
3. `병리 의미 보존`
   benign 거대 클래스에 묻히지 않고, 실제 진단 semantic 축으로 압축된다.

또한 [`multimodal_experiment_matrix.csv`](/home/junkim2603a/proj/paper_ajou_dev/artifacts/eda/isic2024/thesis_plan/multimodal_experiment_matrix.csv) 에서 `multimodal_base_with_aux_oracle` 실험군이 따로 있는 이유도 여기서 설명된다. oracle supervision은 입력 modality 추가가 아니라, `기존 multimodal backbone 위에 얹는 train-time auxiliary task`이기 때문이다.

## 9. 해석 시 주의점

이 문서를 읽을 때 아래 세 가지를 함께 기억해야 한다.

- cluster는 `ground-truth pathology class`가 아니다. 텍스트 기반 semantic grouping이다.
- 현재 결과는 `내부 train split` 기준이다. 본평가에서는 outer fold마다 다시 적합해야 한다.
- `k=8`은 현재 split에서의 최적안이지, 절대 불변의 정답은 아니다. 다만 현재는 silhouette와 cluster size를 동시에 봤을 때 가장 설득력 있는 예비안이다.

한 줄로 요약하면, 14장은 `iddx_full`을 입력 변수로 넣는 대신, 병리 텍스트가 풍부한 subset만 골라 train split 내부에서 semantic cluster target으로 압축하고, 그 target을 auxiliary oracle loss로 연결하기 위한 설계 문서다.
