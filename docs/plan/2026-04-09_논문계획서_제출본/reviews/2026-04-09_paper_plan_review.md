# 2026-04-09 논문계획서 리뷰

검토 대상: [연구A_논문계획서(석사과정)_김준_202533156.pdf](/home/junkim2603a/proj/paper_ajou_dev/docs/plan/연구A_논문계획서(석사과정)_김준_202533156.pdf)

검토 관점:
- 연구문제 정의의 명확성
- `train-time privileged information` 사용 논리의 일관성
- 실험 설계의 식별 가능성
- 평가 프로토콜의 재현 가능성
- 논문 제목과 본문 서술의 정합성

## 총평

현재 버전은 이전 초안 대비 방향이 많이 좋아졌다. 특히 `strict input`과 `auxiliary oracle supervision`의 역할이 분리되어 있고, `iddx_full`을 추론 입력이 아니라 학습 단계의 제한적 신호로 다루겠다는 문제 설정이 선명해졌다. 또한 `Triple Stratified K-Fold`, `X_strict = [X_raw; X_fe]`, `image-only / strict tabular-only / multimodal base / oracle / correction` 식의 단계적 비교 구조가 들어가면서, 실험 스토리도 훨씬 논문형에 가까워졌다.

다만 심사나 구현 handoff 관점에서 보면, 아직 몇 군데는 표현을 더 고정해야 한다. 특히 `pathology-informative subset`의 의미, auxiliary task의 선택 기준, 하이퍼파라미터 선택 규칙, pAUC 정의는 지금보다 더 명시적으로 써 두는 편이 안전하다.

## 주요 findings

### 1. [높음] `pathology-informative subset`의 의미가 아직 약간 넓고, 주 타깃과의 관계가 모호하다

현재 본문은 `iddx_full != "Benign"`이면 `pathology-informative subset`으로 둔다고 정의한다. 이 정의 자체는 단순하고 구현 가능하지만, 주 타깃은 `Malignant vs Benign/Indeterminate` 이진 분류이므로, subset 안에 들어오는 샘플이 곧바로 `malignancy-positive signal`을 의미하는 것은 아니다. 즉 이 보조 과제가 "악성 힌트"를 주는 것인지, 아니면 "병리 구조 일반"을 학습시키는 것인지 한 문장 더 분명히 적어둘 필요가 있다.

리스크:
- 리뷰어가 `oracle task가 사실상 target leakage 아니냐`고 물을 수 있다.
- 반대로 `subset 기준이 너무 넓어 auxiliary task가 negative but complex lesion까지 섞는 것 아니냐`는 질문도 받을 수 있다.

권고:
- `pathology-informative subset`은 `malignancy surrogate`가 아니라 `병리 의미 구조를 가진 train-time privileged subset`이라는 점을 명시한다.
- `Indeterminate` 또는 target-negative인데도 병리적으로 의미 있는 문자열이 어떤 역할을 하는지 짧게 덧붙인다.
- 가능하면 한 줄짜리 ontology 설명이나 예시 표를 추가한다.

권장 문장:

> 본 연구의 auxiliary oracle task는 악성 여부를 직접 재예측하는 보조 분류기가 아니라, `iddx_full`에 포함된 병리 의미 구조를 학습 단계에서만 간접 반영하기 위한 privileged supervision이다. 따라서 `pathology-informative subset`은 `target-positive subset`과 동일한 개념이 아니며, 병리 의미가 존재하는 진단 문자열을 기준으로 정의한다.

### 2. [높음] oracle cluster 관련 선택 자유도가 커서, fold별 선택 편향 우려가 남아 있다

본문은 TF-IDF, Truncated SVD, K-means, silhouette score, minimum cluster size를 잘 적어 두었지만, 실제로는 아래 항목들이 모두 성능에 영향을 줄 수 있다.

- `k` 선택 규칙
- `n-gram` 범위
- SVD 차원 수
- auxiliary loss 가중치
- pathology-informative subset sampling ratio
- correction head 차원 수
- regularization 강도
- `null prototype` 사용 방식

이 값들을 모두 fold마다 유연하게 바꾸면, 결과적으로 본평가 단계에서 설계 자유도가 너무 커 보일 수 있다.

권고:
- `outer fold` 안쪽의 `internal train/validation`에서만 고르는 항목과, 연구 전체에서 고정하는 항목을 분리해서 표로 적는다.
- 가능하면 `k`, `oracle loss weight`, `correction dimension` 정도는 사전 후보 집합을 작게 고정한다.
- `prototype-weighted`와 `Correction Head` 비교가 공정하려면 backbone, optimizer, epoch budget, early stopping rule도 동일하게 둔다고 명시한다.

### 3. [중간] pAUC 정의가 아직 논문 수준으로는 조금 덜 고정되어 있다

본문은 `low-FPR` 구간의 `pAUC`를 주평가 지표로 쓰겠다고 했지만, 정확히 어느 구간인지가 아직 없다. ISIC/Kaggle 맥락에서는 metric 정의를 아주 구체적으로 적지 않으면, 실험 재현성과 비교 가능성이 떨어진다.

권고:
- `pAUC` 계산 구간을 정확한 수치로 명시한다.
- fold별 평균만이 아니라 `mean ± std`, `minimum fold pAUC`를 표준 리포트 형식으로 고정한다.
- 가능하면 주 비교쌍에 대해 paired test 또는 bootstrap CI를 병기할 계획도 한 줄 적는다.

### 4. [중간] baseline 구성이 좋아졌지만, 실무형 강기준선이 하나 더 있으면 더 탄탄하다

현재 `image-only`, `strict tabular-only`, `multimodal base model`, `simple concatenation` 참조 baseline은 논리상 충분히 괜찮다. 다만 피부 병변 메타데이터 문제에서는 여전히 트리 기반 tabular 모델이 강한 기준선으로 받아들여질 가능성이 크다. 지금 구조만으로도 논문은 가능하지만, 리뷰어가 `왜 FT-Transformer만 두고 strong GBDT baseline은 안 봤나`라고 물을 여지가 있다.

권고:
- `CatBoost` 또는 `LightGBM` 기반의 `strict tabular` 강기준선을 부록 또는 reference baseline으로 한 번 포함하는 것을 권한다.
- 최소한 본문에 `tabular strong baseline은 별도 보조 비교로 검토한다`는 문장을 넣어 두면 방어력이 좋아진다.

### 5. [중간] `Correction Head`와 `prototype-weighted` 보정 구조의 수학적 계약이 아직 느슨하다

현재 설명은 방향성 측면에서는 좋지만, 구현 단계에서 질문이 생기기 쉬운 부분이 남아 있다.

- `r̂`의 차원 정의
- `F_base`로의 사상 방식
- add-on residual인지 gated residual인지
- `null prototype`의 생성 방식
- correction branch와 base branch 사이 gradient 흐름
- auxiliary oracle supervision과 correction branch의 결합 순서

권고:
- 도식 1개 또는 수식 3~4줄 정도로 `F_base -> oracle branch / correction branch -> F_final` 계약을 명확히 적는다.
- `prototype-weighted`는 fallback이 아니라 독립 실험군이라는 현재 서술을 유지하되, 구체적인 scoring function 하나는 명시하는 편이 좋다.

### 6. [낮음] 제목의 `피부암 분류`는 본문보다 약간 넓은 표현이다

본문은 일관되게 `악성 여부` 예측 문제로 정리되어 있다. 그런데 제목은 `피부암 분류`로 되어 있어, 독자가 다중 클래스 암종 분류나 melanoma subtype 분류처럼 더 넓게 받아들일 수 있다.

권고:
- 가능하면 제목 또는 부제에 `악성 여부 분류`를 드러내는 표현을 넣는다.
- 예: `... 피부 병변 악성 여부 분류를 위한 ...`

이 부분은 치명적 오류는 아니지만, 제목과 target definition이 더 가까워지면 논문 메시지가 선명해진다.

## 좋아진 점

이번 버전은 이전에 문제였던 큰 축을 대부분 바로잡았다.

- `melanoma-only`처럼 읽히던 문제 정의가 `악성 여부 분류` 중심으로 정리되었다.
- `iddx_full`을 전면 결측 자료처럼 다루지 않고, `Benign` 편중과 `pathology-informative subset` 문제로 재해석했다.
- `strict tabular input`과 `auxiliary oracle supervision`의 역할 분리가 분명해졌다.
- `Triple Stratified K-Fold`와 fold-local fitting 원칙이 들어가 재현성 설명이 좋아졌다.
- `multimodal 자체 효과 -> oracle supervision 효과 -> correction 효과`의 3단 비교가 명확해졌다.
- 설명가능성 분석도 이미지, tabular, oracle cluster 반응으로 나뉘어 있어 논문의 마무리 그림을 그리기 좋다.

## 우선 반영 권고안

논문계획서를 한 번 더 다듬는다면, 아래 다섯 가지만 먼저 고치는 것을 권한다.

1. `pathology-informative subset`이 `target-positive subset`이 아니라는 점을 한 문장으로 못박기
2. `k`, `oracle loss weight`, `correction dimension`의 선택 규칙을 표로 고정하기
3. `pAUC`의 정확한 계산 구간과 fold 집계 규칙을 명시하기
4. `CatBoost/LightGBM` 계열 strong tabular baseline을 부록 또는 보조 실험으로 추가하기
5. `Correction Head`와 `prototype-weighted` 구조를 짧은 수식 또는 도식으로 고정하기

## 결론

현재 계획서는 이미 논문으로 밀어볼 수 있는 수준의 뼈대를 갖추고 있다. 핵심 아이디어도 분명하고, 실제 배치 조건과 학습 단계의 privileged information을 분리해서 다룬다는 메시지도 살아 있다. 지금 남은 과제는 방향 수정이 아니라, 리뷰어가 물을 법한 모호한 부분을 미리 닫아 두는 일이다. 위 항목들만 보강되면, 방법론적 정당성과 실험 재현성이 모두 한 단계 더 안정될 가능성이 높다.
