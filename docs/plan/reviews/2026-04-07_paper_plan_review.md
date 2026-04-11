# 논문계획서 전면 재검토 메모

검토 대상: `(연구A)_논문계획서02.pdf`

검토 기준:
- 사용자가 제공한 정정 정보
- `artifacts/eda/isic2024/splits/train_metadata_with_internal_split.csv`
- `artifacts/eda/isic2024/splits/isic2024_internal_split_summary.csv`
- `artifacts/eda/isic2024/model_inputs/current_model_regime_feature_sets.csv`
- `artifacts/eda/isic2024/iddx_structure_review/iddx_structure_review.md`

## 1. 검증된 사실

### 라벨 정의
- `target = 1` if `iddx_1 == Malignant`
- `target = 0` if `iddx_1 in {Benign, Indeterminate}`
- 저장된 `target` 값은 위 규칙과 전부 일치함

### 전체 train dataset 기준
- 전체 행 수: `401,059`
- `Benign`: `400,552` (`99.873585%`)
- `Malignant`: `393` (`0.097991%`)
- `Indeterminate`: `114` (`0.028425%`)
- `iddx_full` 비결측 행 수: `401,059`
- `iddx_full` 고유 범주 수: `52`

### internal train dataset 기준
- 전체 행 수: `280,335`
- `Benign`: `279,984` (`99.874793%`)
- `Malignant`: `270` (`0.096313%`)
- `Indeterminate`: `81` (`0.028894%`)
- `iddx_full` 비결측 행 수: `280,335`
- `iddx_full` 고유 범주 수: `47`

### internal split 규모
- `train`: `280,335`
- `validation`: `65,684`
- `internal_test`: `55,040`
- split은 `patient_id` 기준으로 생성되어 있음

### 오라클 관련 실제 해석
- `iddx_full` 자체는 희소하지 않음: 전체 train에서 `100%` 기록됨
- 다만 더 깊은 병리 레벨은 희소함
- `iddx_2` 비결측: `1,068` (`0.266295%`)
- `iddx_3` 비결측: `1,065` (`0.265547%`)
- `iddx_4` 비결측: `551` (`0.137386%`)
- `iddx_5` 비결측: `1` (`0.000249%`)

### positive class 해석
- 본 과제의 positive class는 `melanoma only`가 아니라 `Malignant 전체`
- 전체 malignant `393`건 중 `Melanoma` 문자열이 포함된 경우는 `157`건 (`39.949109%`)
- 나머지 `236`건 (`60.050891%`)은 basal cell carcinoma, squamous cell carcinoma 등 비-melanoma malignant임

### feature regime 정의
- Strict: `51`개 입력 컬럼
- Relaxed: `53`개 입력 컬럼 = Strict + `attribution`, `copyright_license`
- Oracle: `65`개 입력 컬럼 = Relaxed + `lesion_id`, `iddx_1~5`, `iddx_full`, `mel_mitotic_index`, `mel_thick_mm`, `tbp_lv_dnn_lesion_confidence`, missing indicator 2개
- 즉 Strict / Relaxed / Oracle은 서로 다른 데이터셋이 아니라, 같은 train metadata 컬럼을 목적별로 분할한 입력 regime임

### `k` 차원 관련 주의점
- 전체 train 기준 `iddx_full` 범주는 `52`개
- internal train 기준 `iddx_full` 범주는 `47`개
- validation / internal_test에만 등장하고 internal train에는 없는 `iddx_full` 범주가 `5`개 존재함
- 따라서 prototype 개수 `k` 또는 Plan-C 출력 차원은 `52`로 둘지 `47`로 둘지 반드시 정의가 필요함

## 2. 핵심 수정 사항

### [최우선] 과제 정의가 "Melanoma 분류"로 잘못 적혀 있음
문제:
- 원고는 여러 곳에서 과제를 `피부암(Melanoma) 분류`처럼 서술함
- 실제 target은 `iddx_1` 기준 `Malignant vs Benign/Indeterminate` 이진 분류임
- 즉 melanoma-only classification이 아님

영향:
- 연구 문제 정의 자체가 바뀜
- 제목, 연구 목표, 실험 설명, 기대효과까지 연쇄적으로 부정확해짐

수정 권고:
- `Melanoma 분류`를 `악성 여부 분류`, `malignancy classification`, `malignant vs benign/indeterminate classification` 계열 표현으로 교체
- melanoma를 예시로 남기고 싶다면 `피부 병변의 악성 여부 분류(멜라노마 포함)`처럼 보조 설명으로 제한

우선 수정이 필요한 페이지:
- p.2
- p.4
- p.6
- p.8

### [최우선] "`iddx_full`이 0.98%만 기록됨" 서술은 사실과 다름
문제:
- 원고는 `iddx_full`이 전체 학습 데이터의 약 `0.98%` 수준이라고 반복 서술함
- 실제로는 전체 train `401,059/401,059`, internal train `280,335/280,335` 모두 `iddx_full`이 기록돼 있음

영향:
- 연구 동기의 핵심 근거가 잘못됨
- `초희소 오라클 데이터`라는 표현의 의미가 본문에서 무너짐

수정 권고:
- `iddx_full availability`를 희소성의 근거로 쓰지 말 것
- 희소성을 유지하고 싶다면 다음 둘 중 하나로 다시 정의할 것
- `Malignant` 양성 클래스가 극단적으로 희소함: 전체 train `0.097991%`, internal train `0.096313%`
- 또는 `iddx_2~iddx_5`와 같은 깊은 병리 계층 정보가 극도로 희소함

우선 수정이 필요한 페이지:
- p.2
- p.3
- p.4
- p.5
- p.7

### [높음] 데이터 분할 방법이 Methods에 명시되지 않았음
문제:
- 현재 본문은 train dataset을 internal `train / validation / internal_test`로 나눠 쓴다는 사실이 거의 드러나지 않음
- 실험 재현성과 해석 가능성 측면에서 큰 공백임

수정 권고:
- Methods 초반에 별도 subsection 추가
- 예시 문장:

> 본 연구는 ISIC 2024 공식 train metadata 401,059행을 `patient_id` 기준으로 internal `train(280,335)`, `validation(65,684)`, `internal_test(55,040)`로 분할하여 사용한다. 모델 학습은 internal train에서 수행하고, validation은 모델 선택 및 early stopping, internal test는 최종 내부 성능 점검에 사용한다.

### [높음] Strict / Relaxed / Oracle 정의가 현재 원고에 충분히 드러나지 않음
문제:
- 본문은 주로 Strict만 강조하지만, 실제로는 train metadata의 컬럼을 목적별로 Strict / Relaxed / Oracle regime에 할당한 구조임
- 이 정의가 빠지면 정보 누수 통제 논리가 불명확해짐

수정 권고:
- 별도 표 또는 짧은 subsection 추가
- 핵심 문장 예시:

> 본 연구는 동일한 train metadata의 컬럼을 leakage risk와 실제 진료 시점의 가용성에 따라 Strict, Relaxed, Oracle 세 regime로 구분하였다. Strict는 추론 시점에도 확보 가능한 기본 임상·형태·색채·위치 정보와 파생변수로 구성된 51개 입력 컬럼이며, Relaxed는 여기에 출처 메타데이터 2개를 추가한 53개 컬럼, Oracle은 병리 및 고위험 참조 컬럼을 포함한 65개 컬럼으로 정의하였다.

### [높음] "추론 시 오라클 데이터가 전면 결측"이라는 표현은 dataset 기준으로 부정확함
문제:
- 실제 dataset에는 `iddx_full`이 존재함
- 다만 deployment를 모사하기 위해 model input에서 제외하거나 masking해야 함

수정 권고:
- `전면 결측` 대신 `추론 시 입력에서 제외`, `배치 추론 상황을 모사하기 위해 oracle columns를 사용하지 않음`, `train-only teacher signal로 제한` 등으로 수정

권장 문장:

> 데이터셋에는 병리 관련 컬럼이 기록되어 있으나, 실제 추론 시나리오를 모사하기 위해 validation 및 internal test 추론 단계에서는 Oracle 컬럼을 모델 입력에서 제외한다. 해당 정보는 학습 단계의 제한적 teacher signal 또는 상한선 비교용 참조 정보로만 사용한다.

### [중간] Plan-C의 출력 차원 정의가 불명확함
문제:
- 본문은 `52-dimensional vector`를 직접 명시함
- 그러나 internal train에서 관측되는 `iddx_full` 범주는 `47`개뿐임

영향:
- softmax vocabulary 정의가 모호함
- unseen class 처리 방식을 묻는 반론이 쉽게 제기될 수 있음

수정 권고:
- 아래 중 하나를 논문에 명시
- `k = 47`: internal train에서 실제 관측된 범주만 예측
- `k = 52`: 전체 train ontology를 고정 vocabulary로 두되, internal train에 없는 5개 class는 zero-shot / unseen class로 취급

## 3. 페이지별 수정 포인트

### p.2
- `iddx_full`이 `0.98%`만 기록된다고 한 문단은 전면 수정 필요
- 연구 공백은 `iddx_full availability scarcity`가 아니라
- `극심한 class imbalance`
- `깊은 병리 계층 정보의 희소성`
- `추론 시 Oracle column 비사용`
  로 다시 써야 함

### p.3
- `극소수의 정답 데이터(0.98%)` 문구 삭제 또는 재정의 필요
- `정답 데이터`라는 표현도 부정확함
- 정답은 모든 row에 존재하며, 희소한 것은 positive class와 깊은 taxonomy임

### p.4
- 연구 목표에서 `Melanoma`를 `malignancy` 기준 표현으로 교체
- `iddx_full의 52개 계층적 범주`는 전체 train 기준이라는 점을 명시
- 바로 뒤에 internal split과 Strict / Relaxed / Oracle 정의를 추가하는 것이 좋음

### p.5
- `0.98%의 샘플에 한정하여 iddx_full 문자열을 분해한다`는 문장은 사실과 다름
- 실제로는 `iddx_full`이 모든 row에 존재하지만, 깊은 계층 정보와 positive class가 희소함
- Teacher/Student 구조를 유지하려면 `Oracle signal을 train-only로 제한한다`는 방향으로 서술을 바꿔야 함

### p.6
- `실제 진단 시 오라클 데이터는 전면 결측`을 `실제 추론 시 Oracle 컬럼은 입력에서 제외`로 바꾸는 편이 정확함
- 최종 예측 대상도 `피부암 예측`보다 `악성 여부 예측`이 맞음

### p.7
- `결측된 1% 미만의 희소 오라클 데이터`는 현재 사실과 충돌
- 이 요약문은 가장 먼저 다시 쓰는 것이 좋음

### p.8
- `52-dimensional vector`는 internal train 기준과 충돌 가능
- `52개 Taxonomy Path` 예시는 유지 가능하지만, vocabulary를 전체 train ontology로 고정한 것인지 설명이 필요

## 4. 추천 서술 방향

현재 원고의 핵심 아이디어는 살릴 수 있다. 다만 논리의 시작점을 다음처럼 바꾸는 것이 안전하다.

1. 문제 정의
- `melanoma 분류`가 아니라 `악성 여부 분류`

2. 데이터 난점
- `iddx_full 자체의 결측`이 아니라
- `극심한 class imbalance`
- `깊은 병리 taxonomy 정보의 희소성`
- `deployment 시 Oracle column 비사용 필요`

3. 데이터 구조
- 동일한 train metadata를 Strict / Relaxed / Oracle column regime으로 분할
- official train만으로 internal train / validation / internal_test 구성

4. 모델 역할 분리
- Student는 Strict 입력으로 실제 추론 상황을 담당
- Oracle 정보는 train-only teacher signal 또는 reference upper bound로 제한

## 5. 바로 반영하면 좋은 문장

### 연구 목표 대체 문장
> 본 연구는 ISIC 2024 공식 train metadata를 patient-level internal split으로 구성한 뒤, 추론 시점에 확보 가능한 Strict feature와 피부 병변 이미지를 중심으로 피부 병변의 악성 여부(`Malignant` vs `Benign/Indeterminate`)를 분류하는 다중모달 구조를 설계한다. 추가로 Oracle 컬럼은 학습 단계에서만 제한적으로 활용하여 정보 누수를 통제하면서도 병리 계층 정보를 distillation signal로 사용한다.

### 데이터 설명 대체 문장
> 전체 train dataset은 401,059행이며, 이 중 internal train은 280,335행, validation은 65,684행, internal test는 55,040행이다. `iddx_full`은 전체 row에 기록되어 있으나, internal train에서 관측되는 고유 범주는 47개이고 전체 train 기준 고유 범주는 52개이다. 반면 positive class(`iddx_1 = Malignant`)는 전체 train의 0.097991%로 극도로 희소하다.

### regime 설명 대체 문장
> Train metadata의 컬럼은 leakage risk와 실제 사용 가능 시점에 따라 Strict, Relaxed, Oracle 세 regime으로 분류하였다. Strict는 실제 추론 입력에 해당하는 51개 컬럼, Relaxed는 출처 메타데이터를 포함한 53개 컬럼, Oracle은 병리 관련 참조 컬럼을 포함한 65개 컬럼으로 정의하였다.
