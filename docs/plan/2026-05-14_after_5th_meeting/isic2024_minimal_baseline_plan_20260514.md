# ISIC2024 Minimal Baseline Plan

작성일: 2026-05-14

## 1. 목적

이 문서는 5차 미팅에서 논의한 baseline 범위를 다시 줄여서, 가장 simple 한 시작점을 고정한다.

핵심 방향은 다음과 같다.

```text
많은 모델 비교가 아니라, protocol 검증 + 최소 비교표 확보부터 시작한다.
```

따라서 v0 baseline은 성능 최고 모델 탐색이 아니다. v0의 목적은 다음 질문에 답하는 것이다.

```text
1. patient-level split과 metric 계산이 믿을 수 있는가?
2. strict metadata만으로 최소 성능이 어느 정도인가?
3. image만으로 최소 성능이 어느 정도인가?
4. metadata + image의 가장 단순한 결합이 single-modal보다 나아지는가?
```

## 2. v0 Minimal Baseline 범위

v0에 포함하는 것은 아래 5개만이다.

| 구분 | 이름 | 역할 | paper-facing 해석 |
|---|---|---|---|
| sanity | `metadata_dummy` | metadata split/metric 확인 | 성능 모델 아님 |
| metadata | `metadata_logreg_balanced` | 첫 metadata actual baseline | v0 metadata 대표 |
| sanity | `image_dummy` | image split/loader/metric 확인 | 성능 모델 아님 |
| image | `image_resnet50_finetune` | 첫 image actual baseline | v0 image 대표 |
| multimodal | `multimodal_concat` | strict metadata feature vector와 ResNet-50 image embedding concat | v0 fusion 대표 |

v0에서 말하는 가장 simple 한 것의 범위는 다음과 같다.

```text
input:
  metadata: strict ordinary metadata only
  image: lesion image only
  multimodal: strict metadata features + ResNet-50 image embedding

split:
  patient-level Triple Stratified Nested CV
  cv_test_fold == outer_test
  outer test fold를 제외한 cv_train 내부에서 inner_train / inner_validation 사용
  outer_test는 final reporting 전용

model:
  metadata: LogisticRegression(class_weight="balanced")
  image: ImageNet pretrained ResNet-50 fine-tuning
  fusion: concat feature vector + shallow LogisticRegression(class_weight="balanced")

threshold:
  validation probabilities에서만 선택

metrics:
  pAUC@TPR>=0.80
  AUC
  F1
  precision
  recall
  balanced accuracy
  Average Precision / PR-AUC
```

## 3. v0에서 제외하는 것

아래 항목들은 baseline 후보로 유효하지만, v0에는 넣지 않는다.

| 항목 | 제외 이유 | 이후 단계 |
|---|---|---|
| `HistGradientBoostingClassifier` | metadata v0보다 한 단계 강한 sklearn boost baseline | v1 metadata 확장 |
| XGBoost / LightGBM / CatBoost | 강한 GBDT baseline이지만 simple 시작점은 아님 | v1/v2 tabular strong baseline |
| DenseNet / EfficientNet / ConvNeXt / ViT / DINOv2 | 여러 image backbone 비교가 되어 v0가 복잡해짐 | v1 image backbone 확장 |
| Resize + Flatten + PCA + LogisticRegression | image signal 모델이라기보다 smoke check에 가까움 | 필요 시 debug-only |
| frozen ResNet embedding + LogisticRegression | 좋은 중간 단계지만 v0 모델 수를 늘림 | v1 image/embedding baseline |
| score stacking LogisticRegression | probability-level trainable fusion이며 v0 concat과 질문이 다름 | v1 score-level fusion |
| multiple embedding concat variants | 여러 image embedding/backbone 비교가 되어 v0가 복잡해짐 | v2 multimodal fusion |
| deep concat / gated fusion / attention fusion | 새로운 multimodal architecture에 해당 | v2/v3 research model |
| imbalance ablation | weighted loss, sampler, focal loss 비교가 필요함 | baseline 안정화 이후 |
| LUPI / `iddx_full` auxiliary | candidate-only 연구 아이디어이며 기본 project identity가 아님 | baselines 이후 research candidate |

## 4. 데이터 처리 원칙

v0는 기존 strict input export 계약을 그대로 따른다.

참조 문서:

```text
docs/eda/isic2024_strict_input_export.md
```

기본 데이터 흐름은 다음과 같다.

```text
raw train metadata/image
  -> strict input export
  -> patient-level Triple Stratified Nested CV split
  -> metadata-only baseline
  -> image-only baseline
  -> image embedding export
  -> metadata feature + image embedding join by isic_id
  -> concat fusion
```

원칙:

1. Raw data는 `data/raw/isic_2024_challenge/`에서 읽기만 한다.
2. Strict input table은 ordinary inference-time metadata만 포함한다.
3. `iddx_full`, diagnosis text, pathology-derived text는 v0 input에 포함하지 않는다.
4. `iddx_full_train_only` sidecar는 v0에서 사용하지 않는다.
5. Metadata 결측치 처리, scaling, encoding은 fold train에서만 fit한다.
6. Validation/test에는 train-fitted transform만 적용한다.
7. Image baseline은 metadata baseline과 같은 nested split CSV를 사용한다.
8. Fusion은 `isic_id` 기준으로 strict metadata feature와 image embedding을 join한다.
9. `outer_test`는 model choice, threshold selection, calibration, feature selection에 사용하지 않는다.

## 5. v0 모델 정의

### 5.1 `metadata_dummy`

목적:

```text
metadata pipeline, patient split, metric implementation sanity check
```

모델:

```text
sklearn.dummy.DummyClassifier
```

해석:

```text
성능 비교 모델이 아니라 "아무것도 안 하는 기준선"이다.
```

### 5.2 `metadata_logreg_balanced`

목적:

```text
strict metadata-only의 첫 actual baseline
```

입력:

```text
strict_main_input
```

모델:

```text
LogisticRegression(class_weight="balanced")
```

전처리:

```text
numeric:
  train median imputation
  StandardScaler

categorical:
  constant "__missing__" imputation
  OneHotEncoder(handle_unknown="ignore")
```

주의:

```text
imputer, scaler, encoder는 fold train에서만 fit한다.
```

### 5.3 `image_dummy`

목적:

```text
image manifest, image split, loader, metric sanity check
```

해석:

```text
성능 비교 모델이 아니라 image pipeline이 정상인지 확인하는 기준선이다.
```

### 5.4 `image_resnet50_finetune`

목적:

```text
image-only의 첫 actual baseline
```

입력:

```text
lesion image only
```

모델:

```text
ImageNet pretrained ResNet-50
classification head replaced for binary classification
fine-tuning
```

기본 설정:

```text
config: experiments/configs/image_baselines/resnet50/config.json
image_size: 224
pretrained weights: torchvision DEFAULT
threshold_source: inner_validation_f1
```

v0에서는 ResNet-50 하나만 쓴다. DenseNet, EfficientNet, ViT 계열은 이후 확장으로 둔다.

### 5.5 `multimodal_concat`

목적:

```text
metadata feature와 image embedding을 concat하는 가장 단순한 feature-level fusion이 single-modal보다 나아지는지 확인
```

입력:

```text
strict metadata feature vector
ResNet-50 image embedding
```

결합:

```text
concat_feature = [metadata_feature_vector, image_embedding]
fusion_classifier = LogisticRegression(class_weight="balanced").fit(concat_feature_train, y_train)
multimodal_probability = fusion_classifier.predict_proba(concat_feature)[:, 1]
```

중요:

```text
probability stacking이 아니다.
metadata feature와 image embedding을 직접 concat한다.
fusion classifier, imputer, scaler는 fold train에서만 fit한다.
validation/test에는 train-fitted fusion pipeline만 적용한다.
```

현재 `run_multimodal_experiment.py`는 미구현 상태이므로, v0 concat fusion은 image embedding과 strict metadata table을 입력으로 받는 작은 script/CLI가 필요하다.

## 6. 실행 순서

### 6.1 Strict input export

```bash
PYTHONPATH=./src python -m isic2024_multimodal.cli.export_strict_input_dataset
```

생성되는 주요 산출물:

```text
data/processed/isic2024_strict_model_input.csv
data/processed/isic2024_iddx_full_train_only_sidecar.csv
data/splits/isic2024_official_train_nested_5x4_seed42.csv
experiments/evidence/validation_protocol/isic2024_strict_input_export_summary_seed42.json
```

### 6.2 Metadata baseline preflight

```bash
PYTHONPATH=./src python -m isic2024_multimodal.cli.run_tabular_baseline \
  --models logistic_regression \
  --feature-sets strict_main_input \
  --preflight-only
```

### 6.3 Metadata baseline run

```bash
PYTHONPATH=./src python -m isic2024_multimodal.cli.run_tabular_baseline \
  --models logistic_regression \
  --feature-sets strict_main_input \
  --run-group-id minimal_v0_metadata_logreg_balanced
```

주의:

```text
현재 repo의 logistic_regression builder는 class_weight="balanced"를 기본 적용한다.
unweighted LogisticRegression은 v0에 포함하지 않는다.
```

### 6.4 Image baseline preflight

```bash
PYTHONPATH=./src python -m isic2024_multimodal.cli.run_image_baseline \
  --config experiments/configs/image_baselines/resnet50/config.json \
  --preflight-only
```

### 6.5 Image baseline run

```bash
PYTHONPATH=./src python -m isic2024_multimodal.cli.run_image_baseline \
  --config experiments/configs/image_baselines/resnet50/config.json \
  --run-group-id minimal_v0_image_resnet50_finetune
```

### 6.6 Simple fusion

필요 입력:

```text
isic_id
target
split
strict metadata feature columns
image_embedding columns
```

필요 산출:

```text
isic_id
target
split
concat model probability
multimodal_probability
```

현재 상태:

```text
multimodal runner is not implemented.
v0 concat fusion requires a small post-processing script/CLI.
```

권장 위치:

```text
src/isic2024_multimodal/cli/run_minimal_concat_fusion.py
```

산출물 위치:

```text
experiments/outputs/multimodal_baselines/minimal_v0_concat/
experiments/tables/multimodal_baselines/minimal_v0_concat/
```

## 7. 평가 기준

모든 v0 결과는 같은 metric function과 같은 threshold protocol을 사용한다.

필수 metric:

```text
pAUC@TPR>=0.80
AUC
F1
precision
recall
balanced accuracy
Average Precision / PR-AUC
false positive count
false negative count
```

Threshold-dependent metric은 validation에서 선택한 threshold만 사용한다.

```text
threshold_source = inner_validation_f1
```

Outer test fold는 final reporting 전용이다.

```text
outer_test는 threshold selection, model choice, calibration, feature selection에 사용하지 않는다.
```

## 8. 추가 진행 범위

v0 결과가 정상적으로 나온 뒤 다음 순서로 확장한다.

### v1: metadata 확장

```text
HistGradientBoostingClassifier(class_weight="balanced")
LightGBM / XGBoost / CatBoost
```

목적:

```text
strict metadata에서 linear baseline보다 비선형 tabular model이 유리한지 확인
```

### v1: image 확장

```text
DenseNet-121
EfficientNet-B0
frozen ResNet-50 embedding + LogisticRegression
```

목적:

```text
ResNet-50 하나의 결과가 image baseline을 대표하기 충분한지 확인
```

### v1: fusion 확장

```text
score stacking: [metadata_probability, image_probability] -> LogisticRegression
fixed probability average: 0.5 * metadata_probability + 0.5 * image_probability
```

조건:

```text
stacking train feature는 leakage-free OOF prediction으로 만들어야 한다.
test prediction은 model choice 이후에만 평가한다.
```

### v2: multimodal architecture

```text
deep concat
gated fusion
attention fusion
```

목적:

```text
단순 score-level fusion보다 feature-level 또는 representation-level fusion이 나은지 확인
```

### v2/v3: imbalance ablation

```text
weighted CE
pos_weight BCE
focal loss
balanced sampler
positive oversampling
validation-selected threshold variants
```

주의:

```text
class weights and samplers are computed from training data only.
```

### research candidate: LUPI / privileged supervision

```text
iddx_full auxiliary target
privileged teacher
diagnosis-text contrastive loss
prototype alignment
```

주의:

```text
candidate-only
not ordinary inference input
must not be required by validation/test/inference dataloaders
```

## 9. v0 완료 기준

v0 baseline은 아래가 모두 충족될 때 완료로 본다.

1. `metadata_dummy`, `metadata_logreg_balanced`, `image_dummy`, `image_resnet50_finetune`, `multimodal_concat` 결과가 있다.
2. 모든 결과가 같은 patient-level Triple Stratified Nested CV split을 사용한다.
3. Patient overlap audit이 0이다.
4. `iddx_full`과 diagnosis/reference columns가 ordinary input에 없다.
5. Metadata preprocessing은 fold train에서만 fit된다.
6. Threshold는 validation에서만 선택된다.
7. Test data는 final metric reporting에만 쓰인다.
8. Fold, seed, config path, split source, threshold source가 결과에 기록된다.
9. Single-modal과 multimodal 결과가 같은 metric table format으로 정리된다.

## 10. 한 줄 결론

v0는 다음 한 문장으로 요약한다.

```text
Strict metadata LogisticRegression, ResNet-50 image model, and feature-level concat fusion under the same patient-level protocol.
```
