# 11.1 feature engineering 방법론 정리

이 문서는 [`notebooks/isic_2024/isic2024_eda_20260411.ipynb`](/home/junkim2603a/proj/paper_ajou_dev/notebooks/isic_2024/isic2024_eda_20260411.ipynb) 의 `11.1 feature engineering 후보 생성과 1차 선별`이 실제로 어떤 규칙으로 구성됐는지를 풀어쓴 설명서다. 핵심은 `임상 문헌의 해석축을 train-metadata가 제공하는 수치 컬럼으로 근사`해서 넓은 후보군을 만든 뒤, `너무 약한 후보`와 `기존 base feature와 거의 같은 후보`만 먼저 걷어내는 것이다.

중요한 점은, 여기서 만드는 값이 임상 문헌의 공식 점수를 그대로 재현하는 것은 아니라는 점이다. 현재 데이터셋은 `ABCD`, `CASH`, `DermNet`이 직접 요구하는 병리적/시각적 annotation을 모두 제공하지 않는다. 따라서 이번 단계는 `문헌의 해석 방향`을 따르되, 실제 구현은 `ISIC 2024 train-metadata`의 색좌표, 형태, 위치, 환자 맥락 컬럼으로 근사한 `proxy feature engineering` 단계라고 보는 것이 맞다.

## 1. 입력 계약과 계산 단위

- 파생변수는 `strict_preprocessing_spec['strict_numeric_columns']`에 포함된 numeric 컬럼만 사용해 만든다.
- 계산은 `train split` 기준으로만 수행한다.
- 결측치는 `strict_preprocessing_spec['numeric_median_imputation']`에 저장된 train median으로 먼저 채운 뒤 파생변수를 계산한다.
- 분모가 0이 될 수 있는 비율식에는 `eps = 1e-6`를 넣어 수치 폭주를 막는다.
- 계산 결과에서 `inf`, `-inf`는 `NaN`으로 바꾼 뒤 `0.0`으로 채운다.
- 따라서 이 문서의 모든 screening 수치(`std diff`, `base corr`, `novelty score`)는 `train split only` 결과다.

즉 `11.1`은 test를 들여다보는 단계가 아니라, `strict_raw_numeric` 안에서 설명 가능한 조합 후보를 넓게 만들고 train 내부에서만 1차 검토하는 단계다.

## 2. 문헌 축을 현재 메타데이터에 어떻게 옮겼는가

| 문헌 축 | 원래 보고 싶은 개념 | 이번 데이터셋에서 사용한 proxy 컬럼 | 대표 feature 예시 |
|---|---|---|---|
| `ABCD rule` | 비대칭, 경계, 색, 직경 | `tbp_lv_symm_2axis`, `tbp_lv_norm_border`, `tbp_lv_norm_color`, `tbp_lv_deltaA/B/L`, `tbp_lv_A/Aext/B/Bext/C/Cext/H/Hext`, `clin_size_long_diam_mm`, `tbp_lv_minorAxisMM`, `tbp_lv_areaMM2`, `tbp_lv_perimeterMM` | `feat_border_color_interaction`, `feat_symmetry_contrast_interaction`, `feat_chroma_normalized_gap`, `feat_long_to_minor_ratio`, `feat_diameter_color_coupling` |
| `CASH` | color, architecture, symmetry, homogeneity | `tbp_lv_norm_color`, `tbp_lv_norm_border`, `tbp_lv_symm_2axis`, `tbp_lv_color_std_mean`, `tbp_lv_radial_color_std_max`, `tbp_lv_stdL`, `tbp_lv_stdLExt` | `feat_architecture_proxy_sum`, `feat_cash_proxy_raw`, `feat_contrast_to_color_variation`, `feat_color_variation_total` |
| `DermNet dermoscopic features` | multi-color, hue 차이, 불균질성, 내부/외부 대비 | `tbp_lv_H/Hext`, `tbp_lv_A/Aext`, `tbp_lv_B/Bext`, `tbp_lv_L/Lext`, `tbp_lv_deltaLB`, `tbp_lv_deltaLBnorm`, `tbp_lv_stdL`, `tbp_lv_stdLExt` | `feat_hue_circular_gap`, `feat_color_contrast_euclidean`, `feat_internal_external_std_ratio` |
| `SLICE-3D / ISIC 2024 구조` | 위치, 해부학적 축, 크기-좌표 결합 | `tbp_lv_x`, `tbp_lv_y`, `tbp_lv_z`, `clin_size_long_diam_mm`, `tbp_lv_areaMM2`, `tbp_lv_symm_2axis` | `feat_xz_radius`, `feat_vertical_size_interaction`, `feat_area_to_xyz_radius` |

이번 구현은 `문헌 개념을 그대로 복사`한 것이 아니라, `문헌에서 중요하다고 보는 방향을 메타데이터 상에서 다시 묶은 것`에 가깝다. 예를 들어 `ABCD proxy raw`나 `CASH proxy raw`는 임상 점수의 공식 구현이 아니라, 현재 데이터셋에 있는 변수들만으로 만든 `설명 가능한 합성 점수`다.

## 3. 문헌별 적용 방식

### 3.1 `ABCD rule`

`ABCD`는 이번 후보군의 중심 축이다.

- `A (Asymmetry)`는 `tbp_lv_symm_2axis`로 근사했다.
- `B (Border)`는 `tbp_lv_norm_border`, `tbp_lv_perimeterMM`, `tbp_lv_area_perim_ratio`로 근사했다.
- `C (Color)`는 `deltaA`, `deltaB`, `deltaL`, 내부/외부 `A/B/C/H/L` 차이, `tbp_lv_norm_color`로 근사했다.
- `D (Diameter)`는 `clin_size_long_diam_mm`, `tbp_lv_minorAxisMM`, `tbp_lv_areaMM2`로 근사했다.

대표 공식은 아래와 같다.

- `feat_color_contrast_euclidean = sqrt(deltaL^2 + deltaA^2 + deltaB^2)`
- `feat_red_green_normalized_gap = (A - Aext) / (|A| + |Aext|)`
- `feat_blue_yellow_normalized_gap = (B - Bext) / (|B| + |Bext|)`
- `feat_border_color_interaction = norm_border * norm_color`
- `feat_symmetry_contrast_interaction = symm_2axis * contrast_euclidean`
- `feat_long_to_minor_ratio = long_diam / minorAxisMM`
- `feat_diameter_color_coupling = long_diam * norm_color`

즉 `ABCD`는 단일 score 하나로 끝내지 않고, `색`, `구조`, `크기`를 여러 조합으로 분해해서 넓게 후보를 만든다.

### 3.2 `CASH`

`CASH`는 `color`, `architecture`, `symmetry`, `homogeneity`를 함께 보려는 문헌 축이다. 이 축은 현재 메타데이터에서 직접적인 texture map이 없기 때문에, `색 분산`과 `경계/대칭`의 상호작용으로 근사했다.

대표 공식은 아래와 같다.

- `feat_contrast_to_color_variation = contrast_euclidean / color_std_mean`
- `feat_contrast_to_radial_variation = contrast_euclidean / radial_color_std_max`
- `feat_color_variation_total = color_std_mean + radial_color_std_max + stdL + stdLExt`
- `feat_architecture_proxy_sum = symm_2axis + norm_border + norm_color`
- `feat_cash_proxy_raw = symm_2axis + norm_border + norm_color + color_std_mean`

여기서 중요한 설계 의도는 두 가지다.

1. `구조가 불규칙한 병변인데 색도 불균질한가`
2. `내부-외부 contrast가 병변 내부 분산보다도 큰가`

즉 `CASH`는 단순한 색차 자체보다, `색차와 균질성의 결합`을 더 보려는 축으로 사용했다.

### 3.3 `DermNet dermoscopic features`

`DermNet`에서 강조하는 다색성, hue 차이, 내부/외부 대비, 밝기 변동성은 `ABCD color`와 일부 겹치지만, 이번 단계에서는 `색차를 어떤 방식으로 요약하느냐`를 더 세분화하는 역할을 맡았다.

대표 공식은 아래와 같다.

- `feat_hue_circular_gap = circular_abs_diff(H, Hext)`
- `feat_color_internal_external_gap = sqrt((A-Aext)^2 + (B-Bext)^2 + (L-Lext)^2)`
- `feat_internal_external_std_ratio = stdL / stdLExt`
- `feat_internal_external_std_balance = (stdL - stdLExt) / (stdL + stdLExt)`

이 축은 `색상환상의 hue 차이`와 `내부/외부 밝기 변동성 비율` 같은 해석 가능한 요약치를 더하는 데 의미가 있었다. 다만 실제 screening에서는 `std ratio`류가 생각보다 약했고, 최종적으로는 `feat_hue_circular_gap`처럼 더 직접적인 색 차이 해석축이 상대적으로 잘 남았다.

### 3.4 `SLICE-3D / train-metadata 구조`

ISIC 2024 메타데이터는 `tbp_lv_x`, `tbp_lv_y`, `tbp_lv_z`를 제공한다. 이는 고전적 dermoscopy 문헌과는 다른 축이지만, 이번 챌린지 데이터 구조를 반영하는 중요한 추가 정보다. 그래서 `문헌 기반 주축`은 아니지만, `현재 데이터셋이 허용하는 위치/좌표 맥락`을 파생변수로 만들어 같이 본다.

대표 공식은 아래와 같다.

- `feat_xyz_radius = sqrt(x^2 + y^2 + z^2)`
- `feat_xz_radius = sqrt(x^2 + z^2)`
- `feat_area_to_xyz_radius = areaMM2 / xyz_radius`
- `feat_vertical_size_interaction = |y| * long_diam`

이 축은 `병변의 절대 위치` 자체보다, `크기나 비대칭성을 위치에 대해 정규화했을 때 어떤 차이가 생기는가`를 보려는 목적에 가깝다.

## 4. 문헌 외에 추가한 보조 축

문헌만으로는 현재 `train-metadata.csv`가 주는 맥락을 충분히 반영하기 어렵기 때문에, 아래 두 축을 보조적으로 추가했다.

### 4.1 `Age-context proxy`

대표 후보:

- `feat_age_size_interaction = age_approx * long_diam`
- `feat_age_area_interaction = age_approx * areaMM2`
- `feat_age_contrast_interaction = age_approx * contrast_euclidean`

의도는 `같은 크기/같은 색차라도 연령 맥락에 따라 의미가 달라질 수 있는가`를 보는 것이다. 다만 이 축은 문헌형 morphology보다 dataset context를 더 탈 수 있으므로, 이후 단계에서 family cap을 더 보수적으로 적용한다.

### 4.2 `Nevus-confidence context`

대표 후보:

- `feat_nevi_border_interaction = nevi_confidence * norm_border`
- `feat_nevi_color_interaction = nevi_confidence * norm_color`
- `feat_nevi_symmetry_interaction = nevi_confidence * symm_2axis`

의도는 `모반 신뢰도`가 morphology와 결합될 때 분리력이 생기는지 보는 것이다. 이 역시 `문헌 기반 핵심 축`은 아니고, `데이터셋 제공 맥락을 보조적으로 활용`하는 축이다.

## 5. family 설계와 실제 후보 수

`11.1`에서는 총 `76개` 파생변수를 만들었고, bookkeeping을 위해 다섯 family로 나눴다.

| family | 역할 | 후보 수 | 1차 통과 수 |
|---|---|---:|---:|
| `color` | 내부/외부 색차, hue 차이, 색 분산, 명도 대비 | 18 | 13 |
| `architecture` | 비대칭, 경계, 색 불균일의 결합 구조 | 18 | 15 |
| `geometry` | 직경, 둘레, 면적, compactness, eccentricity 조합 | 20 | 14 |
| `context` | 연령, nevi confidence 같은 맥락형 조합 | 10 | 9 |
| `spatial` | `x/y/z` 좌표와 크기/형태 결합 | 10 | 8 |
| 합계 |  | 76 | 59 |

이 다섯 family는 `문헌 기반 주력 축`과 `데이터셋 구조 기반 보조 축`을 나눠 보기 위한 관리 단위다. 이후 `11.2`에서 family cap을 적용하는 이유도 여기에 있다.

## 6. 1차 선별 기준과 계산식

`11.1`의 screening은 일부러 단순하게 잡았다. 아직은 `후보끼리의 중복`을 세게 줄이지 않고, 먼저 `너무 약한 후보`와 `base feature와 거의 같은 후보`만 제거한다.

### 6.1 효과 크기

각 후보에 대해 아래 standardized difference를 계산한다.

`std_diff_train = (mean_malignant - mean_benign) / pooled_std`

여기서 `pooled_std = sqrt((var_benign + var_malignant) / 2)` 이다.

이 식을 쓴 이유는 `class 평균 차이`를 `그 feature 자체의 스케일`에서 떼어내고 싶었기 때문이다.

- 분자 `mean_malignant - mean_benign`는 가장 직접적인 class separation이다. malignant 평균이 benign 평균보다 크면 양수, 작으면 음수가 된다.
- 하지만 평균 차이만 보면 단위가 다른 feature끼리 비교가 안 된다. 예를 들어 어떤 후보는 값 범위가 `0~1`이고, 다른 후보는 `0~1000`일 수 있다.
- 그래서 분모에 `pooled_std`를 넣어 `같은 feature 안에서 평균 차이가 표준편차 몇 배인가`로 바꿨다. 이렇게 하면 서로 단위가 다른 후보라도 같은 잣대로 비교할 수 있다.
- `pooled_std = sqrt((var_benign + var_malignant) / 2)`를 쓴 것은, 한쪽 클래스 분산만 보지 않고 benign/malignant 양쪽의 변동성을 같이 반영하려는 의도다. 즉 `평균 차이는 커 보여도 두 클래스 안에서 값이 심하게 흔들리면` 효과 크기를 보수적으로 줄인다.

이번 단계에서 이 식은 `엄밀한 통계 검정`보다 `빠른 screening용 effect size` 역할을 한다. 표본 수가 극단적으로 불균형한 데이터셋이라 p-value류보다 `얼마나 떨어져 보이는가`를 간단히 보는 편이 더 실용적이었다.

문서와 CSV에는 방향을 버린 `abs_std_diff_train`도 같이 저장한다.

`abs_std_diff_train`를 따로 저장한 이유도 분명하다.

- `11.1`에서는 `어느 쪽이 더 큰가`보다 `두 클래스가 얼마나 떨어져 있는가`가 더 중요하다.
- 따라서 malignant에서 높아지는 후보와 benign에서 높아지는 후보를 같은 기준으로 비교하려면 절댓값이 필요하다.
- 방향 정보는 버리지 않고 원래의 `std_diff_train`도 같이 남겨 두기 때문에, 이후 해석 단계에서 어떤 쪽으로 움직이는 feature인지 다시 확인할 수 있다.

### 6.2 기존 base feature와의 중복

각 후보에 대해 `strict_train_numeric_raw_imputed_df`의 base numeric feature들과의 상관 절댓값 최댓값을 구한다.

`max_abs_corr_with_base_train = max(|corr(candidate, each_base_feature)|)`

이 식은 `새 feature가 정말 새 축인가`를 보기 위한 가장 단순한 중복도 지표다.

- 먼저 각 후보를 base numeric feature 하나하나와 상관시킨다.
- 상관에 절댓값을 씌운 이유는 `+0.99`와 `-0.99`가 모두 사실상 같은 정보를 뜻하기 때문이다. 부호가 반대여도 거의 완전한 선형 변환이면 새로운 정보라고 보기 어렵다.
- 그 다음 `max`를 취하는 이유는 `어느 base feature와 가장 비슷한가`만 알아도 screening에는 충분하기 때문이다. 평균 상관을 쓰면 `특정 base 하나와 거의 같은 후보`가 희석될 수 있다.

즉 이 값은 `candidate가 base feature 집합 중 하나와 얼마나 겹치는가`를 보수적으로 요약한 것이다. `max_abs_corr_with_base_train`가 높을수록, 그 후보는 새로운 engineered feature라기보다 `기존 raw feature의 재표현`일 가능성이 커진다.

즉 새 후보가 실제로 새로운 축인지, 아니면 기존 컬럼 하나를 거의 그대로 다시 쓴 것인지를 먼저 본다.

### 6.3 novelty score

통과 후보의 우선순위는 아래 점수로 정렬했다.

`novelty_score = abs_std_diff_train * (1 - clip(max_abs_corr_with_base_train, 0, 0.999))`

이 점수는 `분리력은 좋은데 base와 덜 겹치는 후보`를 위로 올리기 위해 만든 단순 ranking score다.

- 앞부분 `abs_std_diff_train`는 후보의 class separation 강도다.
- 뒷부분 `(1 - max_abs_corr_with_base_train)`는 `비중복 보정항`이다.
- 따라서 어떤 후보가 분리력은 크더라도 base feature와 거의 동일하면, 뒤 항이 거의 `0`에 가까워져서 우선순위가 내려간다.
- 반대로 분리력은 조금 약해도 base와 충분히 다른 축이면 점수가 어느 정도 유지된다.

여기서 `clip(max_abs_corr_with_base_train, 0, 0.999)`를 넣은 이유는 두 가지다.

- 상관계수는 이론적으로 `[-1, 1]` 범위여야 하므로, 계산 오차나 예외 상황 때문에 범위를 벗어나더라도 안전하게 자르기 위함이다.
- 상관이 `1.0`에 매우 가까운 후보는 novelty를 사실상 `0`으로 보내되, 후속 계산에서 극단값이 그대로 전달되는 것을 완화하기 위함이다.

결과적으로 이 식은 `분리력 x 비중복성`의 곱으로 볼 수 있다. 합이 아니라 곱을 쓴 이유는, 둘 중 하나가 매우 부족하면 순위가 확실히 내려가게 만들고 싶었기 때문이다. 즉 `신호는 강하지만 중복이 심한 후보`와 `새롭지만 신호가 거의 없는 후보`를 둘 다 중간 이하로 보내는 효과가 있다.

의도는 간단하다.

- `abs_std_diff_train`이 크면 malignant/benign 분리력이 좋다.
- `max_abs_corr_with_base_train`이 작으면 기존 변수와 덜 겹친다.

즉 `novelty_score`는 `분리력 x 비중복성`의 곱이다.

### 6.4 실제 pass 조건

아래 세 조건을 모두 만족하면 `screen_pass_v2 = True`로 둔다.

1. `abs_std_diff_train >= 0.08`
2. `max_abs_corr_with_base_train < 0.995`
3. `feature_std_train > 0`

각 임계값도 아래 같은 의도로 잡았다.

- `abs_std_diff_train >= 0.08`:
  아주 약한 흔들림 수준의 후보는 초기에 걷어내기 위한 최소 effect size cutoff다. 이번 단계는 최종 feature selection이 아니라 `넓은 후보 유지`가 목적이라 cutoff를 높게 잡지 않고, `완전히 약한 후보만 제외`하는 수준으로 두었다.
- `max_abs_corr_with_base_train < 0.995`:
  engineered feature가 사실상 base feature의 복사본인 경우를 막기 위한 cutoff다. `0.995`처럼 매우 높은 기준을 둔 이유는 `조금 비슷한 정도`는 허용하되, `거의 동일한 선형 재표현`만 제거하려는 1차 screening이기 때문이다.
- `feature_std_train > 0`:
  train에서 값이 상수면 어떤 모델에도 정보가 없고, 상관이나 standardized difference도 불안정해진다. 그래서 zero-variance feature는 무조건 제외한다.

즉 이 세 조건은 각각 `신호 부족`, `기존 변수 복제`, `정보량 0`을 막는 최소 안전장치다. `11.1`이 넓은 screening 단계인 만큼, 여기서는 보수적으로 많이 버리기보다 `버려도 되는 후보만 버리는 방향`으로 threshold를 설계했다.

이 기준으로 `76개 중 59개`가 1차 통과했다.

drop 사유도 단순하다.

- `11개`: `drop_low_signal`
- `6개`: `drop_high_base_overlap_or_zero_variance`

대표 예시는 아래와 같다.

- `feat_deltaLB_to_stdL`, `feat_ellipse_fill_ratio`, `feat_nevi_color_interaction`: 분산은 있지만 `abs_std_diff_train < 0.08`이라 탈락
- `feat_yz_radius`, `feat_abs_y`, `feat_perimeter_sq_to_area`, `feat_color_internal_magnitude`: base feature와의 상관이 너무 커서 탈락

## 7. 실제로 살아남은 문헌 아이디어

이 절의 `살아남은`은 두 단계로 나눠 보는 것이 정확하다.

1. `11.1` screening을 통과한 아이디어
2. 그 다음 `11.2` family cap + 후보-후보 overlap pruning까지 통과한 아이디어

### 7.1 `11.1` screening을 통과한 대표 아이디어

| 문헌/축 | 대표 통과 feature | 의미 |
|---|---|---|
| `ABCD color` | `feat_blue_yellow_normalized_gap`, `feat_chroma_normalized_gap`, `feat_red_green_normalized_gap` | 내부-외부 색 차이를 크기 보정된 비율로 본다 |
| `ABCD border/asymmetry` | `feat_border_color_interaction`, `feat_symmetry_contrast_interaction`, `feat_border_contrast_interaction` | 경계/비대칭과 색차를 따로 보지 않고 결합해서 본다 |
| `ABCD diameter / geometry` | `feat_perimeter_to_long_ratio`, `feat_long_to_minor_ratio`, `feat_diameter_color_coupling` | 크기 자체보다 `크기 x 형태` 또는 `크기 x 색`을 더 본다 |
| `CASH` | `feat_contrast_to_color_variation`, `feat_color_variation_total`, `feat_architecture_proxy_sum` | 색차와 균질성, 구조축을 함께 보는 proxy가 유효했다 |
| `DermNet` | `feat_hue_circular_gap`, `feat_color_contrast_euclidean` | hue 차이와 다색성 강도는 비교적 직접적인 신호로 남았다 |
| `SLICE-3D` | `feat_xz_radius`, `feat_vertical_size_interaction`, `feat_area_to_xyz_radius` | 위치 자체보다는 위치와 크기의 결합이 더 의미 있었다 |
| 문헌 외 보조 축 | `feat_age_contrast_interaction`, `feat_age_size_interaction`, `feat_nevi_border_interaction` | 연령/모반 신뢰도와 morphology 결합이 일부 보조 신호를 제공했다 |

### 7.2 `11.2`까지 이어진 대표 생존 축

후속 단계 `11.2`에서는 family cap과 후보-후보 고상관 제거를 적용한 뒤 `23개`가 남았다. family별 최종 생존 수는 아래와 같다.

- `architecture 6`
- `color 6`
- `geometry 6`
- `context 3`
- `spatial 2`

그중 문헌 축을 대표하는 생존 예시는 아래와 같다.

- `ABCD / CASH`: `feat_architecture_proxy_sum`, `feat_border_color_interaction`
- `ABCD color`: `feat_chroma_normalized_gap`, `feat_red_green_normalized_gap`
- `ABCD color x CASH`: `feat_hue_color_coupling`
- `DermNet`: `feat_hue_circular_gap`
- `Geometry`: `feat_perimeter_to_long_ratio`, `feat_diameter_color_coupling`, `feat_long_to_minor_ratio`
- `SLICE-3D`: `feat_xz_radius`, `feat_vertical_size_interaction`
- `보조 축`: `feat_age_contrast_interaction`, `feat_age_size_interaction`, `feat_nevi_border_interaction`

반대로 `DermNet heterogeneity`의 여러 ratio 계열은 많이 만들었지만, 실제로는 low-signal이어서 상당수가 여기서 정리됐다. 이 점이 이번 screening의 중요한 결론 중 하나다. 즉 `문헌에 등장하는 개념`이라고 해서 모두 살아남는 것은 아니고, `현재 데이터셋에서 실제로 분리력과 비중복성을 같이 갖는가`를 통과해야만 남는다.

## 8. 이 단계의 해석과 한계

`11.1`의 역할은 `최종 답 고르기`가 아니라 `설명 가능한 넓은 후보군 만들기`다. 따라서 아래를 의도적으로 남겨 둔다.

- 통과 후보끼리의 높은 상관
- family 내부의 유사 후보 다수
- context/spatial 계열의 dataset-sensitive 후보

즉 `11.1`만 보고 feature를 확정하면 안 되고, 반드시 다음 단계에서 다시 줄여야 한다. 실제로 노트북도 `11.2`에서 `family cap`, `이미 선택된 engineered feature와의 상관`, `target과의 직접 상관`을 함께 보며 한 번 더 보수적으로 줄인다.

정리하면, 이번 `11.1`의 방법론은 아래 한 문장으로 요약할 수 있다.

`ABCD / CASH / DermNet / SLICE-3D`의 해석축을 `strict_raw_numeric`으로 넓게 근사하고, train split 안에서 `분리력은 거의 없거나 기존 변수와 사실상 같은 후보`만 먼저 제거한 단계다.

## 참고 문헌

1. Dermoscopedia, `ABCD rule`
2. Henning JS et al., `The CASH (color, architecture, symmetry, and homogeneity) algorithm for dermoscopy`
3. DermNet, `Dermatoscopic features`
4. Rotemberg V et al., `SLICE-3D / ISIC 2024 Scientific Data`
