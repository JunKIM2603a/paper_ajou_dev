# 11.1 feature engineering 방법론 정리

## 1. 왜 별도 문서로 분리했는가

`11.1 feature engineering 후보 생성과 1차 선별`은 단순히 파생변수를 많이 만든 단계가 아니다. 이 단계의 핵심은 다음 두 가지를 동시에 만족하는 것이다.

1. 피부병변 진단 문헌에서 반복적으로 등장하는 해석 축을 가져온다.
2. ISIC 2024 `train-metadata.csv`에서 실제로 관측 가능한 컬럼으로만 다시 구성한다.

따라서 이 장은 코드 자체보다도, `어떤 문헌 아이디어를 어떤 수식으로 옮겼는가`를 설명하는 별도 방법론 문서가 함께 있어야 해석이 분명해진다.

## 2. 전체 적용 원칙

이번 feature engineering은 원 논문의 임상 점수식을 그대로 복제하지 않았다. 대신 아래의 절차를 따랐다.

1. 문헌에서 반복적으로 등장하는 진단 축을 먼저 정리한다.
2. 그 축을 현재 메타데이터 컬럼으로 근사할 수 있는지 확인한다.
3. 단일 컬럼이 아니라, 두 개 이상 컬럼의 상호작용 또는 정규화 비율로 재구성한다.
4. 생성된 후보를 train split에서 1차 screening 한다.

즉, 이 단계는 `문헌 기반 아이디어 생성`과 `데이터 기반 1차 정리`를 연결하는 단계로 보는 것이 맞다.

## 3. 문헌별로 어떻게 적용했는가

### 3.1 ABCD rule

ABCD rule은 원래 `Asymmetry`, `Border`, `Color`, `Diameter` 네 축을 함께 보는 규칙이다. 이번 notebook에서는 이 네 축을 다음처럼 메타데이터 컬럼으로 다시 번역했다.

1. `Asymmetry` -> `tbp_lv_symm_2axis`
2. `Border` -> `tbp_lv_norm_border`, `tbp_lv_perimeterMM`
3. `Color` -> `tbp_lv_deltaL`, `tbp_lv_deltaA`, `tbp_lv_deltaB`, `tbp_lv_norm_color`, `tbp_lv_C`, `tbp_lv_Cext`
4. `Diameter` -> `clin_size_long_diam_mm`, `tbp_lv_minorAxisMM`, `tbp_lv_areaMM2`

대표 적용 예시는 다음과 같다.

1. `feat_border_color_interaction = norm_border * norm_color`
2. `feat_symmetry_contrast_interaction = symm_2axis * contrast_euclidean`
3. `feat_abcd_proxy_raw = 1.3*symm_2axis + 0.1*norm_border + 0.5*norm_color + 0.5*contrast_euclidean`
4. `feat_long_to_minor_ratio = long_diam / minorAxisMM`
5. `feat_chroma_normalized_gap = (C-Cext) / (C+Cext)`

핵심은 `ABCD 점수식을 재현`한 것이 아니라, ABCD가 보는 방향을 현재 테이블형 변수로 다시 요약했다는 점이다.

### 3.2 CASH

CASH는 `Color`, `Architecture`, `Symmetry`, `Homogeneity`를 함께 보는 알고리즘이다. 이 축은 현재 데이터에서 아래와 같은 컬럼 조합으로 옮겼다.

1. `color_std_mean`, `radial_color_std_max`로 색 분산을 반영한다.
2. `norm_border`, `norm_color`로 구조 불규칙성과 색 불균일을 반영한다.
3. `symm_2axis`로 대칭성을 반영한다.
4. `stdL`, `stdLExt`로 내부/외부 밝기 변동을 반영한다.

대표 적용 예시는 다음과 같다.

1. `feat_contrast_to_color_variation = contrast_euclidean / color_std_mean`
2. `feat_contrast_to_radial_variation = contrast_euclidean / radial_color_std_max`
3. `feat_color_variation_total = color_std_mean + radial_color_std_max + stdL + stdLExt`
4. `feat_cash_proxy_raw = symm_2axis + norm_border + norm_color + color_std_mean`
5. `feat_structure_dispersion_proxy = norm_color + color_std_mean + radial_color_std_max`

여기서는 특히 `색 자체`보다 `색이 얼마나 고르지 않은가`를 파생변수로 적극적으로 만들었다는 점이 중요하다.

### 3.3 DermNet dermoscopic features

DermNet 쪽 설명은 점수식보다는 `여러 색`, `불균질`, `색 다양성`, `내부/외부 대비` 같은 해석 언어에 가깝다. notebook에서는 이 언어를 수치형 proxy로 다시 구성했다.

대표 적용 예시는 다음과 같다.

1. `feat_color_internal_magnitude = sqrt(A^2 + B^2 + C^2)`
2. `feat_color_external_magnitude = sqrt(Aext^2 + Bext^2 + Cext^2)`
3. `feat_hue_circular_gap = circular_abs_diff(H, Hext)`
4. `feat_deltaLB_to_stdL = deltaLB / stdL`
5. `feat_internal_external_std_ratio = stdL / stdLExt`

즉, DermNet의 시각적 설명을 `색 강도`, `색차`, `밝기 변동`, `hue 차이`의 수치로 치환한 셈이다.

### 3.4 SLICE-3D / ISIC 2024 구조

ISIC 2024는 일반적인 dermoscopy 테이블이 아니라, 3D TBP 기반 데이터라는 특성이 있다. 그래서 단순 색/경계/크기 외에 `위치와 3D 맥락`을 반영하는 축을 별도로 만들었다.

대표 적용 예시는 다음과 같다.

1. `feat_xyz_radius = sqrt(x^2 + y^2 + z^2)`
2. `feat_xz_radius = sqrt(x^2 + z^2)`
3. `feat_area_to_xyz_radius = areaMM2 / xyz_radius`
4. `feat_long_to_xyz_radius = long_diam / xyz_radius`
5. `feat_vertical_size_interaction = |y| * long_diam`

이 축은 임상 점수라기보다, 현재 데이터셋이 가진 `3D 위치 정보`를 반영한 dataset-aware proxy라고 보는 편이 정확하다.

## 4. 문헌 외에 추가된 보조 축

문헌 축만으로는 현재 메타데이터가 제공하는 환자 맥락을 다 쓰지 못한다. 그래서 일부 후보는 `context` 계열 보조 축으로 추가했다.

대표 예시는 다음과 같다.

1. `feat_age_size_interaction = age_approx * long_diam`
2. `feat_age_contrast_interaction = age_approx * contrast_euclidean`
3. `feat_age_perimeter_interaction = age_approx * perimeterMM`
4. `feat_nevi_border_interaction = nevi_confidence * norm_border`
5. `feat_nevi_symmetry_interaction = nevi_confidence * symm_2axis`

이 축은 문헌 점수의 직접 재현은 아니지만, `연령`, `모반 신뢰도`, `병변 크기`, `색차`, `경계`가 결합될 때 악성 구분 신호가 커지는지 보기 위한 보조 후보군이다.

## 5. 1차 선별에는 어떻게 연결됐는가

문헌과 데이터셋 구조를 바탕으로 만든 후보는 총 76개였다. 하지만 이 76개를 그대로 쓰지는 않았다. train split에서 아래 기준으로 1차 screening을 했다.

### 5.1 1차 통과 기준

1. `|standardized difference| >= 0.08`
2. `max |corr with base features| < 0.995`
3. `feature_std_train > 0`

의미는 다음과 같다.

1. 클래스 차이가 너무 약한 후보는 일단 제외한다.
2. 기존 base 변수의 거의 복제본인 후보는 제외한다.
3. 분산이 거의 없어서 정보량이 없는 후보는 제외한다.

### 5.2 이 기준이 필요한 이유

문헌 기반 후보라고 해서 모두 유효한 것은 아니다. 실제 train 데이터에서는 다음 문제가 자주 생긴다.

1. 그럴듯해 보이지만 benign/malignant 평균 차이가 거의 없는 후보
2. 사실상 기존 변수 하나를 조금 비튼 정도라서 새 정보가 없는 후보
3. 극단치에 의해 숫자만 크게 보이지만 분류에 안정적으로 기여하지 않는 후보

따라서 11.1은 문헌의 권위를 확인하는 장이 아니라, 문헌에서 출발한 후보가 이 데이터에서 최소한의 신호와 독립성을 가지는지 확인하는 장이다.

## 6. 실제로 어떤 문헌 아이디어가 살아남았는가

1차 screening을 통과한 후보는 59개였다. 그중 대표적으로 살아남은 축은 아래와 같다.

### 6.1 살아남은 ABCD / CASH 계열

1. `feat_architecture_proxy_sum`
2. `feat_border_color_interaction`
3. `feat_border_contrast_interaction`
4. `feat_hue_color_coupling`
5. `feat_abcd_proxy_raw`

이들은 `비대칭`, `경계`, `색`, `균질성`을 함께 묶는 축이라서, 단일 컬럼보다 더 강한 구조 신호를 보여줬다.

### 6.2 살아남은 색/불균질 계열

1. `feat_contrast_to_color_variation`
2. `feat_contrast_to_radial_variation`
3. `feat_color_variation_total`
4. `feat_hue_circular_gap`
5. `feat_chroma_normalized_gap`

이들은 색 자체보다 `색 차이와 색 분산의 관계`를 보는 방식이라서, 단순 원시 컬럼보다 더 분리력이 있었다.

### 6.3 살아남은 geometry 계열

1. `feat_diameter_color_coupling`
2. `feat_long_to_minor_ratio`
3. `feat_long_minor_difference`
4. `feat_perimeter_to_long_ratio`
5. `feat_area_eccentricity_coupling`

이 축은 크기와 형태를 동시에 본다는 점에서, 임상적으로도 직관적이고 train 데이터에서도 신호가 컸다.

### 6.4 살아남은 spatial / context 계열

1. `feat_xz_radius`
2. `feat_vertical_size_interaction`
3. `feat_area_to_xyz_radius`
4. `feat_age_size_interaction`
5. `feat_age_contrast_interaction`

이 축은 문헌형 feature보다 dataset-aware 성격이 더 강하지만, 실제 train 분리력과 novelty가 충분해 살아남았다.

## 7. 이 방법론을 어떻게 읽어야 하는가

이 단계는 `문헌 요약`이 아니라 `문헌 기반 proxy 설계 단계`로 읽는 것이 맞다.

1. 문헌은 feature 후보를 만드는 출발점이다.
2. 실제 채택 여부는 train split의 신호와 중복성으로 결정한다.
3. 따라서 `문헌에서 중요하다`와 `이 데이터에서 최종 채택된다`는 같은 말이 아니다.

한 줄로 정리하면 다음과 같다.

`11.1은 임상 문헌의 진단 축을 ISIC 2024 metadata에 맞는 파생변수 집합으로 다시 번역하고, 그 후보군을 train split에서 넓게 screening한 단계이다.`
