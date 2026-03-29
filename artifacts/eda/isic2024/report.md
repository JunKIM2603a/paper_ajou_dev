# ISIC2024 Challenge Tabular EDA Report

## 데이터 개요
- 전체 행 수: `401059`
- 양성 수: `393`
- 음성 수: `400666`
- 양성 비율: `0.000980`
- 총 컬럼 수: `58`
- split 정책: `patient_id -> lesion_id -> isic_id`

## 결측 상위 컬럼
- `iddx_5`: `401058`
- `mel_mitotic_index`: `401006`
- `mel_thick_mm`: `400996`
- `iddx_4`: `400508`
- `iddx_3`: `399994`
- `iddx_2`: `399991`
- `lesion_id`: `379001`
- `sex`: `11517`
- `anatom_site_general`: `5756`
- `age_approx`: `2798`

## 수치형 컬럼
- 수치형 컬럼 수: `36`

## feature 그룹 분류 초안
- 이 분류는 feature set 정책을 설명하기 위한 EDA 초안입니다.
- 초기 baseline 검토 가능: `age_approx, anatom_site_general, clin_size_long_diam_mm, image_type, sex, tbp_lv_A, tbp_lv_Aext, tbp_lv_B, tbp_lv_Bext, tbp_lv_C, tbp_lv_Cext, tbp_lv_H, tbp_lv_Hext, tbp_lv_L, tbp_lv_Lext, tbp_lv_areaMM2, tbp_lv_area_perim_ratio, tbp_lv_color_std_mean, tbp_lv_deltaA, tbp_lv_deltaB, tbp_lv_deltaL, tbp_lv_deltaLB, tbp_lv_deltaLBnorm, tbp_lv_dnn_lesion_confidence, tbp_lv_eccentricity, tbp_lv_location, tbp_lv_location_simple, tbp_lv_minorAxisMM, tbp_lv_nevi_confidence, tbp_lv_norm_border, tbp_lv_norm_color, tbp_lv_perimeterMM, tbp_lv_radial_color_std_max, tbp_lv_stdL, tbp_lv_stdLExt, tbp_lv_symm_2axis, tbp_lv_symm_2axis_angle, tbp_lv_x, tbp_lv_y, tbp_lv_z, tbp_tile_type`
- 주의 검토 필요: `attribution, copyright_license, image_exists, image_path, isic_id, lesion_id, patient_id, split_group_id, target`
- leakage 위험 높음: `iddx_1, iddx_2, iddx_3, iddx_4, iddx_5, iddx_full, mel_mitotic_index, mel_thick_mm`
