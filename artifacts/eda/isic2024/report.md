# ISIC2024 Tabular EDA Report

## 데이터 개요
- 전체 행 수: `401059`
- 양성 수: `393`
- 음성 수: `400666`
- 양성 비율: `0.000980`
- 이미지 경로 확인 성공 수: `401059`
- 이미지 경로 확인 실패 수: `0`
- 총 컬럼 수: `16`

## 결측 상위 컬럼
- `iddx_5`: `401058`
- `mel_mitotic_index`: `401006`
- `mel_thick_mm`: `400996`
- `iddx_4`: `400508`
- `iddx_3`: `399994`
- `iddx_2`: `399991`
- `lesion_id`: `379001`
- `attribution`: `0`
- `copyright_license`: `0`
- `iddx_1`: `0`

## 수치형 후보 컬럼
- `mel_mitotic_index`: 유효값 `0`개
- `mel_thick_mm`: 유효값 `63`개
- `tbp_lv_dnn_lesion_confidence`: 유효값 `401059`개

## feature 그룹 분류 초안
- 이 분류는 최종 확정이 아니라 EDA 해석을 위한 초안입니다.
- 초기 baseline 검토 가능: `mel_mitotic_index, mel_thick_mm, tbp_lv_dnn_lesion_confidence`
- 주의 검토 필요: `attribution, copyright_license, image_exists, image_path, isic_id, lesion_id, malignant`
- leakage 위험 높음: `iddx_1, iddx_2, iddx_3, iddx_4, iddx_5, iddx_full`

## 산출물
- `dataset_overview.json`
- `preview_rows.csv`
- `missingness_summary.csv`
- `numeric_summary.csv`
- `categorical_summary.csv`
- `target_rate_by_*.csv`
- `feature_groups.json`
