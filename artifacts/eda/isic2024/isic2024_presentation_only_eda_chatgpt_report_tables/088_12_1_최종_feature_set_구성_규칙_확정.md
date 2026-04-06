| col_1 | feature | family | literature_anchor | source_columns | goal | abs_target_corr | vif_with_v2_numeric | max_abs_corr_with_final_numeric_train |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | feat_border_color_interaction | architecture | ABCD border x color | tbp_lv_norm_border, tbp_lv_norm_color | 경계 불규칙성과 색 불균일을 함께 반영 | 0.0268 | 426.2530 | 0.8911 |
| 1 | feat_architecture_proxy_sum | architecture | ABCD / CASH proxy | tbp_lv_symm_2axis, tbp_lv_norm_border, tbp_lv_... | 비대칭-경계-색 축을 합산한 간단한 구조 점수 | 0.0253 | 4941.1696 | 0.7653 |
| 2 | feat_border_contrast_interaction | architecture | ABCD border x color contrast | tbp_lv_norm_border, tbp_lv_deltaL, tbp_lv_delt... | 경계 불규칙성과 색차 강도를 함께 반영 | 0.0222 | 5900.8545 | 0.9077 |
| 3 | feat_symmetry_contrast_interaction | architecture | ABCD asymmetry x color contrast | tbp_lv_symm_2axis, tbp_lv_deltaL, tbp_lv_delta... | 비대칭성과 색차 강도를 함께 반영 | 0.0135 | 260.2856 | 0.9077 |
| 4 | feat_color_to_border_ratio | architecture | ABCD color vs border balance | tbp_lv_norm_border, tbp_lv_norm_color | 색 불균일성이 경계 불규칙성보다 상대적으로 큰지 확인 | 0.0127 | 6544.0105 | 0.8401 |
| 5 | feat_hue_color_coupling | color | ABCD color x CASH homogeneity | tbp_lv_H, tbp_lv_Hext, tbp_lv_norm_color | hue 차이와 색 불균일을 동시에 반영 | 0.0280 | 77.1910 | 0.8918 |
| 6 | feat_chroma_normalized_gap | color | ABCD color | tbp_lv_C, tbp_lv_Cext | 채도 차이를 크기 보정된 비율로 표현 | 0.0255 | 146.6028 | 0.7684 |
| 7 | feat_color_variation_total | color | CASH homogeneity | tbp_lv_color_std_mean, tbp_lv_radial_color_std... | 색과 명도의 이질성을 한 축으로 합산 | 0.0249 | 181.6103 | 0.9170 |
| 8 | feat_hue_circular_gap | color | DermNet color variety | tbp_lv_H, tbp_lv_Hext | 내부와 외부의 hue 차이를 원형 축에서 계산 | 0.0185 | 26.1762 | 0.7566 |
| 9 | feat_contrast_to_color_variation | color | CASH color-homogeneity | tbp_lv_deltaL, tbp_lv_deltaA, tbp_lv_deltaB, t... | 색차 강도를 병변 내부 색 분산으로 다시 스케일링 | 0.0172 | 2329.3516 | 0.9919 |
| 10 | feat_red_green_normalized_gap | color | ABCD color | tbp_lv_A, tbp_lv_Aext | A축(적-녹) 차이를 안정적인 정규화 비율로 표현 | 0.0095 | 393.6578 | 0.8884 |
| 11 | feat_age_size_interaction | context | Age-context proxy | age_approx, clin_size_long_diam_mm | 연령과 장축 크기를 함께 반영 | 0.0361 | 5661.9117 | 0.8017 |
| 12 | feat_age_contrast_interaction | context | Age-context proxy | age_approx, tbp_lv_deltaL, tbp_lv_deltaA, tbp_... | 연령과 색차 강도를 함께 반영 | 0.0178 | 4872.2933 | 0.7576 |
| 13 | feat_nevi_border_interaction | context | Nevus-confidence context | tbp_lv_nevi_confidence, tbp_lv_norm_border | 모반 신뢰도와 경계 불규칙성을 결합 | 0.0116 | 5.6269 | 0.8714 |
| 14 | feat_diameter_color_coupling | geometry | Diameter x color | clin_size_long_diam_mm, tbp_lv_norm_color | 크기와 색 불균일을 함께 반영 | 0.0481 | 935.2737 | 0.9266 |
| 15 | feat_area_eccentricity_coupling | geometry | Size x eccentricity | tbp_lv_areaMM2, tbp_lv_eccentricity | 큰 병변이면서 길쭉한 형태를 함께 반영 | 0.0431 | 468.3824 | 0.9731 |
| 16 | feat_diameter_symmetry_coupling | geometry | Diameter x asymmetry | clin_size_long_diam_mm, tbp_lv_symm_2axis | 크기와 비대칭성을 함께 반영 | 0.0185 | 267.3951 | 0.8731 |
| 17 | feat_perimeter_to_long_ratio | geometry | Border length / diameter | tbp_lv_perimeterMM, clin_size_long_diam_mm | 둘레가 장축 대비 얼마나 큰지 확인 | 0.0160 | 1239.4638 | 0.7079 |
| 18 | feat_long_minor_difference | geometry | Diameter spread | clin_size_long_diam_mm, tbp_lv_minorAxisMM | 장축과 단축 차이 | 0.0158 | 45.4864 | 0.7444 |
| 19 | feat_long_to_minor_ratio | geometry | ABCD diameter / geometry | clin_size_long_diam_mm, tbp_lv_minorAxisMM | 장축 대비 단축 비율 | 0.0062 | 286.3948 | 0.8795 |
| 20 | feat_vertical_size_interaction | spatial | SLICE-3D spatial proxy | tbp_lv_y, clin_size_long_diam_mm | 세로 위치와 크기를 함께 반영 | 0.0303 | 10.6053 | 0.8195 |
| 21 | feat_xz_radius | spatial | SLICE-3D spatial proxy | tbp_lv_x, tbp_lv_z | x-z 평면 반경 | 0.0098 | 1.1388 | 0.2002 |
