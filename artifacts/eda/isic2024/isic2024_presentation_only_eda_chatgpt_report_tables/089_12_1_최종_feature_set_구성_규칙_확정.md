| col_1 | feature | family | literature_anchor | source_columns | goal | abs_target_corr | vif_with_v2_numeric | max_abs_corr_with_final_numeric_train |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | feat_border_color_interaction | architecture | ABCD border x color | tbp_lv_norm_border, tbp_lv_norm_color | 경계 불규칙성과 색 불균일을 함께 반영 | 0.0268 | 426.2530 | 0.8911 |
| 1 | feat_architecture_proxy_sum | architecture | ABCD / CASH proxy | tbp_lv_symm_2axis, tbp_lv_norm_border, tbp_lv_... | 비대칭-경계-색 축을 합산한 간단한 구조 점수 | 0.0253 | 4941.1696 | 0.7653 |
| 2 | feat_border_contrast_interaction | architecture | ABCD border x color contrast | tbp_lv_norm_border, tbp_lv_deltaL, tbp_lv_delt... | 경계 불규칙성과 색차 강도를 함께 반영 | 0.0222 | 5900.8545 | 0.9077 |
| 3 | feat_hue_color_coupling | color | ABCD color x CASH homogeneity | tbp_lv_H, tbp_lv_Hext, tbp_lv_norm_color | hue 차이와 색 불균일을 동시에 반영 | 0.0280 | 77.1910 | 0.8918 |
| 4 | feat_chroma_normalized_gap | color | ABCD color | tbp_lv_C, tbp_lv_Cext | 채도 차이를 크기 보정된 비율로 표현 | 0.0255 | 146.6028 | 0.7684 |
| 5 | feat_color_variation_total | color | CASH homogeneity | tbp_lv_color_std_mean, tbp_lv_radial_color_std... | 색과 명도의 이질성을 한 축으로 합산 | 0.0249 | 181.6103 | 0.9170 |
| 6 | feat_age_size_interaction | context | Age-context proxy | age_approx, clin_size_long_diam_mm | 연령과 장축 크기를 함께 반영 | 0.0361 | 5661.9117 | 0.8017 |
| 7 | feat_diameter_color_coupling | geometry | Diameter x color | clin_size_long_diam_mm, tbp_lv_norm_color | 크기와 색 불균일을 함께 반영 | 0.0481 | 935.2737 | 0.9266 |
| 8 | feat_area_eccentricity_coupling | geometry | Size x eccentricity | tbp_lv_areaMM2, tbp_lv_eccentricity | 큰 병변이면서 길쭉한 형태를 함께 반영 | 0.0431 | 468.3824 | 0.9731 |
| 9 | feat_long_minor_difference | geometry | Diameter spread | clin_size_long_diam_mm, tbp_lv_minorAxisMM | 장축과 단축 차이 | 0.0158 | 45.4864 | 0.7444 |
| 10 | feat_vertical_size_interaction | spatial | SLICE-3D spatial proxy | tbp_lv_y, clin_size_long_diam_mm | 세로 위치와 크기를 함께 반영 | 0.0303 | 10.6053 | 0.8195 |
