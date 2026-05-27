# 번역 누락 및 LaTeX 수식 재검토 감사

이 감사 파일은 PDF 원문 기준 섹션형 번역본 생성 후의 구조/수식/그림 OCR 검수 기록이다.

## PDF별 구조 체크

| 번역 파일 | PDF | 페이지 | 추출 단위 | 섹션 수 | 그림 | LaTeX 수식 | 깨진 수식 패턴 |
|---|---|---:|---:|---:|---:|---:|---|
| `sensors_22_07530_v4_ko.md` | `sensors-22-07530-v4.pdf` | 24 | 360 | 19 | 17 | 12 | 없음 |
| `s41746_025_02070_7_ko.md` | `s41746-025-02070-7.pdf` | 11 | 278 | 21 | 4 | 0 | 없음 |
| `s41598_025_33536_z_1_ko.md` | `s41598-025-33536-z-1.pdf` | 11 | 161 | 4 | 6 | 4 | 없음 |
| `s41598_025_26392_4_ko.md` | `s41598-025-26392-4.pdf` | 15 | 208 | 10 | 13 | 4 | 없음 |
| `fsurg_09_1029991_ko.md` | `fsurg-09-1029991.pdf` | 9 | 168 | 15 | 4 | 7 | 없음 |
| `experimental_dermatology_2018_yap_multimodal_skin_lesion_classification_ko.md` | `Experimental Dermatology - 2018 - Yap - Multimodal skin lesion classification using deep learning.pdf` | 7 | 469 | 35 | 4 | 0 | 없음 |
| `an_attention_based_mechanism_to_combine_images_and_metadata_ko.md` | `An_Attention-Based_Mechanism_to_Combine_Images_and_Metadata_in_Deep_Learning_Models_Applied_to_Skin_Cancer_Classification.pdf` | 10 | 211 | 21 | 7 | 7 | 없음 |
| `2506_03420v1_hybrid_ensemble_segmentation_gbdt_ko.md` | `2506.03420v1.pdf` | 9 | 119 | 20 | 6 | 5 | 없음 |

## 수식 감사표

| 번역 파일 | 수식 번호 | 설명 | LaTeX | 상태 |
|---|---:|---|---|---|
| `sensors_22_07530_v4_ko.md` | 1 | Soft-Attention | `$f_{\mathrm{sa}}=\gamma^{t}\sum_{k=1}^{K}\mathrm{softmax}(W_k*t)$` | PDF 원문 대조 후 LaTeX 복원 |
| `sensors_22_07530_v4_ko.md` | 2 | 가중 범주형 교차엔트로피 | `$L(\theta,x_n)=-\frac{1}{N}\sum_{n=1}^{N}\sum_{c=1}^{C}W_c\,y_n^c\log(\hat{y}_n^c)$` | PDF 원문 대조 후 LaTeX 복원 |
| `sensors_22_07530_v4_ko.md` | 3 | 클래스 가중치 | `$W=N\odot D$` | PDF 원문 대조 후 LaTeX 복원 |
| `sensors_22_07530_v4_ko.md` | 4 | 불균형 보정 행렬 | `$D=\left[\frac{1}{C N_1},\frac{1}{C N_2},\ldots,\frac{1}{C N_n}\right]$` | PDF 원문 대조 후 LaTeX 복원 |
| `sensors_22_07530_v4_ko.md` | 5 | 거짓 양성 | `$FP_c=\sum_{i=1}^{C}a_{ic}-TP_c$` | PDF 원문 대조 후 LaTeX 복원 |
| `sensors_22_07530_v4_ko.md` | 6 | 거짓 음성 | `$FN_c=\sum_{j=1}^{C}a_{cj}-TP_c$` | PDF 원문 대조 후 LaTeX 복원 |
| `sensors_22_07530_v4_ko.md` | 7 | 참 음성 | `$TN_c=\sum_{i=1}^{C}\sum_{j=1}^{C}a_{ij}-TP_c-FP_c-FN_c$` | PDF 원문 대조 후 LaTeX 복원 |
| `sensors_22_07530_v4_ko.md` | 8 | 민감도 | `$\mathrm{Sensitivity}=\frac{TP}{TP+FN}$` | PDF 원문 대조 후 LaTeX 복원 |
| `sensors_22_07530_v4_ko.md` | 9 | 특이도 | `$\mathrm{Specificity}=\frac{TN}{TN+FP}$` | PDF 원문 대조 후 LaTeX 복원 |
| `sensors_22_07530_v4_ko.md` | 10 | 정밀도 | `$\mathrm{Precision}=\frac{TP}{TP+FP}$` | PDF 원문 대조 후 LaTeX 복원 |
| `sensors_22_07530_v4_ko.md` | 11 | F1 점수 | `$F1=\frac{2TP}{2TP+FP+FN}$` | PDF 원문 대조 후 LaTeX 복원 |
| `sensors_22_07530_v4_ko.md` | 12 | 정확도 | `$\mathrm{Accuracy}=\frac{TP+TN}{TP+FP+FN+TN}$` | PDF 원문 대조 후 LaTeX 복원 |
| `s41598_025_33536_z_1_ko.md` | 1 | 영상/임상 특징 | `$f_{\mathrm{image}}=\mathrm{CNN}_{\mathrm{image}}(I),\quad f_{\mathrm{clinical}}=[x_1,x_2,\ldots,x_n]$` | PDF 원문 대조 후 LaTeX 복원 |
| `s41598_025_33536_z_1_ko.md` | 2 | XGBoost 결합 예측 | `$y=\mathrm{XGBoost}\left(\mathrm{Concatenate}(f_{\mathrm{image}},f_{\mathrm{clinical}})\right)$` | PDF 원문 대조 후 LaTeX 복원 |
| `s41598_025_33536_z_1_ko.md` | 3 | 누적 로짓 | `$\mathrm{logit}\left(P(Y\leq k\mid X)\right)=\beta_0^{(k)}+\sum_{i=1}^{n}\beta_i^{(k)}X_i$` | PDF 원문 대조 후 LaTeX 복원 |
| `s41598_025_33536_z_1_ko.md` | 4 | 선형 점수 | `$\beta_0+\beta_1X_1+\beta_2X_2+\cdots+\beta_nX_n$` | PDF 원문 대조 후 LaTeX 복원 |
| `s41598_025_26392_4_ko.md` | 1 | 민감도 | `$\mathrm{Sensitivity}=\frac{TP}{TP+FN}$` | PDF 원문 대조 후 LaTeX 복원 |
| `s41598_025_26392_4_ko.md` | 2 | 특이도 | `$\mathrm{Specificity}=\frac{TN}{FP+TN}$` | PDF 원문 대조 후 LaTeX 복원 |
| `s41598_025_26392_4_ko.md` | 3 | 정확도 | `$\mathrm{ACC}=\frac{\mathrm{SEN}+\mathrm{SPC}}{2}$` | PDF 원문 대조 후 LaTeX 복원 |
| `s41598_025_26392_4_ko.md` | 4 | AUC 확률 해석 | `$\mathrm{AUC}=P\left(\mathrm{Score}(TP)>\mathrm{Score}(TN)\right)$` | PDF 원문 대조 후 LaTeX 복원 |
| `fsurg_09_1029991_ko.md` | 1 | 클래스 확률 | `$\hat{y}=p\left(y=c\mid x_{\mathrm{img}},x_{\mathrm{meta}}\right)$` | PDF 원문 대조 후 LaTeX 복원 |
| `fsurg_09_1029991_ko.md` | 2 | Self-attention 투영 | `$Q=W_qx,\quad K=W_kx,\quad V=W_vx$` | PDF 원문 대조 후 LaTeX 복원 |
| `fsurg_09_1029991_ko.md` | 3 | Scaled dot-product attention | `$x'=\mathrm{softmax}\left(\frac{QK^{T}}{\sqrt{d}}\right)V$` | PDF 원문 대조 후 LaTeX 복원 |
| `fsurg_09_1029991_ko.md` | 4 | 영상 경로 cross-attention | `$Q_1=W_{q1}x'_{\mathrm{img}},\quad K_1=W_{k1}x'_{\mathrm{meta}},\quad V_1=W_{v1}x'_{\mathrm{img}}$` | PDF 원문 대조 후 LaTeX 복원 |
| `fsurg_09_1029991_ko.md` | 5 | 영상 경로 출력 | `$x''_{\mathrm{img}}=\mathrm{softmax}\left(\frac{Q_1K_1^{T}}{\sqrt{d}}\right)V_1$` | PDF 원문 대조 후 LaTeX 복원 |
| `fsurg_09_1029991_ko.md` | 6 | 메타데이터 경로 cross-attention | `$Q_2=W_{q2}x'_{\mathrm{meta}},\quad K_2=W_{k2}x'_{\mathrm{img}},\quad V_2=W_{v2}x'_{\mathrm{meta}}$` | PDF 원문 대조 후 LaTeX 복원 |
| `fsurg_09_1029991_ko.md` | 7 | 메타데이터 경로 출력 | `$x''_{\mathrm{meta}}=\mathrm{softmax}\left(\frac{Q_2K_2^{T}}{\sqrt{d}}\right)V_2$` | PDF 원문 대조 후 LaTeX 복원 |
| `an_attention_based_mechanism_to_combine_images_and_metadata_ko.md` | 1 | 특징 추출 | `$\tilde{x}_{\mathrm{img}}=\psi_{\mathrm{img}}(x_{\mathrm{img}}),\quad \tilde{x}_{\mathrm{meta}}=\psi_{\mathrm{meta}}(x_{\mathrm{meta}})$` | PDF 원문 대조 후 LaTeX 복원 |
| `an_attention_based_mechanism_to_combine_images_and_metadata_ko.md` | 2 | 분류 확률 | `$\hat{y}=p\left(y=c\mid \tilde{x}_{\mathrm{img}},\tilde{x}_{\mathrm{meta}}\right)$` | PDF 원문 대조 후 LaTeX 복원 |
| `an_attention_based_mechanism_to_combine_images_and_metadata_ko.md` | 3 | MetaBlock 출력 | `$\tilde{x}=\sigma\left(\tanh\left(f_b(\tilde{x}_{\mathrm{meta}})\odot \tilde{x}_{\mathrm{img}}\right)+g_b(\tilde{x}_{\mathrm{meta}})\right)$` | PDF 원문 대조 후 LaTeX 복원 |
| `an_attention_based_mechanism_to_combine_images_and_metadata_ko.md` | 4 | 게이트 함수 | `$f_b(\tilde{x}_{\mathrm{meta}})=W_f^{T}\tilde{x}_{\mathrm{meta}}+w_{0f}$` | PDF 원문 대조 후 LaTeX 복원 |
| `an_attention_based_mechanism_to_combine_images_and_metadata_ko.md` | 5 | 게이트 함수 | `$g_b(\tilde{x}_{\mathrm{meta}})=W_g^{T}\tilde{x}_{\mathrm{meta}}+w_{0g}$` | PDF 원문 대조 후 LaTeX 복원 |
| `an_attention_based_mechanism_to_combine_images_and_metadata_ko.md` | 6 | 쌍곡탄젠트 게이트 | `$T_{\mathrm{gate}}=\tanh\left(f_b(\tilde{x}_{\mathrm{meta}})\odot \tilde{x}_{\mathrm{img}}\right)$` | PDF 원문 대조 후 LaTeX 복원 |
| `an_attention_based_mechanism_to_combine_images_and_metadata_ko.md` | 7 | 시그모이드 게이트 | `$S_{\mathrm{gate}}=\sigma\left(T_{\mathrm{gate}}+g_b(\tilde{x}_{\mathrm{meta}})\right)$` | PDF 원문 대조 후 LaTeX 복원 |
| `2506_03420v1_hybrid_ensemble_segmentation_gbdt_ko.md` | 1 | 분류 손실 | `$L_{\mathrm{cls}}=-\left[y\log(\hat{y})+(1-y)\log(1-\hat{y})\right]$` | PDF 원문 대조 후 LaTeX 복원 |
| `2506_03420v1_hybrid_ensemble_segmentation_gbdt_ko.md` | 2 | CAM Dice 손실 | `$L_{\mathrm{cam}}=1-\frac{2\sum M_{\mathrm{cam}}\cdot M_{\mathrm{gt}}}{\sum M_{\mathrm{cam}}+\sum M_{\mathrm{gt}}+\epsilon}$` | PDF 원문 대조 후 LaTeX 복원 |
| `2506_03420v1_hybrid_ensemble_segmentation_gbdt_ko.md` | 3 | 분할 손실 | `$L_{\mathrm{seg}}=L_{\mathrm{mask}}^{\mathrm{BCE}}+L_{\mathrm{mask}}^{\mathrm{Dice}}$` | PDF 원문 대조 후 LaTeX 복원 |
| `2506_03420v1_hybrid_ensemble_segmentation_gbdt_ko.md` | 4 | 전체 손실 | `$L_{\mathrm{total}}=L_{\mathrm{cls}}+L_{\mathrm{cam}}+L_{\mathrm{seg}}$` | PDF 원문 대조 후 LaTeX 복원 |
| `2506_03420v1_hybrid_ensemble_segmentation_gbdt_ko.md` | 5 | 80% TPR 이상 pAUC | `$\mathrm{pAUC}=\int_{0.8}^{1.0}\mathrm{ROC}(t)\,dt$` | PDF 원문 대조 후 LaTeX 복원 |

## 검수 메모

- `## 1쪽` 형태의 페이지 기준 제목은 사용하지 않았다.
- 수식은 번역 모델에 맡기지 않고 별도 LaTeX 매핑으로 복원했다.
- 그림 내부 텍스트는 RapidOCR로 판독 가능한 문자열을 추출해 번역 블록에 기록했다.
- OCR이 판독하지 못한 순수 이미지 영역은 이미지 자체를 보존했고, 캡션은 본문 번역에 포함했다.
