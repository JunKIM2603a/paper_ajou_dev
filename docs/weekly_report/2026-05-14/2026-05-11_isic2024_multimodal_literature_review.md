# ISIC 2024 Train-Only Multimodal Literature Review

조사일: 2026-05-04  
주제: ISIC 2024 Kaggle train dataset만 사용한 skin cancer detection 멀티모달 모델 연구를 위한 선행논문 조사  
핵심 조사 대상: dataset 불균형 극복 방법, image-tabular multimodal fusion 방법

---

## 1. 연구 배경

ISIC 2024 Kaggle Challenge는 3D Total Body Photography(3D-TBP)에서 추출한 피부 병변 crop image와 환자/병변 metadata를 함께 제공하는 binary classification 문제이다. 공식 train dataset은 SLICE-3D dataset이며, 총 401,059개 lesion tile로 구성된다.

가장 중요한 특징은 극단적인 class imbalance이다.

| 항목 | 개수 | 비율 |
|---|---:|---:|
| Benign / target 0 | 400,666 | 99.902% |
| Malignant / target 1 | 393 | 0.098% |
| Total | 401,059 | 100% |

따라서 accuracy 중심 평가는 부적절하며, malignant를 놓치지 않는 high-sensitivity 영역의 성능 평가가 중요하다. Kaggle 공식 metric도 `pAUC > 80% TPR`를 사용했다.

ISIC 2024 train dataset의 modality는 크게 두 가지이다.

- Image: 15mm x 15mm field-of-view lesion crop image
- Tabular metadata: age, sex, anatomical site, lesion size/color/shape 관련 WB360 measurements, patient_id, lesion_id, attribution 등

---

## 2. 논문 분석 요약표

| 논문/자료 | 목표 & 핵심 기여 | Dataset 정보 | Imbalanced data 극복방법 | Tabular model | Image model | Fusion 방식 | 평가 지표 | 평가(성능) | 최종결과 |
|---|---|---|---|---|---|---|---|---|---|
| SLICE-3D Dataset, Scientific Data 2024 | ISIC 2024 공식 train dataset 기술 및 공개 | ISIC 2024 train; 401,059 lesion tiles; binary malignant/benign; 3D-TBP image + metadata | benign 400,666 vs malignant 393; 모델 학습 없음, 데이터/학습 조작 없음 | metadata 제공: demographics, anatomical site, WB360 measurements, patient_id | 3D-TBP lesion crop image | 해당 없음 | 해당 없음 | dataset descriptor | ISIC 2024 train-only 연구의 1차 dataset 인용 자료 |
| ISIC 2024 Automated Triage, npj Digital Medicine 2025 | Kaggle ISIC 2024 상위 모델과 ablation 분석 | ISIC 2024 Challenge dataset; binary malignant/benign; 3D-TBP tile + metadata + patient-context | class 축소 없음; pAUC metric, patient-context, 3개 GBT late fusion 중심 | 3개 Gradient Boosting Tree, metadata + engineered feature + patient-context feature | EVA 2개 + EdgeNeXt ensemble, 각 모델 5-fold aggregation | neural network output vector + metadata를 3개 GBT에 입력 후 output aggregation | 우선: pAUC>80% TPR; 보조: AUC, SE top-15, NNT@80/90% sensitivity | pAUC 0.1726/0.2, AUC 0.9668, NNT80 51.57, NNT90 98.20 | metadata와 patient-context가 image-only보다 강한 성능 기여 |
| Wang et al., Scientific Reports 2025 | explainable multimodal AI: ISIC 2024 3D TBP image + clinical data 결합 | ISIC 2024; 1,075 patients; 6-class lesion risk prediction; 41 clinical/lesion-specific features + 3D TBP images | 6-class 유지; non-nevus class targeted augmentation; 5-fold CV; ISIC binary benchmark로 pFPR 비교 | clinical-only ML 비교, XGBoost fusion, multinomial logistic regression scoring + VIF/nomogram | HAM10000 transfer learning CNN, 3 conv blocks + 2 Conv2D fine-tuning | CNN six-class probability vector + clinical feature vector concat/standardize -> XGBoost late fusion; SHAP/CAM/nomogram | 우선: AUC, recall/F1, pFPR; 보조: accuracy/confusion matrix | multimodal AUC > 0.95, recall/F1 > 95%, pFPR 0.17343; clinical XGBoost Acc 0.6837, image-only nevus Acc 87.10% | ISIC 2024에서 late fusion + XAI 설명 가능성의 직접 근거 |
| MetaBlock, JBHI 2021 | metadata로 CNN feature map을 scale/shift/gating하는 MetaBlock 제안 | PAD-UFES-20 6-class clinical image+metadata; ISIC 2019 8-class dermoscopy+metadata | class 수 조절 없음; weighted cross-entropy와 stratified 5-fold CV 사용 | metadata feature에서 `f_b`, `g_b` modifier 생성 | CNN backbone의 last feature maps | Eq. (3)의 metadata-conditioned feature map modulation | 우선: BACC; 보조: ACC, AUC | MetaBlock이 10개 실험 중 6개에서 최고 BACC, ISIC/PAD 각각 3개 CNN에서 최고 | ISIC 2024 중간 fusion baseline으로 적합 |
| MMF-Net, Frontiers in Surgery 2022 | smartphone clinical image + metadata fusion | PAD-UFES-20; 6-class smartphone clinical image + metadata | class 수 유지; stratified 5-fold CV + on-the-fly augmentation; 명시적 over/under sampling 없음 | numeric/Boolean/categorical metadata 전처리 + MLP encoder | ResNet-50 | intra-modality self-attention + 양방향 inter-modality cross-attention | 우선: BACC, aggregated AUC; 보조: ACC | BACC 0.775±0.022, AUC 0.947±0.007, ACC 0.768±0.022 | cross-attention fusion 근거로 적합 |
| Yap et al., Experimental Dermatology 2018 | dermoscopy + clinical image + patient metadata 결합 | 2917 cases; dermoscopic + macroscopic image + patient metadata; binary melanoma 및 5-class task | task 목적상 binary와 5-class를 별도 평가; imbalance-specific sampling/loss는 핵심 아님 | patient metadata 사용 | CNN 기반 dermoscopic/clinical image model | multimodal classifier | 우선: binary AUC, multiclass mAP | AUC 0.866 vs 0.784, mAP 0.729 vs 0.598 | image-only보다 multimodal이 우수 |
| Islam et al., Scientific Reports 2026 | patient metadata + DER/DSLR image fusion으로 suspicious lesion triage | 79,246 images; 39,623 lesions; 19,295 patients; suspicious/non-suspicious; DER/DSLR image + collected 22 meta-features | binary triage label 정의; patient-separated split, augmentation, decision-level majority voting | 22개 중 7 C4C risk factors + C4C risk score 중심 metadata model | EfficientNet-B2 계열 | image vector + 8개 metadata feature concat, 최종 decision-level majority voting | 우선: sensitivity, specificity; ACC는 (SEN+SPC)/2 | fused SEN 99.66±0.28%, SPC 74.45±0.80%; voting SEN 99.50±1.18%, SPC 82.72±1.64% | metadata fusion과 decision fusion이 specificity 개선 |
| Nguyen et al., Sensors 2022 | imbalanced skin lesion classification에서 soft-attention + weighted loss + metadata 사용 | HAM10000; 10,015 images; 7 classes; dermoscopy image + age/gender/localization metadata | class 수 유지; augmentation to 53,573 images; categorical cross-entropy에 class weight 적용 | age, gender, localization metadata dense branch | InceptionResNetV2, MobileNetV3Large 등 + Soft-Attention | Soft-Attention output + metadata dense feature concat | 우선: AUC, recall/F1; 보조: accuracy, precision | abstract: ACC 0.90, Precision/F1/Recall/AUC 0.81/0.81/0.82/0.99; conclusion: F1/Recall/AUC 0.86/0.81/0.975 | imbalance-aware loss와 attention의 결합 근거 |
| Focal Loss 2017 / Class-Balanced Loss 2019 | 일반 long-tail/class imbalance 학습의 대표 loss | 특정 dataset 고정 없음; long-tail classification 일반 loss | easy negative down-weighting, effective number 기반 class weight | 해당 없음 | 모든 CNN/ViT에 적용 가능 | 해당 없음 | task별 metric | long-tailed dataset에서 성능 개선 | ISIC 2024 image branch의 BCE 대체 loss 근거 |
| GAN/Diffusion augmentation 관련 연구 | minority skin lesion image 합성으로 imbalance 완화 | 연구별 skin lesion dataset; minority synthetic image 생성 | GAN/diffusion synthetic augmentation | 보통 없음 | CNN classifier | 해당 없음 | accuracy, AUC 등 | dataset별 개선 보고 | train-only 조건에서는 train positive만으로 생성해야 함 |

---

## 2.1 모델 구조 수식 공통 notation

이 문서의 모델 구조 수식은 원문 equation을 그대로 옮긴 것이 아니라, 각 논문의 figure, method 설명, 공개 코드 구조를 바탕으로 이해를 돕기 위해 정리한 구조적 표현이다. 원문에 명시된 수식이 아닌 경우에는 논문별로 별도 표시했다.

| 기호 | 의미 |
|---|---|
| `I` | lesion image input |
| `m` | patient 또는 lesion-level metadata vector |
| `z_p` | patient-context feature |
| `h_img` | image encoder가 만든 image embedding |
| `h_meta` | metadata encoder가 만든 metadata embedding |
| `y_hat` | predicted probability 또는 class score |

공통적으로 multimodal skin lesion classifier는 다음처럼 요약할 수 있다.

$$
\begin{aligned}
h_{\text{img}} &= f_{\theta}(I), \\
h_{\text{meta}} &= g_{\phi}(m), \\
h_{\text{fuse}} &= \mathcal{F}(h_{\text{img}}, h_{\text{meta}}, z_p), \\
\hat{y} &= c_{\psi}(h_{\text{fuse}})
\end{aligned}
$$

여기서 식의 각 구성요소는 다음과 같다.

| 구성요소 | 의미 |
|---|---|
| `f_theta` | CNN/ViT 계열 image encoder |
| `g_phi` | MLP/GBDT/tabular encoder |
| `F` | concat, late fusion, metadata modulation, cross-attention 같은 fusion 연산 |
| `c_psi` | 최종 classifier |

---

## 3. 주요 논문별 상세 분석

### 3.1 SLICE-3D Dataset: 400,000 Skin Lesion Image Crops Extracted from 3D TBP

출처: Scientific Data, 2024  
링크: https://www.nature.com/articles/s41597-024-03743-w

#### 주요 Figure

원문 라이선스: CC BY 4.0

**Figure 1. Examples of image types**

ISIC 2024의 tile image가 dermoscopic image보다 morphologic detail이 적고, 3D-TBP에서 추출된 low-resolution clinical crop이라는 점을 보여준다. 논문 dataset section에서 가장 유용하다.

![SLICE-3D Fig 1](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41597-024-03743-w/MediaObjects/41597_2024_3743_Fig1_HTML.png)

**Figure 2. Dataset curation workflow**

strong label, weak label, tile sub-selection, QA 과정을 설명한다. train-only 연구에서 label noise와 weak benign label 문제를 설명할 때 유용하다.

![SLICE-3D Fig 2](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41597-024-03743-w/MediaObjects/41597_2024_3743_Fig2_HTML.png)

1. 논문의 목표 & 핵심 기여

   ISIC 2024 Challenge의 공식 train dataset인 SLICE-3D를 소개한 dataset descriptor이다. 3D-TBP에서 자동 추출한 lesion crop image와 metadata를 공개하여, dermoscopy 중심 기존 데이터셋의 selection bias를 줄이고 primary care 또는 telehealth 환경에 가까운 저해상도/비전문 촬영 이미지 기반 모델 개발을 가능하게 했다.

2. Dataset 정보

   - Dataset: ISIC 2024 train dataset
   - Sample 수: 401,059 lesion tiles
   - Task: malignant/benign binary classification
   - Modality: 3D-TBP lesion image + patient/lesion metadata

3. Imbalanced data 극복방법

   - 불균형 정도: benign 400,666개 vs malignant 393개, 약 1020:1
   - 데이터 조작: 제안 없음
   - 학습 조작: 모델 학습 논문이 아니므로 없음
   - class 조절: binary target 자체를 기술하며 class 수 조절 없음

4. Tabular model

   모델은 없음. 다만 age, sex, anatomical site, lighting modality, lesion size/color/shape 관련 WB360 measurements, patient_id 등이 제공된다.

5. Image model

   모델은 없음. image는 15mm x 15mm lesion tile이며 평균 크기는 약 133px x 133px이다.

6. Fusion 방식

   해당 없음.

7. 모델 구조 수식

   원문에 학습 모델 구조 수식은 없으므로, 아래는 SLICE-3D sample 구성을 이해하기 위한 표기이다.

   $$
   x_i = (I_i, m_i, p_i), \quad y_i \in \{0, 1\}
   $$

   - `I_i`: 3D-TBP에서 추출된 lesion tile image
   - `m_i`: age, sex, anatomical site, WB360 measurement 등 metadata
   - `p_i`: patient identifier 또는 patient-level grouping 정보
   - `y_i`: malignant 여부

   따라서 이 논문은 위 식처럼 sample 구성을 정의한 dataset descriptor이며, 후속 연구가 사용할 multimodal input space를 제공한 것으로 해석해야 한다.

8. 평가 지표

   - 공식 classification metric: 없음
   - 참고: 후속 ISIC 2024 Challenge에서는 high sensitivity 영역의 `pAUC > 80% TPR`가 핵심 평가 지표로 사용됨

9. 평가(성능)

   - 모델 성능 결과: 해당 없음
   - 주요 정량값: benign 400,666개, malignant 393개

10. 최종결과

   ISIC 2024 train-only 연구의 dataset section에서 반드시 인용해야 할 1차 자료이다.

11. 추가 논의/생각해볼 점

   - benign label에 weak label이 포함되므로 label noise를 고려해야 한다.
   - patient-level clustering이 존재하므로 train-only 실험에서는 patient-level split과 leakage 방지가 중요하다.
   - image crop만으로는 정보가 제한적이어서 metadata와 patient-context feature의 필요성이 크다.

---

### 3.2 Automated Triage of Cancer-Suspicious Skin Lesions with 3D Total-Body Photography

출처: npj Digital Medicine, 2025  
링크: https://www.nature.com/articles/s41746-025-02070-7

#### 주요 Figure

원문 라이선스: CC BY 4.0

**Figure 1. Public/private leaderboard score distribution**

ISIC 2024가 positive가 매우 적고 leaderboard shake-up이 큰 문제였음을 보여준다. 데이터 불균형과 validation instability를 설명할 때 좋다.

![ISIC 2024 Automated Triage Fig 1](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41746-025-02070-7/MediaObjects/41746_2025_2070_Fig1_HTML.png)

**Figure 2. Lesion risk scores stratified by patient**

patient-context, ugly duckling sign, 환자 내 outlier lesion 개념을 설명하는 데 가장 중요한 그림이다.

![ISIC 2024 Automated Triage Fig 2](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41746-025-02070-7/MediaObjects/41746_2025_2070_Fig2_HTML.png)

**Figure 3. Association of lesion characteristics and ML-modelled risk**

WB360 metadata 중 어떤 feature가 모델 risk score와 관련되는지 보여준다. tabular feature engineering의 근거로 사용하기 좋다.

![ISIC 2024 Automated Triage Fig 3](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41746-025-02070-7/MediaObjects/41746_2025_2070_Fig3_HTML.png)

**Figure 4. Winning model diagram**

image-only model output과 metadata/patient-context feature를 boosting model로 결합한 late fusion 구조를 보여준다. ISIC 2024 train-only 연구의 강한 baseline으로 삼기 좋다.

![ISIC 2024 Automated Triage Fig 4](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41746-025-02070-7/MediaObjects/41746_2025_2070_Fig4_HTML.png)

1. 논문의 목표 & 핵심 기여

   ISIC 2024 Kaggle competition 결과를 분석하고, winning model의 ablation study를 통해 image, basic metadata, WB360 appearance metadata, patient-context feature가 성능에 미치는 영향을 비교했다.

2. Dataset 정보

   - Dataset: ISIC 2024 Challenge dataset
   - Task: malignant/benign binary classification
   - Modality: 3D-TBP lesion tile + metadata + patient-context feature

3. Imbalanced data 극복방법

   - class 조절: binary target 유지, class 수 축소/재정의 없음
   - 데이터 조작: 핵심 방법으로 oversampling/undersampling을 제안하지 않음
   - 학습/모델 조작: patient-context feature, metadata feature, GBDT late fusion 활용
   - 평가 기반 대응: `pAUC > 80% TPR`로 high-sensitivity 영역을 우선 평가
   - 주의: 일부 상위 image model은 external dermoscopy data를 사용했으므로 train-only 연구에서는 분리 해석 필요

4. Tabular model

   metadata branch는 basic demographics, WB360 appearance metadata, interaction terms, patient-context terms를 사용했다. 최종 단계에서는 neural network outputs와 metadata feature를 3개 Gradient Boosting Tree 모델에 입력하고, 그 출력들을 aggregate해 lesion risk estimate를 만들었다.

5. Image model

   image branch는 EVA model 2개와 EdgeNeXt 1개로 구성된 ensemble이다. 일부 image model은 external dermoscopy data도 사용했으나, train-only 연구에서는 외부 데이터 사용 부분을 제외하고 tile-only image branch를 참고하는 것이 적절하다.

6. Fusion 방식

   image model ensemble의 neural network output vector와 metadata/patient-context feature를 결합한 뒤, 3개 GBT 모델에 넣고 GBT output을 aggregate하는 late fusion 구조이다.

7. 모델 구조 수식

   아래 수식은 winning solution의 ablation 설명을 바탕으로 한 이해용 구조 표현이다. 원문은 neural network outputs와 metadata feature를 3개 GBT 모델에 넣고, GBT outputs를 aggregate한다고 설명한다.

   $$
   \begin{aligned}
   s_{k,i}^{(r)} &= f_{\theta_{k,r}}(I_i), \quad k=1,2,3,\; r=1,\dots,5, \\
   \bar{s}_{k,i} &= \operatorname{Agg}_{\text{fold}}\left(s_{k,i}^{(1)},\dots,s_{k,i}^{(5)}\right), \\
   z_{p,i} &= \rho(m_i, \mathcal{P}_i), \\
   u_i &= [\bar{s}_{1,i};\;\bar{s}_{2,i};\;\bar{s}_{3,i};\;m_i;\;z_{p,i}], \\
   r_{g,i} &= G_g(u_i), \quad g=1,2,3, \\
   \hat{y}_i &= \operatorname{Agg}_{\text{GBT}}\left(r_{1,i}, r_{2,i}, r_{3,i}\right)
   \end{aligned}
   $$

   - `k`: EVA 2개와 EdgeNeXt 1개로 구성된 neural network model index
   - `r`: 각 image model의 cross-validation fold index
   - `P_i`: 같은 patient에 속한 병변 집합
   - `rho`: ugly duckling 또는 patient-wise rank/z-score 같은 patient-context feature 생성 함수
   - `G_g`: late fusion에 사용되는 3개 Gradient Boosting Tree model

   핵심은 image score를 하나로 평균낸 뒤 쓰는 것이 아니라, 여러 neural network output과 metadata/patient-context feature를 함께 사용해 GBT ensemble risk estimate를 만든다는 점이다.

8. 평가 지표

   - 우선순위 지표: `pAUC > 80% TPR`. `TPR >= 0.8`인 ROC 구간의 partial AUC이며, score range는 `[0, 0.2]`이다.
   - 보조 지표: AUC.
   - clinical utility 지표: `SE top-15`, `NNT 80% sensitivity`, `NNT 90% sensitivity`.
   - NNT는 해당 sensitivity threshold에서 true positive 1개를 찾기 위해 expert review가 필요한 lesion 수로 해석할 수 있다.

9. 평가(성능)

   - 우선순위 지표: `pAUC 0.1726/0.2`
   - 보조 지표: full AUC `0.9668`
   - clinical utility: `NNT80 51.57`, `NNT90 98.20`
   - ablation: patient-context 제외 시 AUC가 0.967에서 0.956으로 감소
   - 추가 결과: WB360 metadata-only model이 tile-only model보다 좋은 성능을 보임

10. 최종결과

   ISIC 2024에서는 image-only보다 metadata와 patient-context를 결합한 multimodal late fusion이 핵심적이다. train-only 논문에서도 image-only, tabular-only, late fusion, patient-context ablation을 반드시 포함하는 것이 좋다.

11. 추가 논의/생각해볼 점

   - train-only 연구에서는 외부 dermoscopy data를 제외하고도 metadata/patient-context 효과가 유지되는지 확인해야 한다.
   - patient-context feature는 강력하지만, patient-level split을 잘못 잡으면 leakage로 과대평가될 수 있다.

---

### 3.3 Wang et al.: Explainable Multimodal AI for Skin Lesion Risk Prediction via 3D Imaging and Clinical Data

출처: Scientific Reports, 2025  
링크: https://www.nature.com/articles/s41598-025-33536-z  
PDF: `s41598-025-33536-z-1.pdf`

#### 주요 Figure

원문 라이선스: CC BY-NC-ND 4.0. 본 문서에서는 원문 이미지를 수정하지 않고 링크 삽입한다. 최종 논문/배포물에서 상업적 이용 또는 이미지 수정이 필요한 경우 별도 권한 확인이 필요하다.

**Figure 1. Explainable multimodal AI framework**

data preparation, image-only model, clinical model, fusion model, model analysis를 한 흐름으로 보여준다. ISIC 2024 train-only 연구에서 전체 pipeline을 설명할 때 유용하다.

![Wang et al. Fig 1](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-025-33536-z/MediaObjects/41598_2025_33536_Fig1_HTML.png)

**Figure 2. Multimodal fusion workflow**

CNN-derived six-class probability vector와 clinical feature vector를 결합해 XGBoost에 입력하는 late fusion 구조를 보여준다. ISIC 2024에서 image score + tabular feature를 결합하는 직접 선행근거로 중요하다.

![Wang et al. Fig 2](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-025-33536-z/MediaObjects/41598_2025_33536_Fig2_HTML.png)

**Figure 3. Performance and interpretability**

multimodal model의 5-fold ROC, SHAP feature importance, CAM visualization을 함께 보여준다. 성능 개선뿐 아니라 어떤 feature와 image region이 판단에 기여했는지 설명하는 근거로 사용할 수 있다.

![Wang et al. Fig 3](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-025-33536-z/MediaObjects/41598_2025_33536_Fig3_HTML.png)

**Figure 4. ISIC 2024 binary benchmark**

6-class 예측 결과를 benign/malignant로 collapse한 뒤 ISIC 2024 challenge top teams와 pFPR을 비교한다. train-only 연구의 pAUC 또는 high-sensitivity benchmark와 연결하기 좋다.

![Wang et al. Fig 4](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-025-33536-z/MediaObjects/41598_2025_33536_Fig4_HTML.png)

1. 논문의 목표 & 핵심 기여

   ISIC 2024 기반 3D-TBP image와 structured clinical/lesion feature를 결합해 6개 skin lesion type을 분류하고, SHAP, CAM, nomogram으로 모델 판단을 설명하는 explainable multimodal AI framework를 제안했다. 기존 ISIC 2024 competition 분석 논문이 leaderboard와 patient-context 효과를 강조했다면, 이 논문은 multimodal fusion 결과를 clinician-friendly scoring system과 XAI로 설명하는 데 초점이 있다.

2. Dataset 정보

   - Dataset: ISIC 2024 dataset subset
   - Patient 수: 1,075명
   - Feature 수: 41 clinical 및 lesion-specific feature
   - Class setting: 6-class lesion risk prediction
   - Class 구성: invasive melanoma 157, basal cell carcinoma 163, squamous cell carcinoma 73, nevus 443, benign NOS 200, solar/actinic keratosis 39
   - Modality: 3D TBP image + clinical/lesion metadata

3. Imbalanced data 극복방법

   - 불균형 정도: nevus 443개 vs solar/actinic keratosis 39개, 약 11.4:1
   - class 조절: 6-class setting 유지
   - 데이터 조작: nevus를 제외한 모든 lesion type에 targeted augmentation 적용
   - augmentation 종류: random rotation, horizontal flip, color variation
   - 평가 기반 대응: 5-fold cross-validation 및 ISIC 2024 binary benchmark pFPR 비교
   - 주의: class weighting과 oversampling은 고려했지만 최종 핵심 방법은 targeted augmentation으로 설명됨

4. Tabular model

   clinical-only model로 logistic regression, decision tree, random forest, gradient boosting, XGBoost, CatBoost, SVM을 비교했고, XGBoost가 가장 robust한 모델로 사용되었다. 추가로 VIF 기반 multicollinearity screening 후 multinomial logistic regression과 nomogram을 이용해 임상적으로 해석 가능한 scoring system을 구성했다.

5. Image model

   image branch는 HAM10000으로 pretraining한 CNN을 3D-TBP image에 transfer learning한 구조이다. CNN은 3개 convolutional block을 기반으로 하며, 3D-TBP fine-tuning을 위해 Conv2D layer 2개를 추가했다. 이미지는 128 x 128 x 3으로 resize했고, cross-entropy loss, Adam optimizer, learning rate 1e-4, batch size 128, 200 epochs로 학습했다.

6. Fusion 방식

   late fusion 구조이다. CNN이 만든 six-class probability vector를 structured visual feature로 사용하고, clinical feature vector와 concat 및 standardization 후 XGBoost classifier에 입력한다. 이후 SHAP으로 feature contribution을 설명하고, CAM으로 image region contribution을 시각화하며, multinomial logistic regression 기반 nomogram으로 clinical scoring panel을 제공한다.

7. 모델 구조 수식

   아래 수식은 원문 Eq. (1)~(2)를 문서 내 표기와 맞춘 구조 표현이다.

   $$
   \begin{aligned}
   f_{\text{image}} &= CNN_{\theta}(I), \quad f_{\text{image}} \in \mathbb{R}^{6}, \\
   f_{\text{clinical}} &= [x_1, x_2, \ldots, x_n], \\
   u &= \operatorname{Standardize}([f_{\text{image}};\;f_{\text{clinical}}]), \\
   \hat{y} &= XGBoost_{\psi}(u)
   \end{aligned}
   $$

   - `f_image`: CNN이 출력한 6-class probability vector
   - `f_clinical`: age, lesion size, TBP-derived numerical feature 등 clinical/lesion feature vector
   - `u`: image-derived prediction vector와 clinical feature를 결합한 late fusion vector
   - `y_hat`: 최종 6-class lesion prediction

   scoring system에서는 VIF로 redundant feature를 제거한 뒤 multinomial logistic regression을 사용한다.

   $$
   VIF_i = \frac{1}{1 - R_i^2}
   $$

   $$
   \operatorname{logit}(P(Y \le k \mid X))
   =
   \beta_0^{(k)} + \sum_{i=1}^{n}\beta_i^{(k)}X_i
   $$

   이 수식은 final classifier 자체라기보다, model output과 clinical/TBP feature를 clinician-friendly risk score로 변환하기 위한 해석 가능 scoring layer로 보는 것이 적절하다.

8. 평가 지표

   - 우선순위 지표: multiclass AUC, recall, F1
   - ISIC benchmark 지표: 6-class prediction을 benign/malignant로 collapse한 pFPR
   - 보조 지표: accuracy, precision, confusion matrix
   - 설명 가능성 지표/도구: SHAP feature importance, CAM, nomogram, VIF

9. 평가(성능)

   - Clinical-only XGBoost: overall accuracy `0.6837`, recall `0.4090`, F1 `0.4582`
   - Clinical-only class result: BCC `78.6%`, nevus `72.6%`, melanoma invasive `43.8%`, actinic keratosis `12.5%`, SCC `16.7%`
   - 3D-TBP image-only CNN: nevus `87.10%`, benign NOS `75.34%`, invasive melanoma `71.88%`, BCC `54.05%`, SCC `60.32%`, actinic/solar keratosis `65.62%`
   - Multimodal fusion: class별 AUC `> 0.95`, nevus 및 actinic keratosis AUC `0.98`
   - Multimodal fusion: recall 및 F1 score `> 95%`
   - ISIC 2024 binary benchmark: pFPR `0.17343`, top 5 team range `0.17210-0.17264`

10. 최종결과

   이 논문은 ISIC 2024에서 image-derived prediction vector와 clinical/TBP metadata를 결합하는 late fusion이 unimodal model보다 강하다는 근거를 제공한다. 특히 SHAP, CAM, nomogram을 통해 성능 결과를 설명 가능하게 제시하므로, train-only 논문에서 단순 pAUC 개선뿐 아니라 어떤 metadata와 image region이 malignant risk에 기여했는지 설명하는 XAI section의 핵심 선행연구로 쓸 수 있다.

11. 추가 논의/생각해볼 점

   - image branch가 HAM10000 transfer learning을 사용하므로 엄격한 ISIC 2024 train-only 조건에서는 외부 pretraining 사용 여부를 별도 분리해야 한다.
   - 6-class setting을 binary challenge metric으로 collapse하므로, ISIC 2024 원래 binary target과 직접 동일한 실험은 아니다.
   - 논문 자체도 external cohort validation이 없고, augmentation으로 인한 optimistic performance 가능성을 limitation으로 언급한다.
   - 그럼에도 41개 feature, SHAP, CAM, nomogram을 결합한 구조는 train-only 연구의 explainability 설계에 매우 직접적으로 유용하다.

---

### 3.4 MetaBlock: An Attention-Based Mechanism to Combine Images and Metadata

출처: IEEE Journal of Biomedical and Health Informatics, 2021  
링크: https://doi.org/10.1109/JBHI.2021.3062002
코드: https://github.com/paaatcha/MetaBlock

#### 주요 Figure

주의: IEEE 원문 Fig. 2와 Fig. 3 이미지는 직접 삽입하지 않는다. IEEE 논문 figure는 별도 재사용 권한이 필요할 수 있으므로, 본 문서에는 원문 구조를 설명하기 위한 자체 schematic을 넣는다. 원본 figure가 꼭 필요하면 DOI 링크 또는 IEEE Xplore에서 확인하고, 최종 논문/학위논문에는 IEEE RightsLink 권한 확인 후 삽입하는 것이 안전하다.

**자체 Figure 2. MetaBlock 내부 구조**

원문 Fig. 2는 비교 구조가 아니라 MetaBlock의 내부 구조를 보여준다. metadata feature에서 `f_b`와 `g_b` modifier를 만들고, image feature map에 hyperbolic tangent gate와 sigmoid gate를 차례로 적용해 block output을 만든다.

```mermaid
flowchart LR
    XI["Image feature x_img_tilde"] --> TG["T_gate = tanh(f_b(x_meta_tilde) * x_img_tilde)"]
    XM["Metadata feature x_meta_tilde"] --> FB["f_b(x_meta_tilde)"]
    XM --> GB["g_b(x_meta_tilde)"]
    FB --> TG
    TG --> SG["S_gate = sigmoid(T_gate + g_b(x_meta_tilde))"]
    GB --> SG
    SG --> XO["MetaBlock output x_tilde"]
```

**자체 Figure 3. CNN에 삽입된 MetaBlock layer**

원문 Fig. 3은 MetaBlock layer가 CNN model에 attached된 모습을 보여준다. CNN의 last feature maps와 metadata feature가 MetaBlock으로 들어가고, MetaBlock output이 classification layer로 전달된다.

```mermaid
flowchart LR
    I["Input image x_img"] --> CNN["CNN feature extractor phi_img"]
    CNN --> XI["Feature maps x_img_tilde"]
    M["Metadata x_meta"] --> PHI["Metadata feature extractor phi_meta"]
    PHI --> XM["Metadata feature x_meta_tilde"]
    XI --> MB["MetaBlock"]
    XM --> MB
    MB --> XO["Output feature x_tilde"]
    XO --> CLF["Classification layer"]
    CLF --> Y["p(y = c | x_img_tilde, x_meta_tilde)"]
```

핵심 해석:

- 원문은 image와 metadata를 각각 `x_img`, `x_meta`로 두고, feature extractor를 거친 값을 `x_img_tilde`, `x_meta_tilde`로 설명한다.
- `f_b(x_meta_tilde)`와 `g_b(x_meta_tilde)`는 metadata feature에서 생성되는 modifier coefficient이다.
- Eq. (3)의 MetaBlock output은 `sigmoid(tanh(f_b(x_meta_tilde) * x_img_tilde) + g_b(x_meta_tilde))` 형태이다.
- 이 output은 class probability가 아니라 classification layer로 전달되는 중간 feature map이다.

ISIC 2024 적용 예:

```text
image tile -> ConvNeXt/EfficientNet feature
metadata + WB360 + patient-context -> scale/shift modifier
MetaBlock -> metadata-aware image feature
classifier -> malignant probability
```

1. 논문의 목표 & 핵심 기여

   skin lesion classification에서 image feature와 patient metadata를 단순 concat하지 않고, metadata feature가 CNN의 last feature maps를 scale/shift/gating하도록 하는 Metadata Processing Block(MetaBlock)을 제안했다.

2. Dataset 정보

   - Dataset 1: PAD-UFES-20
   - PAD-UFES-20 구성: 6-class clinical image + metadata
   - Dataset 2: ISIC 2019
   - ISIC 2019 구성: 8-class dermoscopy image + metadata

3. Imbalanced data 극복방법

   - 불균형 정도: PAD-UFES-20은 MEL 52개 vs BCC 845개, 약 16:1
   - 불균형 정도: ISIC 2019는 DF 239개 vs NV 12,875개, 약 54:1
   - class 조절: class 수 조절 없음
   - 데이터 조작: common image augmentation 사용, 5-fold CV는 label frequency 기준 stratified
   - 학습 조작: class frequency 기반 weighted cross-entropy 사용

4. Tabular model

   metadata는 feature extractor `phi_meta`를 거쳐 `x_meta_tilde`가 되고, MetaBlock 내부의 `f_b`와 `g_b` single-layer neural network가 image feature map을 조절하는 modifier coefficient를 만든다.

5. Image model

   EfficientNet-B4, DenseNet-121, MobileNet-v2, ResNet-50, VGG-13 등 CNN backbone의 last feature map layer를 image feature로 사용했다.

6. Fusion 방식

   intermediate fusion이다. metadata가 CNN feature extraction 과정 중간에 개입해 중요한 visual feature를 강화한다.

7. 모델 구조 수식

   아래 수식은 원문 Eq. (1)~(7)을 문서용으로 정리한 표현이다. 원문은 image와 metadata를 `x_img`, `x_meta`로 두고, feature extractor를 거친 값을 각각 `x_img_tilde`, `x_meta_tilde`로 설명한다.

   $$
   \begin{aligned}
   \tilde{x}_{\text{img}} &= \phi_{\text{img}}(x_{\text{img}}), \\
   \tilde{x}_{\text{meta}} &= \phi_{\text{meta}}(x_{\text{meta}}), \\
   f_b(\tilde{x}_{\text{meta}}) &= W_f^{T}\tilde{x}_{\text{meta}} + w_f^0, \\
   g_b(\tilde{x}_{\text{meta}}) &= W_g^{T}\tilde{x}_{\text{meta}} + w_g^0, \\
   T_{\text{gate}} &= \tanh\left(f_b(\tilde{x}_{\text{meta}}) \odot \tilde{x}_{\text{img}}\right), \\
   S_{\text{gate}} &= \sigma\left(T_{\text{gate}} + g_b(\tilde{x}_{\text{meta}})\right), \\
   \tilde{x} &= S_{\text{gate}}, \\
   \hat{y} &= \operatorname{classifier}(\tilde{x})
   \end{aligned}
   $$

   - `x_img_tilde`: CNN에서 추출한 last feature maps
   - `x_meta_tilde`: metadata feature extractor가 만든 metadata feature
   - `f_b`, `g_b`: metadata feature에서 modifier coefficient를 만드는 single-layer neural network
   - `T_gate`: hyperbolic tangent gate
   - `S_gate`: sigmoid gate이자 MetaBlock output feature
   - `x_tilde`: `x_img_tilde`와 같은 shape를 유지하는 MetaBlock output

   단순 concat baseline은 아래처럼 image embedding과 metadata embedding을 이어 붙이는 형태이다.

   $$
   h_{\text{concat}} = [h_{\text{img}};\;h_{\text{meta}}]
   $$

   이에 비해 MetaBlock은 metadata feature `x_meta_tilde`가 CNN feature map `x_img_tilde`의 변환에 직접 들어간다는 점에서 intermediate fusion에 가깝다.

8. 평가 지표

   - 우선순위 지표: balanced accuracy(BACC)
   - 의미: multiclass BACC는 class별 recall의 평균

   ```text
   BACC = (recall_1 + recall_2 + ... + recall_K) / K
   ```

   - binary case: `BACC = (Sensitivity + Specificity) / 2`

9. 평가(성능)

   - 원 논문 비교: CNN-only, concatenation, MetaNet과 비교했으며 10개 실험 중 6개에서 가장 좋은 BACC 보고
   - ISIC 2019: MetaBlock은 5개 CNN 중 3개에서 최고 BACC를 보였고, Friedman/Wilcoxon test에서도 차이를 보고
   - PAD-UFES-20: MetaBlock은 5개 CNN 중 3개에서 최고 BACC를 보였고, 평균적으로 metadata를 쓰지 않는 baseline보다 높은 성능을 보임
   - 참고: 기존 문서에 있던 `BACC 0.77 ± 0.02`, `BACC 0.77 ± 0.01` 같은 단일 대표값은 원문 Table 전체를 단순화한 값이라 본문에서는 제거

10. 최종결과

   ISIC 2024 train-only 논문에서 "simple concat vs metadata modulation" 비교 실험의 근거로 적합하다.

11. 추가 논의/생각해볼 점

   - MetaBlock은 불균형 자체를 해결하는 논문이라기보다 metadata-conditioned fusion 논문이다.
   - ISIC 2024처럼 1020:1 수준의 binary imbalance에서는 MetaBlock만으로는 부족하다.
   - ISIC 2024 적용 시 balanced sampler, focal/class-balanced loss, pAUC/AUPRC 평가를 함께 설계해야 한다.

---

### 3.5 MMF-Net: Deep Learning Based Multimodal Fusion Using Smartphone Images and Metadata

출처: Frontiers in Surgery, 2022  
링크: https://www.frontiersin.org/articles/10.3389/fsurg.2022.1029991/full

#### 주요 Figure

원문 라이선스: CC BY 4.0

**Figure 1. Overall network architecture**

image encoder, meta encoder, multimodal fusion module, classifier의 전체 흐름을 보여준다. multimodal fusion model architecture section에 유용하다.

![MMF-Net Fig 1](https://www.frontiersin.org/files/Articles/1029991/fsurg-09-1029991-HTML/image_m/fsurg-09-1029991-g001.jpg)

**Figure 2. Multimodal fusion module**

self-attention과 cross-attention을 이용해 image feature와 metadata feature를 양방향으로 결합하는 구조를 보여준다. MetaBlock보다 적극적인 attention fusion을 설명할 때 중요하다.

![MMF-Net Fig 2](https://www.frontiersin.org/files/Articles/1029991/fsurg-09-1029991-HTML/image_m/fsurg-09-1029991-g002.jpg)

**Figure 3. ROC curves**

class별 AUC 성능을 보여준다.

![MMF-Net Fig 3](https://www.frontiersin.org/files/Articles/1029991/fsurg-09-1029991-HTML/image_m/fsurg-09-1029991-g003.jpg)

**Figure 4. Confusion matrix**

multiclass skin lesion classification에서 어떤 class가 혼동되는지 보여준다.

![MMF-Net Fig 4](https://www.frontiersin.org/files/Articles/1029991/fsurg-09-1029991-HTML/image_m/fsurg-09-1029991-g004.jpg)

1. 논문의 목표 & 핵심 기여

   smartphone으로 수집된 clinical image와 metadata를 결합하여 skin lesion type을 분류하는 multimodal fusion network를 제안했다.

2. Dataset 정보

   - Dataset: PAD-UFES-20
   - Class setting: 6-class skin lesion classification
   - Modality: smartphone clinical image + metadata
   - Metadata: numeric/categorical metadata 제공

3. Imbalanced data 극복방법

   - class 조절: 6-class 설정 유지
   - 데이터 조작: on-the-fly image augmentation 사용
   - split: stratified 5-fold cross-validation
   - sampling: 명시적 oversampling/undersampling 없음
   - 학습 조작: weighted loss는 핵심 방법으로 보고하지 않음
   - 평가 기반 대응: BACC와 aggregated AUC 사용

4. Tabular model

   numeric feature는 그대로 사용하고, categorical feature는 one-hot encoding 후 MLP encoder를 통과시켰다.

5. Image model

   ResNet-50을 image encoder로 사용했다.

6. Fusion 방식

   intra-modality self-attention으로 각 modality 내부의 중요 feature를 강화하고, inter-modality cross-attention으로 image feature와 metadata feature가 서로를 guide하도록 했다.

7. 모델 구조 수식

   아래 수식은 MMF-Net의 self-attention과 cross-attention fusion을 이해하기 위한 구조적 표현이다. 원문 설명에 맞춰 첫 cross-attention path는 Query/Value를 image feature에서, Key를 metadata feature에서 만들고, 두 번째 path는 Query/Value를 metadata feature에서, Key를 image feature에서 만든다.

   $$
   \operatorname{Attn}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^{\top}}{\sqrt{d}}\right)V
   $$

   $$
   \begin{aligned}
   x_{\text{img}} &= f_{\theta}(I), \\
   x_{\text{meta}} &= g_{\phi}(m), \\
   x_{\text{img}}' &= \operatorname{SelfAttn}(x_{\text{img}}), \\
   x_{\text{meta}}' &= \operatorname{SelfAttn}(x_{\text{meta}}), \\
   x_{\text{img}}'' &= \operatorname{Attn}(W_Q^i x_{\text{img}}', W_K^m x_{\text{meta}}', W_V^i x_{\text{img}}'), \\
   x_{\text{meta}}'' &= \operatorname{Attn}(W_Q^m x_{\text{meta}}', W_K^i x_{\text{img}}', W_V^m x_{\text{meta}}'), \\
   x_{\text{final}} &= [x_{\text{img}}'';\;x_{\text{meta}}''], \\
   \hat{y} &= \operatorname{softmax}(W_o x_{\text{final}} + b_o)
   \end{aligned}
   $$

   - self-attention은 각 modality 내부의 irrelevant information을 낮추는 단계이다.
   - 첫 cross-attention path는 metadata가 image feature selection을 guide한다.
   - 둘째 cross-attention path는 image feature가 metadata feature selection을 guide한다.
   - ISIC 2024에 적용하면 metadata feature에는 WB360 및 patient-context feature를 포함할 수 있다.

8. 평가 지표

   - 우선순위 지표: BACC, aggregated AUC
   - 보조 지표: ACC
   - BACC: 원문에서는 sensitivity와 specificity의 산술평균으로 설명
   - aggregated AUC: 6-class 문제에서 class pair별 AUC를 평균한 값

9. 평가(성능)

   - BACC: `0.775 ± 0.022`
   - aggregated AUC: `0.947 ± 0.007`
   - ACC: `0.768 ± 0.022`

10. 최종결과

   metadata 포함이 image-only보다 성능을 유의미하게 개선했다. ISIC 2024에서 cross-attention fusion을 제안할 때 직접적인 선행연구로 사용할 수 있다.

11. 추가 논의/생각해볼 점

   - cross-attention은 fusion 기여를 설명하기 좋다.
   - positive가 극도로 적은 ISIC 2024에서는 end-to-end 학습이 불안정할 수 있다.
   - train-only 조건에서는 late fusion baseline과 비교해 cross-attention의 실제 pAUC 개선폭을 검증해야 한다.

---

### 3.6 Yap et al.: Multimodal Skin Lesion Classification Using Deep Learning

출처: Experimental Dermatology, 2018  
링크: https://doi.org/10.1111/exd.13777

#### 주요 Figure

주의: Wiley 원문 figure는 직접 삽입하지 않고, multimodal concept schematic으로 대체한다.

**자체 Figure. Multiple image modalities + metadata fusion**

```mermaid
flowchart LR
    A["Macroscopic / clinical image"] --> B["CNN image encoder"]
    C["Dermoscopic image"] --> D["CNN image encoder"]
    E["Patient metadata"] --> F["Metadata branch"]

    B --> G["Feature fusion"]
    D --> G
    F --> G
    G --> H["Classifier"]
    H --> I["Binary melanoma / multiclass lesion prediction"]
```

핵심 해석:

- 단일 macroscopic image보다 dermoscopy + clinical image + metadata 조합이 더 좋은 성능을 보였다.
- ISIC 2024는 dermoscopy가 없지만, "여러 source의 정보를 결합하면 image-only보다 성능이 좋아진다"는 선행근거로 사용할 수 있다.

1. 논문의 목표 & 핵심 기여

   하나의 macroscopic image만 사용하는 기존 방식에서 벗어나, dermoscopic image, clinical/macroscopic image, patient metadata를 결합한 multimodal classifier를 제안했다.

2. Dataset 정보

   - Dataset: 자체 dataset
   - Sample 수: 2917 cases
   - Modality: dermoscopic image + macroscopic image + patient metadata
   - Task: binary melanoma detection, 5-class classification

3. Imbalanced data 극복방법

   - class 조절: binary task와 5-class task를 별도 평가
   - 해석: task 목적에 따른 label setting이며, 하나의 binary target에서 class 수를 임의 축소한 것은 아님
   - 데이터 조작: imbalance-specific sampling은 핵심 기여로 보고되지 않음
   - 학습 조작: imbalance-specific loss는 핵심 기여로 보고되지 않음
   - 평가 기반 대응: AUC, mAP 중심 평가

4. Tabular model

   patient metadata를 multimodal classifier에 포함했다.

5. Image model

   CNN 기반 image model을 사용했다.

6. Fusion 방식

   multiple image modalities와 patient metadata를 결합하는 multimodal classifier 구조이다.

7. 모델 구조 수식

   아래 수식은 dermoscopic image, macroscopic image, patient metadata를 결합하는 multimodal classifier를 이해하기 위한 구조적 표현이다.

   $$
   \begin{aligned}
   h_{\text{derm}} &= f_{\theta_d}(I_{\text{derm}}), \\
   h_{\text{macro}} &= f_{\theta_c}(I_{\text{macro}}), \\
   h_{\text{meta}} &= g_{\phi}(m), \\
   h_{\text{fuse}} &= [h_{\text{derm}};\;h_{\text{macro}};\;h_{\text{meta}}], \\
   \hat{y}_{\text{bin}} &= \sigma(w_b^{\top}h_{\text{fuse}} + b_b), \\
   \hat{\mathbf{y}}_{\text{multi}} &= \operatorname{softmax}(W_m h_{\text{fuse}} + b_m)
   \end{aligned}
   $$

   - `I_derm`: dermoscopic image
   - `I_macro`: macroscopic 또는 clinical image
   - `y_hat_bin`: binary melanoma prediction
   - `y_hat_multi`: multiclass lesion type prediction

   ISIC 2024는 dermoscopy input이 없으므로 직접 적용식은 아래처럼 tile image와 metadata를 결합하는 형태로 단순화된다.

   $$
   h_{\text{fuse}} = [h_{\text{tile}};\;h_{\text{meta}}]
   $$

8. 평가 지표

   - 우선순위 지표: binary melanoma detection의 AUC
   - 우선순위 지표: multiclass classification의 mAP
   - mAP: class별 precision-recall curve의 average precision을 계산한 뒤 평균한 값

9. 평가(성능)

   - Binary melanoma detection: multimodal classifier `AUC 0.866`
   - Binary baseline: single macroscopic image `AUC 0.784`
   - Multiclass classification: multimodal classifier `mAP 0.729`
   - Multiclass baseline: `mAP 0.598`

10. 최종결과

   피부 병변 진단에서 image + metadata 결합이 image-only보다 우수하다는 초기 근거 논문이다.

11. 추가 논의/생각해볼 점

   - 여러 image modality가 있는 환경의 연구라 ISIC 2024의 single tile image 상황과는 입력 조건이 다르다.
   - metadata가 image-only 성능을 보완한다는 방향성은 ISIC 2024 train-only 연구의 근거로 사용할 수 있다.

---

### 3.7 Islam et al.: Fusion of Patient Metadata and Skin Lesion Images

출처: Scientific Reports, 2026  
링크: https://www.nature.com/articles/s41598-025-26392-4

#### 주요 Figure

원문 라이선스: CC BY 4.0

**Figure 6. Proposed AI framework**

metadata와 image를 함께 사용해 suspicious vs non-suspicious skin lesion을 분류하는 전체 framework를 보여준다.

![Islam et al. Fig 6](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-025-26392-4/MediaObjects/41598_2025_26392_Fig6_HTML.png)

**Figure 8. Fusion of metadata with image data**

image feature와 metadata feature를 결합하는 fusion 구조를 보여준다.

![Islam et al. Fig 8](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-025-26392-4/MediaObjects/41598_2025_26392_Fig8_HTML.png)

**Figure 11. Performance of image + metadata fusion**

metadata fusion이 specificity를 개선하는 결과를 보여준다. ISIC 2024에서 high sensitivity를 유지하면서 false positive를 줄이는 목표와 연결하기 좋다.

![Islam et al. Fig 11](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-025-26392-4/MediaObjects/41598_2025_26392_Fig11_HTML.png)

1. 논문의 목표 & 핵심 기여

   teledermatology triage에서 suspicious vs non-suspicious lesion을 분류하기 위해 patient metadata와 dermoscopic/DSLR image를 결합한 AI framework를 제안했다.

2. Dataset 정보

   - Dataset: UK private skin cancer diagnostic centres dataset
   - Patient 수: 19,295명
   - Lesion 수: 39,623개
   - Image 수: 79,246개
   - Task: suspicious/non-suspicious binary triage
   - Modality: DER/DSLR image + 22 meta-features

3. Imbalanced data 극복방법

   - 불균형 정도: suspicious 11,258개 vs non-suspicious 67,988개
   - class 조절: class 축소라기보다 clinical triage 목적의 binary label 정의
   - split: 80/20 patient-separated split
   - 데이터 조작: image preprocessing/augmentation
   - 모델 조작: model decision majority voting
   - 평가 기반 대응: sensitivity와 specificity 중심 평가

4. Tabular model

   전체 dataset에는 lesion별 22개 meta-feature가 수집되었지만, multimodal fusion 설명에서는 7개 C4C risk factor와 overall C4C risk score, 총 8개 meta-feature를 image output과 결합하는 것으로 설명한다.

5. Image model

   EfficientNet-B2 기반 image model을 사용했다.

6. Fusion 방식

   image vector와 8개 metadata feature를 concat한 뒤 dropout/linear layer를 거쳐 suspicious 여부를 분류했다. 별도로 DER, SLR, DER+metadata, SLR+metadata, DER+SLR, DER+SLR+metadata 등 6개 EfficientNet-B2 계열 model의 outcome decision을 majority voting으로 결합했다.

7. 모델 구조 수식

   아래 수식은 원문 Fig. 8의 feature-level metadata fusion과 Fig. 9의 decision-level majority voting을 분리해 정리한 이해용 표현이다.

   $$
   \begin{aligned}
   h_{\text{img},i} &= f_{\theta}(I_i), \\
   h_{\text{meta},i} &= g_{\phi}(m_i^{\text{C4C}}), \\
   h_{\text{fuse},i} &= [h_{\text{img},i};\;h_{\text{meta},i}], \\
   \hat{y}_{i}^{\text{fuse}} &= \sigma(c_{\psi}(\operatorname{Dropout}(h_{\text{fuse},i})))
   \end{aligned}
   $$

   $$
   \begin{aligned}
   d_i^{(r)} &= \mathbb{1}(\hat{y}_i^{(r)} \ge \tau_r), \quad r=1,\dots,6, \\
   \hat{d}_i &= \operatorname{majority}\left(d_i^{(1)},\dots,d_i^{(6)}\right)
   \end{aligned}
   $$

   - `m_i_C4C`: 7개 C4C risk factor와 C4C risk score로 구성된 metadata vector
   - `y_hat_fuse`: image와 metadata를 feature-level concat한 단일 fused model output
   - `d_i_r`: DER, SLR, DER+Meta 등 입력 조합이 다른 개별 model의 binary decision
   - `d_hat_i`: majority voting 이후 최종 suspicious/non-suspicious decision

8. 평가 지표

   - 우선순위 지표: sensitivity(SEN), specificity(SPC)
   - SEN: suspicious/malignant 계열을 놓치지 않는 정도
   - SPC: non-suspicious/benign을 불필요하게 의심하지 않는 정도
   - ACC: 논문 수식상 `ACC = (SEN + SPC) / 2`, binary balanced accuracy에 가까움

9. 평가(성능)

   - Image + metadata fused model: `SEN 99.66 ± 0.28%`, `SPC 74.45 ± 0.80%`
   - Majority voting: `SEN 99.50 ± 1.18%`, `SPC 82.72 ± 1.64%`
   - Metadata-only model: `SEN 85.24 ± 2.20%`, `SPC 61.12 ± 0.90%`
   - 핵심 효과: majority voting에서 sensitivity를 유지하면서 specificity 개선

10. 최종결과

   metadata fusion은 sensitivity를 유지하면서 specificity를 개선하는 데 유리하다. ISIC 2024에서도 high sensitivity 조건에서 false positive를 줄이는 fusion 목표와 잘 맞는다.

11. 추가 논의/생각해볼 점

   - specificity 개선은 ISIC 2024의 high-sensitivity pAUC 목표와 잘 연결된다.
   - private clinic 기반 데이터라 population bias가 있을 수 있다.
   - hair removal과 resizing 같은 preprocessing이 lesion shape를 왜곡할 가능성을 저자도 논의한다.

---

### 3.8 Nguyen et al.: Skin Lesion Classification on Imbalanced Data Using Soft Attention

출처: Sensors, 2022  
링크: https://www.mdpi.com/1424-8220/22/19/7530

#### 주요 Figure

원문 라이선스: CC BY 4.0

**Figure 2. Overall model architecture**

image input, metadata branch, soft-attention layer, classifier가 결합되는 전체 구조를 보여준다.

![Nguyen et al. Fig 2](https://www.mdpi.com/sensors/sensors-22-07530/article_deploy/html/images/sensors-22-07530-g002.png)

**Figure 6. Soft-Attention layer**

soft-attention이 feature map에서 중요한 영역을 강화하는 방식을 설명한다.

![Nguyen et al. Fig 6](https://www.mdpi.com/sensors/sensors-22-07530/article_deploy/html/images/sensors-22-07530-g006.png)

**Figure 7. Soft-Attention module**

soft-attention module의 세부 구조를 보여준다.

![Nguyen et al. Fig 7](https://www.mdpi.com/sensors/sensors-22-07530/article_deploy/html/images/sensors-22-07530-g007.png)

**Figure 9. AUC curves**

imbalance-aware loss와 soft-attention 기반 모델의 class별 ROC/AUC 결과를 보여준다.

![Nguyen et al. Fig 9](https://www.mdpi.com/sensors/sensors-22-07530/article_deploy/html/images/sensors-22-07530-g009.png)

1. 논문의 목표 & 핵심 기여

   imbalanced skin lesion classification에서 soft-attention과 imbalance-aware loss function을 결합한 deep learning model을 제안했다.

2. Dataset 정보

   - Dataset: HAM10000
   - Image 수: 10,015개 dermoscopy images
   - Class 수: 7 classes
   - Metadata: age, gender 등 patient information

3. Imbalanced data 극복방법

   - 불균형 정도: NV 6705개 vs DF 115개, 약 58:1
   - class 조절: class 수 조절 없음
   - 데이터 조작: augmentation으로 training image를 53,573개까지 확장
   - 학습 조작: weighted/new loss function 사용
   - 모델 조작: soft-attention으로 lesion 중심 feature 강화

4. Tabular model

   age, gender 같은 personal information을 추가로 사용했다.

5. Image model

   InceptionResNetV2, MobileNetV3Large 등 다양한 backbone을 사용했다.

6. Fusion 방식

   image feature 중심 구조에 personal information을 함께 사용했다.

7. 모델 구조 수식

   아래 수식은 원문 Soft-Attention 설명과 metadata branch를 함께 정리한 이해용 구조 표현이다. 원문은 attention maps를 만든 뒤 이를 aggregate한 weight function을 feature tensor에 곱하고, 원래 feature tensor와 scaled attention feature를 concatenate한다고 설명한다.

   $$
   \begin{aligned}
   T &= f_{\theta}(I), \\
   A_{1:K} &= \operatorname{softmax}(\operatorname{Conv3D}(T)), \\
   A &= \operatorname{Aggregate}(A_{1},\dots,A_{K}), \\
   T_{\text{att}} &= \gamma (T \odot A), \\
   h_{\text{SA}} &= \operatorname{Pool}([T;\;T_{\text{att}}]), \\
   h_{\text{meta}} &= g_{\phi}(m_{\text{age}}, m_{\text{sex}}, m_{\text{localization}}), \\
   h_{\text{fuse}} &= [h_{\text{SA}};\;h_{\text{meta}}], \\
   \hat{\mathbf{y}} &= \operatorname{softmax}(W_c h_{\text{fuse}} + b_c)
   \end{aligned}
   $$

   - `T`: backbone이 만든 feature tensor
   - `A_1:K`: softmax로 정규화된 K개 attention map
   - `gamma`: scaled attention feature에 곱해지는 learnable scalar
   - `h_SA`: original feature와 scaled attention feature를 concat한 뒤 pooling한 image feature
   - `h_meta`: age, sex, localization metadata branch output

   구조적으로는 class imbalance를 weighted loss만으로 다루는 것이 아니라, Soft-Attention으로 visual feature를 재가중한 뒤 metadata dense branch와 결합한다는 점이 중요하다.

8. 평가 지표

   - 우선순위 지표: AUC, recall, F1
   - 보조 지표: accuracy, precision
   - recall: minority class를 놓치지 않는 정도
   - F1: precision과 recall의 조화평균

9. 평가(성능)

   - Abstract 요약: InceptionResNetV2 + Soft-Attention + new loss에서 `ACC 0.90`, `Precision 0.81`, `F1 0.81`, `Recall 0.82`, `AUC 0.99`
   - 본문 conclusion: best model의 핵심 균형 지표로 `F1 0.86`, `Recall 0.81`, `AUC 0.975`를 제시
   - Appendix detailed table: InceptionResNetV2 with Metadata and WeightLoss의 mean `F1 0.81`, mean `Recall 0.81`; MobileNetV3Large는 `ACC 0.86`, `F1 0.79`, `Sensitivity 0.80`, `AUC 0.96`로 더 가벼운 대안으로 제시됨

10. 최종결과

   ISIC 2024 image branch에서 focal loss, class-balanced loss, attention block을 실험할 근거로 활용 가능하다.

11. 추가 논의/생각해볼 점

   - augmentation과 weighted loss가 함께 들어가 있어 성능 향상의 원인을 분리해 봐야 한다.
   - loss 때문인지 attention 때문인지 ablation을 세밀히 확인해야 한다.
   - ISIC 2024에서는 binary pAUC/AUPRC 기준으로 같은 전략이 유지되는지 별도 검증이 필요하다.

---

## 4. Dataset 불균형 극복 방법 정리

ISIC 2024 train-only 조건에서 사용할 수 있는 방법은 다음과 같다.

### 4.1 Sampling 기반

- Positive oversampling
- Negative undersampling
- Balanced batch sampler
- patient-level split을 유지한 stratified sampling

장점:

- 구현이 쉽고 image branch 학습에 바로 적용 가능하다.

주의점:

- positive가 393개뿐이므로 단순 oversampling은 overfitting 위험이 크다.
- 동일 patient가 train/validation에 섞이면 leakage가 발생할 수 있다.

### 4.2 Loss 기반

- Weighted BCE
- Focal Loss
- Class-Balanced Loss
- Class-Balanced Focal Loss
- Asymmetric Loss

ISIC 2024에서는 easy negative가 압도적으로 많으므로 focal 계열 loss가 적합하다. 다만 positive가 매우 적어 loss weight를 과도하게 키우면 calibration이 무너질 수 있으므로 validation pAUC와 AUPRC를 함께 확인해야 한다.

#### 참고 Figure. Loss 기반 imbalance 대응 구조

Focal Loss와 Class-Balanced Loss 논문은 skin lesion 전용은 아니지만, ISIC 2024의 extreme imbalance를 다룰 때 loss 설계 근거로 중요하다.

```mermaid
flowchart LR
    A["Extremely imbalanced batch"] --> B["Image encoder"]
    B --> C["Logit"]
    C --> D["BCE / Weighted BCE"]
    C --> E["Focal Loss\nCE * (1 - p_t)^gamma"]
    C --> F["Class-Balanced Loss\nweight by effective number"]
    D --> G["Compare validation pAUC / AUPRC"]
    E --> G
    F --> G
```

논문 작성 시 권장 사용:

- Focal Loss: easy negative가 압도적으로 많은 상황에서 negative loss contribution을 낮추는 근거
- Class-Balanced Loss: sample 수가 적은 malignant class에 effective number 기반 weight를 주는 근거
- ISIC 2024에서는 accuracy가 아니라 pAUC, AUPRC, sensitivity-specificity trade-off로 비교해야 한다.

### 4.3 Feature engineering 기반

ISIC 2024에서 특히 중요한 방법이다.

- patient별 lesion count
- patient별 feature mean/std
- lesion size/color feature의 patient-wise z-score
- anatomical site별 rank/percentile
- ugly duckling feature: 같은 환자 내에서 얼마나 outlier인지 측정
- WB360 feature interaction
- explainable feature importance: SHAP으로 확인한 `visual_classifier`, `tbp_lv_symm_2axis`, `tbp_lv_color_std_mean` 등

선행연구상 ISIC 2024에서는 image tile보다 WB360 metadata와 patient-context feature가 더 강한 정보를 제공할 수 있다.

### 4.4 Synthetic augmentation

- GAN 또는 diffusion으로 malignant image 생성
- train positive만 사용한 lesion-preserving augmentation

주의점:

- 사용자가 "train dataset만 사용"한다고 했으므로 external data나 pretrained generative dataset을 사용하면 연구 조건에서 벗어날 수 있다.
- synthetic image는 artifact가 생길 수 있어 final model보다 ablation 또는 보조 실험으로 사용하는 것이 안전하다.

---

## 5. Multimodal Fusion 방식 정리

### 5.1 Late Fusion

구조:

1. image model을 학습한다.
2. train OOF prediction 또는 image embedding을 만든다.
3. tabular metadata와 image prediction을 합쳐 GBDT 또는 MLP에 입력한다.

장점:

- ISIC 2024 winning solution과 가장 유사하다.
- tabular feature가 강한 데이터셋에 적합하다.
- image branch와 tabular branch를 독립적으로 검증하기 쉽다.

단점:

- end-to-end multimodal model이라고 주장하기는 약하다.

### 5.2 Early Fusion / Concatenation

구조:

1. CNN/ViT image encoder에서 embedding 추출
2. metadata MLP embedding과 concat
3. classifier로 최종 예측

장점:

- 구현이 쉽고 논문 baseline으로 적합하다.

단점:

- image feature와 tabular feature의 scale 차이 때문에 metadata가 무시되거나 과도하게 지배할 수 있다.

### 5.3 Metadata Modulation / FiLM / MetaBlock

구조:

1. metadata encoder가 scale/shift/gating vector를 생성한다.
2. image feature map 또는 embedding을 metadata 조건으로 조절한다.

장점:

- 단순 concat보다 논문 기여로 설명하기 좋다.
- 환자 정보가 image feature extraction 과정에 직접 영향을 줄 수 있다.

단점:

- 구현과 ablation이 필요하다.

### 5.4 Cross-Attention Fusion

구조:

1. image token과 metadata token을 각각 encoder에 통과시킨다.
2. self-attention으로 modality 내부 feature를 정제한다.
3. cross-attention으로 image와 metadata가 서로를 guide하도록 한다.

장점:

- multimodal fusion 기여를 가장 명확히 보여줄 수 있다.
- MMF-Net 등 직접적인 선행연구가 있다.

단점:

- 데이터가 극단적으로 불균형이므로 end-to-end training이 불안정할 수 있다.
- train-only 조건에서는 positive 부족으로 과적합 위험이 크다.

---

## 6. ISIC 2024 Train-Only 논문 실험 설계 제안

### 6.1 권장 모델 구성

최소 실험군:

1. Image-only model
   - EfficientNetV2, ConvNeXt, EVA, EdgeNeXt 중 하나
   - weighted BCE / focal loss / class-balanced focal loss 비교

2. Tabular-only model
   - LightGBM, CatBoost, XGBoost
   - basic metadata + WB360 metadata
   - patient-context feature 포함/제외 ablation

3. Simple multimodal baseline
   - image embedding 또는 image OOF prediction + metadata concat
   - MLP 또는 GBDT classifier

4. Proposed multimodal model
   - MetaBlock/FiLM 또는 cross-attention 기반 fusion
   - patient-context feature를 포함한 tabular branch

5. Explainability layer
   - SHAP으로 tabular/WB360 feature contribution 분석
   - CAM으로 image branch가 주목한 lesion region 확인
   - nomogram 또는 coefficient-based score로 clinical decision support 해석 보강

### 6.2 필수 Ablation

| 실험 | 목적 |
|---|---|
| image-only | image branch의 한계 확인 |
| tabular-only | metadata/WB360 feature의 독립 성능 확인 |
| image + basic metadata | 기본 임상정보 효과 확인 |
| image + WB360 metadata | lesion measurement 효과 확인 |
| image + patient-context feature | ugly duckling feature 효과 확인 |
| concat fusion vs attention fusion | 제안 fusion 방식의 기여 검증 |
| BCE vs focal vs class-balanced focal | imbalance 극복 방법 검증 |
| SHAP/CAM/nomogram 포함 vs 제외 | 성능뿐 아니라 설명 가능성 기여 검증 |

### 6.3 권장 평가 지표

ISIC 2024의 class imbalance를 고려하면 다음 지표를 함께 제시하는 것이 좋다.

- pAUC > 80% TPR
- AUROC
- AUPRC
- sensitivity
- specificity
- F1-score
- balanced accuracy
- NNT@80% sensitivity
- NNT@90% sensitivity

특히 pAUC와 AUPRC를 중심으로 제시해야 한다. accuracy는 참고 지표로만 사용한다.

### 6.4 논문 주장 방향

가능한 논문 메시지:

> ISIC 2024 train dataset은 malignant 비율이 0.098%에 불과한 극단적 불균형 데이터이며, image-only deep learning은 rare malignant detection에서 한계가 있다. 본 연구는 patient-context-aware tabular feature와 lesion image representation을 결합하는 multimodal fusion을 통해 high-sensitivity 영역의 pAUC와 AUPRC를 개선한다.

---

## 6.5 수식 검토 순서도

각 논문에 추가한 모델 구조 수식은 아래 순서로 검토한다. 특히 원문 equation이 아니라 이해용 구조 표현인 경우, figure와 method 설명에 맞는 수준으로 단순화했는지 확인한다.

```mermaid
flowchart TD
    A["논문 구조 확인"] --> B["입력 modality 정의"]
    B --> C["Image / metadata encoder 수식 작성"]
    C --> D["Fusion 방식 수식 작성"]
    D --> E["Classifier / prediction 수식 작성"]
    E --> F["원문 Figure, 설명, 코드와 대조"]
    F --> G{"구조가 원문과 일치하는가?"}
    G -- "Yes" --> H["변수 설명과 해석 추가"]
    G -- "No" --> I["수식 단순화 또는 구조적 표현으로 표시"]
    I --> F
    H --> J["LaTeX 렌더링 및 번호 흐름 검토"]
```

---

## 6.6 원문 대조 검토 메모

- ISIC 2024 Automated Triage: 원문 Methods는 neural network outputs와 metadata features를 3개 GBT model에 넣고 output을 aggregate한다고 설명하므로, 단일 평균 image score 수식을 GBT ensemble 수식으로 수정했다.
- Wang et al. 2025: 원문 Eq. (1)~(2)의 CNN six-class probability vector + clinical feature vector + XGBoost late fusion 수식을 반영했고, VIF/nomogram logistic scoring은 해석 가능 scoring layer로 구분했다.
- MetaBlock: 원문 Eq. (1)~(7)의 `x_img`, `x_meta`, `x_img_tilde`, `x_meta_tilde`, `T_gate`, `S_gate` 표기로 수정했다.
- MMF-Net: 원문 cross-attention 설명에 맞춰 image-guided/meta-guided 양방향 path의 Query, Key, Value 출처를 수정했고, BACC 설명을 sensitivity-specificity 평균으로 정리했다.
- Islam et al.: 전체 수집 metadata는 22개지만 multimodal fusion 설명은 7개 C4C risk factor와 C4C risk score, 총 8개 feature 중심이므로 본문과 요약표를 구분해 수정했다.
- Nguyen et al.: Soft-Attention은 단순 feature multiplication이 아니라 original feature tensor와 scaled attention feature의 concatenation 구조로 수정했고, abstract 수치와 본문/appendix 수치를 구분했다.

---

## 7. 참고문헌 및 링크

1. Kurtansky, N. R. et al. The SLICE-3D dataset: 400,000 skin lesion image crops extracted from 3D TBP for skin cancer detection. Scientific Data, 2024.  
   https://www.nature.com/articles/s41597-024-03743-w

2. Kurtansky, N. R. et al. Automated triage of cancer-suspicious skin lesions with 3D total-body photography. npj Digital Medicine, 2025.  
   https://www.nature.com/articles/s41746-025-02070-7

3. Wang, Z. et al. Explainable multimodal AI for skin lesion risk prediction via 3D imaging and clinical data. Scientific Reports, 2025.  
   https://www.nature.com/articles/s41598-025-33536-z

4. Pacheco, A. G. C. and Krohling, R. A. An Attention-Based Mechanism to Combine Images and Metadata in Deep Learning Models Applied to Skin Cancer Classification. IEEE JBHI, 2021.  
   https://doi.org/10.1109/JBHI.2021.3062002

5. Ou, C. et al. A deep learning based multimodal fusion model for skin lesion diagnosis using smartphone collected clinical images and metadata. Frontiers in Surgery, 2022.  
   https://www.frontiersin.org/articles/10.3389/fsurg.2022.1029991/full

6. Yap, J., Yolland, W., and Tschandl, P. Multimodal skin lesion classification using deep learning. Experimental Dermatology, 2018.  
   https://doi.org/10.1111/exd.13777

7. Islam, S. et al. Advancing skin cancer detection through deep learning and fusion of patient metadata and skin lesion images. Scientific Reports, 2026.  
   https://www.nature.com/articles/s41598-025-26392-4

8. Nguyen, V. D., Bui, N. D., and Do, H. K. Skin Lesion Classification on Imbalanced Data Using Deep Learning with Soft Attention. Sensors, 2022.  
   https://www.mdpi.com/1424-8220/22/19/7530

9. Lin, T.-Y. et al. Focal Loss for Dense Object Detection. ICCV 2017 / TPAMI 2020.  
   https://pubmed.ncbi.nlm.nih.gov/30040631/

10. Cui, Y. et al. Class-Balanced Loss Based on Effective Number of Samples. CVPR 2019.  
   https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html

11. Goceri, E. GAN based augmentation using a hybrid loss function for dermoscopy images. Artificial Intelligence Review, 2024.  
    https://link.springer.com/article/10.1007/s10462-024-10897-x

12. Souza Jr., L. A. et al. LiwTERM: A Lightweight Transformer-Based Model for Dermatological Multimodal Lesion Detection. SIBGRAPI, 2024.  
    https://doi.org/10.1109/SIBGRAPI62404.2024.10716324
