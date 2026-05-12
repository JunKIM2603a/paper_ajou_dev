# Wang et al.: Explainable Multimodal AI for Skin Lesion Risk Prediction via 3D Imaging and Clinical Data

## 출처/링크

출처: Scientific Reports, 2025  
링크: https://www.nature.com/articles/s41598-025-33536-z  
PDF: `s41598-025-33536-z-1.pdf`

## 우리 연구에서의 위치

3D-TBP image-derived prediction vector와 clinical feature를 XGBoost로 결합하는 late fusion 및 SHAP/CAM 기반 설명 가능성 설계의 근거이다.

---

## 주요 Figure

**Figure 1. Explainable multimodal AI framework**

data preparation, image-only model, clinical model, fusion model, model analysis를 한 흐름으로 보여준다. ISIC 2024 train-only 연구에서 전체 pipeline을 설명할 때 유용하다.

![Wang et al. Fig 1](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-025-33536-z/MediaObjects/41598_2025_33536_Fig1_HTML.png)
> 1. 데이터 준비 (Data Preparation)
> 2. AI 모델 만들기 (Model Development)
> - 영상만 쓰는 모델 (Image-only model):
> - 임상 정보만 쓰는 모델 (Clinical model):
> - 영상 + 임상 정보 융합 모델 (Fusion):
> 3. 모델 성능 평가 및 설명 (Model Analysis)
> - 성능 평가 (Performance):
> - 설명 가능성 (Explanation):
>   - 노모그램: 어떤 정보가 예측에 얼마나 중요한지 시각적으로 보여주는 그래프입니다.
>   - SHAP 시각화: AI가 각 환자의 어떤 정보(특징) 때문에 그런 예측을 했는지 수치로 분석하여 보여줍니다.
>   - CAM: AI가 영상의 어느 부분을 보고 판단했는지 영상 위에 색깔로 표시해 줍니다.

**Figure 2. Multimodal fusion workflow**

CNN-derived six-class probability vector와 clinical feature vector를 결합해 XGBoost에 입력하는 **late fusion** 구조를 보여준다. ISIC 2024에서 image score + tabular feature를 결합하는 직접 선행근거로 중요하다.

![Wang et al. Fig 2](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-025-33536-z/MediaObjects/41598_2025_33536_Fig2_HTML.png)

**Figure 3. Performance and interpretability**

multimodal model의 5-fold ROC, SHAP feature importance, CAM visualization을 함께 보여준다. 성능 개선뿐 아니라 어떤 feature와 image region이 판단에 기여했는지 설명하는 근거로 사용할 수 있다.

![Wang et al. Fig 3](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-025-33536-z/MediaObjects/41598_2025_33536_Fig3_HTML.png)

**Figure 4. ISIC 2024 binary benchmark**

6-class 예측 결과를 benign/malignant로 collapse한 뒤 ISIC 2024 challenge top teams와 pFPR을 비교한다. train-only 연구의 pAUC 또는 high-sensitivity benchmark와 연결하기 좋다.

![Wang et al. Fig 4](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-025-33536-z/MediaObjects/41598_2025_33536_Fig4_HTML.png)

## 목표와 기여
ISIC 2024 기반 3D-TBP image와 structured clinical/lesion feature를 결합해 6개 skin lesion type을 분류하고, SHAP, CAM, nomogram으로 모델 판단을 설명하는 explainable multimodal AI framework를 제안 
기존 ISIC 2024 competition 분석 논문이 leaderboard와 patient-context 효과를 강조했다면, 이 논문은 multimodal fusion 결과를 clinician-friendly scoring system과 XAI로 설명하는 데 초점

## Dataset 정보
- Dataset: 논문은 `ISIC 2024 dataset`이라고 표현하지만, 숫자와 label 구성상 Kaggle/SLICE-3D public train 전체가 아니라 선별된 subset 또는 재구성된 6-class subset으로 봐야 한다.
- Patient 수: 1,075명
- Feature 수: 41 clinical 및 lesion-specific feature
- Class setting: 6-class lesion risk prediction
- Class 구성: invasive melanoma 157, basal cell carcinoma 163, squamous cell carcinoma 73, nevus 443, benign NOS 200, solar/actinic keratosis 39
- Modality: 3D TBP image + clinical/lesion metadata

### Dataset 해석 주의

- 공식 ISIC 2024 Kaggle/SLICE-3D train dataset은 401,059개 lesion tile과 binary malignant target을 제공하는 challenge dataset이다.
- Wang et al.의 6-class 구성은 총 1,075개 case로, Kaggle public train 전체 401,059개 lesion tile과 규모가 맞지 않는다.
- 특히 invasive melanoma 157 + basal cell carcinoma 163 + squamous cell carcinoma 73 = 393으로 공식 train의 malignant positive 수와 일치하지만, benign 쪽은 400,666개 전체가 아니라 nevus/benign NOS/solar-or-actinic keratosis 일부 682개만 사용한 형태이다.
- 따라서 이 논문은 ISIC 2024 전체 binary benchmark와 직접 동일한 데이터셋 실험이 아니라, ISIC 2024에서 granular diagnosis가 있는 일부 case를 6-class risk prediction으로 재구성한 related experiment로 해석하는 것이 안전하다.

## Imbalance 처리
- 불균형 정도: nevus 443개 vs solar/actinic keratosis 39개, 약 11.4:1
- class 조절: 6-class setting 유지
- 데이터 조작: nevus를 제외한 모든 lesion type에 targeted augmentation 적용
- augmentation 종류: random rotation, horizontal flip, color variation
- multimodal pairing 해석: **증강은 image에만 적용되며 diagnostic label은 바꾸지 않는다.** 따라서 증강 이미지가 multimodal 학습 sample로 사용될 때는 같은 원본 lesion의 clinical/tabular metadata vector를 그대로 연결하는 방식으로 이해하는 것이 자연스럽다.
- 평가 기반 대응: 5-fold cross-validation 및 ISIC 2024 binary benchmark pFPR 비교
- 주의: class weighting과 oversampling은 고려했지만 최종 핵심 방법은 targeted augmentation으로 설명됨

## Tabular model
clinical-only model로 logistic regression, decision tree, random forest, gradient boosting, XGBoost, CatBoost, SVM을 비교했고, XGBoost가 가장 robust한 모델로 사용되었다. 추가로 VIF 기반 multicollinearity screening 후 multinomial logistic regression과 nomogram을 이용해 임상적으로 해석 가능한 scoring system을 구성했다.

## Image model
image branch는 HAM10000으로 pretraining한 CNN을 3D-TBP image에 transfer learning한 구조이다. CNN은 3개 convolutional block을 기반으로 하며, 3D-TBP fine-tuning을 위해 Conv2D layer 2개를 추가했다. 이미지는 128 x 128 x 3으로 resize했고, cross-entropy loss, Adam optimizer, learning rate 1e-4, batch size 128, 200 epochs로 학습했다.

## Fusion 방식
late fusion 구조이다. CNN이 만든 six-class probability vector를 structured visual feature로 사용하고, clinical feature vector와 concat 및 standardization 후 XGBoost classifier에 입력한다. 이후 SHAP으로 feature contribution을 설명하고, CAM으로 image region contribution을 시각화하며, multinomial logistic regression 기반 nomogram으로 clinical scoring panel을 제공한다.

## 모델 구조 수식
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
\operatorname{logit}(P(Y \le k \mid X)) = \beta_0^{(k)} + \sum_{i=1}^{n}\beta_i^{(k)}X_i
$$

이 수식은 final classifier 자체라기보다, model output과 clinical/TBP feature를 clinician-friendly risk score로 변환하기 위한 해석 가능 scoring layer로 보는 것이 적절하다.

## 평가 지표
- 우선순위 지표: multiclass AUC, recall, F1
- ISIC benchmark 지표: 6-class prediction을 benign/malignant로 collapse한 pFPR
- 보조 지표: accuracy, precision, confusion matrix
- 설명 가능성 지표/도구: SHAP feature importance, CAM, nomogram, VIF

## 평가 결과
- Clinical-only XGBoost: overall accuracy `0.6837`, recall `0.4090`, F1 `0.4582`
- Clinical-only class result: BCC `78.6%`, nevus `72.6%`, melanoma invasive `43.8%`, actinic keratosis `12.5%`, SCC `16.7%`
- 3D-TBP image-only CNN: nevus `87.10%`, benign NOS `75.34%`, invasive melanoma `71.88%`, BCC `54.05%`, SCC `60.32%`, actinic/solar keratosis `65.62%`
- Multimodal fusion: class별 AUC `> 0.95`, nevus 및 actinic keratosis AUC `0.98`
- Multimodal fusion: recall 및 F1 score `> 95%`
- ISIC 2024 binary benchmark: pFPR `0.17343`, top 5 team range `0.17210-0.17264`

## ISIC2024 strict multimodal 연구에 주는 시사점
이 논문은 ISIC 2024에서 image-derived prediction vector와 clinical/TBP metadata를 결합하는 late fusion이 unimodal model보다 강하다는 근거를 제공한다. 특히 SHAP, CAM, nomogram을 통해 성능 결과를 설명 가능하게 제시하므로, train-only 논문에서 단순 pAUC 개선뿐 아니라 어떤 metadata와 image region이 malignant risk에 기여했는지 설명하는 XAI section의 핵심 선행연구로 쓸 수 있다.

## 추가 논의/생각해볼 점
- image branch가 HAM10000 transfer learning을 사용하므로 엄격한 ISIC 2024 train-only 조건에서는 외부 pretraining 사용 여부를 별도 분리해야 한다.
- 6-class setting을 binary challenge metric으로 collapse하므로, ISIC 2024 원래 binary target과 직접 동일한 실험은 아니다.
- 데이터셋 역시 Kaggle/SLICE-3D public train 전체가 아니라 1,075개 case의 6-class subset으로 보이므로, 성능 수치를 ISIC 2024 binary challenge baseline과 직접 비교하면 안 된다.
- 논문 자체도 external cohort validation이 없고, augmentation으로 인한 optimistic performance 가능성을 limitation으로 언급한다.
- 우리 구현에서 이 방식을 참고할 경우, train fold 안에서만 image augmentation을 수행하고, 증강된 image view에는 원본 lesion의 ordinary tabular metadata와 label만 연결해야 한다. validation/test fold나 `iddx_full` 같은 privileged field를 augmentation pairing에 섞으면 안 된다.
- 그럼에도 41개 feature, SHAP, CAM, nomogram을 결합한 구조는 train-only 연구의 explainability 설계에 매우 직접적으로 유용하다.

---

[메인 문서로 돌아가기](../2026-05-12_isic2024_multimodal_literature_review.md#3-주요-논문별-상세-분석)
