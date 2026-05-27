# Nguyen et al.: Skin Lesion Classification on Imbalanced Data Using Soft Attention

## 출처/링크

출처: Sensors, 2022  
링크: https://www.mdpi.com/1424-8220/22/19/7530

## 우리 연구에서의 위치

soft attention, metadata branch, class-weighted loss를 함께 사용한 imbalance-aware image-tabular baseline 및 ablation 근거이다.

---

## 주요 Figure
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

## 목표와 기여
imbalanced skin lesion classification에서 soft-attention과 imbalance-aware loss function을 결합한 deep learning model을 제안했다.

## Dataset 정보
- Dataset: HAM10000
- Image 수: 10,015개 dermoscopy images
- Class 수: 7 classes
- Metadata: age, gender 등 patient information

## Imbalance 처리
- 불균형 정도: NV 6705개 vs DF 115개, 약 58:1
- class 조절: class 수 조절 없음
- 데이터 조작: augmentation으로 training image를 53,573개까지 확장
- 학습 조작: weighted/new loss function 사용
- 모델 조작: soft-attention으로 lesion 중심 feature 강화

## Tabular model
age, gender 같은 personal information을 추가로 사용했다.

## Image model
InceptionResNetV2, MobileNetV3Large 등 다양한 backbone을 사용했다.

## Fusion 방식
image feature 중심 구조에 personal information을 함께 사용했다.

## 모델 구조 수식
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

## 평가 지표
- 우선순위 지표: AUC, recall, F1
- 보조 지표: accuracy, precision
- recall: minority class를 놓치지 않는 정도
- F1: precision과 recall의 조화평균

## 평가 결과
- Abstract 요약: InceptionResNetV2 + Soft-Attention + new loss에서 `ACC 0.90`, `Precision 0.81`, `F1 0.81`, `Recall 0.82`, `AUC 0.99`
- 본문 conclusion: best model의 핵심 균형 지표로 `F1 0.86`, `Recall 0.81`, `AUC 0.975`를 제시
- Appendix detailed table: InceptionResNetV2 with Metadata and WeightLoss의 mean `F1 0.81`, mean `Recall 0.81`; MobileNetV3Large는 `ACC 0.86`, `F1 0.79`, `Sensitivity 0.80`, `AUC 0.96`로 더 가벼운 대안으로 제시됨

## ISIC2024 strict multimodal 연구에 주는 시사점
ISIC 2024 image branch에서 focal loss, class-balanced loss, attention block을 실험할 근거로 활용 가능하다.

## 추가 논의/생각해볼 점
- augmentation과 weighted loss가 함께 들어가 있어 성능 향상의 원인을 분리해 봐야 한다.
- loss 때문인지 attention 때문인지 ablation을 세밀히 확인해야 한다.
- ISIC 2024에서는 binary pAUC/AUPRC 기준으로 같은 전략이 유지되는지 별도 검증이 필요하다.

---

[메인 문서로 돌아가기](../2026-05-12_isic2024_multimodal_literature_review.md#3-주요-논문별-상세-분석)
