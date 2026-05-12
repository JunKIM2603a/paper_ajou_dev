# MMF-Net: Deep Learning Based Multimodal Fusion Using Smartphone Images and Metadata

## 출처/링크

출처: Frontiers in Surgery, 2022  
링크: https://www.frontiersin.org/articles/10.3389/fsurg.2022.1029991/full

## 우리 연구에서의 위치

image branch와 metadata branch 사이의 self-attention 및 cross-attention fusion을 비교 대상으로 설계할 때 사용할 수 있는 선행 연구이다.

---

## 주요 Figure
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

## 목표와 기여
smartphone으로 수집된 clinical image와 metadata를 결합하여 skin lesion type을 분류하는 multimodal fusion network를 제안했다.

## Dataset 정보
- Dataset: PAD-UFES-20
- Class setting: 6-class skin lesion classification
- Modality: smartphone clinical image + metadata
- Metadata: numeric/categorical metadata 제공

## Imbalance 처리
- class 조절: 6-class 설정 유지
- 데이터 조작: on-the-fly image augmentation 사용
- split: stratified 5-fold cross-validation
- sampling: 명시적 oversampling/undersampling 없음
- 학습 조작: weighted loss는 핵심 방법으로 보고하지 않음
- 평가 기반 대응: BACC와 aggregated AUC 사용

## Tabular model
numeric feature는 그대로 사용하고, categorical feature는 one-hot encoding 후 MLP encoder를 통과시켰다.

## Image model
ResNet-50을 image encoder로 사용했다.

## Fusion 방식
intra-modality self-attention으로 각 modality 내부의 중요 feature를 강화하고, inter-modality cross-attention으로 image feature와 metadata feature가 서로를 guide하도록 했다.

## 모델 구조 수식
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

## 평가 지표
- 우선순위 지표: BACC, aggregated AUC
- 보조 지표: ACC
- BACC: 원문에서는 sensitivity와 specificity의 산술평균으로 설명
- aggregated AUC: 6-class 문제에서 class pair별 AUC를 평균한 값

## 평가 결과
- BACC: `0.775 ± 0.022`
- aggregated AUC: `0.947 ± 0.007`
- ACC: `0.768 ± 0.022`

## ISIC2024 strict multimodal 연구에 주는 시사점
metadata 포함이 image-only보다 성능을 유의미하게 개선했다. ISIC 2024에서 cross-attention fusion을 제안할 때 직접적인 선행연구로 사용할 수 있다.

## 추가 논의/생각해볼 점
- cross-attention은 fusion 기여를 설명하기 좋다.
- positive가 극도로 적은 ISIC 2024에서는 end-to-end 학습이 불안정할 수 있다.
- train-only 조건에서는 late fusion baseline과 비교해 cross-attention의 실제 pAUC 개선폭을 검증해야 한다.

---

[메인 문서로 돌아가기](../2026-05-12_isic2024_multimodal_literature_review.md#3-주요-논문별-상세-분석)
