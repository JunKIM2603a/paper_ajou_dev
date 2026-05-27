# Islam et al.: Fusion of Patient Metadata and Skin Lesion Images

## 출처/링크

출처: Scientific Reports, 2026  
링크: https://www.nature.com/articles/s41598-025-26392-4

## 우리 연구에서의 위치

patient-separated split, metadata fusion, decision-level voting으로 high-sensitivity triage 성능을 개선한 최근 multimodal triage reference이다.

---

## 주요 Figure
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

## 목표와 기여
teledermatology triage에서 suspicious vs non-suspicious lesion을 분류하기 위해 patient metadata와 dermoscopic/DSLR image를 결합한 AI framework를 제안했다.

## Dataset 정보
- Dataset: UK private skin cancer diagnostic centres dataset
- Patient 수: 19,295명
- Lesion 수: 39,623개
- Image 수: 79,246개
- Task: suspicious/non-suspicious binary triage
- Modality: DER/DSLR image + 22 meta-features

## Imbalance 처리
- 불균형 정도: suspicious 11,258개 vs non-suspicious 67,988개
- class 조절: class 축소라기보다 clinical triage 목적의 binary label 정의
- split: 80/20 patient-separated split
- 데이터 조작: image preprocessing/augmentation
- 모델 조작: model decision majority voting
- 평가 기반 대응: sensitivity와 specificity 중심 평가

## Tabular model
전체 dataset에는 lesion별 22개 meta-feature가 수집되었지만, multimodal fusion 설명에서는 7개 C4C risk factor와 overall C4C risk score, 총 8개 meta-feature를 image output과 결합하는 것으로 설명한다.

## Image model
EfficientNet-B2 기반 image model을 사용했다.

## Fusion 방식
image vector와 8개 metadata feature를 concat한 뒤 dropout/linear layer를 거쳐 suspicious 여부를 분류했다. 별도로 DER, SLR, DER+metadata, SLR+metadata, DER+SLR, DER+SLR+metadata 등 6개 EfficientNet-B2 계열 model의 outcome decision을 majority voting으로 결합했다.

## 모델 구조 수식
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

## 평가 지표
- 우선순위 지표: sensitivity(SEN), specificity(SPC)
- SEN: suspicious/malignant 계열을 놓치지 않는 정도
- SPC: non-suspicious/benign을 불필요하게 의심하지 않는 정도
- ACC: 논문 수식상 `ACC = (SEN + SPC) / 2`, binary balanced accuracy에 가까움

## 평가 결과
- Image + metadata fused model: `SEN 99.66 ± 0.28%`, `SPC 74.45 ± 0.80%`
- Majority voting: `SEN 99.50 ± 1.18%`, `SPC 82.72 ± 1.64%`
- Metadata-only model: `SEN 85.24 ± 2.20%`, `SPC 61.12 ± 0.90%`
- 핵심 효과: majority voting에서 sensitivity를 유지하면서 specificity 개선

## ISIC2024 strict multimodal 연구에 주는 시사점
metadata fusion은 sensitivity를 유지하면서 specificity를 개선하는 데 유리하다. ISIC 2024에서도 high sensitivity 조건에서 false positive를 줄이는 fusion 목표와 잘 맞는다.

## 추가 논의/생각해볼 점
- specificity 개선은 ISIC 2024의 high-sensitivity pAUC 목표와 잘 연결된다.
- private clinic 기반 데이터라 population bias가 있을 수 있다.
- hair removal과 resizing 같은 preprocessing이 lesion shape를 왜곡할 가능성을 저자도 논의한다.

---

[메인 문서로 돌아가기](../2026-05-12_isic2024_multimodal_literature_review.md#3-주요-논문별-상세-분석)
