# Automated Triage of Cancer-Suspicious Skin Lesions with 3D Total-Body Photography

## 출처/링크

출처: npj Digital Medicine, 2025  
링크: https://www.nature.com/articles/s41746-025-02070-7

## 우리 연구에서의 위치

ISIC 2024 challenge metric, metadata/patient-context ablation, image score와 tabular feature late fusion을 직접 뒷받침하는 핵심 baseline reference이다.

---

## 주요 Figure
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

## 목표와 기여
ISIC 2024 Kaggle competition 결과를 분석하고, winning model의 ablation study를 통해 image, basic metadata, WB360 appearance metadata, patient-context feature가 성능에 미치는 영향을 비교했다.

## Dataset 정보
- Dataset: ISIC 2024 Challenge dataset
- Task: malignant/benign binary classification
- Modality: 3D-TBP lesion tile + metadata + patient-context feature

## Imbalance 처리
- class 조절: binary target 유지, class 수 축소/재정의 없음
- 데이터 조작: 핵심 방법으로 oversampling/undersampling을 제안하지 않음
- 학습/모델 조작: patient-context feature, metadata feature, GBDT late fusion 활용
- 평가 기반 대응: `pAUC > 80% TPR`로 high-sensitivity 영역을 우선 평가
- 주의: 일부 상위 image model은 external dermoscopy data를 사용했으므로 train-only 연구에서는 분리 해석 필요

## Tabular model
metadata branch는 basic demographics, WB360 appearance metadata, interaction terms, patient-context terms를 사용했다. 최종 단계에서는 neural network outputs와 metadata feature를 3개 Gradient Boosting Tree 모델에 입력하고, 그 출력들을 aggregate해 lesion risk estimate를 만들었다.

## Image model
image branch는 EVA model 2개와 EdgeNeXt 1개로 구성된 ensemble이다. 일부 image model은 external dermoscopy data도 사용했으나, train-only 연구에서는 외부 데이터 사용 부분을 제외하고 tile-only image branch를 참고하는 것이 적절하다.

## Fusion 방식
image model ensemble의 neural network output vector와 metadata/patient-context feature를 결합한 뒤, 3개 GBT 모델에 넣고 GBT output을 aggregate하는 late fusion 구조이다.

## 모델 구조 수식
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

## 평가 지표
- 우선순위 지표: `pAUC > 80% TPR`. `TPR >= 0.8`인 ROC 구간의 partial AUC이며, score range는 `[0, 0.2]`이다.
- 보조 지표: AUC.
- clinical utility 지표: `SE top-15`, `NNT 80% sensitivity`, `NNT 90% sensitivity`.
- NNT는 해당 sensitivity threshold에서 true positive 1개를 찾기 위해 expert review가 필요한 lesion 수로 해석할 수 있다.

## 평가 결과
- 우선순위 지표: `pAUC 0.1726/0.2`
- 보조 지표: full AUC `0.9668`
- clinical utility: `NNT80 51.57`, `NNT90 98.20`
- ablation: patient-context 제외 시 AUC가 0.967에서 0.956으로 감소
- 추가 결과: WB360 metadata-only model이 tile-only model보다 좋은 성능을 보임

## ISIC2024 strict multimodal 연구에 주는 시사점
ISIC 2024에서는 image-only보다 metadata와 patient-context를 결합한 multimodal late fusion이 핵심적이다. train-only 논문에서도 image-only, tabular-only, late fusion, patient-context ablation을 반드시 포함하는 것이 좋다.

## 추가 논의/생각해볼 점
- train-only 연구에서는 외부 dermoscopy data를 제외하고도 metadata/patient-context 효과가 유지되는지 확인해야 한다.
- patient-context feature는 강력하지만, patient-level split을 잘못 잡으면 leakage로 과대평가될 수 있다.

---

[메인 문서로 돌아가기](../2026-05-12_isic2024_multimodal_literature_review.md#3-주요-논문별-상세-분석)
