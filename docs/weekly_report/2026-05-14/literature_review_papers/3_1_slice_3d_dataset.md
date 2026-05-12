# SLICE-3D Dataset: 400,000 Skin Lesion Image Crops Extracted from 3D TBP

## 출처/링크

출처: Scientific Data, 2024  
링크: https://www.nature.com/articles/s41597-024-03743-w

## 우리 연구에서의 위치

ISIC 2024 train dataset의 공식 데이터셋 근거이며, ultra-rare malignant target, weak benign label, patient-level split 필요성을 정당화하는 1차 자료이다.

---

## 주요 Figure
원문 라이선스: CC BY 4.0

**Figure 1. Examples of image types**

ISIC 2024의 tile image가 dermoscopic image보다 morphologic detail이 적고, 3D-TBP에서 추출된 low-resolution clinical crop이라는 점을 보여준다. 논문 dataset section에서 가장 유용하다.

![SLICE-3D Fig 1](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41597-024-03743-w/MediaObjects/41597_2024_3743_Fig1_HTML.png)

**Figure 2. Dataset curation workflow**

strong label, weak label, tile sub-selection, QA 과정을 설명한다. train-only 연구에서 label noise와 weak benign label 문제를 설명할 때 유용하다.

![SLICE-3D Fig 2](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41597-024-03743-w/MediaObjects/41597_2024_3743_Fig2_HTML.png)

## 목표와 기여
ISIC 2024 Challenge의 공식 train dataset인 SLICE-3D를 소개한 dataset descriptor이다. 3D-TBP에서 자동 추출한 lesion crop image와 metadata를 공개하여, dermoscopy 중심 기존 데이터셋의 selection bias를 줄이고 primary care 또는 telehealth 환경에 가까운 저해상도/비전문 촬영 이미지 기반 모델 개발을 가능하게 했다.

## Dataset 정보
- Dataset: ISIC 2024 train dataset
- Sample 수: 401,059 lesion tiles
- Task: malignant/benign binary classification
- Modality: 3D-TBP lesion image + patient/lesion metadata

## Imbalance 처리
- **불균형 정도: benign 400,666개 vs malignant 393개, 약 1020:1**
- 데이터 조작: 제안 없음
- 학습 조작: 모델 학습 논문이 아니므로 없음
- class 조절: binary target 자체를 기술하며 class 수 조절 없음

## Tabular model
모델은 없음. 다만 age, sex, anatomical site, lighting modality, lesion size/color/shape 관련 WB360 measurements, patient_id 등이 제공된다.

## Image model
모델은 없음. image는 15mm x 15mm lesion tile이며 평균 크기는 약 133px x 133px이다.

## Fusion 방식
해당 없음.

## 모델 구조 수식
원문에 학습 모델 구조 수식은 없으므로, 아래는 SLICE-3D sample 구성을 이해하기 위한 표기이다.

$$
x_i = (I_i, m_i, p_i), \quad y_i \in \{0, 1\}
$$

- `I_i`: 3D-TBP에서 추출된 lesion tile image
- `m_i`: age, sex, anatomical site, WB360 measurement 등 metadata
- `p_i`: patient identifier 또는 patient-level grouping 정보
- `y_i`: malignant 여부

따라서 이 논문은 위 식처럼 sample 구성을 정의한 dataset descriptor이며, 후속 연구가 사용할 multimodal input space를 제공한 것으로 해석해야 한다.

## 평가 지표
- 공식 classification metric: 없음
- 참고: 후속 ISIC 2024 Challenge에서는 high sensitivity 영역의 `pAUC > 80% TPR`가 핵심 평가 지표로 사용됨

## 평가 결과
- 모델 성능 결과: 해당 없음
- 주요 정량값: benign 400,666개, malignant 393개

## ISIC2024 strict multimodal 연구에 주는 시사점
ISIC 2024 train-only 연구의 dataset section에서 반드시 인용해야 할 1차 자료이다.

## 추가 논의/생각해볼 점
- benign label에 weak label이 포함되므로 label noise를 고려해야 한다.
- patient-level clustering이 존재하므로 train-only 실험에서는 patient-level split과 leakage 방지가 중요하다.
- image crop만으로는 정보가 제한적이어서 metadata와 patient-context feature의 필요성이 크다.

---

[메인 문서로 돌아가기](../2026-05-12_isic2024_multimodal_literature_review.md#3-주요-논문별-상세-분석)
