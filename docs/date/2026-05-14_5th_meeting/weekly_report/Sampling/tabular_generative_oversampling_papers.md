# Tabular Dataset Oversampling with VAE and Diffusion Models

작성일: 2026-05-08

## 핵심 요약

Tabular dataset의 불균형 문제는 단순히 minority class를 복제하는 방식보다, **class-conditional VAE**, **tabular diffusion model**, **score-based generative model** 등을 이용해 synthetic minority samples를 생성하는 방식으로 접근할 수 있다.

Stable Diffusion을 tabular data에 그대로 적용하는 것은 일반적이지 않다. 대신 Stable Diffusion의 핵심 아이디어인 **latent space에서의 diffusion** 또는 **conditional generation**을 tabular row vector에 맞게 변형한 모델을 사용한다.

기본 흐름은 다음과 같다.

```text
train data만 사용
-> minority class 또는 class-conditional generative model 학습
-> synthetic minority samples 생성
-> original train + synthetic train 구성
-> classifier 학습
-> real validation/test set에서 평가
```

## 왜 VAE나 Diffusion을 쓰는가?

### VAE 기반 접근

VAE는 tabular row를 latent space로 압축한 뒤 다시 복원하도록 학습한다. 학습 후 latent space에서 새로운 값을 샘플링하면 기존 데이터 분포와 유사한 synthetic row를 만들 수 있다.

장점:

- 구현이 비교적 단순하다.
- 연속형 변수 생성에 잘 맞는다.
- class-conditional 구조를 붙이기 쉽다.

한계:

- 평균적인 샘플을 많이 생성할 수 있다.
- minority class의 복잡한 경계나 rare pattern을 충분히 살리지 못할 수 있다.
- categorical feature 처리를 별도로 설계해야 한다.

### Diffusion 기반 접근

Diffusion model은 tabular row에 노이즈를 추가한 뒤, 이를 역으로 제거하면서 실제 데이터와 유사한 row를 생성한다. 이미지 diffusion과 원리는 비슷하지만 입력이 이미지가 아니라 feature vector라는 점이 다르다.

장점:

- 복잡한 feature distribution을 모델링하는 데 강점이 있다.
- VAE보다 다양한 synthetic sample을 생성할 가능성이 있다.
- class-conditional generation과 잘 맞는다.

한계:

- 구현과 튜닝이 더 어렵다.
- 데이터가 작으면 과적합 synthetic sample이 생길 수 있다.
- numerical/categorical/missing value/clinical constraint 처리가 중요하다.

## 직접 관련 논문

| 논문 | 방법 | 관련성 |
|---|---|---|
| [On oversampling imbalanced data with deep conditional generative models](https://www.sciencedirect.com/science/article/pii/S0957417420311155) | conditional VAE/GAN | minority class synthetic sample 생성으로 oversampling을 수행하는 대표 논문 |
| [A conditional variational autoencoder based self-transferred algorithm for imbalanced classification](https://www.sciencedirect.com/science/article/abs/pii/S0950705121000198) | CVAE | majority/minority 정보를 활용해 imbalanced classification을 개선 |
| [SOS: Score-based Oversampling for Tabular Data](https://arxiv.org/abs/2206.08555) | score-based diffusion 계열 | tabular oversampling에 score-based generative model을 직접 적용 |
| [CTVAE: Contrastive Tabular Variational Autoencoder for imbalance data](https://link.springer.com/article/10.1007/s10115-025-02377-7) | conditional VAE + contrastive learning | latent space에서 class separation을 강화해 synthetic sample 품질 개선 |
| [Diffusion GAN-based Oversampling for Imbalanced Tabular Data](https://scholars.hkbu.edu.hk/en/publications/diffusion-gan-based-oversampling-for-imbalanced-tabular-data/) | diffusion + GAN | diffusion generator와 GAN discriminator를 결합한 tabular oversampling 접근 |
| [CTTVAE: Latent Space Structuring for Conditional Tabular Data Generation on Imbalanced Datasets](https://arxiv.org/abs/2602.03641) | transformer-based tabular VAE | severe imbalance와 rare-event tabular generation에 초점 |

## Tabular Diffusion 기반 Synthetic Data 논문

| 논문 | 방법 | 메모 |
|---|---|---|
| [TabDDPM: Modelling Tabular Data with Diffusion Models](https://proceedings.mlr.press/v202/kotelnikov23a.html) | DDPM for tabular data | mixed continuous/categorical feature를 다루는 대표 tabular diffusion 논문 |
| [CoDi: Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis](https://proceedings.mlr.press/v202/lee23i.html) | continuous/discrete 별도 diffusion | mixed-type tabular data synthesis에 강점 |
| [Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space / TabSyn](https://arxiv.org/abs/2310.09656) | VAE latent space + diffusion | Stable Diffusion의 latent diffusion 발상과 가장 가까운 tabular 논문 |
| [TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation](https://arxiv.org/abs/2410.20626) | mixed-type diffusion | numerical/categorical feature를 joint diffusion으로 처리 |
| [Balanced Mixed-Type Tabular Data Synthesis with Diffusion Models](https://arxiv.org/abs/2404.08254) | balanced tabular diffusion | label과 sensitive attribute 균형을 고려한 synthetic generation |

## 기본 Baseline 및 배경 논문

| 논문 | 방법 | 메모 |
|---|---|---|
| [Modeling Tabular Data using Conditional GAN](https://proceedings.neurips.cc/paper/2019/hash/254ed7d2de3b23ab10936522dd547b78-Abstract.html) | CTGAN/TVAE | tabular synthetic data generation의 대표 baseline |
| [Tabular and latent space synthetic data generation: a literature review](https://link.springer.com/article/10.1186/s40537-023-00792-7) | review | tabular synthetic data 전반을 정리한 리뷰 |
| [Synthetic Tabular Data Generation for Imbalanced Classification: The Surprising Effectiveness of an Overlap Class](https://arxiv.org/abs/2412.15657) | generative oversampling analysis | imbalanced classification에서 synthetic data 품질과 overlap class 문제를 다룸 |

## 추천 읽기 순서

Diffusion 계열을 먼저 보고 싶다면:

1. [SOS: Score-based Oversampling for Tabular Data](https://arxiv.org/abs/2206.08555)
2. [TabDDPM: Modelling Tabular Data with Diffusion Models](https://proceedings.mlr.press/v202/kotelnikov23a.html)
3. [Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space / TabSyn](https://arxiv.org/abs/2310.09656)
4. [TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation](https://arxiv.org/abs/2410.20626)
5. [Diffusion GAN-based Oversampling for Imbalanced Tabular Data](https://scholars.hkbu.edu.hk/en/publications/diffusion-gan-based-oversampling-for-imbalanced-tabular-data/)

VAE 계열을 먼저 보고 싶다면:

1. [On oversampling imbalanced data with deep conditional generative models](https://www.sciencedirect.com/science/article/pii/S0957417420311155)
2. [A conditional variational autoencoder based self-transferred algorithm for imbalanced classification](https://www.sciencedirect.com/science/article/abs/pii/S0950705121000198)
3. [CTVAE: Contrastive Tabular Variational Autoencoder for imbalance data](https://link.springer.com/article/10.1007/s10115-025-02377-7)
4. [CTTVAE: Latent Space Structuring for Conditional Tabular Data Generation on Imbalanced Datasets](https://arxiv.org/abs/2602.03641)

## 실험 설계 시 주의점

1. Synthetic generator는 반드시 **train set만** 사용해 학습한다.
2. Validation/test set에는 synthetic sample을 섞지 않는다.
3. 성능 평가는 AUROC만 보지 말고 AUPRC, sensitivity, specificity, F1, calibration을 함께 본다.
4. SMOTE, ADASYN, class weight, focal loss 등 단순 baseline과 반드시 비교한다.
5. 의료/임상 tabular data라면 가능한 feature 범위와 feature 간 규칙을 후처리로 강제해야 한다.
6. Synthetic sample이 원본 minority sample을 거의 복사하는지 nearest-neighbor distance 등으로 확인한다.
7. Categorical feature는 one-hot, embedding, multinomial diffusion 등 모델별 처리 방식이 성능에 큰 영향을 준다.

## 검색 키워드

논문을 추가로 찾을 때 유용한 검색어:

- `tabular diffusion oversampling imbalanced classification`
- `score-based oversampling tabular data`
- `conditional VAE imbalanced tabular data`
- `deep generative oversampling imbalanced data`
- `synthetic tabular data generation imbalanced classification`
- `latent diffusion tabular data generation`
- `mixed-type tabular diffusion model`

## 한 줄 결론

Tabular dataset의 oversampling에는 이미지용 Stable Diffusion을 직접 쓰기보다, **class-conditional VAE**, **TVAE/CTGAN**, **TabDDPM/TabSyn/TabDiff 같은 tabular diffusion model**을 사용하는 것이 더 적절하다. 특히 minority class의 feature correlation과 rare pattern이 중요한 문제라면 generative oversampling을 실험해볼 가치가 있다.
