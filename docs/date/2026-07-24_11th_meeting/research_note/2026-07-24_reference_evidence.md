# 근거자료 — 참고논문 검증 & 요약 (2026-07-24)

> 이 문서는 **내 근거자료(작업용)**다. W11 보고용·연구노트에서 인용한 논문 각각에 대해 ① 서지 ② 요약 ③ **내가 인용하는 부분** ④ 주의(preprint/미확인)를 정리한다.
> 각 논문 원문을 직접 확인해 검증했다. peer-review 여부와 검증 상태를 명시한다.

## ⚠️ 검증 노트 (먼저 읽기)

- **인용 오류 정정:** 소표본 불안정성 논문(PMC8360533)의 저자는 **An et al. (PLOS ONE 2021)**이다. 이전에 "Kim & Cho"로 잘못 적었던 것을 W11·연구노트에서 모두 정정했다.
- **Cui(2019) "95%/5% 의료 예시"는 논문에 없다.** 논문 본문·Figure 1은 iNaturalist를 예시로 쓴다. 그 의료 예시는 발표 슬라이드에만 있고 논문 저자의 논문 내용이라는 근거가 없으므로, **인용하지 않는다.** Cui는 논문에 실제로 있는 내용(유효 샘플 공식·iNaturalist 프레이밍)으로만 인용한다.
- **arXiv:2602.10315의 "ordinal이 recall·kappa를 함께 개선" 주장은 본문 §V(Discussion)에 실제로 있다** ("ordinal modeling consistently increases both recall and kappa"). 다만 논문 Table II의 비교 대상은 전부 순서형 변형(evidential 유무)이라, "vs 비순서형"은 본문 서술로만 제시되고 표로 직접 보이진 않는다. → 인용 가능하되 이 뉘앙스와 preprint 표기 병기.
- **PEFTDiff 세부는 절반만 확인.** 방법(레이어 아님) 선택·"diffusion=매니폴드 랜덤워크"는 확인. 9종 방법·백본·PCA 논증·τ 수치는 CVF PDF 봇 차단으로 **미확인** → 인용 시 "저자 요약 기준"으로 완충.
- **모든 arXiv ID가 실존 논문으로 확인됨**(2026 preprint 4건 포함).

---

## Part 1. 손실 · 불균형 · 지표

### [1] Focal Loss — Lin et al., ICCV 2017
- **서지:** T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár, "Focal Loss for Dense Object Detection," ICCV 2017. arXiv:1708.02002.
- **요약:** CE에 modulating factor (1−p_t)^γ를 곱해 쉬운 샘플 기여를 낮추고 어려운 샘플에 집중. 1-stage 검출기의 극단적 **전경-배경 불균형(~1:1000)**을 겨냥해 설계됨(RetinaNet).
- **인용 부분:** focal loss의 원조이며 **검출의 전경-배경 불균형용**이지 다중분류용이 아니라는 점.
- **주의:** peer-reviewed, 신뢰 가능.

### [2] Class-Balanced Loss — Cui et al., CVPR 2019
- **서지:** Y. Cui, M. Jia, T.-Y. Lin, Y. Song, S. Belongie, CVPR 2019, pp. 9268–9277. DOI 10.1109/CVPR.2019.00949. arXiv:1901.05555.
- **요약:** 샘플이 늘수록 새 샘플의 한계 효용이 감소한다는 관점에서 **유효 샘플 수 E_n=(1−β^n)/(1−β)** 를 정의하고, 그 역수로 클래스 손실을 재가중. iNaturalist·ImageNet-LT·long-tailed CIFAR에서 검증. class-balanced focal은 "60 epoch 이후 이점이 나타난다"고 명시.
- **인용 부분:** 유효 샘플 수 기반 재가중, class-balanced focal, **이득이 극심한 불균형에서 집중**된다는 점. (※ "95%/5% 의료 예시"는 논문에 없어 인용하지 않음.)
- **주의:** peer-reviewed, 신뢰 가능.

### [3] CORAL — Cao, Mirjalili, Raschka, Pattern Recognition Letters 2020
- **서지:** arXiv:1901.07884. DOI 10.1016/j.patrec.2020.11.008. (`coral-pytorch`)
- **요약:** K등급 순서형을 **K−1개 "등급 초과?" 이진 subtask**로 바꾸고, 출력층 **가중치 공유**로 순위 단조성(consistency)을 이론적으로 보장(Niu et al. 대비 순위 불일치 0).
- **인용 부분:** aptos(순서형 0~4)용 손실. 순위 일관성 보장.
- **주의:** peer-reviewed, 신뢰 가능.

### [4] CORN — Shi, Cao, Raschka, 2021/2023
- **서지:** arXiv:2111.08851. 저널본: Pattern Analysis and Applications 26:941–955 (2023), DOI 10.1007/s10044-023-01181-9.
- **요약:** CORAL의 가중치 공유 제약이 출력층 표현력을 제한한다고 보고, **조건부확률(chain rule)**로 제약을 제거. "CORAL 대비 성능이 상당히 개선"된다고 명시.
- **인용 부분:** aptos 순서형 손실 대안, CORAL 개선판.
- **주의:** peer-reviewed(2023 저널본), 신뢰 가능.

### [5] LDAM-DRW — Cao et al., NeurIPS 2019
- **서지:** arXiv:1906.07413. (`kaidic/LDAM-DRW`)
- **요약:** 클래스별 마진 **Δ_j = C / n_j^{1/4}**(희귀할수록 큰 마진, 마진 기반 일반화 바운드 최소화에서 유도) + **재가중 지연(DRW)**(먼저 ERM으로 표현 학습 후 후반부에 재가중). 원문: LDAM이 "순수 CE 및 그 변형인 **focal loss보다 우수**"하다.
- **인용 부분:** Oral(명목형) 주력 손실. **focal보다 우수**하다는 직접 근거.
- **주의:** peer-reviewed, 신뢰 가능.

### [6] Logit Adjustment — Menon et al., ICLR 2021
- **서지:** arXiv:2007.07314.
- **요약:** 로짓을 **라벨 빈도 로그**로 조정 — **post-hoc**(학습 후) 또는 **손실 내** 두 형태. 희귀-양성 vs 다수-음성 간 상대 마진 확보. 핵심: 조정 목적함수가 **balanced error에 대해 Bayes-consistent**.
- **인용 부분:** Oral 명목형 손실(대안), 하이퍼파라미터 가벼운 강한 기본값.
- **주의:** peer-reviewed, 신뢰 가능.

### [7] LMFLoss — Sadi et al., 2022/2024 — 핵심 불균형 근거
- **서지:** arXiv:2212.12741 (v1 2022-12, v2 2024-09).
- **요약:** **focal + LDAM 선형결합** 손실(α, β 가중). 4개 불균형 의료 데이터셋(ODIR-5K, HAM-10K, ISIC-2019, COVID-19) × 3 백본에서 macro-F1 2–9% 개선 보고.
- **인용 부분(수치 직접 확인):** ISIC-2019/EfficientNetV2에서 macro-F1 = **CE 75.46 / Focal 71.73(CE보다 나쁨) / LDAM 78.50(CE보다 나음)**; ResNet50 = CE 68.42 / Focal 70.60 / LDAM 75.64. 원문: "focal loss가 ISIC-2019에서 저조했고 때때로 표준 CCE보다 F1이 낮았다. 반면 LDAM은 CCE 대비 개선을 보였다." → **"focal은 baseline만"** 판단의 직접 근거.
- **주의:** **preprint**(peer-review 미확인). 단 수치는 PDF 표에서 직접 확인해 신뢰 가능.

### [8] Metrics Reloaded — Maier-Hein/Reinke et al., Nature Methods 2024
- **서지:** DOI 10.1038/s41592-023-02151-z. 동반논문(pitfalls): 10.1038/s41592-023-02150-0. arXiv:2206.01653.
- **요약:** 대규모 국제 컨소시엄(Delphi)이 **"problem fingerprint"** 기반 문제 인지 지표 선택 프레임워크 구축. 동반논문이 **prevalence/클래스 불균형 의존성**을 주요 pitfall로 정리.
- **인용 부분:** prevalence-robust 지표 선택의 근거(accuracy/MCC가 prevalence 시프트에 오도됨). ※ 명시적 prevalence 경고는 **동반논문(…02150-0)**에 있으므로 두 DOI 병기.
- **주의:** peer-reviewed, 권위 있음.

### [9] DR 순서형 / QWK 논문
- **(a) arXiv:2604.17341** — "Robust DR Grading … Dual-Resolution Attention … Ordinal Regression"(2026-04-19 제출). 이중 해상도 EfficientNet + 어텐션 융합 + **CORAL cumulative-link 순서형 헤드**. **APTOS2019 검증 QWK 0.88, 외부 Messidor-2 QWK 0.68** 보고. → **2026 preprint, peer-review 아님.**
- **(b) arXiv:2602.10315** — "Uncertainty-Aware Ordinal Deep Learning for cross-Dataset DR Grading"(El Bellaj et al., Mississippi State + Univ. Rabat, 2026-02-10). **ConvNeXt-Base + lesion-query attention pooling + evidential Dirichlet 순서형 헤드**. Table II: 최종 모델 **Acc 0.876 / QWK 0.940 / Recall 0.75 / Precision 0.80**. **본문 §V에 "ordinal modeling consistently increases both recall and kappa"가 실제로 있음(확인).** 단 표의 비교군은 순서형 변형(evidential 유무)이라 "vs 비순서형"은 서술로만 제시. → **2026 preprint.**
- **(c) DR|GRADUATE — arXiv:1910.11777** — Araújo, Aresta et al., **Medical Image Analysis 2020**(DOI 10.1016/j.media.2020.300797). 불확실성 인지 순서형 DR, **QWK 0.71–0.84**(5개 데이터셋). → **peer-reviewed 앵커.**
- **QWK가 DR 표준 지표**임 — 확인(APTOS 2019 공식 지표).

---

## Part 2. Landscape (novelty)

### [10] PanDerm — Yan et al., Nature Medicine 2025
- **서지:** DOI 10.1038/s41591-025-03747-y; arXiv:2410.15038. (`SiyuanYan1/PanDerm`)
- **요약(원문 확인):** self-supervised(masked latent + CLIP 정렬)로 **"11개 기관, 4개 이미징 모달리티, 200만+ 실제 피부질환 이미지"**에 사전학습. 28개 벤치마크 평가, "라벨 10%만으로도 기존 모델 능가" 다수.
- **인용 부분:** 우리 base model. **linear probing이 full fine-tuning과 유사**하다는 점(→ 레이어/용량 질문과 직결). ※ 정확한 수치 동등성은 논문 결과 그림을 직접 인용.
- **주의:** peer-reviewed.

### [11] LogME — You et al., ICML 2021
- **서지:** arXiv:2102.11005. (`thuml/LogME`)
- **요약:** frozen feature의 **최대 라벨 evidence** 추정으로 fine-tuning 없이 모델 순위화. maximum likelihood와 달리 **"over-fitting 면역"**.
- **인용 부분(확인):** "immune to over-fitting"; **"최대 3000× wall-clock 가속, 1% 메모리"**(본문: vision 메모리 ~120×↓, NLP ~86× 가속).
- **주의:** peer-reviewed.

### [12] LEEP — Nguyen et al., ICML 2020
- **서지:** arXiv:2002.12462.
- **요약:** 소스 분류기에 타깃 데이터를 **1회 forward**해 얻는 Log Expected Empirical Prediction. NCE·H-score보다 우수.
- **인용 부분(확인):** "**small or imbalanced data에서도**" 전이 성능 예측; ImageNet→CIFAR100 "**최대 30% 개선**". ※ 소스 **분류 헤드 필요**(PanDerm 적용 시 고려).
- **주의:** peer-reviewed.

### [13] PEFTDiff — Khoba et al., ICCV 2025 — 핵심 경쟁작
- **서지:** DOI 10.1109/ICCV51701.2025.00143. 관련 WACV 2025: arXiv:2502.16471.
- **요약/확인:** 공유 백본에서 **최적 PEFT '방법' 선택**(기존 TE는 서로 다른 백본 비교용이라 실패). **diffusion maps·랜덤워크 거리**로 특징 연결성 포착("Euclidean/선형 분리 가정 너머").
- **인용 부분(축별 확인):**
  - (a) **방법 선택·레이어 아님 — 확인.**
  - (b) **9종 방법·백본 — 미확인**(저자 요약은 "19 VTAB × 9 methods"; PDF 봇 차단으로 본문 미확인, 백본은 관행상 ViT-B/16 추정).
  - (c) **"diffusion=diffusion-maps, 생성모델 아님" — 확인.**
  - (d) **PCA 논증(PEFT≠CNN 임베딩) — 직접 미확인**(일반 동기는 확인).
  - (e) **weighted Kendall τ 수치 — 미확인.**
  - → **차별 3축(우리 자리):** ① 방법만 vs 우리 레이어+방법, ② diffusion-maps vs 우리 LogME/LEEP, ③ 일반비전 ViT vs 우리 의료 FM 저자원 OOD.
  - WACV본: "Spread/Attract" 특징 섭동으로 TE 강건화, **LogME +28.84%**(확인).
- **주의:** ICCV(peer-reviewed)이나 위 (b)(d)(e)는 CVF/IEEE PDF로 직접 확인 후 단정. 그 전엔 "저자 요약 기준"으로 완충.

### [14] TE의 저자원 신뢰성
- **(a) arXiv:2204.01403** — "How stable are Transferability Metrics evaluations?" **ECCV 2022**. **715k 설정 변형** 대규모 연구, **"단일 TE 지표가 모든 시나리오에서 최선이지 않음"**(소스 데이터셋 선택엔 LogME, 아키텍처엔 N-LEEP, 타깃 이득엔 GBC). peer-reviewed.
- **(b) arXiv:2603.00478** — few-shot 전이 벤치(FEWTRANS 10개)로 **"validation set illusion"** 명시, 저자원에서 사전학습 모델 선택이 지배적. **2026 preprint.**
- **(c) Scientific Reports 2024** — Abou Baker & Handmann, DOI 10.1038/s41598-024-81752-w. **14개 TE 점수 × 11 데이터셋 × 21 모델**, weighted Kendall-τ; 효과가 데이터셋·아키텍처에 크게 좌우, **단일 최선 없음**. peer-reviewed.
- **인용 부분:** 저자원·few-shot에서 TE 결론이 불안정·맥락 의존 → LogME/LEEP를 맹신 말고 우리 도메인에서 검증해야 한다는 근거.

### [15] 음성 LogME 레이어 선택 — Chen et al., Interspeech 2023
- **서지:** arXiv:2306.01015.
- **요약:** 사전학습 음성 모델의 **cross-layer/cross-model** 전이성을 fine-tuning 없이 점수화(LogME 변형 + OT).
- **인용 부분(확인):** LogME로 레이어 선택, **"correlation 0.87, p=6×10⁻⁶"**(sliced-Wasserstein 변형 0.81, overall Spearman 0.94). → 우리 P1 **레이어 축** 선례.
- **주의:** peer-reviewed.

### [16] LLM PEFT 레이어 배치 — arXiv:2602.04019 (2026)
- **요약:** projected residual 관점 + per-layer 진단 **"Layer Card"**(projected residual norm·activation energy·layer coupling). Qwen3-8B에서 선택 레이어만 적응해도 full-layer LoRA에 근접(저비용).
- **인용 부분:** "어느 층을 적응" 축의 개념적 지지(단 LLM 기반).
- **주의:** **2026 preprint.**

### [17] 도메인 거리 지표
- **CKA — Kornblith et al., ICML 2019.** 정규화 HSIC, 직교/등방 스케일 불변. **축: 레이어.** 한계: 표현 유사도지 전이성 아님, 커널 선택 민감.
- **OTDD — Alvarez-Melis & Fusi, NeurIPS 2020.** 라벨 인지 OT 데이터셋 거리(라벨을 특징 분포로). **축: 데이터셋(소스↔타깃).** 한계: 대용량 무거움.
- **s-OTDD — arXiv:2501.18901 (2025).** sliced-OT, 근사 선형 복잡도·클래스 수 독립·disjoint 라벨 처리. **축: 데이터셋.** 한계: MC 근사, **preprint.**
- **Task2Vec — Achille et al., ICCV 2019.** 공유 probe의 Fisher 대각으로 태스크 임베딩. **축: 태스크.** 한계: probe 필요, 태스크 단위(레이어 아님).
- **인용 결론: CKA(레이어) + OTDD/s-OTDD(데이터셋)**, Task2Vec 보조.

### [18] 소표본 평가 불안정 — An et al., PLOS ONE 2021 (정정)
- **서지:** C. An, Y. W. Park, S. S. Ahn, K. Han, H. Kim, S.-K. Lee, PLOS ONE 16(8):e0256152 (2021). PMC8360533.
- **요약:** 뇌종양 MRI radiomics로 **1,000회 무작위 train/test 분할**, 분할마다 AUC 변동 측정. 예: 어려운 태스크에서 한 분할 train 0.882/test 0.667, 다른 분할 train 0.709/test 0.911. 간극이 크면 어떤 검증법도 충분히 못 줄임.
- **인용 부분:** 단일 무작위 분할은 신뢰 불가 → 우리 소표본(Oral ~54, OLP ~10) 성능 기반 선택 불가의 앵커.
- **정정·주의:** 저자는 **An et al.**(Kim & Cho 아님). peer-reviewed.

### [19] 의료 FM의 OOD 저하
- **MFM-DA — arXiv:2503.00802** (2025-03). few-shot UDA: DDPM으로 타깃 스타일 변환 후 채널-공간 정렬 LoRA. **RETFound(망막 FM)가 fine-tuning 후에도 도메인 시프트에서 성능 저하** 실증. **preprint.**
- **FM-CT — arXiv:2502.02779.** head-CT 3D FM, **361,663 스캔** 사전학습. **소규모 타깃 CQ500(~1,120)에서 in-domain fine-tuning이 전이보다 저조**, 특히 극불균형 하위유형(EDH/SDH). 저널본: Nature Biomedical Engineering(DOI 10.1038/s41551-026-01668-w) 존재(arXiv는 preprint).
- **인용 부분:** "저자원·불균형 OOD에서 in-domain fine-tuning이 전이보다 나쁠 수 있다"는 우리 동기.

---

## 참고: 인용 시 원칙 (내 메모)

1. **preprint는 preprint로 표기.** 2026 DR 2건, LMFLoss, FEWTRANS, LLM-PEFT-layer, MFM-DA. CORN·FM-CT는 저널본 인용.
2. **PEFTDiff 미확인 4항(9종·백본·PCA·τ)은 "저자 요약 기준"으로 완충** 후, CVF/IEEE PDF 확보 시 단정으로 승격.
3. **2602.10315의 recall·kappa 주장**은 본문 §V 근거로 인용하되, 표 비교군이 순서형 변형임을 병기.
4. **Cui 95%/5% 의료 예시 인용 금지**(논문 미포함).
5. **An et al.**(Kim & Cho 아님).
