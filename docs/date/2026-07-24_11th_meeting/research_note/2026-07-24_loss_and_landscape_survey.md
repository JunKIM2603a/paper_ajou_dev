# 연구노트 — 손실함수 & 논문 Landscape 서베이 (2026-07-24)

> 이 문서는 **내 연구노트(작업용)**다. W11 보고용(§2 P3·§4)에 넣은 방법·레퍼런스의 상세 근거를 여기 모아둔다.
> 교수님 지시 대응: **W8-1**("multi-class 불균형, 관련 연구 찾아보고 weighted CE 외 방법 시도") · **W9-2**("focal loss, 순서형 특성 살린 손실 고려") · **W10**("방법론을 관련 최신 연구·레퍼런스 기반으로").
> ⚠️ preprint 수치는 인용 전 peer-review 여부 확인. PEFTDiff 세부 수치는 CVF PDF에서 직접 재확인 필요(자동 fetch 차단됨).
> 📎 **논문별 상세 검증·서지·인용 근거**는 [근거자료 — 참고논문 검증 & 요약](./2026-07-24_reference_evidence.md) 참조.

---

## 0. 한 줄 요약 (내 판단)

- 손실은 **태스크별로 다르게 간다**: aptos(순서형 0~4)는 **CORAL/CORN + QWK**, Oral(명목형 7클래스)은 **LDAM-DRW / logit adjustment**. **focal은 baseline으로만** 쓴다 — 의료 다중분류에서 이득이 불안정하다는 근거가 반복적으로 나온다.
- 소표본에서 손실 교체 이득은 작고 들쑥날쑥하다 → **CI + 다중 시드 없이는 "개선"이라고 쓰지 않는다.**
- 내 novelty는 PEFTDiff와 **3축에서 다르다**(레이어까지 선택 / LogME·LEEP 사용 / 의료 FM 저자원 OOD). TE 자체의 저자원 신뢰성은 아직 덜 검증된 영역이라, 이게 하위 기여가 된다.

---

## 1. 불균형 손실함수 (태스크별 선택)

### 1-1. Focal loss — baseline로만 쓴다
- **Lin, Goyal, Girshick, He, Dollár, "Focal Loss for Dense Object Detection," ICCV 2017** (arXiv:1708.02002). CE에 (1−p_t)^γ modulating factor를 붙여 쉬운 샘플 기여를 낮추고 어려운 샘플에 집중. 원래 **객체탐지의 전경-배경 불균형**용이지 다중분류용이 아니다 — 이 점은 논문에 명시할 것.
- **Cui et al., "Class-Balanced Loss Based on Effective Number of Samples," CVPR 2019.** 유효 샘플 수 E_n=(1−β^n)/(1−β)로 클래스 재가중(**class-balanced focal**). 논문 본문·Fig.1은 **iNaturalist 롱테일**을 예시로 삼는다. **이득이 극심한 불균형에서만 크고 완만하면 급감** → Oral 극소수 클래스(OLP)엔 유효, 덜 치우친 클래스엔 미미할 것으로 예상. ⚠️ 흔히 인용되는 "의료 95%/5%" 예시는 **논문에 없다**(발표 슬라이드 전용) → **인용하지 않는다**.
- ⚠️ **한계 근거(정직하게 넣을 것):** *LMFLoss* (arXiv:2212.12741)에서 focal이 ISIC-2019에서 **일반 CE보다 F1이 낮게 나온** 경우 보고. 큰 백본(EfficientNetV2/ResNet50)에서 CE·focal 모두 저조. PLOS ONE(journal.pone.0261307)도 "단일 손실이 지배적이지 않고 CE가 강한 baseline"이라는 결론.
- **내 결론:** focal은 γ/α 튜닝·오라벨에 민감. baseline으로 CI와 함께 보고, 헤드라인 손실로 쓰지 않는다.

### 1-2. 순서형 손실 (aptos DR 0~4에 직결) — 근거 가장 강함
- **CORAL — Cao, Mirjalili, Raschka, "Rank consistent ordinal regression…," Pattern Recognition Letters 2020** (arXiv:1901.07884; `coral-pytorch`). K등급을 K−1개 "등급 초과?" 이진 subtask로 바꾸고 **가중치 공유로 순위 단조성(consistency) 보장**. 추가 학습비용 없음. 순위 불일치 문제를 이론적 보장으로 해결.
- **CORN — Shi, Cao, Raschka, "…Rank-Consistent Ordinal Regression Based on Conditional Probabilities"** (arXiv:2111.08851). CORAL의 가중치 공유 제약을 조건부확률(chain rule)로 제거 → **CORAL 대비 성능 상당 개선**.
- **DR 특화 근거:**
  - CORAL cumulative-link 헤드로 **aptos2019 검증셋 QWK 0.88**(외부 Messidor-2 0.68) 보고 — arXiv:2604.17341.
  - 불확실성 인지 순서형 DR 연구: 순서형 모델이 "recall과 kappa를 일관되게 개선" — arXiv:2602.10315(**2026 preprint**, 본문 §V 확인. ⚠️ 단 Table II 비교군은 순서형 변형이라 "vs 비순서형"은 서술로만 제시).
  - QWK가 DR 표준인 이유: 예측-정답 등급 거리를 **제곱으로 벌함**. accuracy/F1/AUC는 순서성과 오차 크기를 무시. (DR|GRADUATE arXiv:1910.11777; QWK transfer/ensemble PMC11323616.)
- ⚠️ QWK 0.88~0.9대 최고 수치는 **preprint** 다수. 인용 전 검증.
- **내 결론:** aptos는 CORAL부터, 여유 되면 CORN. 주 지표 QWK.

### 1-3. Margin / logit-adjustment 손실 (Oral 7클래스 명목형에 적합)
- **Menon et al., "Long-tail learning via logit adjustment," ICLR 2021.** 로짓을 라벨 빈도 로그로 조정(post-hoc 또는 손실 내). 희귀 vs 다수 클래스 간 상대 마진 확보. **Bayes-consistent**, 하이퍼파라미터 가벼움 → 강한 기본값.
- **LDAM-DRW — Cao, Wei, Gaidon, Arechiga, Ma, NeurIPS 2019** (arXiv:1906.07413; `kaidic/LDAM-DRW`). 클래스별 마진 **Δ_j = C/n_j^{1/4}**(희귀할수록 큰 마진) + 재가중 지연(DRW). 원논문: "LDAM이 순수 CE 및 그 변형인 **focal loss보다 우수**하다(재밸런싱 스케줄 없이도)."
- **의료 근거:** LMFLoss에서 **LDAM은 CE 대비 개선, focal은 아님** → Oral 주력 손실로 LDAM-DRW/logit adjustment가 focal보다 낫다는 직접 근거.
- **내 결론:** Oral 주력은 LDAM-DRW 또는 logit adjustment. class-balanced 재가중(Cui)은 ablation.

### 1-4. 샘플링 · 데이터 증강
- 서베이: *Artificial Intelligence Review* 2024 (10.1007/s10462-024-10884-2); *Sensors* 2025 (10.3390/s26061998). GAN 합성: *Pattern Recognition* 2025 (S0031320325003401).
- 트레이드오프: 오버샘플링>언더샘플링(의료, PMC11300732)이나 단순 복제는 과적합 위험. SMOTE/ADASYN은 소수 다양성↑이나 비현실적 특징 삽입 가능. GAN은 고품질이나 계산·임상타당성·규제 이슈.
- **내 결론(중요):** OLP ≈ 10장 규모에선 **GAN 합성 금지**(54장으로 합성 품질 검증 불가). **클래스 균형 샘플러 + 보수적·임상적으로 타당한 증강**(플립·소각도 회전·약한 대비)이 안전. Cui class-balanced 재가중은 극단 클래스에만 이득 기대.

### 1-5. 지표
- **Metrics Reloaded — Maier-Hein, Reinke et al., Nature Methods 2024** (10.1038/s41592-023-02151-z) + companion pitfalls (10.1038/s41592-023-02150-0). prevalence에 따라 accuracy·MCC가 크게 왜곡됨을 실증 → prevalence-robust 지표 선택 근거.
- **내 결론:** **DR = QWK**(순서형·표준), **Oral = macro AUPRC + balanced accuracy**(명목·극불균형·소표본), 보조로 macro-F1·per-class recall. 소표본이라 **항상 bootstrap CI 병기**. (교수님 W9-1 지시와 일치.)

---

## 2. 논문 Landscape (novelty 확인)

### 2-1. PanDerm 자체
- Yan, Yu, Primiero et al., "A multimodal vision foundation model for clinical dermatology," **Nature Medicine 2025**, 31(8):2691–2702 (10.1038/s41591-025-03747-y; arXiv:2410.15038; `SiyuanYan1/PanDerm`). 4개 피부과 모달리티 ~2.1M 이미지, masked-latent + CLIP 정렬. **linear-probing이 full fine-tuning과 유사**하다고 보고(Supp. Table 2) → 내 레이어/용량 질문과 직결. ViT-base 변형 존재. **fundus·oral 전이 근거는 없음** → 그 미지의 거리가 내 실증 기여.

### 2-2. 의료 FM의 OOD 전이
- RETFound(망막 FM)도 다운스트림 도메인 시프트에서 성능 저하 — **MFM-DA** arXiv:2503.00802.
- 3D head-CT FM: 소규모 타깃(CQ500 1,120장)에서 **in-domain fine-tuning이 전이보다 저조**, 특히 극불균형 클래스 — arXiv:2502.02779.
- → 내 전제 뒷받침: 의료 FM 전이는 시프트에 민감하고, 저자원 OOD에서 **적응 전략이 결정적**.

### 2-3. TE for layer/PEFT 선택 (핵심 차별점)
- **LogME — You et al., ICML 2021** (arXiv:2102.11005; `thuml/LogME`). frozen feature의 log maximum evidence 추정, "over-fitting 면역", brute-force 대비 최대 3000× 속도·1% 메모리. PanDerm 특징에 바로 적용 가능.
- **LEEP — Nguyen et al., ICML 2020** (arXiv:2002.12462). forward 1회. 저자: "small or imbalanced data에서도" 전이 성능 예측 가능(ImageNet→CIFAR100 최대 30% 개선). 단 자연이미지 벤치 검증.
- **PEFTDiff — Khoba et al., "Diffusion-Guided Transferability Estimation for PEFT," ICCV 2025** (10.1109/iccv51701.2025.00143). **가장 가까운 경쟁작.**
  - 문제: PEFT **'방법' 9종 중** 최적을 fine-tuning 없이 선택(백본 고정). **레이어는 안 고름.**
  - 메커니즘: "diffusion"=**diffusion-maps(특징 매니폴드 랜덤워크 거리)**, 생성 모델 아님. PCA로 "PEFT는 CNN과 임베딩 구조가 근본적으로 다르다" → LogME/LEEP 판별력 저하 주장.
  - 평가: **VTAB 19개 × ViT**, weighted Kendall τ 보고. VTAB Specialized에 의료인접(Camelyon·DR) 포함되나 **일반비전 논문**.
  - **차별 3축(내 자리):** ① 방법만 vs **나는 레이어+방법 동시**, ② diffusion-maps vs **나는 LogME/LEEP**, ③ 일반비전 ViT vs **나는 의료 FM 저자원 OOD**.
  - ⚠️ 세부 τ_w·9종 목록은 CVF PDF/IEEE에서 직접 확인(자동 fetch 차단).
  - 인접: Khoba et al. WACV 2025 (arXiv:2502.16471); Wang et al. ICCV 2023(neural collapse & transferability).

### 2-4. TE의 저자원 신뢰성 (내 빈틈)
- "How stable are Transferability Metrics evaluations?" arXiv:2204.01403 — TE 랭킹 불안정.
- "Benchmarking Few-shot Transferability… Improved Evaluation Protocols" arXiv:2603.00478 — "validation-set illusion", 소표본 단일 샘플 변동 큼.
- "One size does not fit all…," **Scientific Reports 2024** (s41598-024-81752-w) — TE 점수 효과가 데이터셋·아키텍처에 크게 좌우; 일부는 저자원에서 오히려 더 나음. → **TE는 큰 벤치서 검증됐고 정작 필요한 저자원에선 덜 검증**이라는 내 프레이밍 근거.

### 2-5. 도메인 거리 지표 (H2-층의 x축)
- **CKA — Kornblith et al., ICML 2019.** Gram 행렬 정규화 HSIC, 0~1, 직교/등방 스케일 불변. **레이어 축에 적합**·저렴. 단 매칭 샘플 필요, 高 CKA≠전이성 보장 → 표현 신호로만.
- **OTDD — Alvarez-Melis & Fusi, NeurIPS 2020.** 라벨 인지 OT 거리, 서로 다른 라벨셋도 비교, "전이성 예측력 높음". 무거움 → **s-OTDD**(sliced, arXiv:2501.18901)로 스케일. 소표본 추정 분산 큼.
- **Task2Vec — Achille et al., ICCV 2019.** Fisher 정보 기반 태스크 임베딩 거리. 태스크 단위·보조용, CKA보다 거칠음. (FID/MMD는 단순하나 라벨 무시.)
- **내 결론: CKA(레이어) + OTDD/s-OTDD(데이터셋) 조합**, Task2Vec 보조.

### 2-6. 소표본 평가 불안정 (핵심 문제 진술)
- **An et al., "Radiomics ML with a small sample size…," PLOS ONE 2021** (PMC8360533). 뇌종양 MRI 1,000회 분할: train/test AUC가 분할마다 크게 변동, 소표본·어려운 태스크일수록 심함 → "단일 분할은 신뢰 불가, nested CV/bootstrap 권장." **내 앵커 인용.** 보완: "Externally validated yet undertrained…"(feature-selection 방법 간 일관된 우위 없음).

---

## 3. 논문 반영 판단 (내 의견 · 다음 단계)

1. **손실 단계적 적용:** weighted CE·focal은 baseline만. aptos→CORAL(→CORN), Oral→LDAM-DRW/logit adjustment. **벤치마크 기준:** 순서형이 QWK를 CI 폭 이상으로 못 올리면 CE+증강으로 회귀하고 null 정직 보고.
2. **샘플링:** 균형 샘플러 + 보수적 증강. GAN 금지. class-balanced 재가중은 ablation.
3. **지표 사전 고정:** DR=QWK, Oral=macro AUPRC+balanced accuracy, bootstrap CI 필수.
4. **novelty 서술:** "의료 FM 저자원 OOD에서 LogME/LEEP로 레이어+PEFT 동시 선택" + 하위 기여 "저자원 TE 신뢰성 감사"(arXiv:2204.01403 / 2603.00478 / Sci Rep 2024).
5. **도메인 거리(H2):** CKA+OTDD로 x축 구성, 최적층/최적방법과 상관. **기준:** 3개 도메인에서 |ρ|<0.5면 H2는 확증이 아니라 탐색적으로 표기하고 중간거리 도메인(병리 등) 추가 검토.

---

## 참고문헌 (링크)

**손실·불균형·지표**
1. Lin et al., Focal Loss, ICCV 2017 — arXiv:1708.02002
2. Cui et al., Class-Balanced Loss, CVPR 2019 — DOI 10.1109/CVPR.2019.00949
3. Cao/Mirjalili/Raschka, CORAL, Pattern Recognition Letters 2020 — arXiv:1901.07884
4. Shi/Cao/Raschka, CORN — arXiv:2111.08851
5. Cao et al., LDAM-DRW, NeurIPS 2019 — arXiv:1906.07413
6. Menon et al., Logit Adjustment, ICLR 2021
7. LMFLoss (focal 한계 근거) — arXiv:2212.12741
8. Maier-Hein/Reinke et al., Metrics Reloaded, Nature Methods 2024 — 10.1038/s41592-023-02151-z (+ pitfalls 10.1038/s41592-023-02150-0)
9. DR CORAL QWK 0.88 — arXiv:2604.17341 · 순서형 DR — arXiv:2602.10315 · DR|GRADUATE — arXiv:1910.11777
10. 불균형 서베이 — 10.1007/s10462-024-10884-2 · 10.3390/s26061998

**Landscape**
11. Yan et al., PanDerm, Nature Medicine 2025 — 10.1038/s41591-025-03747-y · arXiv:2410.15038
12. You et al., LogME, ICML 2021 — arXiv:2102.11005
13. Nguyen et al., LEEP, ICML 2020 — arXiv:2002.12462
14. Khoba et al., PEFTDiff, ICCV 2025 — 10.1109/iccv51701.2025.00143 (+ WACV 2025 arXiv:2502.16471)
15. TE 신뢰성 — arXiv:2204.01403 · arXiv:2603.00478 · Sci Rep 2024 s41598-024-81752-w
16. CKA — Kornblith et al., ICML 2019 · OTDD — Alvarez-Melis & Fusi NeurIPS 2020 · s-OTDD arXiv:2501.18901 · Task2Vec — Achille et al. ICCV 2019
17. An et al., 소표본 불안정, PLOS ONE 2021 — PMC8360533
18. 의료 FM OOD — MFM-DA arXiv:2503.00802 · 3D CT FM arXiv:2502.02779
