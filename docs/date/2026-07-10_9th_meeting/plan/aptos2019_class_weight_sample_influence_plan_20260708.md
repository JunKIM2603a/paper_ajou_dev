# Plan: aptos2019 클래스 가중치 적용 & 샘플 영향도 분석 노트북

## Context

9차 미팅 체크리스트의 첫 번째 우선순위 항목(②)은 다음 두 가지다:
1. Linear Evaluation에 클래스 가중치 손실(Weighted CE) 적용 → 기존 baseline과 비교
2. aptos2019에 대해 Oral_Diseases와 동일한 방법론(LOO 샘플 영향도 분석)을 적용

사용자는 `PanDerm/` 내부의 기존 코드는 전혀 수정하고 싶지 않고, 대신 새 `.ipynb`를 만들어 실험하면서 마크다운으로 결과를 함께 보고하길 원한다. 이는 이미 존재하는 `notebooks/PanDerm/Linear Evaluation/panderm_sample_influence_oral_diseases_20260702.ipynb`의 스타일(코드 셀 다음에 "결과 해석" 마크다운 셀)을 그대로 따르는 것과 일치한다.

## 조사로 확인된 핵심 사실

- **PanDerm/ 는 벤더링된 fork**(origin: JunKIM2603a/PanDerm, upstream: SiyuanYan1/PanDerm). 라이브러리로만 import해서 쓸 것 — 파일 수정 없음.
- **aptos2019 baseline 수치(Accuracy 0.814 / BAcc 0.628 / Macro AUPR 0.679 / Severe Recall 0.37 / Proliferative Recall 0.44)의 출처**: `notebooks/PanDerm/Linear Evaluation/panderm_linear_eval_analysis_20260628.ipynb` (+ 렌더된 `docs/eda/panderm_linear_eval_analysis.md`). 이 노트북의 322번째 줄 부근에 이미 `criterion = nn.CrossEntropyLoss(weight=class_weights)` 주석(계획만 있고 미구현)이 있다 — 이번 작업이 그걸 실제로 구현하는 것.
- **aptos2019 메타데이터**: `PanDerm/data/aptos2019/Linear Evaluation/aptos2019_multiclass.csv` (컬럼: `image,label,split`) — train 2562 / val 546 / test 554. 라벨맵(`aptos2019_label_map.csv`): 0=no_dr, 1=mild, 2=moderate, 3=severe, 4=proliferative_dr.
- **기존 예측 결과**: `PanDerm/output_dir/aptos2019_panderm_large_lp/aptos2019_multiclass.csv` — 이미지별 예측 확률(5클래스)까지는 있지만 **원본 1024차원 backbone feature는 저장되어 있지 않음** → 가중치 재학습과 LOO를 하려면 feature를 새로 추출해야 함.
- **Feature 캐시**: Oral_Diseases는 `PanDerm/output_dir/oral_diseases_features_cache/*.npy`로 캐시되어 있지만, **aptos2019는 캐시가 없음** — 새로 추출 필요.
- **⚠️ 블로커 — 체크포인트 파일 없음**: `PanDerm/checkpoint/`에는 `info.txt`(Google Drive 링크)만 있고 실제 `panderm_ll_data6_checkpoint-499.pth`는 이 머신에 없음(`.gitignore`에 의해 제외됨, 저장소/홈 디렉터리 전체 검색해도 없음). **Feature 추출은 이 체크포인트 없이는 불가능** — 노트북 실행 전에 사용자가 `info.txt`에 적힌 Google Drive 링크에서 다운받아 `PanDerm/checkpoint/panderm_ll_data6_checkpoint-499.pth`에 위치시켜야 한다. (다운로드는 내가 대신 할 수 없음 — 브라우저로 받아야 하는 외부 리소스)
- **Linear Probe 라이브러리 코드**: `PanDerm/classification/panderm_model/downstream/eval_features/logistic_regression.py`의 `LogisticRegression` 클래스 — `torch.nn.CrossEntropyLoss()`(가중치 없음), LBFGS로 `fit()`. **이 파일은 수정하지 않고**, 노트북에서 이 클래스를 상속한 `WeightedLogisticRegression`을 새로 정의해서 가중치 손실만 오버라이드.
- **Feature 추출 재사용 패턴** (Oral 노트북 그대로 재사용): `from datasets.derm_data import Derm_Dataset`, `from models import get_encoder`, `from panderm_model.downstream.extract_features import extract_features_from_dataloader`.
- **⚠️ LOO 참조 코드가 실제로는 존재하지 않음**: Oral_Diseases LOO 노트북은 `from panderm_model.downstream.sample_influence import run_loo_analysis`를 import하지만, **`sample_influence.py`는 저장소 어디에도 (git 이력 포함) 존재하지 않는다** — 즉 그 노트북은 현재 그대로 재실행이 불가능한 상태(죽은 import)다. "동일한 방법론 적용"은 이 파일을 그대로 재사용하는 게 아니라, **그 노트북의 마크다운에 서술된 방법론(baseline 학습 → 샘플 1개씩 제거 후 재학습 → BAcc/클래스별 recall 변화量 = influence)을 새 노트북 안에 자체적으로(self-contained) 재구현**하는 것을 의미한다. PanDerm 트리에 새 파일을 추가하지 않고 노트북 셀 안에 함수로 정의한다.
- **이산(discrete) recall 기반 influence의 한계**: Oral 분석에서 테스트 샘플이 적은 클래스(OLP=10개)는 recall이 0.1 단위로만 움직여 407개 중 389개(96%)의 influence가 정확히 0으로 나오는 문제가 있었음. aptos2019도 test 세트에서 Severe=30, Proliferative=45로 표본이 적어 **동일한 양자화 문제가 처음부터 발생할 것** → 이번엔 "연속 확률 기반 influence"(정답 클래스에 대한 평균 예측확률 변화)를 사후 patch가 아니라 **처음부터 기본 지표로 설계**.
- **규모 차이**: Oral LOO는 407회 재학습, aptos2019 train은 2562개 → 풀 LOO는 약 6배 느림. Severe(135)+Proliferative(206) = 341개로 좁히면 Oral과 비슷한 규모.
- **문서 컨벤션**: `docs/date/*/` 아래에는 `notebooks/`나 `analysis/` 서브폴더가 없음. 실제 노트북은 `notebooks/PanDerm/Linear Evaluation/`에, 렌더된 마크다운 보고서는 `docs/eda/`에 위치하고, 미팅 minutes에서 상대경로로 링크한다(예: 8차 minutes → `panderm_sample_influence_oral_diseases_20260702.ipynb`). 이번에도 동일 컨벤션을 따른다.

## 실행 결정 (기본값 — 실행 전 확인 필요, `AskUserQuestion` 도구가 이번 세션에서 일시적으로 오류가 나서 직접 못 여쭤봤습니다)

1. **체크포인트**: 사용자가 `PanDerm/checkpoint/info.txt`의 Google Drive 링크에서 `panderm_ll_data6_checkpoint-499.pth`를 받아 `PanDerm/checkpoint/`에 두는 것을 전제로 진행. (이미 다른 경로에 있다면 그 경로로 대체)
2. **LOO 범위**: **풀 LOO(2562회) + 소수 클래스(Severe/Proliferative) 집중 딥다이브**를 기본으로 제안. 시간이 너무 오래 걸리면(1차 100회 실행 후 추정 시간 보고) 소수 클래스 집중 분석만으로 축소하는 옵션을 노트북 내 마크다운에 명시.
3. **LOO 기준 모델**: Oral_Diseases 선례와의 직접 비교를 위해 **가중치 미적용 baseline 모델 기준**으로 LOO 수행(선례와 동일 조건 유지).

## 새 노트북 구조

**파일**: `notebooks/PanDerm/Linear Evaluation/panderm_class_weight_sample_influence_aptos2019_20260708.ipynb`
(Oral 노트북과 동일하게 "코드 셀 → 결과 해석 마크다운 셀" 패턴 반복, 한글 마크다운 설명 포함)

1. **0. 설정**: 경로, `CLASS_NAMES=[no_dr, mild, moderate, severe, proliferative_dr]`, `CHECKPOINT`, 캐시 디렉터리 `PanDerm/output_dir/aptos2019_features_cache/` 신규 생성.
2. **1. Feature 추출 & 캐싱**: Oral 노트북의 `Derm_Dataset` + `get_encoder` + `extract_features_from_dataloader` 패턴을 aptos2019 메타 CSV로 그대로 재사용, train/test(및 val) feature를 `.npy`로 캐시.
3. **2. Baseline 재현(검증용)**: 미수정 `LogisticRegression`으로 학습·평가 → `docs/eda/panderm_linear_eval_analysis.md`의 기존 수치(0.814/0.628/0.679/0.37/0.44)와 비교해 파이프라인이 정확히 일치하는지 먼저 검증(sanity check).
4. **3. 클래스 가중치 비교**: `WeightedLogisticRegression(LogisticRegression)` 서브클래스를 노트북에서 정의(`compute_class_weight("balanced", ...)`로 가중치 계산, `loss_func`만 교체) → 학습·평가 → 미팅 문서 2-1 표(Accuracy/BAcc/Macro AUPR/Severe Recall/Proliferative Recall, 가중치 전후) 그대로 생성 + 전체 클래스별 recall/AUPR 비교 그래프, confusion matrix 2개.
5. **4. 샘플 영향도(LOO) 분석**: 
   - 마크다운으로 "원본 `run_loo_analysis` import가 실제로는 존재하지 않아 방법론을 자체 재구현했다"는 점 명시.
   - 연속 확률 기반 influence를 기본 지표로 사용.
   - 전체 2562개 대상 aggregate 분포(히스토그램, top-10 helpful/harmful 표+이미지 그리드) — Oral 리포트와 동일 구성.
   - Severe/Proliferative 집중 딥다이브(OLP 분석과 동일한 형태) + 클래스간 influence 히트맵.
   - 결과를 `.npz`로 캐시(재실행 시 즉시 로드).
6. **5. 종합 요약**: Oral 노트북의 "쉽게 풀어서 설명" 스타일 마크다운 + 9차 미팅 minutes의 "2-1 표"에 바로 붙여넣을 수 있는 최종 결과 표.

## 검증 방법

- `PanDerm` conda 환경에서 노트북을 처음부터 끝까지 실행.
- 2번 섹션(baseline 재현) 수치가 `docs/eda/panderm_linear_eval_analysis.md`의 기존 값과 거의 일치하는지 확인 — 불일치 시 feature 추출/전처리 로직을 먼저 디버깅.
- LOO 실행 전 100개 샘플만으로 1차 실행해 예상 소요 시간을 추정해 마크다운에 기록.
- 최종적으로 모든 셀이 에러 없이 실행되고, 그래프/표가 기대한 형태로 출력되는지, 캐시 파일들이 `PanDerm/output_dir/`에 정상 저장되는지 확인.
- (범위 밖) `docs/date/2026-07-10_9th_meeting/minutes/...md`의 TBD 값 채우기·체크박스 갱신은 노트북 결과가 나온 후 사용자가 원하면 별도로 진행.
