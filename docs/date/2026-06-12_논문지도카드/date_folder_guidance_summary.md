# 날짜별 논문 지도내용 요약

## 지도일: 2026-03-19 / 2026-03-25
### 2026-03-19_1st_meeting_kick-off

* 논문제목
- 의료 영상 분류 모델 비교 및 설명가능성 분석

* 지도내용
    * 논문 계획 검토
        * 기존 연구 계획인 '안저 영상 분류 모델 비교 및 설명가능성 분석' 진행 시 예상되는 기술적 한계점 및 난이도 분석 보고
        * 교수님과의 피드백을 통해 유방암 및 폐 CT 데이터셋을 대안으로 제안받아 데이터 도메인 변경 검토
    * 논문 방향성
        * 설명 가능한 AI(XAI) 방향성 수립
            * 모델 분류 성공/실패 샘플(Top-k 및 Error Analysis) 간의 비교 분석 진행 필요
            * 어텐션(Attention) 기반 시각화 기법과 전통적 XAI 방법론 간의 정성적·정량적 차이 분석 방향 검토 필요
    * 다음 회차 연구 계획 및 지시사항
        * 베이스라인 및 벤치마크 환경 구축
            * 유방암 데이터셋 기반 DINOv2 모델의 초기 파이프라인 구동 및 기본 성능 지표(Accuracy, Precision, Recall, F1-score, AUC-ROC) 확보
            * CNN, ImageNet 기반 사전학습 모델, Vanilla ViT 계열의 특성 조사 및 비전 베이스라인(Vision Baseline) 선정
        * 최신 문헌 조사 및 타당성 검토
            * 메디컬 이미지 특화 모델 및 최신 비전 파운데이션 모델(Vision Foundation Model) 선행 연구 조사 (GitHub 소스코드 및 인용 수가 확보된 대표 논문 2~3개 엄선)
            * Google Scholar를 통해 대상 데이터셋 기준 기존 연구의 소타(SOTA) 성능 수준 및 대표 아키텍처 파악

### 2026-03-25_after_class

* 논문제목
- ISIC2024 이미지-메타데이터 멀티모달 피부 병변 분류

* 지도내용  
    * 연구 주제 및 방향성 전환
        * 연구 주제 선정 어려움을 겪는 상황 보고
        * 연구의 핵심 초점을 단일 모달 중심의 XAI에서 이미지(Vision)와 정형 메타데이터(Tabular Metadata)를 결합하는 다중모달(Multimodal) 융합 네트워크 구조로 연구 방향 전환 설정
    * 다음 회차 연구 계획 및 지시사항
        * 정형 데이터 분석 및 단일 모달 기준선(Baseline) 수립
            * Tabular 데이터 대상 탐색적 데이터 분석(EDA) 선행: 정답 레이블 및 주요 메타데이터 피처의 분포 특성 파악
            * 정형 데이터 단일 머신러닝/딥러닝 기준선 확보: XGBoost, CatBoost, SVM, Logistic Regression, MLP 간의 성능 비교 평가 설계
            * 이미지 단일 기준선 확보: 백본 네트워크의 미세조정(Fine-tuning) 및 동결(Freeze) 전략별 비교 실험 조건 통일
        * 다중모달 융합(Multimodal Fusion) 메커니즘 검토
            * 단순 연결(Simple Concatenation) 방식과 고도화된 융합 방법론(Advanced Fusion Methods)을 대조군으로 설정하여 성능 개선 가능성 검토

## 지도일: 2026-04-02
###  2026-04-02_2nd_meeting

* 논문제목
- ISIC2024 train metadata와 lesion image 기반 멀티모달 피부 병변 악성 여부 분류

* 지도내용  
    * 데이터셋 선정 보고
        * ISIC2024 kaggle dataset 선정 및 EDA 보고
        * test dataset sample 부족으로 train dataset 만 사용하기로 결정
        * 불균형 데이터에 따른 target sample 부족 보고
        * EDA 분석이 부족했다고 생각되어 다음 회차에 다시 EDA 분석 보고하기로 함
        * 교수님께서 Dataset 연구논문을 레퍼런스 논문으로 제시 ("The SLICE-3D dataset: 400,000 skin lesion image crops extracted from 3D TBP for skin cancer detection")  
    * 다음 회차 연구 계획 및 지시사항
        * 연구 가설 정의 
            * 명확한 연구 목적, 핵심 가설, 학술적 기여점(Contribution)을 선제적으로 정의한 후 실험 프레임워크 구조화
        * 최신 연구 분석
            * 레퍼런스 논문 분석
            * 도메인 특화 선행 연구의 불균형 데이터 처리 기법 검색
            * 공개 코드 테스트

## ~~지도일: 2026-04-09~~
### 2026-04-09_논문계획서_제출

1. 논문제목
- ISIC2024 이미지 및 strict metadata 기반 피부 병변 악성 여부 분류와 train-only privileged supervision 후보

2. 지도내용
- 연구 문제는 melanoma-only 분류가 아니라 `Malignant` vs `Benign/Indeterminate` 악성 여부 이진 분류로 명확히 수정한다.
- `iddx_full` 자체는 전체 row에 존재하므로 희소성 근거로 쓰지 않는다. 핵심 난점은 malignant 양성 클래스의 극단적 희소성과 깊은 병리 계층 정보의 희소성이다.
- Methods에는 patient-level internal split, Strict/Relaxed/Oracle feature regime, train/validation/internal test 역할을 명확히 적는다.
- `iddx_full`, 진단 텍스트, pathology-derived field는 ordinary inference input이 아니며, 사용한다면 train-only privileged supervision 또는 candidate experiment로만 표시한다.
- pAUC 계산 구간, fold-wise mean/std/min, validation 기반 threshold 선택 규칙을 명시한다.
- CatBoost/LightGBM 같은 강한 tabular baseline을 보조 기준선으로 포함하고, correction head 또는 prototype 구조는 수식이나 도식으로 고정한다.

## 지도일: 2026-04-16
### 2026-04-16_3rd_meeting

* 논문제목
- ISIC2024 patient-level protocol 기반 image-tabular multimodal 악성 여부 분류

* 지도내용
    * 제출한 논문계획서 리뷰
        * 직접하는 피처엔지니어링 지양: 딥러닝이 아닌 방법으로 도메인지식 넣지말라고 지시
        * Imbalance 극복, 성능 향상, 기타 기여 요소 등에서 중심 목표 정리 필요성 제안
        * 논문에 포함되는 각 단계의 타당성과 필요성 제안
    * Tabular 베이스라인 시험 결과 공유
        * 피처엔지니어링에 따른 재시험 필요
    * 다음 회차 연구 계획 및 지시사항
        * 최신 멀티모달 논문 조사
            * SLICE-3D 및 ISIC2024 Triage 연구를 벤치마킹한 최근 3년 이내의 다중모달 논문 검색 및 심층 분석
            * 문헌 조사 항목의 정형화 기재: 연구 목표, 핵심 기여점, 불균형 대응책, 정형/비정형 백본 모델, 융합 메커니즘, 평가지표, 최종 성능 지표를 표 형태로 일괄 요약 정리
        * 모달리티 별 다각적 베이스라인 구축
            * Tabular 베이스라인 재시험
            * Image 베이스라인 시험: 사전학습 등 여러 backbone을 같은 평가 기준에서 비교 
            * 멀티모달 베이스라인 시험
        * 평가 프레임워크 및 기여도 논리 정립

## 지도일: 2026-04-30
### 2026-04-30_4th_meeting

* 논문제목
- ISIC2024 ultra-rare malignant classification을 위한 imbalance-aware image-tabular fusion 연구

* 지도내용
    * 조사 논문 리뷰
        * Dataset 레퍼런스 논문 리뷰
            * dataset 의 구성 과정 및 구성 요소 보고
        * ISIC 2024 kaggle 분석 및 1등 솔루션 리뷰 논문 리뷰
            * 주요 평가지표는 pAUC >80% TPR 보고
            * 대부분의 kaggle 모델 및 1등 솔루션에서 multimodal late fusion 적용 보고
            * 불균형 처리 방법 보고(RandomOverSampler, RandomUnderSampler, scale_pos_weight, ...)
        * ISIC 2024 dataset 사용 멀티모달 논문 리뷰
        * 평가 지표의 학술적 타당성 확보
            * 고성능 하이-센서티비티(High-Sensitivity) 영역 검증을 위해 pAUC 지표 중심의 모델 해석 메커니즘을 고수하며, 불균형 데이터셋에 취약한 단순 정확도(Accuracy) 중심의 정량적 주장 배제 조치
    * 다음 회차 연구 계획 및 지시사항
        * 극단적 불균형 데이터셋 대응 전략 수립
            * 초희귀 악성 클래스(Malignant Class Imbalance, 약 0.1% 미만) 분포 특성을 극복하기 위한 데이터 증강 및 SOTA 다중모달 융합 기법 문헌 분석
            * 메타데이터 기반의 Late Fusion 메커니즘 및 환자 단위 컨텍스트(Patient-level Context) 통합의 유효성을 선행 문헌 결론 연구와 대조하여 정리
        * 논문 방향 
            * 성능 향상 제시

## 지도일: 2026-05-14
### 2026-05-14_5th_meeting

* 논문제목
- ISIC2024 strict input 기반 image-tabular multimodal baseline 연구

* 지도내용
    * 조사 논문 4개 리뷰 
        * 논문 별 보고: 목표와 기여, Dataset 정보, Imbalance 처리, Tabular model, Image model, Fusion 방식, 평가 지표, 평가 결과, ISIC2024 multimodal 연구에 주는 시사점
    * 다음 회차 연구 계획 및 지시사항
        * Baseline 시험: simple baseline부터 시작한다
            * metadata-only: sklearn/boosting 계열
            * image-only: ResNet 등 기본 backbone부터 비교
            * Multimodal baselines (metadata + image)
        * 멀티모달 fusion 방법 논문 조사
            * 최근 멀티모달 연구 조사
            * 도메인의 멀티모달 논문이 부족하면 싱글 모달 논문도 조사
            * 구글 스칼라 이용해서 인용 모델들 조사
    * 논문 방향
        * NEW fusion 방법론 제시

## 지도일: 2026-05-28
### 2026-05-28_6th_meeting

* 논문제목
- ISIC2024 strict protocol 기반 이미지-메타데이터 baseline 및 PanDerm 일반화 성능 검토

* 지도내용
    * Baseline 시험 결과 공유
        * metadata-only: xgboost 성능 가장 좋음
        * image-only: 하드웨어 리소스 부족으로 학습 단계에서 어려움 보고
            * 학습 시간을 줄이기 위한 데이터 샘플링 등의 방법 제시
        * Multimodal baselines (metadata + image): early, joint, late 단계로 구조화
        * 대학원 여분 하드웨어 리소스 진행현황 확인 후 제공 검토
    * 조사 논문 리뷰
        * PanDerm 논문 리뷰
            * 좋은 모델로서 요즘 논문의 추세에 맞춰서, 이 모델을 다른 도메인에 사용해서 일반화 성능 확인하는 방향으로 논문 방향 수정 제시
    * 다음 회차 연구 계획 및 지시사항 
        * PanDerm 의 일반화 성능 확인 방안 조사
    * 논문 방향
        * PanDerm 의 일반화 성능 확인 

