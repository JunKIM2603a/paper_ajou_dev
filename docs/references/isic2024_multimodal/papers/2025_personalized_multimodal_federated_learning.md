# Personalized Multimodal Federated Learning for Skin Cancer Diagnosis

- 인용 정보: Fan et al., 2025, `Electronics`, https://doi.org/10.3390/electronics14142880
- 출판 유형: peer-reviewed(동료심사 논문)
- Seed 논문 인용 관계: ISIC 2024 challenge citation; 정확한 seed citation 확인 필요
- 데이터셋: custom ISIC 2018-2024 mixture(맞춤 혼합 데이터셋)
- 과제: federated(연합학습), heterogeneous(이질적), missing-modality(누락 모달리티) setting에서의 skin cancer diagnosis
- 모달리티: image와 tabular metadata(표형 메타데이터)
- 추론 입력: client(클라이언트)별 image/tabular 사용 가능성에 따라 달라짐
- Strict-contract 호환성: 부분 호환

## 목표와 기여
- heterogeneous client 전반의 personalized multimodal federated learning(개인화 멀티모달 연합학습)을 위한 PMM-FL을 제안한다.
- missing modality와 privacy-preserving cross-institutional training(개인정보 보호 기관 간 학습)을 목표로 한다.
- multitask learning(다중과제 학습)을 사용해 diagnosis(진단)와 missing-tabular-modality prediction(누락 표형 모달리티 예측)을 결합한다.

## 불균형 처리
- 여러 ISIC 연도를 결합하고 모든 ISIC 2024 malignant example(악성 예시)을 선택해 custom dataset(맞춤 데이터셋)을 만든다.
- 이전 ISIC 연도의 positive(양성)와 sampling(표본 추출)한 benign example(양성 병변 예시)을 추가해 imbalance(불균형)를 줄인다.
- 논문 본문에서 ISIC 2024 count가 일관되지 않게 보고된 것으로 보이므로 인용 전에 검증해야 한다.

## 모델
- Tabular: tabular encoder(표형 인코더)와 missing-modality prediction module(누락 모달리티 예측 모듈).
- Image: CNN image encoder.
- Fusion: concatenated image/tabular feature(연결된 영상/표형 특징) 뒤에 multi-head attention(다중 헤드 어텐션)과 classifier(분류기)를 적용.

## 지표와 결과
- 지표: diagnostic accuracy(진단 정확도), missing-modality robustness(누락 모달리티 강건성), communication overhead(통신 부담).
- 최고 보고 결과: 92.32% diagnostic accuracy; 30% modality missingness(모달리티 누락률)에서 2% 하락.
- 검증/테스트 protocol(절차): federated simulation(연합학습 시뮬레이션); 정확한 patient split(환자 분할) 확인 필요.
- Threshold(임계값) 선택: 핵심 사항 아님 / 확인 필요.

## 한계
- dataset을 혼합하고 federated learning을 최적화하므로 직접적인 ISIC 2024 pAUC baseline(기준 모델)은 아니다.
- Ultra-rare malignant detection(초희귀 악성 탐지)에서는 accuracy가 pAUC/AP보다 정보량이 적다.
- Custom balancing(맞춤 균형화) 때문에 strict ISIC 2024 protocol과 직접 비교하기 부적절하다.

## 우리 연구와의 관련성
- missing metadata(누락 메타데이터)와 heterogeneous deployment(이질적 배포)에 대한 유용한 related work(관련 연구)이다.
- 첫 paper-facing baseline(논문용 기준 모델)이 아니라 향후 robustness experiment(강건성 실험)에 참고할 수 있다.
- strict image+tabular ISIC 2024 baseline과는 별도로 인용해야 한다.

## 검증 메모
- 출처: MDPI article page와 abstract snippet.
