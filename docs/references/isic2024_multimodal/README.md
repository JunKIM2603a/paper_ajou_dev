# ISIC2024 멀티모달 문헌 조사

이 폴더는 ISIC2024 / SLICE-3D / 3D-TBP 기반 멀티모달 피부암 논문조사 결과를 축적하는 공간이다.

기본 관점은 `lesion image + ordinary inference-time tabular metadata -> malignant probability`이다. LUPI, privileged supervision(특권 정보 기반 감독), diagnosis text(진단 텍스트), pathology-derived text(병리 유래 텍스트), `iddx_full` 기반 방법은 기본 baseline(기준 모델)이 아니라 candidate(후보 방법) 또는 related idea(관련 아이디어)로 분리한다.

## 파일

- `papers/`: 논문별 Markdown 요약
- `comparison_table.md`: 논문 간 비교표
- `search_log.md`: 검색 쿼리, 후보 논문, 제외 이유, 확인 필요 사항

## 요약 길이

논문별 요약은 기본 45-55라인을 목표로 하고, 최대 70라인 미만으로 유지한다.

## 필수 추출 항목

- 인용 정보
- Seed 논문 인용 관계
- 데이터셋 / 과제 / 모달리티
- 추론 입력
- Strict-contract 호환성
- 목표와 기여
- 불균형 처리
- Tabular 모델
- Image 모델
- Fusion
- 지표와 결과
- 한계
- 우리 연구와의 관련성
- 검증 메모

## Strict-Contract 규칙

`iddx_full`, diagnosis text(진단 텍스트), pathology-derived context(병리 유래 맥락), oracle diagnosis label(정답 진단 label)을 inference input(추론 입력)으로 요구하는 논문은 이 프로젝트의 strict multimodal baseline(엄격한 멀티모달 기준 모델)과 직접 비교하지 않는다. 이런 논문은 related work(관련 연구), limitation discussion(한계 논의), 또는 candidate method context(후보 방법 맥락)로만 사용한다.
