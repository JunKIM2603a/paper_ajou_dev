# ISIC2024 문서 길잡이

이 디렉터리는 논문 실험 protocol, 재현 순서, 회의 기록, 발표 자료, 관련 연구 정리를 보관한다. README는 짧게 유지하고, 세부 실행법과 근거는 아래 문서로 나눈다.

## 먼저 볼 문서

| 문서 | 언제 보는가 |
|---|---|
| [reproducibility.md](reproducibility.md) | 새 환경에서 strict input, split, baseline을 재현할 때 |
| [eda/isic2024_strict_input_export.md](eda/isic2024_strict_input_export.md) | strict input, `iddx_full` sidecar, nested split export 계약을 확인할 때 |
| [eda/isic2024_tabular_baselines.md](eda/isic2024_tabular_baselines.md) | tabular baseline, missing value policy, GPU/CPU 실행, nested CV summary를 확인할 때 |
| [plan/2026-05-14_after_5th_meeting/isic2024_strict_input_data_protocol_presentation_20260514.md](plan/2026-05-14_after_5th_meeting/isic2024_strict_input_data_protocol_presentation_20260514.md) | strict input 데이터 처리 내용을 발표용으로 설명할 때 |
| [weekly_report/2026-05-14/](weekly_report/2026-05-14/) | 관련 연구와 진행 메모를 확인할 때 |

## 디렉터리 역할

```text
docs/eda/             # 데이터 계약, EDA, validation protocol 문서
docs/plan/            # 논문 계획, 발표 요약, 다이어그램
docs/minutes/         # 회의 기록과 연구 결정
docs/weekly_report/   # 주간 보고서와 진행 메모
docs/reports/         # 보고서 산출물
docs/checkpoint_base/ # image baseline checkpoint 관련 참고 문서
```

## 문서 작성 원칙

- paper-facing protocol은 patient-level split, train-only preprocessing, validation-only selection을 명시한다.
- `iddx_full`과 diagnosis text는 ordinary inference-time input처럼 쓰지 않는다.
- 큰 산출물, raw data, processed CSV, split CSV는 문서에 링크하지 않고 경로와 생성 명령만 기록한다.
- 결과를 보고할 때는 split source, outer/inner fold, threshold source, metric, refit 여부를 함께 남긴다.
