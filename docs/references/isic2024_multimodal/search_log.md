# Literature Search Log

검색일: 2026-04-27

## Scope

- 기간: 2023-04-27 이후 발표 논문 또는 preprint
- 대상: ISIC 2024 / SLICE-3D / 3D-TBP 기반 multimodal skin lesion / skin cancer 연구
- Seed papers:
  - `The SLICE-3D dataset: 400,000 skin lesion image crops extracted from 3D TBP for skin cancer detection`
  - `Automated triage of cancer-suspicious skin lesions with 3D total-body photography`

## Search Queries

| Date | Query | Source | Notes |
|---|---|---|---|
| 2026-04-27 | `"ISIC 2024" "multimodal" "tabular" "image" skin cancer arXiv` | web search | arXiv and publisher candidates |
| 2026-04-27 | `"SLICE-3D" "image" "tabular" "fusion" skin cancer` | web search | SLICE-3D direct candidates |
| 2026-04-27 | `"The SLICE-3D dataset" "cited by" "multimodal"` | web search | citation candidates, ResearchGate snippets |
| 2026-04-27 | `"Automated triage of cancer-suspicious" "cited by"` | web search | triage-paper citation candidates |
| 2026-04-27 | `"FusionNetX" "partial" "0.18380"` | web search | journal article found |
| 2026-04-27 | `"Explainable multimodal AI for skin lesion risk prediction via 3D imaging and clinical data"` | web search | Scientific Reports article found |

## Candidate Papers

| Paper | Source | Seed Citation | Include? | Reason |
|---|---|---|---|---|
| Automated triage of cancer-suspicious skin lesions with 3D total-body photography | Nature / npj Digital Medicine | cites SLICE-3D; is seed paper 2 | yes | ISIC 2024 official outcome/ablation reference |
| FusionNetX: A highly effective multimodal framework for skin cancer detection | Journal of Computer Science and Cybernetics | SLICE-3D / ISIC 2024 citation 확인 필요 | yes | direct ISIC 2024 image+metadata multimodal paper |
| Multimodal system for skin cancer detection | arXiv 2601.14822 | cites SLICE-3D via references 확인 필요 | yes | direct image+metadata multimodal system with pAUC |
| Hybrid Ensemble of Segmentation-Assisted Classification and GBDT... | arXiv 2506.03420 | direct SLICE-3D use; explicit citation 확인 필요 | yes | hybrid image+metadata ensemble and synthetic lesion augmentation |
| Explainable multimodal AI for skin lesion risk prediction via 3D imaging and clinical data | Scientific Reports | likely cites SLICE-3D / triage context; exact reference 확인 필요 | yes | peer-reviewed explainable image+clinical fusion |
| Dual-stage segmentation and classification framework for skin lesion analysis using deep neural network | Digital Health / SAGE | SLICE-3D use; exact seed citation 확인 필요 | yes | includes SLICE tabular-only vs fusion pAUC contrast |
| A Personalized Multimodal Federated Learning Framework for Skin Cancer Diagnosis | MDPI Electronics | ISIC 2024 challenge citation; seed citation 확인 필요 | yes, related | federated/missing-modality setting, not direct strict baseline |
| A Novel Transfer Learning Approach for Skin Cancer Classification on ISIC 2024 3D Total Body Photographs | International Journal of Imaging Systems and Technology | ISIC 2024 / SLICE-3D citation 확인 필요 | no | image-only or unclear multimodal status; useful only as weak related work |
| Improved Skin Cancer Detection with 3D Total Body Photography: Integrating AI Algorithms for Precise Diagnosis | Research Square | ISIC 2024 / 3D-TBP; seed citation 확인 필요 | low-confidence candidate | preprint detail insufficient; keep in verification queue only |
| Skin region images extracted from 3D total body photographs for lesion detection / iToBoS dataset | Scientific Data / arXiv | related 3D-TBP dataset; positions against SLICE-3D | no | dataset descriptor, not a multimodal classification model |
| HCHS-Net: A Multimodal Handcrafted Feature and Metadata Framework... | MDPI Biomimetics | no ISIC 2024; PAD-UFES-20 | no | useful general multimodal metadata paper but outside current dataset scope |
| SkinEHDLF hybrid deep learning approach | Scientific Reports | ISIC 2024 mention | no | appears image-centric and dataset description has balance claims requiring caution |

## Verification Queue

| Item | Status | Notes |
|---|---|---|
| Google Scholar cited-by relation for each candidate | 확인 필요 | Web snippets identify direct dataset use; formal cited-by should be checked manually if used in paper text |
| Exact patient-level split for FusionNetX | partial | abstract states stratified group CV; detailed split source should be checked in PDF |
| Exact split/threshold source for arXiv 2601.14822 | partial | arXiv abstract and review/source snippets report metrics; PDF details should be checked before paper claim |
| Diagnosis-informed relabeling leakage risk in arXiv 2506.03420 | 확인 필요 | should be separated from strict baseline because it uses diagnosis-informed external-data harmonization |
| Pathology-derived feature risk in Scientific Reports 2025 XAI paper | major risk | reported feature list includes `mel_thick_mm`; treat as not strict-compatible unless removed and re-audited |
| pFPR wording in Scientific Reports 2025 XAI paper | 확인 필요 | paper reports pFPR 0.17343; map carefully to pAUC@TPR>=0.80 before comparison |
| PMM-FL ISIC 2024 lesion count typo/risk | 확인 필요 | source text reports 40,159 not 401,059 in one place; do not use as dataset fact without correction |
