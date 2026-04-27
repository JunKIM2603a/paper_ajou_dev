# Project Rules For Agents

This repository is for **ISIC2024 multimodal research on an ultra-rare malignant target**.

The confirmed project direction is:

```text
image + tabular modeling for ISIC2024 malignant classification
```

LUPI, privileged supervision, diagnosis-text supervision, and `iddx_full`-based auxiliary learning are **candidate research ideas**, not the default project identity.

Agents must protect the scientific validity of the project. In this repository, patient-level leakage, test-time leakage, incorrect pAUC computation, and misuse of privileged diagnosis fields are critical failures.

---

## 1. Project Identity

This project studies multimodal skin lesion classification using ISIC2024 / SLICE-3D style data.

The main modeling direction is:

```text
lesion image + ordinary inference-time tabular metadata -> malignant probability
```

The project is not automatically a LUPI project.

LUPI-style ideas are allowed only as explicitly labeled candidate experiments:

```text
src/isic2024_multimodal/research/lupi_candidates/
experiments/configs/research_candidates/
```

Do not treat `iddx_full`, diagnosis text, pathology-derived text, or similar privileged fields as ordinary inference-time inputs.

---

## 2. Critical Research Rules

### 2.1 Patient-Level Leakage Is a Critical Failure

Splits must respect this grouping:

```text
patient_id -> lesion_id -> isic_id
```

Rules:

- Never use row-level random train/test split for paper experiments.
- The same `patient_id` must not appear across train, validation, and test within a fold.
- Do not let lesions from the same patient leak across folds.
- Always audit patient overlap before trusting any score.
- If patient-level split is impossible for a proposed experiment, label the result as exploratory only.

### 2.2 Raw Data Must Not Be Modified

Raw ISIC2024 data belongs under:

```text
data/raw/isic_2024_challenge/
```

This directory is read-only for agents.

Do not modify, overwrite, clean, normalize, or reorganize files in this directory.

Existing raw-data contents may include:

```text
sample_submission.csv
test-image.hdf5
test-metadata.csv
train-image.hdf5
train-metadata.csv
```

New processed artifacts must go outside the raw directory, preferably under:

```text
data/processed/
experiments/outputs/
experiments/evidence/
```

New split outputs should go under:

```text
data/splits/
```

However, split files should not be tracked in Git unless the user explicitly decides otherwise. Track the split-generation code and small summary evidence instead.

### 2.3 Train-Only Preprocessing

Anything that learns from data distribution must be fit on training data only.

Train-only preprocessing includes:

```text
imputation
scaling
normalization
categorical encoding
feature selection
PCA / SVD
text vectorization
TF-IDF vocabulary
privileged-information transforms
class weights
sampling plans
threshold selection
calibration models
```

Validation and test data may only receive transforms fitted on the training portion of the fold.

Never do this:

```text
fit_transform(full_dataframe)
scaler.fit(all_data)
encoder.fit(all_data)
feature_selector.fit(all_data)
threshold selected on test fold
```

### 2.4 Privileged Diagnosis Fields

The following are not ordinary inference-time inputs:

```text
iddx_full
diagnosis text
pathology-derived text
oracle diagnosis label
target-derived diagnosis grouping
```

Allowed only when explicitly framed as a candidate experiment:

```text
training-only auxiliary target
training-only privileged teacher input
training-only representation alignment signal
analysis-only interpretability evidence
```

Not allowed:

```text
predict.py requiring iddx_full
model.forward requiring iddx_full for inference
test-time feature extraction from iddx_full
full-data text vectorizer fitted on iddx_full
diagnosis text merged into ordinary tabular baseline
```

When using `iddx_full` or related diagnosis text, clearly mark the experiment as:

```text
LUPI candidate
privileged supervision candidate
auxiliary training-only candidate
```

---

## 3. Required Metrics

Primary baseline metrics must include:

```text
pAUC above TPR 0.80
AUC
F1
precision
recall
balanced accuracy
```

Because the target is ultra-rare, agents should also consider reporting:

```text
Average Precision / PR-AUC
confusion matrix
false negative count
false positive count
fold-wise score
mean ± std
minimum fold score
```

Accuracy alone is not acceptable evidence for model quality.

Threshold-dependent metrics such as F1, precision, recall, and balanced accuracy must use a threshold chosen from validation data only.

Do not choose thresholds on the test fold.

---

## 4. Repository Structure Rules

Use only this Python package:

```text
src/isic2024_multimodal
```

Do not create another Python package at repository root.

Runnable commands must live under:

```text
src/isic2024_multimodal/cli/
```

Current command modules include:

```text
define_feature_sets.py
export_tabular_dataset.py
generate_reports.py
run_all_image_models.py
run_all_tabular_models.py
run_image_baseline.py
run_multimodal_experiment.py
run_tabular_baseline.py
```

Image and tabular baseline code belongs under:

```text
src/isic2024_multimodal/baselines/
```

Specifically:

```text
src/isic2024_multimodal/baselines/image/
src/isic2024_multimodal/baselines/tabular/
```

Future image + tabular fusion models belong under:

```text
src/isic2024_multimodal/models/fusion/
```

Uncertain LUPI-style ideas belong under:

```text
src/isic2024_multimodal/research/lupi_candidates/
```

Do not promote LUPI candidate code into the main model path until the paper direction is explicitly finalized.

Notebooks belong under:

```text
notebooks/isic_2024/
```

Do not put notebooks under `src/`.

Legacy exploratory notebooks may remain under:

```text
notebooks/exploratory/
```

but new ISIC2024 research notebooks should go under:

```text
notebooks/isic_2024/
```

---

## 5. Important Directory Map

Use this map when deciding where to read or write files.

### Source Code

```text
src/isic2024_multimodal/
```

Purpose:

```text
main Python package
reusable code
training logic
evaluation logic
model definitions
CLI commands
```

### Data

```text
data/raw/isic_2024_challenge/
```

Purpose:

```text
raw ISIC2024 downloaded data
read-only
do not modify in place
```

```text
data/processed/
```

Purpose:

```text
processed datasets
derived tables
safe transformed data
```

```text
data/splits/
```

Purpose:

```text
patient-level split artifacts
fold definitions
split summaries
```

### Experiments

```text
experiments/configs/
```

Purpose:

```text
experiment configuration files
```

Subdirectories:

```text
experiments/configs/image_baselines/
experiments/configs/tabular_baselines/
experiments/configs/multimodal/
experiments/configs/research_candidates/
```

Use them as follows:

```text
image_baselines/       -> image-only baseline configs
tabular_baselines/     -> tabular-only baseline configs
multimodal/            -> image + tabular fusion configs
research_candidates/   -> LUPI, privileged supervision, or uncertain candidate configs
```

```text
experiments/evidence/
```

Purpose:

```text
small CSV/Markdown evidence
EDA tables
feature-selection evidence
validation-protocol evidence
```

```text
experiments/logs/
```

Purpose:

```text
MLflow logs and generated run logs
do not track in Git
```

```text
experiments/outputs/
```

Purpose:

```text
generated predictions
checkpoints
plots
temporary outputs
do not track large generated artifacts
```

```text
experiments/tables/
```

Purpose:

```text
paper-ready result tables
small summarized tables may be tracked
```

### Documentation

```text
docs/eda/
```

Purpose:

```text
EDA methodology
feature engineering notes
oracle-supervision notes
validation-protocol notes
```

```text
docs/minutes/
```

Purpose:

```text
meeting notes
research decisions
timeline evidence
```

```text
docs/plan/
```

Purpose:

```text
paper plan
proposal
figures
reviews
presentation files
```

```text
docs/reports/
docs/weekly_report/
```

Purpose:

```text
progress reports
weekly summaries
```

---

## 6. Experiment Protocol

The default experiment protocol is:

```text
patient-level split
train-only preprocessing
validation-based model selection
test fold for final reporting only
```

Do not use test results for:

```text
early stopping
hyperparameter selection
feature selection
threshold selection
calibration fitting
model choice
paper-claim selection
```

If an experiment violates the protocol, label it:

```text
exploratory
not paper-valid
```

---

## 7. Required Experiment Families

Agents working on experiments should organize comparisons in the following order.

### 7.1 Protocol Sanity Check

Before model comparisons, verify:

```text
patient_id overlap across splits
lesion_id consistency
isic_id uniqueness
class distribution by fold
malignant count by fold
metric implementation
train-only preprocessing
```

Expected evidence:

```text
experiments/evidence/validation_protocol/
```

### 7.2 Tabular-Only Baselines

Purpose:

```text
measure the signal strength of ordinary inference-time tabular metadata
```

Allowed inputs:

```text
ordinary tabular metadata
strict/main tabular feature set
features defined under src/isic2024_multimodal/features/
```

Disallowed inputs:

```text
iddx_full
diagnosis text
target-derived features
test-derived statistics
full-data-fitted transforms
```

Expected locations:

```text
src/isic2024_multimodal/baselines/tabular/
experiments/configs/tabular_baselines/
```

Candidate models:

```text
logistic regression
random forest
gradient boosting
tabular MLP
tabular torch estimator
```

### 7.3 Image-Only Baselines

Purpose:

```text
measure the visual signal strength of lesion images
```

Expected locations:

```text
src/isic2024_multimodal/baselines/image/
src/isic2024_multimodal/models/image/
experiments/configs/image_baselines/
```

Control variables:

```text
same patient-level folds
same metric function
same evaluation script
same reporting format
same threshold-selection protocol
```

Image backbones may differ, but protocol must not.

### 7.4 Multimodal Baselines

Purpose:

```text
test whether image + tabular modeling improves over unimodal baselines
```

Required comparisons:

```text
image-only best
tabular-only best
late fusion
feature concatenation fusion
gated fusion
attention-based fusion, if implemented
```

Expected locations:

```text
src/isic2024_multimodal/models/fusion/
src/isic2024_multimodal/cli/run_multimodal_experiment.py
experiments/configs/multimodal/
```

Do not add LUPI or `iddx_full` at this stage unless the experiment is explicitly labeled as a research candidate.

### 7.5 Imbalance, Loss, and Threshold Ablation

Purpose:

```text
handle the ultra-rare malignant target
```

Compare:

```text
plain BCE
weighted BCE
focal loss
class-balanced loss
positive oversampling
balanced batch sampler
validation-selected threshold
fixed threshold baseline
```

Rules:

```text
class weights are computed from training data only
samplers are built from training data only
thresholds are selected from validation data only
test data is never used for tuning
```

### 7.6 LUPI / Privileged Supervision Candidates

Purpose:

```text
test whether training-only privileged information improves a strict image + tabular model
```

This is optional and candidate-only.

Allowed candidate forms:

```text
auxiliary iddx_full-derived classification head
privileged teacher -> strict student distillation
train-only diagnosis-text contrastive loss
train-only prototype alignment
```

Required comparison:

```text
strict image + tabular baseline
strict image + tabular + candidate auxiliary training
```

Inference must remain:

```text
image + ordinary tabular metadata only
```

Expected locations:

```text
src/isic2024_multimodal/research/lupi_candidates/
experiments/configs/research_candidates/
```

Do not place uncertain candidate logic in the main fusion model path.

### 7.7 Final Reporting

Final reports must include:

```text
fold-wise metrics
mean ± std
config path
split version
seed
threshold source
model checkpoint source, if applicable
```

Required metrics:

```text
pAUC above TPR 0.80
AUC
F1
precision
recall
balanced accuracy
```

Recommended additional metrics:

```text
Average Precision / PR-AUC
confusion matrix
false negative count
false positive count
calibration summary
```

---

## 8. Feature Engineering Rules

Feature engineering must be reproducible and leakage-safe.

Feature definitions should live under:

```text
src/isic2024_multimodal/features/
```

Current feature-related modules include:

```text
final_tabular_inputs.py
tabular_feature_sets.py
tabular_terms.py
```

Rules:

- Feature sets must be named clearly.
- Do not silently add diagnosis or target-derived columns.
- Do not use `iddx_full` in ordinary tabular feature sets.
- Feature selection must be fitted on training data only.
- Store small feature-selection evidence under `experiments/evidence/`.
- Store documentation under `docs/eda/` when the feature choice affects paper claims.

---

## 9. Model Organization Rules

Image model code:

```text
src/isic2024_multimodal/models/image/
```

Tabular model code:

```text
src/isic2024_multimodal/models/tabular/
```

Fusion model code:

```text
src/isic2024_multimodal/models/fusion/
```

Heads:

```text
src/isic2024_multimodal/models/heads/
```

Research-only candidates:

```text
src/isic2024_multimodal/research/lupi_candidates/
```

Do not mix candidate research code into production baseline code before the paper direction is finalized.

Model inputs must be explicit.

Good naming:

```text
image
tabular
target
patient_id
lesion_id
isic_id
fold_id
```

For candidate experiments only:

```text
iddx_full_train_only
auxiliary_target
privileged_signal
teacher_logits
```

Bad naming:

```text
metadata
extra
oracle
diagnosis
```

unless the role is fully documented.

---

## 10. CLI Rules

Runnable commands must be implemented under:

```text
src/isic2024_multimodal/cli/
```

Prefer running modules with:

```text
python -m isic2024_multimodal.cli.<command_name>
```

Do not put major runnable experiment logic in notebooks.

Notebooks may inspect data, visualize outputs, or prototype ideas, but reusable logic belongs in `src/isic2024_multimodal`.

---

## 11. Logging and Reporting Rules

Use consistent logging for all experiments.

Generated logs belong under:

```text
experiments/logs/
```

Generated outputs belong under:

```text
experiments/outputs/
```

Paper-ready summary tables belong under:

```text
experiments/tables/
```

Small evidence files may go under:

```text
experiments/evidence/
```

MLflow logs, checkpoints, caches, and large generated outputs must not be tracked in Git.

When reporting a result, include:

```text
model name
config path
fold id
seed
split source
preprocessing source
metric function
threshold source
```

---

## 12. Git Rules

Track:

```text
source code
configs
docs
small CSV evidence
small Markdown evidence
paper-ready small result tables
```

Do not track:

```text
raw data
processed data
split files
checkpoints
MLflow logs
cache directories
__pycache__
large generated outputs
temporary plots
```

Prefer English `snake_case` for new files and folders.

Avoid Korean filenames for new source code, configs, and experiment outputs.

Generated experiment outputs go under:

```text
experiments/outputs/
```

Generated logs go under:

```text
experiments/logs/
```

---

## 13. Naming Rules

Use English `snake_case` for:

```text
new Python files
new folders
config files
experiment IDs
model names
metric names
```

Good examples:

```text
run_tabular_baseline.py
patient_level_split.py
multimodal_late_fusion.json
weighted_bce_ablation.json
lupi_auxiliary_diagnosis_candidate.json
```

Avoid:

```text
새실험.py
final_final.py
test2.py
copy_model.py
```

---

## 14. Agent Workflow

For major work, agents should be used in this order.

### 14.1 Researcher

Use for:

```text
paper framing
related work
research question
claim wording
method comparison
```

The researcher must not treat LUPI as the project identity unless the user explicitly changes the project direction.

### 14.2 Data Guardian

Use for:

```text
patient-level split audit
leakage check
train-only preprocessing check
iddx_full misuse check
```

Any change involving split, preprocessing, feature selection, `iddx_full`, or diagnosis text should be reviewed by the data guardian.

### 14.3 Engineer

Use for:

```text
implementation
refactoring
model code
dataset code
training code
evaluation code
CLI commands
```

The engineer must keep code inside `src/isic2024_multimodal`.

### 14.4 Experimenter

Use for:

```text
experiment design
ablation planning
metric protocol
config organization
result table planning
```

The experimenter must separate:

```text
required baseline experiments
candidate LUPI experiments
exploratory experiments
paper-valid experiments
```

### 14.5 Reviewer

Use for:

```text
final protocol review
paper claim review
metric review
leakage review
ablation completeness review
```

The reviewer should be skeptical. High scores do not matter if leakage risk exists.

---

## 15. Required Response Style For Agents

When modifying code, summarize:

```text
Changed files
Why changed
How leakage is avoided
How patient-level split is preserved
How train-only preprocessing is preserved
How inference inputs are controlled
How to run
How to verify
Remaining risks
```

When designing experiments, summarize:

```text
Hypothesis
Compared models
Allowed inputs
Disallowed inputs
Controlled variables
Metrics
Expected config location
Expected output location
Failure interpretation
```

When reviewing, summarize:

```text
Verdict: PASS / MINOR RISK / MAJOR RISK / FAIL
Critical issues
Leakage risks
Metric risks
Required fixes
Paper-claim readiness
```

When handling LUPI or privileged supervision candidates, summarize:

```text
Why this is candidate-only
What privileged signal is used
Where it is used during training
Why it is not used at inference
How leakage is prevented
What baseline it must beat
```

---

## 16. Prohibited Patterns

Do not introduce these patterns:

```text
train_test_split(rows, random_state=...)
random row-level split for paper experiments
fit_transform(full_dataframe)
scaler.fit(all_data)
encoder.fit(all_data)
feature_selector.fit(all_data)
class_weight computed from full data
threshold chosen on test fold
calibration fitted on test fold
iddx_full used in ordinary tabular baseline
iddx_full required by predict.py
diagnosis text merged into default inference features
public leaderboard used as final paper evidence
```

Do not write new files into:

```text
data/raw/isic_2024_challenge/
```

Do not put reusable experiment logic into notebooks.

Do not create a new Python package outside:

```text
src/isic2024_multimodal/
```

---

## 17. Paper Claim Discipline

Do not write claims stronger than the evidence.

Allowed wording:

```text
suggests
indicates
improves under the defined patient-level protocol
supports the usefulness of multimodal modeling
improves validation-selected threshold performance
shows promise as a candidate auxiliary method
```

Avoid unsupported wording:

```text
clinically proven
causal
universally superior
state-of-the-art
diagnostically definitive
leakage-free without audit
```

A claim is paper-ready only if it has:

```text
patient-level split
train-only preprocessing
fold-wise result
baseline comparison
metric definition
config path
reproducible command
leakage audit
```

---

## 18. Current Project Priorities

The current priority order is:

```text
1. Validate patient-level split and metric implementation
2. Stabilize tabular-only baselines
3. Stabilize image-only baselines
4. Build clean image + tabular multimodal baselines
5. Run imbalance/loss/threshold ablations
6. Consider LUPI or privileged supervision candidates only after baselines are stable
7. Prepare paper-ready result tables and evidence
```

Do not start with complex LUPI models before the baseline protocol is reliable.

---

## 19. Minimal Scientific Standard

Every paper-facing experiment must answer:

```text
What input was used?
Was patient-level leakage prevented?
Was preprocessing fitted on train only?
Was iddx_full excluded from inference?
What metric was used?
How was the threshold chosen?
Which fold and seed were used?
Where is the config?
Where is the result?
```

If any answer is missing, the result is not paper-ready.

---