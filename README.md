# CBIS-DDSM Model Benchmark

`1st_after` ?꾨옒 媛??대뜑???섎굹??紐⑤뜽 ?ㅽ뿕 ?⑥쐞?낅땲?? 紐⑤뱺 紐⑤뜽? ?숈씪??`CBIS-DDSM: Breast Cancer Image Dataset` ?щ∼ ?대?吏瑜??ъ슜???댁쭊 遺꾨쪟(`BENIGN/BENIGN_WITHOUT_CALLBACK` vs `MALIGNANT`)瑜??섑뻾?섍퀬, `Accuracy`, `Precision`, `Recall`, `F1-Score`, `AUC-ROC`瑜?`MLflow`??湲곕줉?⑸땲??

## 援ъ꽦

- `cbis_ddsm_benchmark/`
- `1st_after/<紐⑤뜽紐?/config.json`
- `1st_after/<紐⑤뜽紐?/run.ps1`
- `run_all_models.py`

## ?ㅽ뻾 以鍮?
```powershell
python -m pip install -r requirements.txt
```

## ?⑥씪 紐⑤뜽 ?ㅽ뻾

```powershell
powershell -ExecutionPolicy Bypass -File ".\1st_after\ResNet-50\run.ps1"
```

## ?꾩껜 紐⑤뜽 ?ㅽ뻾

```powershell
python .\run_all_models.py
```

## MLflow UI

```powershell
python -m mlflow ui --backend-store-uri ".\mlruns"
```

遺紐???`tags.role = model_parent`)? 紐⑤뜽蹂?理쒓퀬 寃곌낵 鍮꾧탳, ?먯떇 ??`tags.role = hyperparameter_trial`)? ?섏씠?쇳뙆?쇰??곕퀎 ?곸꽭 寃곌낵 ?뺤씤?⑹엯?덈떎.

## 由щ뜑蹂대뱶 CSV 異붿텧

```powershell
python -m cbis_ddsm_benchmark.mlflow_report
```

## 泥댄겕?ъ씤??硫붾え

?쇰? ?섎즺 ?ъ쟾?숈뒿 紐⑤뜽? 異붽? ?⑦궎吏 ?먮뒗 ?몃? 泥댄겕?ъ씤?멸? ?꾩슂?⑸땲??

- `BioMedCLIP`, `CheXzero`: `open_clip_torch`
- `MedCLIP`: `transformers`
- `DeiT-S`, `DINOv2 ViT-S`, `RETFound`: `timm`
- `EyePACS`, `HAM10000`, `TorchXRayVision`: ?ㅼ젣 ?꾩슜 泥댄겕?ъ씤?멸? ?덉쑝硫?`config.json`??`checkpoint_path`瑜?梨꾩썙 ?ｌ쑝硫??⑸땲??

