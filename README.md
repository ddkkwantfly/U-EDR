## Overview
This repository contains a minimal implementation of an asynchronous BCI
state detection pipeline using geometry-aware evidence fusion.

## Files
- train_multiple_enn_realoodcalib.py  
  Train EEGNet with evidential output and real-OOD evidence calibration.

- ood_eval_classwise_fusion_enn_maha.py  
  Evaluate NC/IC detection using class-wise Mahalanobis distance and
  geometry-aware evidence fusion.

- data_loader.py  
  Data loading and windowing utilities (expects .mat files, not included).

## Usage
1. Prepare .mat files in `data/mat/`
2. Train the model:
   ```bash
   python train_multiple_enn_realoodcalib.py


## Evaluate
1. python ood_eval_classwise_fusion_enn_maha.py
