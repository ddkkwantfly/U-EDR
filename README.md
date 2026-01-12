# U-EDR: Uncertainty-aware Evidential Decision with Geometry Fusion for SSVEP-BCI OOD Detection

## 项目简介

本项目实现并评估了一种 **基于 Evidential Neural Network（ENN）与几何距离融合的 SSVEP-BCI OOD 检测框架（U-EDR）**。  
核心目标是在保证 ID（in-distribution）分类性能的同时，对 **未知或非目标输入（OOD）** 提供可靠、可解释的拒识能力，以贴近真实 BCI 应用场景。

方法基于 EEGNet 架构，引入 Dirichlet evidential 输出，并在评估阶段融合特征空间中的 Mahalanobis 几何距离，实现 **证据（uncertainty）与几何（distance）** 的联合决策。

---

## 方法概述

### 1. ID 建模：Evidential Learning

- 网络输出 Dirichlet 参数 `α`
- 使用 **期望交叉熵（Expected CE）+ KL 正则项** 进行 ID 训练  
  - KL 项约束 Dir(α) 向均匀先验 Dir(1) 收缩
- 证据总量  
  \[
  S = \sum_k \alpha_k
  \]
  作为不确定性与拒识信号（证据越低，越倾向 OOD）

---

### 2. OOD 校准（训练阶段，可选）

考虑到真实场景中不可能穷举所有 OOD，本项目引入 **有限但受控的 OOD 校准机制**：

- 在训练阶段，仅使用 **部分已知 OOD 类（calib\_ood）**
- 通过低证据损失（如 KL 到 Dir(1)）约束模型在这些样本上输出低证据
- 明确保留 **unseen OOD**（训练中完全未出现）用于测试泛化能力

该设计避免将 OOD 问题退化为“换标签的 IID 分类”。

---

### 3. 几何信息与证据融合

- 在特征空间中，对 ID 数据拟合 **共享协方差的 class-wise Mahalanobis 距离**
- 对每个测试样本计算其到各 ID 类中心的距离
- 基于距离构造权重，对 evidential 输出进行加权融合：
  - **evidence-only**：仅使用证据总量
  - **maha-only**：仅使用最小 Mahalanobis 距离
  - **fusion\_geo**：证据 × 几何权重（核心方法）

该融合机制在不引入额外训练参数的情况下，显著提升拒识稳定性。

---

## OOD 设定（核心设计）

考虑真实应用中 OOD 的多样性与不可枚举性，本项目将 OOD 明确划分为两类：

### 1. calib\_ood（Seen OOD）
- 训练阶段用于低证据校准
- 例如：类别 (8, 9)

### 2. unseen\_ood（Unseen OOD）
- 训练阶段完全未使用
- 仅在测试阶段评估模型的泛化拒识能力
- 例如：类别 (10, 11)

评估中分别报告：
- **ID vs calib\_ood**
- **ID vs unseen\_ood**

以避免将 OOD 任务简化为披着 OOD 外壳的 IID 分类。

---

## 实验设置

- 数据划分：**6 / 2 / 2 跨被试（train / val / test）**
- ID 类别：0–7
- 训练窗口长度：2s  
  - 支持 **multi-start window** 训练接口（如 1.0 / 1.5 / 2.0s）
- 评估阶段：single-start window（用于稳定对比）
- 评估指标：
  - ID accuracy
  - OOD AUROC
  - FPR @ TPR = {0.80, 0.90, 0.95}

---

## 当前结果概览（应用导向）

- **ID 分类准确率**：约 0.69（跨被试）
- **Seen OOD（calib\_ood）**：
  - `fusion_geo` AUROC ≈ **0.92**
  - 在高 TPR 区域具有较低 FPR，工程上具备实用性
- **Unseen OOD**：
  - `fusion_geo` AUROC ≈ **0.79**
  - 在高召回率条件下仍存在一定误报，符合真实 OOD 的挑战性

整体结果表明：
- 证据与几何信号之间存在一定 trade-off
- 融合策略在已知 OOD 上显著提升拒识能力
- 对真正未知 OOD 仍具挑战，但明显优于单一信号

---

## 下一步计划

为进一步评估方法在更真实 OOD 场景下的鲁棒性，计划引入以下两类新 OOD：

### 1. 机制型 OOD（Mechanistic OOD）
- 随机或非目标频率刺激（或等价频域扰动）
- 用于检验模型是否真正学习到 SSVEP 的结构性特征，而非类别记忆

### 2. 时间 / 信息不足型 OOD（Temporal OOD）
- 准备阶段或极短窗口输入（如 0.5s）
- 评估 evidential 不确定性在低信息条件下的保守性与安全性

同时，将补充：
- 不同 OOD 类型下的系统化对比表
- 小样本条件下的 bootstrap 置信区间分析

---

## 备注

本项目当前以 **应用验证与系统性评估** 为主要目标，  
证据与几何信号之间的理论 trade-off 问题仍有较大研究空间，留待后续深入探讨。


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
