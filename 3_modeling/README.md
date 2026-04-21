# 3_modeling：Brent 30 日残差建模与结果导出层

本目录负责基于 `2_encoding_feature` 生成的时间窗口完成基线模型、单模态 TimeMixer 和多模态主线模型的训练、评估与导出。当前最终展示主线固定为：

```text
TimeMixer + fusion + late.gru_gate
```

## 目录结构

```text
3_modeling/
├─ README.md
├─ run_modeling_benchmarks.py
├─ baselines/
│  ├─ README.md
│  ├─ __init__.py
│  ├─ har/
│  │  └─ run_har_benchmark.py
│  └─ lstm/
│     └─ run_lstm_benchmark.py
├─ common/
│  ├─ README.md
│  ├─ __init__.py
│  ├─ comparison.py
│  ├─ metrics.py
│  ├─ paths.py
│  ├─ reporting.py
│  └─ window_data.py
├─ TimeMixer/
│  ├─ README.md
│  ├─ official_benchmark.py
│  ├─ run_fusion_timemixer.py
│  ├─ run_horizon30_late_gru_gate_mainline_tuning.py
│  ├─ build_modality_comparison.py
│  ├─ data_provider/
│  ├─ exp/
│  ├─ layers/
│  ├─ models/
│  └─ utils/
└─ results/
   ├─ official/
   │  └─ daily_horizon30/
   └─ export/
      └─ daily_horizon30_late_gru_gate_mainline_final/
```

## 文件作用

| 文件或目录 | 作用 |
| --- | --- |
| `run_modeling_benchmarks.py` | 快速批量运行基线和 TimeMixer 对照。 |
| `baselines/har/run_har_benchmark.py` | HAR-no-leak 残差基线。 |
| `baselines/lstm/run_lstm_benchmark.py` | LSTM-window 残差基线。 |
| `common/metrics.py` | RMSE、MAE、MAPE、direction accuracy 等指标计算。 |
| `common/window_data.py` | 读取编码窗口、拼接 reference、构造 rolling 计划。 |
| `common/comparison.py` | 对比表、排行榜和对比图生成。 |
| `common/reporting.py` | 结果落盘、图表保存与目录清理。 |
| `TimeMixer/official_benchmark.py` | TimeMixer official 训练与结果落盘主入口。 |
| `TimeMixer/run_fusion_timemixer.py` | 轻量运行 TimeMixer 的命令行入口。 |
| `TimeMixer/run_horizon30_late_gru_gate_mainline_tuning.py` | 当前主线调优、rolling 复核和 export 生成脚本。 |
| `results/official/` | 各模型族 official 结果目录。 |
| `results/export/` | 当前对外交付用的最终主线结果目录。 |

## 数据如何流动

1. `3_modeling` 读取 `2_encoding_feature/outputs/daily/time_series_horizon30*` 中的 train/valid/test 窗口。
2. `baselines/` 跑 Naive、HAR 和 LSTM 残差基线。
3. `TimeMixer/official_benchmark.py` 分别跑 structured、text、image 和 fusion 输入的 TimeMixer。
4. `run_horizon30_late_gru_gate_mainline_tuning.py` 对 `late.gru_gate` 主线做验证集选择、rolling 复核和导出。
5. `results/official/` 保留各模型的正式结果。
6. `results/export/daily_horizon30_late_gru_gate_mainline_final/` 保留当前可交付版本的最终结果表和正式图。
7. `4_decision_rl` 直接读取 export 与 official 的主线结果生成前端。

## 仓库同步说明

建模层区分两类结果：

- `results/export/`：最终展示与交付资产，保留在仓库中，用于论文、答辩和前端展示。
- `results/official/` 中的训练检查点：由训练脚本本地生成，不上传仓库。

因此，仓库中保留的是最终结果表、正式图和前端直接依赖的结果摘要；训练过程中的模型检查点和大型中间产物由代码按需再生成。

## 当前任务与评估口径

### 任务

```text
target = target_brent_avg_next_30d - reference_brent
```

### 模型选择口径

- 配置选择优先看 validation
- fixed test 只做最终确认
- rolling 采用 6-fold out-of-time 设计，每个 fold 为 60 天

### 当前主线

- 模型：`TimeMixer`
- 输入：`fusion`
- 融合器：`late.gru_gate`
- export 名称：`daily_horizon30_late_gru_gate_mainline_final`
- 当前状态：`validation_status = PASS`

## 可运行脚本与调整方向

| 脚本 | 常用参数或调整点 | 影响 |
| --- | --- | --- |
| `run_modeling_benchmarks.py` | `--models`、`--frequency`、`--window-length` | 快速刷新基线与单模态对照。 |
| `baselines/har/run_har_benchmark.py` | 历史窗口长度、rolling 计划 | 控制 HAR 对照口径。 |
| `baselines/lstm/run_lstm_benchmark.py` | `hidden_size`、`num_layers`、`dropout`、`lr` | 影响 LSTM 容量与稳定性。 |
| `TimeMixer/official_benchmark.py` | `window_length`、`learning_rate`、`batch_size`、`loss`、`dropout`、`d_model`、`down_sampling_layers` | 影响 TimeMixer 收敛速度、容量和泛化。 |
| `TimeMixer/run_fusion_timemixer.py` | 输入模态、窗口长度、seed | 便于快速跑 fusion 对照。 |
| `TimeMixer/run_horizon30_late_gru_gate_mainline_tuning.py` | gate bias、modal dropout、window、seed ensemble、rolling fold | 影响当前主线 late.gru_gate 的最终表现。 |

## official 结果目录

```text
3_modeling/results/official/daily_horizon30/
```

当前保留的核心模型目录：

| 目录 | 作用 |
| --- | --- |
| `naive_reference/` | Naive 基线。 |
| `har_no_leak_residual/` | HAR-no-leak 基线。 |
| `lstm_residual/` | LSTM-window 基线。 |
| `timemixer_structured/` | 结构化单模态 TimeMixer。 |
| `timemixer_text/` | 文本单模态 TimeMixer。 |
| `timemixer_image/` | 图片单模态 TimeMixer。 |
| `timemixer_late_gru_gate_mainline_final/` | 当前正式主线。 |

## 最终 export 结构

```text
3_modeling/results/export/daily_horizon30_late_gru_gate_mainline_final/
├─ EXPORT_SUMMARY.json
├─ tables/
│  ├─ final_selection_basis.csv
│  ├─ final_test_leaderboard.csv
│  ├─ rolling_leaderboard.csv
│  ├─ rolling_predictions.csv
│  ├─ fold_metrics.csv
│  ├─ market_focus_days.csv
│  └─ validation_selection_by_fold.csv
└─ figures/
   ├─ figure1_multimodal_selection_error.png
   ├─ figure2_timemixer_directional_hit_rate.png
   ├─ figure3_timemixer_cumulative_direction_calls.png
   ├─ figure4_test_prediction_overlay.png
   └─ figure5_advantage_matrix.png
```

## 当前主线结果解读

| 指标 | 主线 | 最强对照 | 解读 |
| --- | ---: | ---: | --- |
| fixed test RMSE | `15.4705` | Image `15.8806` | 多模态 fusion 在最终测试窗口领先全部单模态与传统基线。 |
| rolling score | `6.0236` | late.gru_concat `6.2267` | `late.gru_gate` 在跨期滚动验证中保持第一。 |
| rolling RMSE mean | `4.7957` | late.gru_concat `4.9688` | 主线在多个 out-of-time 时段都维持更低平均误差。 |
| rolling direction accuracy | `72.5%` | 约 `54.7%` | 方向信号优势明显，适合风险预警。 |
| fixed test direction accuracy | `95.0%` | HAR/LSTM `5.0%` | 主线在最终窗口的方向识别能力非常强。 |

当前结果说明两点：

- `fusion` 输入确实优于当前单模态输入
- `TimeMixer + late.gru_gate` 在 test 和 rolling 两个口径下都形成了当前项目最强主线

## 结果文件的使用方式

| 文件 | 用途 |
| --- | --- |
| `EXPORT_SUMMARY.json` | 读取主线名称、模型状态和核心指标。 |
| `final_test_leaderboard.csv` | 用于固定 test 口径比较。 |
| `rolling_leaderboard.csv` | 用于 rolling 稳定性比较。 |
| `final_selection_basis.csv` | 用于说明为什么最终选择 `late.gru_gate`。 |
| `rolling_predictions.csv` | 用于跨期走势解释。 |
| `fold_metrics.csv` | 用于逐 fold 指标解读。 |
| `market_focus_days.csv` | 用于前端风险复盘。 |
| `figures/*.png` | 用于论文、答辩和前端中的模型依据展示。 |

## 当前结论

对当前项目来说，`3_modeling` 已经完成从单模态到多模态、从传统基线到 TimeMixer 主线的完整对照，并且正式结果已经稳定收敛到 `TimeMixer + fusion + late.gru_gate`。后续如果继续做前端增强或展示封装，不需要再改建模口径，只需读取当前 export 即可。
