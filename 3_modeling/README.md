# 3_modeling

## 目录定位

`3_modeling` 负责把编码层产出的时间窗口转化为可复核的正式建模结果。这里既保留了 Naive、HAR、LSTM 等对照模型，也保留了当前主线 `TimeMixer + fusion + late.gru_gate` 的正式结果和导出图表。

当前主任务固定为：

```text
日频未来 30 日 Brent 均价残差预测
target = target_brent_avg_next_30d - reference_brent
```

## 目录结构

```text
3_modeling/
├─ README.md
├─ run_modeling_benchmarks.py
├─ baselines/
│  ├─ README.md
│  ├─ har/
│  │  └─ run_har_benchmark.py
│  └─ lstm/
│     └─ run_lstm_benchmark.py
├─ common/
│  ├─ README.md
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
│  ├─ models/
│  ├─ layers/
│  ├─ exp/
│  ├─ data_provider/
│  └─ utils/
└─ results/
   ├─ official/
   │  └─ daily_horizon30/
   │     ├─ naive_reference/
   │     ├─ har_no_leak_residual/
   │     ├─ lstm_residual/
   │     ├─ timemixer_image/
   │     ├─ timemixer_text/
   │     ├─ timemixer_structured/
   │     └─ timemixer_late_gru_gate_mainline_final/
   └─ export/
      └─ daily_horizon30_late_gru_gate_mainline_final/
         ├─ EXPORT_SUMMARY.json
         ├─ figures/
         └─ tables/
```

## 建模数据流

```text
2_encoding_feature/time_series_horizon30_mainline
  -> baselines / TimeMixer
  -> results/official/daily_horizon30
  -> results/export/daily_horizon30_late_gru_gate_mainline_final
  -> 4_decision_rl
```

## 模型组成

### 基线模型
- `Naive`：参考价延续基线
- `HAR-no-leak`：无泄漏历史波动回归基线
- `LSTM-window`：序列建模神经网络基线

### 单模态 TimeMixer
- `timemixer_text`
- `timemixer_image`
- `timemixer_structured`

### 主线模型
- 输入：`fusion`
- 融合器：`late.gru_gate`
- 主模型：`TimeMixer`
- 最终形式：validation-selected family-diverse config ensemble

## 关键文件说明

| 文件 | 描述性说明 |
| --- | --- |
| `run_modeling_benchmarks.py` | 这是建模层的统一启动器，用同一套数据窗口和结果目录把 Naive、HAR、LSTM 与 TimeMixer 拉到同一评估口径上。它的价值在于把“主线是否真的更强”变成可复核的统一对照。 |
| `common/metrics.py` | 该文件定义了 RMSE、MAE、MAPE、方向命中率等核心指标，是所有模型结果可比较的基础。没有这层统一指标实现，不同实验之间就无法形成正式 leaderboard。 |
| `common/window_data.py` | 这里封装了时间窗口读取和切分逻辑，负责把编码层产物稳定地送入各类模型。它保证基线模型和主线模型看到的是同一份样本切分。 |
| `TimeMixer/official_benchmark.py` | 这个文件承担正式 TimeMixer 评测流程的封装，用来连接模型训练、验证选择和结果落盘。单模态与多模态 TimeMixer 的官方结果都要经过这一层。 |
| `TimeMixer/run_fusion_timemixer.py` | 这是单次 TimeMixer 实验的核心入口，负责读取某一输入变体并完成训练、验证和测试。调模型超参数时，最常直接操作的就是这个脚本。 |
| `TimeMixer/run_horizon30_late_gru_gate_mainline_tuning.py` | 这是当前主线总控文件，负责 late.gru_gate 主线的调优、rolling 复核、结果汇总和图表导出。最终主线之所以能形成一套完整的正式结果，依赖的就是这条脚本链。 |
| `results/official/daily_horizon30/timemixer_late_gru_gate_mainline_final/official_metrics.json` | 这是主线官方指标的落盘文件，记录最终确认的 test 与 rolling 指标。对外引用“正式结果”时，最权威的数值来源就是它。 |
| `results/export/daily_horizon30_late_gru_gate_mainline_final/EXPORT_SUMMARY.json` | 这是导出包的总索引，集中汇总当前主线的核心指标、图表和表格路径。前端、汇报材料和人工复核通常都从这份摘要开始定位结果资产。 |

## 关键脚本

### `run_modeling_benchmarks.py`
负责统一跑官方对照。

常用参数：
- `--frequency daily|monthly|both`
- `--models har,lstm,timemixer|all`
- `--window-length`

### `TimeMixer/run_fusion_timemixer.py`
负责单次官方 TimeMixer 训练与评估。

常用参数：
- `--mode official|debug`
- `--window-length`
- `--input-variant fusion|text|image|structured`
- `--frequency daily|monthly`
- `--smoke`
- `--fusion-method`
- `--batch-size`
- `--max-epochs`
- `--patience`
- `--learning-rate`
- `--weight-decay`
- `--loss mse|l1|huber`
- `--target-mode level|residual`
- `--d-model`
- `--e-layers`
- `--d-ff`
- `--dropout`
- `--down-sampling-layers`
- `--moving-avg`

### `TimeMixer/run_horizon30_late_gru_gate_mainline_tuning.py`
负责当前主线的调优、rolling 复核、正式导出和图表生成。

常用参数：
- `--mode all|gate-feature|visuals-only`
- `--force`

## 当前正式结果

正式导出目录：

`results/export/daily_horizon30_late_gru_gate_mainline_final/`

当前主线摘要：
- `selected_method = late.gru_gate`
- `model = TimeMixer`
- `input_variant = fusion`
- `test_rmse = 15.4705`
- `test_mae = 12.2412`
- `test_mape = 0.1245`
- `test_direction_acc = 0.95`
- `rolling_score = 6.0236`
- `rolling_rmse_mean = 4.7957`
- `rolling_direction_acc_mean = 0.725`

## 结果解读

固定测试窗口上，当前主线优于主要对照：
- 主线：`15.4705`
- `Image`: `15.8806`
- `late.gru_concat`: `16.1250`
- `Naive`: `16.2388`
- `HAR-no-leak`: `17.0130`
- `LSTM`: `19.6625`

6-fold rolling 上，当前主线同样保持最低 `rolling_score`：
- 主线：`6.0236`
- `late.gru_concat`: `6.2267`
- `Text`: `6.3557`
- `Image`: `6.4442`
- `Naive`: `6.4677`
- `HAR-no-leak`: `6.7441`
- `LSTM`: `7.2290`

这说明：
- 多模态融合确实优于单模态输入
- `late.gru_gate` 在主线里比主要融合器对照更稳定
- TimeMixer 在当前任务上兼顾了误差和方向判断能力，因此成为最终导出模型

## 导出结果资产

### 核心表格
- `tables/final_selection_basis.csv`
- `tables/final_test_leaderboard.csv`
- `tables/rolling_leaderboard.csv`
- `tables/rolling_predictions.csv`
- `tables/fold_metrics.csv`
- `tables/market_focus_days.csv`
- `tables/validation_selection_by_fold.csv`

### 核心图表
- `figures/figure1_multimodal_selection_error.png`
- `figures/figure2_timemixer_directional_hit_rate.png`
- `figures/figure3_timemixer_cumulative_direction_calls.png`
- `figures/figure4_test_prediction_overlay.png`
- `figures/figure5_advantage_matrix.png`

这些资产同时服务于：
- 主线模型复核
- 论文与答辩展示
- 前端证据面板展示
