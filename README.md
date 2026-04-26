# 多模态 Brent 30 日残差预测项目

## 项目简介

本项目围绕 Brent 日频价格序列构建多模态预测流程，数据包括结构化市场变量、公开文本和公开图像。当前任务定义为未来 30 日 Brent 均价相对参考价的残差预测：

```text
未来 30 日 Brent 均价残差 = target_brent_avg_next_30d - reference_brent
```

当前正式主线固定为：

```text
TimeMixer + fusion + late.gru_gate
```

活跃时间范围为 `2022-04-17` 至 `2026-04-16`，共 `1461` 个自然日。结构化、文本、图片、建模与展示数据均按该时间范围组织。

## 全流程概览

```text
1_data_handling
  结构化 / 文本 / 图片 / event 数据维护
  -> 生成 DATA_INVENTORY

2_encoding_feature
  structured 特征处理
  multilingual BERT 文本编码
  CLIP 图像编码
  late.gru_gate 融合
  -> 构造 horizon30 时间窗口

3_modeling
  Naive / HAR / LSTM / TimeMixer 对照
  单模态 / 多模态对照
  late.gru_gate 主线训练与正式导出

4_decision_rl
  构建静态结果数据包和展示页面

project_shared
  提供日期、频率、路径、目标和 IO 约定
```

## 当前数据面

| 模态 | 活跃文件 | 规模 | 说明 |
| --- | --- | --- | --- |
| 结构化 | `1_data_handling/raw/structured/structured_daily_merged.csv` | `1461 × 15` | 真日频 Brent/WTI/美元指数/EPU/GPR/汇率与基础派生列 |
| 月度派生 | `1_data_handling/raw/structured/structured_monthly_derived.csv` | `49 × 17` | 从日表月末锚点派生 |
| 文本 | `1_data_handling/raw/text/text_documents_multisource_cleaned.jsonl.gz` | `8852` 条 | 多源新闻与开放文本，按日覆盖 `1461/1461` |
| 图片 | `1_data_handling/raw/image/image_manifest_commons_cleaned.jsonl.gz` | `1461` 条 | 一天一图的图片清单，缩略图存于 `thumbnails_webp/` |
| Event | `1_data_handling/raw/event/event_manifest.jsonl` | `18` 条 | 事件清单 |

主要真实来源包括：
- 结构化：FRED、NBU、CBR、CBUAE、Iacoviello GPR
- 文本：开放新闻源、多源公开网页抓取结果
- 图片：GDELT DOC、Wikimedia、NASA GIBS、Copernicus、Unsplash、Pexels

## 当前主线结果

正式导出目录为：

`3_modeling/results/export/daily_horizon30_late_gru_gate_mainline_final/`

当前主线指标摘要：
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

### 结果摘要

当前正式主线 `TimeMixer + fusion + late.gru_gate` 在固定测试窗口和 `6` 折 rolling 两个口径下均为当前结果集中的最优配置，因此作为正式导出版本。

在固定测试窗口上：
- 主线 `test_rmse = 15.4705`
- 强单模态对照 `Image = 15.8806`
- 强融合器对照 `late.gru_concat = 16.1250`
- `Naive = 16.2388`
- `HAR-no-leak = 17.0130`
- `LSTM = 19.6625`

在 rolling 稳定性上：
- 主线 `rolling_score = 6.0236`
- `late.gru_concat = 6.2267`
- `Text = 6.3557`
- `Image = 6.4442`
- `Naive = 6.4677`
- `HAR-no-leak = 6.7441`
- `LSTM = 7.2290`

结果表明，多模态融合在当前样本划分下优于单模态输入和主要基线；`late.gru_gate` 相对主要融合器对照具有更低的 rolling score。导出目录中的表格和图形用于记录模型选择、固定测试结果、rolling 评估和展示页面数据来源。

导出资产包括：
- `tables/final_selection_basis.csv`
- `tables/final_test_leaderboard.csv`
- `tables/rolling_leaderboard.csv`
- `tables/rolling_predictions.csv`
- `figures/figure1_multimodal_selection_error.png`
- `figures/figure2_timemixer_directional_hit_rate.png`
- `figures/figure3_timemixer_cumulative_direction_calls.png`
- `figures/figure4_test_prediction_overlay.png`
- `figures/figure5_advantage_matrix.png`

## 顶层目录说明

| 目录 | 作用 |
| --- | --- |
| `0_docs/` | 文献、立项材料、比赛资料与历史归档，不进入主线运行 |
| `1_data_handling/` | 原始数据主表、覆盖统计、字段字典与盘点 |
| `2_encoding_feature/` | 结构化、文本、图像编码与融合特征、时间窗口构建 |
| `3_modeling/` | 基线模型、TimeMixer 主线、正式结果与导出图表 |
| `4_decision_rl/` | 展示页面、前端数据构建和结果索引 |
| `project_shared/` | 全流程共享的日期、频率、路径、目标和 IO 工具 |

## 根目录关键文件说明

| 文件 | 描述性说明 |
| --- | --- |
| `README.md` | 根说明文档，记录任务定义、数据范围、正式结果和运行入口。 |
| `.gitignore` | 版本控制排除规则，主要覆盖本地环境、缓存、大型编码输出和训练检查点。 |

## 运行入口

```powershell
python 1_data_handling/_collection_scripts/profile_dataset_inventory.py
python 2_encoding_feature/run_all_real_data_pipeline.py
python 2_encoding_feature/time_series/run_time_series_pipeline.py --frequency daily --variants fusion,text,image,structured
python 3_modeling/run_modeling_benchmarks.py --frequency daily
python 4_decision_rl/build_decision_dashboard_data.py
```
