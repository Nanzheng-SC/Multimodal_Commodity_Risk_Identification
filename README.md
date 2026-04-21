# 多模态 Brent 风险预测与企业套保决策系统

## 项目简介

本项目面向企业原油价格风险管理场景，构建了一条从真实多源数据采集、三模态编码、融合建模到前端决策展示的完整链路。当前主线围绕 Brent 日频价格风险展开，核心任务是预测：

```text
未来 30 日 Brent 均价残差 = target_brent_avg_next_30d - reference_brent
```

当前正式主线固定为：

```text
TimeMixer + fusion + late.gru_gate
```

活跃时间范围为 `2022-04-17` 至 `2026-04-16`，共 `1461` 个自然日。结构化、文本、图片、建模与前端全部围绕这一时间范围组织。

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
  将模型输出转成企业风险预警、套保比例和行动建议

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
| Event | `1_data_handling/raw/event/event_manifest.jsonl` | `18` 条 | 事件扩展清单 |

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

### 结果解读

当前正式主线 `TimeMixer + fusion + late.gru_gate` 在固定测试窗口和 `6` 折 rolling 两个口径下都保持领先，因此被确定为最终展示版本。

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

这些结果说明两点：
- 多模态融合确实带来了稳定增益，主线同时优于单模态文本、图像和结构化输入
- TimeMixer 与 `late.gru_gate` 的组合在波动区间内保持了更好的方向判断能力和跨期稳定性，因而适合继续作为企业套保前端的核心模型

从业务角度看，主线不仅给出未来 30 日 Brent 均价残差的点预测，还把这种预测进一步转成前端中的风险灯、套保比例建议、利润波动暴露和一句话管理结论，因此结果目录中的表格和图形同时服务于模型复核与前端展示。

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
| `4_decision_rl/` | 企业风险套保前端、业务解释和前端数据构建 |
| `project_shared/` | 全流程共享的日期、频率、路径、目标和 IO 工具 |

## 根目录关键文件说明

| 文件 | 描述性说明 |
| --- | --- |
| `README.md` | 根说明文档用于概括整条主线，从任务定义、数据范围到正式结果都在这里集中呈现。它的作用不是替代各子目录说明，而是帮助读者先建立全局认知，再进入数据、编码、建模和前端的细节。 |
| `.gitignore` | 这个文件控制哪些本地产物不进入版本库，重点排除了大型编码输出、训练检查点和其他可再生中间结果。它直接决定仓库中哪些内容属于可交付资产，哪些内容保留为本地运行时产物。 |

## 运行入口

```powershell
python 1_data_handling/_collection_scripts/profile_dataset_inventory.py
python 2_encoding_feature/run_all_real_data_pipeline.py
python 2_encoding_feature/time_series/run_time_series_pipeline.py --frequency daily --variants fusion,text,image,structured
python 3_modeling/run_modeling_benchmarks.py --frequency daily
python 4_decision_rl/build_decision_dashboard_data.py
```
