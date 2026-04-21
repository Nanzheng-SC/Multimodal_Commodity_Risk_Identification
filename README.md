# FireFlower：多模态 Brent 风险预测与企业套保决策系统

FireFlower 面向企业原油价格风险管理场景，构建了一个从数据采集、模态编码、时序建模到前端决策展示的完整闭环。项目当前主线固定为：

```text
TimeMixer + fusion + late.gru_gate
```

核心任务是预测未来 30 日 Brent 均价相对当前 Brent 的残差，用于支持企业的采购、库存、套保和现金流风险判断。

当前活跃时间范围为 `2022-04-17` 至 `2026-04-16`，共 `1461` 个自然日。数据、编码、建模和前端全部围绕该范围组织。

## 项目全流程

```text
1_data_handling
  维护结构化、文本、图片和 event 数据
  -> 生成 DATA_INVENTORY

2_encoding_feature
  structured 标准化
  multilingual BERT 文本编码
  CLIP 图像编码
  late.gru_gate 融合特征
  -> 生成 horizon30 建模窗口

3_modeling
  Naive / HAR / LSTM / TimeMixer 对照
  单模态 / 多模态对照
  late.gru_gate 主线训练、rolling 复核和 export

4_decision_rl
  读取最终结果
  -> 企业 Brent 风险套保控制台

project_shared
  提供共享日期、频率、路径、目标和 IO 约定
```

## 顶层目录说明

| 目录 | 地位 | 主要内容 |
| --- | --- | --- |
| `0_docs/` | 资料区 | 文献、立项材料、展示素材和历史参考资料，不进入主流程交付。 |
| `1_data_handling/` | 数据资产层 | 结构化主表、文本主表、图片 manifest、缩略图、event 扩展和数据盘点。 |
| `2_encoding_feature/` | 编码层 | 结构化、文本、图片、融合编码，以及日频 horizon30 时间窗口构建。 |
| `3_modeling/` | 建模层 | Naive、HAR、LSTM、TimeMixer、主线导出结果和正式图表。 |
| `4_decision_rl/` | 应用层 | 企业 Brent 风险套保前端、前端数据构建逻辑和业务解释文档。 |
| `project_shared/` | 公共约定层 | 全流程通用的日期、频率、路径、目标与 IO 工具。 |

每个顶层目录都配有独立 `README.md`。根 README 负责总览，模块 README 负责展开该模块的文件结构、数据流、脚本参数和结果解释。

## 仓库保留范围

GitHub 仓库保留源代码、数据盘点、前端文件以及论文和答辩使用的最终结果表与图。体积较大的编码派生文件、时间窗口样本和训练检查点由代码在本地生成，不作为仓库内容上传。

当前保留用于展示与交付的结果资产集中在：

```text
3_modeling/results/export/daily_horizon30_late_gru_gate_mainline_final/
4_decision_rl/
```

其中 `figures/`、`tables/`、`decision_dashboard_data.json/js` 和前端页面属于最终展示资产，会随仓库保留。

## 当前数据资产

| 模态 | 主文件 | 规模 | 覆盖 | 作用 |
| --- | --- | ---: | --- | --- |
| 结构化日频 | `1_data_handling/raw/structured/structured_daily_merged.csv` | `1461` 行，`15` 列 | `2022-04-17` -> `2026-04-16` | Brent/WTI、宏观与汇率、不确定性、价格派生特征和数据层监督标签。 |
| 结构化月频 | `1_data_handling/raw/structured/structured_monthly_derived.csv` | `49` 行，`17` 列 | `2022-04-30` -> `2026-04-16` | 由日频主表派生的月末锚点表。 |
| 文本 | `1_data_handling/raw/text/text_documents_multisource_cleaned.jsonl.gz` | `8852` 条，`25` 字段 | `1461/1461` 天 | 新闻、官方能源文本和公开日频文本，供 multilingual BERT 编码。 |
| 图片 | `1_data_handling/raw/image/image_manifest_commons_cleaned.jsonl.gz` | `1461` 条，`35` 字段 | `1461/1461` 天 | 新闻图、遥感图、开放图库和免费素材图，供 CLIP 编码。 |
| 图片缩略图 | `1_data_handling/raw/image/thumbnails_webp/` | `1461` 张 WebP | 与 manifest 完全对齐 | 图像编码实际读取对象。 |
| Event | `1_data_handling/raw/event/event_manifest.jsonl` | `18` 条，`11` 字段 | `2022-07-27` -> `2025-06-03` | 扩展事件资产，当前不进入主线训练。 |

数据来源摘要：

- 结构化：FRED、Iacoviello GPR、NBU、CBR、CBUAE
- 文本：GDELT 发现链路、EIA Today in Energy、NASA APOD
- 图片：GDELT 新闻配图、NASA GIBS、Copernicus OGC、Wikimedia Commons、Unsplash、Pexels

当前主线文本与图片资产均为真实外部来源记录。

## 数据流

1. `1_data_handling` 刷新并维护原始日频数据，输出 `DATA_INVENTORY.json/md`。
2. `2_encoding_feature` 从原始数据生成 structured/text/image/fusion 四类特征。
3. `2_encoding_feature/time_series` 生成 horizon30 的日频建模窗口。
4. `3_modeling` 基于同一窗口口径训练 Naive、HAR、LSTM、TimeMixer 和主线融合模型。
5. `3_modeling/results/export/daily_horizon30_late_gru_gate_mainline_final/` 输出最终表格与正式图。
6. `4_decision_rl/build_decision_dashboard_data.py` 读取 export、official 预测和结构化主表，生成前端数据。
7. `4_decision_rl/enterprise_risk_dashboard.html` 将模型信号转成企业风险预警、套保测算、行业传导和政策建议。

## 主线任务定义

当前主任务是日频未来 30 日 Brent 均价残差预测：

```text
target = target_brent_avg_next_30d - reference_brent
```

其中：

- `reference_brent` 表示当前日参考 Brent 价格
- `target_brent_avg_next_30d` 表示未来 30 个自然日 Brent 的平均价格

这个目标比单日价格点预测更贴近企业套保，因为企业更关心未来一个采购周期或库存周期内的均价水平。

## 当前主线结果

最终主线配置：

- 模型：`TimeMixer`
- 输入：`fusion`
- 融合器：`late.gru_gate`
- 任务：日频未来 30 日 Brent 均价残差
- 选择原则：验证集优先，固定 test 只做最终确认
- 当前状态：`validation_status = PASS`

核心结果如下：

| 指标 | 主线 | 最强对照 | 解释 |
| --- | ---: | ---: | --- |
| fixed test RMSE | `15.4705` | Image `15.8806` | 多模态主线在最终测试窗口优于全部单模态与传统基线。 |
| 6-fold rolling score | `6.0236` | late.gru_concat `6.2267` | `late.gru_gate` 在跨期滚动验证中保持最佳综合表现。 |
| rolling RMSE mean | `4.7957` | late.gru_concat `4.9688` | 主线在多个 out-of-time 窗口下维持更低平均误差。 |
| rolling direction accuracy | `72.5%` | 约 `54.7%` | 方向判断优势清晰，适合企业风险预警。 |
| fixed test direction accuracy | `95.0%` | HAR/LSTM `5.0%` | 主线在最终窗口的方向识别能力非常突出。 |

适合对外表述的结论是：

- 多模态融合能够稳定提升 Brent 风险预测质量
- TimeMixer 比传统时序基线更适合当前 30 日均价残差任务
- `late.gru_gate` 能在结构化、文本和图片之间形成更稳定的风险信号整合

## 最终结果目录

最终导出目录：

```text
3_modeling/results/export/daily_horizon30_late_gru_gate_mainline_final/
```

关键结果文件：

| 文件 | 作用 |
| --- | --- |
| `EXPORT_SUMMARY.json` | 主线结果摘要和胜出检查。 |
| `tables/final_test_leaderboard.csv` | fixed test 口径对比表。 |
| `tables/rolling_leaderboard.csv` | 6-fold rolling 对比表。 |
| `tables/final_selection_basis.csv` | 主线相对单模态、基线和融合方法的最终选择依据。 |
| `tables/rolling_predictions.csv` | rolling 预测明细。 |
| `tables/fold_metrics.csv` | 各 rolling fold 的指标表。 |
| `tables/market_focus_days.csv` | 重点市场日期明细。 |
| `figures/figure1_multimodal_selection_error.png` | 多模态相对单模态的优势图。 |
| `figures/figure2_timemixer_directional_hit_rate.png` | TimeMixer 相对基线模型的方向命中率优势图。 |
| `figures/figure3_timemixer_cumulative_direction_calls.png` | 最终测试窗口累计方向命中图。 |
| `figures/figure4_test_prediction_overlay.png` | 真实值与预测值对比图。 |
| `figures/figure5_advantage_matrix.png` | 主线优势矩阵图。 |

## 企业套保落地逻辑

前端中的核心风险值来自模型预测残差：

```text
predicted_risk = predicted_future_30d_brent_average - reference_brent
```

其业务解释为：

- `predicted_risk > 0`：未来 30 日均价预计高于当前价，采购型企业成本压力上升
- `predicted_risk < 0`：未来 30 日均价预计低于当前价，上游企业更关注收入保护

前端会结合行业传导系数、能源成本占比、利润率、负债率和现有套保比例，把模型结果转成：

- 风险预警灯
- 一句话管理层结论
- 建议套保比例
- 利润波动敞口
- 行业与政策建议

## 文档入口

- 数据层：`1_data_handling/README.md`
- 编码层：`2_encoding_feature/README.md`
- 建模层：`3_modeling/README.md`
- 前端层：`4_decision_rl/README.md`
- 公共约定：`project_shared/README.md`

本 README 已吸收当前阶段的工作记录和结果分析内容，作为整个项目的总入口。
