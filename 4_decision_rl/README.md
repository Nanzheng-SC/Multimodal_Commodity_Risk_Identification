# 4_decision_rl：企业 Brent 风险套保前端

本目录保存项目的最终展示前端。页面接入当前主线 `TimeMixer + fusion + late.gru_gate`，把未来 30 日 Brent 均价残差预测转成企业可理解、可讨论、可执行的风险预警与套保建议。

## 目录结构

```text
4_decision_rl/
├─ README.md
├─ build_decision_dashboard_data.py
├─ decision_dashboard_data.json
├─ decision_dashboard_data.js
├─ enterprise_risk_dashboard.html
└─ FRONTEND_RISK_HEDGING_GUIDE.md
```

## 文件作用

| 文件 | 作用 |
| --- | --- |
| `build_decision_dashboard_data.py` | 从 export、official prediction 和结构化主表组装前端数据。 |
| `decision_dashboard_data.json` | 标准 JSON 数据文件，便于调试、接口化和外部读取。 |
| `decision_dashboard_data.js` | 静态页面直接引用的数据文件，定义 `window.DECISION_DASHBOARD_DATA`。 |
| `enterprise_risk_dashboard.html` | 企业 Brent 风险套保控制台页面。 |
| `FRONTEND_RISK_HEDGING_GUIDE.md` | 风险定义、价格传导、行业敏感点和套保业务解释。 |

## 数据如何流动

1. `3_modeling/results/export/daily_horizon30_late_gru_gate_mainline_final/EXPORT_SUMMARY.json` 提供主线模型名称、核心指标和 rolling 摘要。
2. `3_modeling/results/export/daily_horizon30_late_gru_gate_mainline_final/tables/*.csv` 提供 leaderboard、fold 指标、重点市场日期和 rolling 预测。
3. `3_modeling/results/official/daily_horizon30/timemixer_late_gru_gate_mainline_final/window_validation_selected/best_run/predictions_test.csv` 提供官方测试窗口预测序列。
4. `1_data_handling/raw/structured/structured_daily_merged.csv` 提供最新 Brent、WTI、USD、EPU、GPR 等市场背景。
5. `build_decision_dashboard_data.py` 将这些结果整合成 `decision_dashboard_data.json/js`。
6. `enterprise_risk_dashboard.html` 读取 `decision_dashboard_data.js` 完成页面渲染。

前端展示依赖的正式图表和结果表来自：

```text
3_modeling/results/export/daily_horizon30_late_gru_gate_mainline_final/
```

该目录中的 `figures/` 与 `tables/` 属于最终展示资产，保留在仓库中。

## 前端页面功能

| 模块 | 功能 |
| --- | --- |
| 风险驾驶舱 | 展示风险灯、管理层一句话总结、核心风险值、价格路径和行动建议。 |
| 企业测算 | 输入行业、收入、能源成本占比、利润率、负债率、现有套保比例，输出建议套保比例和利润波动敞口。 |
| 行业传导 | 说明航空航运、化工材料、炼化、物流运输、上游油气等行业的 Brent 敏感点。 |
| 模型依据 | 展示主线模型相对单模态与传统基线的验证依据，用于增强业务部门的可解释性。 |

## 风险定义

前端核心风险值来自模型预测残差：

```text
predicted_risk = predicted_future_30d_brent_average - reference_brent
```

业务解释如下：

- `predicted_risk > 0`：未来 30 日 Brent 均价预计高于当前价，采购型企业成本压力上升。
- `predicted_risk < 0`：未来 30 日 Brent 均价预计低于当前价，采购压力下降，上游企业收入保护重要性上升。

前端会把该风险值结合：

- 行业传导系数
- 能源成本占比
- 利润率
- 负债率
- 现有套保比例

进一步转成企业层面的建议套保比例、调整后风险和利润波动敞口。

## `build_decision_dashboard_data.py` 可调整内容

| 配置或逻辑 | 作用 | 调整影响 |
| --- | --- | --- |
| `EXPORT_ROOT` | 最终 export 路径 | 切换展示的模型版本。 |
| `OFFICIAL_ROOT` | official prediction 根目录 | 切换预测序列来源。 |
| `STRUCTURED_DAILY` | 结构化日表路径 | 更新市场背景与最新行情。 |
| `PREDICTION_PATHS` | 前端展示的模型预测文件列表 | 增减对比曲线。 |
| `INDUSTRY_PROFILES` | 行业参数、默认企业参数和行业文案 | 调整行业传导与企业测算。 |
| `SCENARIO_CONFIG` | 温和、基准、压力、自定义情景系数 | 调整企业测算情景。 |
| `sanitize_json()` | 清洗 NaN / Infinity | 保证前端 JSON 可被浏览器正常解析。 |

## 当前前端数据结构

`decision_dashboard_data.json` 的核心字段如下：

| 字段 | 含义 |
| --- | --- |
| `data_scope` | 数据范围和样本规模。 |
| `task` | 任务定义、频率和 horizon。 |
| `model` | 当前主线模型、输入模态、融合器和验证状态。 |
| `latest_market` | 最新 Brent、WTI、USD Index 和近 30 日变化。 |
| `model_forecast` | 最新预测均价、参考 Brent、残差、方向和风险等级。 |
| `risk_signal` | 风险灯等级、阈值和业务解释。 |
| `historical_distribution` | 历史残差分布，用于判断风险位置。 |
| `scenarios` | 企业测算情景。 |
| `official_metrics` | 主线 official 指标。 |
| `leaderboards` | fixed test、rolling 和 selection basis 对比表。 |
| `prediction_series` | 测试窗口预测曲线。 |
| `rolling` | rolling fold 指标、预测和重点市场日期。 |
| `industry_profiles` | 各行业传导参数与默认企业设定。 |
| `figures` | 前端引用的模型依据图片路径。 |
| `source_files` | 前端数据来源文件清单。 |

## 运行方式

```powershell
python 4_decision_rl/build_decision_dashboard_data.py
```

然后直接打开：

```text
4_decision_rl/enterprise_risk_dashboard.html
```

## 当前结果解读

前端当前展示的主线结果来自 `TimeMixer + fusion + late.gru_gate`：

- fixed test RMSE `15.4705`
- rolling score `6.0236`
- rolling direction accuracy `72.5%`
- fixed test direction accuracy `95.0%`

这些结果意味着当前前端中的风险灯、套保建议和行业解释，不是孤立的静态展示，而是建立在已经通过 test 和 rolling 双口径验证的主线模型之上。

## 前端在整个项目中的位置

`4_decision_rl` 不是训练模块，而是主线结果的业务落地层。它把建模输出转成企业能直接阅读和讨论的决策界面，因此承担的是：

- 把模型指标转成风险语言
- 把 Brent 价格变化转成行业传导逻辑
- 把预测信号转成可执行的套保建议

这个目录与 `FRONTEND_RISK_HEDGING_GUIDE.md` 一起，构成项目最终落地展示的核心部分。
