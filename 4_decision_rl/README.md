# 4_decision_rl

## 目录定位

`4_decision_rl` 是项目的展示与业务落地层。它把 `3_modeling` 导出的主线结果转成企业管理者能够直接讨论和使用的风险预警、套保测算和行业传导解释。

当前接入的正式模型为：
- 模型：`TimeMixer`
- 输入：`fusion`
- 融合器：`late.gru_gate`
- 任务：未来 `30` 日 Brent 均价残差预测

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

## 关键文件说明

| 文件 | 作用 |
| --- | --- |
| `build_decision_dashboard_data.py` | 这是前端数据总装脚本，负责把结构化主表、官方模型结果、rolling 指标和行业画像整理成页面可直接消费的数据包。前端之所以能同时展示风险灯、证据页和企业测算，靠的就是这一步统一组装。 |
| `decision_dashboard_data.json` | 这是标准数据源文件，保存页面运行所需的全部结构化内容。它的角色类似前端的数据底稿，便于独立检查页面展示是否与模型导出保持一致。 |
| `decision_dashboard_data.js` | 这是为静态页面直接挂载准备的数据镜像，把 JSON 转成浏览器可立即读取的全局变量形式。这样前端可以在不依赖后端服务的情况下完成展示。 |
| `enterprise_risk_dashboard.html` | 这是最终交互页面本体，承载风险预警、套保测算、行业传导和证据展示四个核心视图。它不是单纯的结果看板，而是整个项目业务落地的展示出口。 |
| `FRONTEND_RISK_HEDGING_GUIDE.md` | 这份文档解释了前端业务逻辑背后的金融含义，包括风险如何定义、Brent 价格如何向企业成本传导、行业敏感点如何区分以及套保建议为什么成立。它为页面中的警示灯、建议比例和行业画像提供业务解释。 |

## 前端数据流

```text
3_modeling/results/export/daily_horizon30_late_gru_gate_mainline_final
  + 3_modeling/results/official/daily_horizon30/*
  + 1_data_handling/raw/structured/structured_daily_merged.csv
  -> build_decision_dashboard_data.py
  -> decision_dashboard_data.json / decision_dashboard_data.js
  -> enterprise_risk_dashboard.html
```

## 页面结构

前端当前包含四个主要视图：

### 1. Dashboard
- 英雄区风险预警灯
- 30 日 Brent 路径监测
- 行动建议区

### 2. Assessment
- 企业套保测算面板
- 风险等级徽章
- 一句话结论
- 建议套保比例
- 风险下降幅度
- 结果摘要复制按钮

### 3. Industry
- 行业价格传导图谱
- 各行业的敏感点、套保重点和政策关注点

### 4. Evidence
- 固定 test RMSE 对比
- Rolling 稳健性对比
- 模型验证图谱

## 前端核心业务逻辑

### 风险灯
风险灯由 `risk_signal` 驱动，核心字段包括：
- `base_risk_value`
- `predicted_residual_pct`
- `level_key`
- `level_label`
- `direction`

页面会把主线模型给出的未来 30 日均价残差，转成：
- 低风险
- 观察
- 偏高
- 高风险

### 企业套保测算
企业测算使用以下输入：
- 年收入
- 能源成本占比
- 净利率
- 资产负债率
- 当前已有套保比例
- 行业 profile
- 情景系数

输出包括：
- 企业调整后风险值
- 建议套保比例
- 新增套保比例
- 风险下降幅度
- 利润波动暴露
- 管理层一句话摘要

### 行业传导
当前预置行业包括：
- 航空与航运
- 物流与公路运输
- 化工与塑料
- 炼化企业
- 油气上游

每个行业 profile 都包含：
- `exposure_type`
- `sensitivity`
- `exposure_multiplier`
- `basis_multiplier`
- `hedge_efficiency`
- `transmission`
- `hedging_focus`
- `policy_focus`
- 默认经营参数

## build_decision_dashboard_data.py 产出内容

当前前端数据包包含：
- `task`
- `model`
- `official_metrics`
- `rolling`
- `risk_signal`
- `leaderboards`
- `prediction_series`
- `latest_market`
- `industry_profiles`
- `historical_distribution`
- `figures`
- `source_files`

这意味着前端同时具备：
- 风险预警
- 企业测算
- 行业解释
- 证据展示

## 前端解读

当前前端不是只展示单个预测点，而是把模型结果完整映射到企业风险管理流程：

- 风险灯解决“当前该不该警觉”
- Brent 路径监测解决“价格风险往哪走”
- 企业套保测算解决“企业该覆盖多少敞口”
- 行业传导图谱解决“风险为何会从 Brent 传导到具体行业”
- 证据页解决“为什么当前这条主线可以作为经营讨论依据”

因此，这个前端既是项目的可视化出口，也是模型结果向业务决策传导的最后一层。
