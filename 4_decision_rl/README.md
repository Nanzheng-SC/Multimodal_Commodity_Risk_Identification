# 4_decision_rl

## 目录定位

`4_decision_rl` 保存静态展示页面和前端数据构建脚本。该目录读取 `3_modeling` 与 `5_statistical_analysis` 的正式结果，生成可复核的展示数据包。

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
| `build_decision_dashboard_data.py` | 前端数据构建脚本，整合结构化主表、官方模型结果、rolling 指标、风险阶段表现和评估图表。 |
| `decision_dashboard_data.json` | 页面运行使用的结构化数据源。 |
| `decision_dashboard_data.js` | 与 JSON 同步的浏览器加载文件，提供静态页面全局数据对象。 |
| `enterprise_risk_dashboard.html` | 静态交互页面，包含 Dashboard、Assessment、Industry 和 Evidence 四个视图。 |
| `FRONTEND_RISK_HEDGING_GUIDE.md` | 前端指标口径说明，记录风险信号、行业参数和使用边界。 |

## 前端数据流

```text
3_modeling/results/export/daily_horizon30_late_gru_gate_mainline_final
  + 3_modeling/results/official/daily_horizon30/*
  + 5_statistical_analysis/outputs/tables
  + 5_statistical_analysis/outputs/figures
  + 1_data_handling/raw/structured/structured_daily_merged.csv
  -> build_decision_dashboard_data.py
  -> decision_dashboard_data.json / decision_dashboard_data.js
  -> enterprise_risk_dashboard.html
```

## 页面结构

前端当前包含四个视图：

### 1. Dashboard
- 英雄区风险预警灯
- 30 日 Brent 路径监测
- 风险信号摘要

### 2. Assessment
- 企业套保测算面板
- 风险等级徽章
- 暴露测算摘要
- 套保比例测算
- 风险下降幅度
- 结果摘要复制按钮

### 3. Industry
- 行业价格传导图谱
- 各行业的敏感点、套保重点和政策关注点

### 4. Evidence
- 固定 test RMSE 对比
- Rolling 稳健性对比
- 当前评估图表

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
- 测算套保比例
- 新增套保比例
- 风险下降幅度
- 利润波动暴露
- 摘要文本

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
- `risk_regime_performance`
- `leaderboards`
- `prediction_series`
- `latest_market`
- `industry_profiles`
- `historical_distribution`
- `figures`
- `source_files`

## 使用口径

- 页面展示基于已落盘的模型结果和统计分析输出。
- `risk_signal` 由最新可验证预测、参考价和 rolling 残差分布阈值生成。
- Assessment 视图中的企业参数属于情景输入，不改变模型预测结果。
- Evidence 视图用于核对固定测试、rolling 稳健性和风险阶段评估结果。
