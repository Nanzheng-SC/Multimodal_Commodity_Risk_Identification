# 前端风险指标口径

## 1. 文件定位

本文件记录 `4_decision_rl` 静态页面使用的风险信号、行业参数和测算边界。页面数据由 `build_decision_dashboard_data.py` 从正式模型结果、统计分析表格和结构化主表生成。

当前模型口径：

- 模型：`TimeMixer`
- 输入：`fusion`
- 融合器：`late.gru_gate`
- 频率：日频
- 预测跨度：未来 `30` 日
- 目标：`target_brent_avg_next_30d - reference_brent`

页面只使用已落盘结果和可验证样本，不构造 `2026-04-16` 之后的真实标签。

## 2. 风险信号

核心预测残差定义为：

```text
predicted_residual = predicted_future_30d_brent_average - reference_brent
```

风险等级由 `predicted_residual / reference_brent` 的绝对值与 rolling 样本分位数比较得到。方向字段仅记录残差符号：

- `predicted_residual > 0`：未来 30 日均价高于参考价。
- `predicted_residual < 0`：未来 30 日均价低于参考价。
- `predicted_residual = 0`：预测值与参考价一致。

页面同时展示固定测试误差、rolling score、方向命中率和风险阶段表现，用于核对模型信号的历史评估结果。

## 3. 行业参数

行业参数用于把同一 Brent 残差信号映射到不同暴露类型。当前预置行业包括：

- 航空与航运
- 物流与公路运输
- 化工与塑料
- 炼化企业
- 油气上游

每个行业 profile 包含：

- `exposure_type`
- `sensitivity`
- `exposure_multiplier`
- `basis_multiplier`
- `hedge_efficiency`
- `transmission`
- `hedging_focus`
- `policy_focus`
- 默认经营参数

这些字段属于页面测算参数，不参与模型训练或模型选择。

## 4. 测算公式

Assessment 视图中的暴露测算使用：

```text
estimated_exposure = notional_barrels * predicted_residual * industry_basis_multiplier * hedge_ratio
```

其中 `notional_barrels` 来自页面输入参数换算，`industry_basis_multiplier` 和 `hedge_ratio` 来自行业 profile 与页面情景设置。测算结果用于统一展示同一模型信号下的行业差异，不作为交易指令或会计确认值。

## 5. 使用边界

- 模型选择以 validation 与 rolling 评估结果为准。
- 页面指标只引用真实标签可验证区间内的预测与评估结果。
- RMSE、MAE、MAPE 与方向命中率分别保留原始评价含义，不混合作为单一指标。
- 行业参数和情景系数仅用于前端测算展示，不改变模型输出。
