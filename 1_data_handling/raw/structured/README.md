# Structured Data

本目录保存近 4 年 Brent 预测任务的结构化数据，活跃范围为 `2022-04-17` 至 `2026-04-16`。

## 文件

| 文件 | 说明 |
| --- | --- |
| `structured_daily_merged.csv` | 日频结构化主表，每个自然日一行。 |
| `structured_monthly_derived.csv` | 由日频主表派生的月末锚点表。 |
| `structured_dataset_manifest.json` | 字段来源、规模和构建摘要。 |
| `field_registry.py` | 活跃字段注册表，供采集和编码读取。 |

## 日频字段

| 字段 | 含义 |
| --- | --- |
| `date` | 自然日主键。 |
| `Brent` | Brent 原油现货价格。 |
| `WTI` | WTI 原油现货价格。 |
| `USD_Index` | 美元广义指数。 |
| `EPU` | 美国经济政策不确定性 7-day 指数。 |
| `GPR` | 地缘政治风险日频指数。 |
| `UAH_per_USD` | 乌克兰格里夫纳兑美元。 |
| `RUB_per_USD` | 俄罗斯卢布兑美元。 |
| `AED_per_USD` | 迪拉姆兑美元官方固定汇率。 |
| `reference_brent` | Brent 参考价格。 |
| `brent_step_return` | Brent 相邻日收益率。 |
| `brent_step_volatility_7` | Brent 7 日滚动波动。 |
| `market_closed_flag` | 非交易日或价格前向延续标记。 |
| `target_brent_avg_next_7d` | 未来 7 个自然日 Brent 均价。 |
| `target_brent_day7` | 第 7 个自然日 Brent 价格。 |

## 当前规模

- 日频主表：`1461` 行，`15` 列。
- 月频派生表：`49` 行，`17` 列。
- 日频日期范围：`2022-04-17` 至 `2026-04-16`。
- 月频日期范围：`2022-04-30` 至 `2026-04-16`。

建模主线使用 horizon30 目标，目标在窗口构建阶段由 Brent 序列动态生成。
