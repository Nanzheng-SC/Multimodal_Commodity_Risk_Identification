# raw/structured

## 目录定位

本目录保存结构化数据主表、字段注册表和结构化数据清单。它是全项目的日频时间轴基准。

## 文件结构

| 文件 | 作用 |
| --- | --- |
| `structured_daily_merged.csv` | 真日频结构化主表，按 `date` 对齐 |
| `structured_monthly_derived.csv` | 由日表按月末锚点派生的月表 |
| `structured_dataset_manifest.json` | 结构化表的来源、范围与规模摘要 |
| `field_registry.py` | 字段注册表，记录来源、频率、填充规则和是否参与建模 |
| `__init__.py` | 包入口 |

## 字段分组

### 价格基准
- `Brent`
- `WTI`

### 宏观、汇率与不确定性
- `USD_Index`
- `EPU`
- `GPR`
- `UAH_per_USD`
- `RUB_per_USD`
- `AED_per_USD`

### 派生特征
- `reference_brent`
- `brent_step_return`
- `brent_step_volatility_7`
- `brent_return_3d`
- `brent_return_7d`
- `brent_return_14d`
- `brent_return_30d`
- `brent_volatility_14d`
- `brent_volatility_30d`
- `wti_brent_spread`
- `usd_index_return_7d`
- `gpr_change_7d`
- `epu_change_7d`
- `brent_zscore_30d`
- `high_volatility_regime`
- `fast_uptrend_regime`
- `market_closed_flag`

### 标签
- `target_brent_avg_next_7d`
- `target_brent_day7`

其中日表当前实际保存 `15` 列；更长的派生字段清单由 `field_registry.py` 维护，编码阶段按该清单生成特征。

## 数据来源与规则

- FRED：Brent、WTI、USD_Index、EPU
- Iacoviello GPR：GPR
- NBU / CBR / CBUAE：汇率
- 非交易日采用前向延续，同时保留 `market_closed_flag`
- 月表不单独抓取，而是由日表月末锚点派生

## 与下游的关系

- `2_encoding_feature/structured/process_structured.py` 基于本目录生成结构化特征数组和标签数组
- `project_shared/targets.py` 中的价格标签逻辑直接作用于这里的日表
