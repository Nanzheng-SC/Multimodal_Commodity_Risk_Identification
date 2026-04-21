# structured

本目录负责把 `1_data_handling/raw/structured/structured_daily_merged.csv` 转成建模可直接使用的结构化特征矩阵。

## 文件

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 包标记。 |
| `process_structured.py` | 结构化特征处理入口。 |

## 数据流

1. 读取 `structured_daily_merged.csv` 或按频率读取月表。
2. 根据 `project_shared.targets` 构造 reference、步长收益、波动与目标相关字段。
3. 生成编码层使用的历史派生特征，例如 `brent_return_3d`、`brent_volatility_30d`、`wti_brent_spread`、`brent_zscore_30d` 等。
4. 输出 `structured_features.npy`、`structured_labels.npy`、`structured_reference.npy` 和索引文件。

## 可调整项

`process_structured.py` 里最值得调整的部分：

| 项目 | 影响 |
| --- | --- |
| `REGIME_DERIVED_COLUMNS` | 控制是否加入历史收益、波动、价差和 regime 特征。 |
| `STRUCTURED_DROP_COLUMNS` | 控制哪些原始列不进入建模特征。 |
| 频率参数 `daily/monthly` | 控制读取日表还是月表。 |

## 作用

结构化模块是整个项目的价格主干。它提供：

- Brent / WTI / USD / EPU / GPR / FX 等市场背景
- 历史收益率、波动率与 regime 状态
- 建模所需 reference 与标签对齐基准

下游 TimeMixer 和融合器都依赖这里的 reference 与结构化上下文，因此这个模块决定了整个项目的价格解释基础。
