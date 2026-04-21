# project_shared：全流程共享配置与工具

本目录保存数据、编码、建模和前端共同依赖的路径、日期、频率、目标和 IO 工具。它的作用是减少各模块重复写路径和任务口径，保证全流程使用同一套约定。

## 目录结构与文件作用

```text
project_shared/
├─ README.md
├─ __init__.py
├─ date_scope.py
├─ frequency.py
├─ io_utils.py
├─ paths.py
└─ targets.py
```

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | Python 包标记。 |
| `date_scope.py` | 统一日期范围解析，默认近 4 年。 |
| `frequency.py` | 统一 daily/monthly 频率、索引列和窗口长度配置。 |
| `io_utils.py` | 通用文件读写工具。 |
| `paths.py` | 项目根目录、raw 数据、编码输出、建模结果和前端文件路径。 |
| `targets.py` | Brent 目标列和目标构造相关工具。 |

## 数据如何流动

`project_shared` 不直接生产数据，但所有主线模块都会读取这里的约定：

1. `1_data_handling` 使用日期范围和路径约定维护 raw 数据。
2. `2_encoding_feature` 使用 `paths.py` 定位 raw 数据和输出目录，使用 `frequency.py` 选择 daily/monthly 窗口。
3. `3_modeling` 使用同一套窗口长度、目标和结果目录，保证基线与 TimeMixer 对比口径一致。
4. `4_decision_rl` 使用最终结果路径和 raw 结构化数据路径生成前端数据。

## `date_scope.py` 参数

| 参数或函数 | 当前口径 | 影响 |
| --- | --- | --- |
| `DEFAULT_YEARS_BACK` | `4` | 默认数据范围向前回看 4 年。 |
| `DEFAULT_END_DATE` | `today` | 默认结束日期为运行当天。 |
| `resolve_end_date()` | 将 `today` 或日期字符串解析为标准日期 | 控制全流程结束日期。 |
| `default_start_date_for_end()` | `end_date - years_back + 1 day` | 自动生成默认起始日期。 |
| `resolve_date_scope()` | 返回 `(start_date, end_date)` | 采集、盘点和编码统一日期范围。 |
| `date_scope_metadata()` | 返回范围、年数和总天数 | 写入盘点和报告。 |

如需把主线改成近 3 年或近 5 年，优先调整调用参数 `years_back`，不要在各脚本中手写日期。

## `frequency.py` 参数

| 参数或函数 | 当前口径 | 影响 |
| --- | --- | --- |
| `SUPPORTED_FREQUENCIES` | `daily`, `monthly` | 全流程支持的频率。 |
| `SUPPORTED_FREQUENCY_CHOICES` | `daily`, `monthly`, `both` | CLI 可选频率。 |
| `WINDOW_LENGTHS_BY_FREQUENCY` | daily `[14,30,90]`，monthly `[3,6,12]` | time-series 窗口构建默认集合。 |
| `DEFAULT_WINDOW_LENGTH_BY_FREQUENCY` | daily `30`，monthly `6` | 默认窗口长度。 |
| `INDEX_COLUMN_BY_FREQUENCY` | daily `date`，monthly `month` | 数据表索引列名。 |
| `expand_frequency_choice()` | 展开 `both` | 批量运行 daily/monthly。 |
| `default_window_lengths()` | 返回频率对应窗口 | 编码和建模共享窗口集合。 |

如果要加新频率，应先在这里扩展，再同步调整数据派生、编码输出和建模路径。

## `paths.py` 路径约定

| 常量或函数 | 指向 |
| --- | --- |
| `PROJECT_ROOT` | 项目根目录。 |
| `DATA_HANDLING_ROOT` | `1_data_handling/`。 |
| `STRUCTURED_DAILY_PATH` | `raw/structured/structured_daily_merged.csv`。 |
| `STRUCTURED_MONTHLY_DERIVED_PATH` | `raw/structured/structured_monthly_derived.csv`。 |
| `TEXT_DOCUMENTS_CLEANED_GZ_PATH` | 文本 gzip 主表。 |
| `IMAGE_MANIFEST_CLEANED_GZ_PATH` | 图片 manifest gzip 主表。 |
| `IMAGE_THUMBNAIL_DIR` | WebP 缩略图目录。 |
| `EVENT_MANIFEST_PATH` | event manifest。 |
| `ENCODING_OUTPUT_ROOT` | `2_encoding_feature/outputs/`。 |
| `MODELING_RESULTS_ROOT` | `3_modeling/results/`。 |
| `DASHBOARD_JSON_PATH` / `DASHBOARD_JS_PATH` | 前端数据文件。 |
| `structured_source_path(frequency)` | 根据频率返回结构化日表或月表。 |
| `feature_root(kind, frequency)` | 返回 structured/text/image/fusion 特征目录。 |
| `time_series_root(input_variant, frequency)` | 返回对应 time-series 窗口目录。 |
| `official_model_dir()` | official 建模结果目录。 |

路径修改会影响多个模块。若移动数据或结果目录，应优先改 `paths.py`，再运行各模块测试。

## `targets.py`

`targets.py` 保存 Brent 预测目标相关工具，供编码和建模阶段复用。当前项目数据层保留 7 日标签，建模主线使用 horizon30 动态目标。目标构造应遵循以下原则：

- 只在建模窗口中使用可验证的未来 Brent 序列生成监督标签。
- 特征只使用当前日和历史信息，避免泄漏。
- 前端解释统一使用 `predicted_future_30d_brent_average - reference_brent`。

## `io_utils.py`

`io_utils.py` 存放通用读写工具，适合放置 JSON、JSONL、gzip、CSV 和目录创建相关函数。后续新增模块时优先复用这里的工具，避免每个脚本重复实现。

## 调整建议

- 改时间范围：优先改 `date_scope.py` 或 CLI 调用参数。
- 改窗口长度：优先改 `frequency.py`。
- 改数据/结果路径：优先改 `paths.py`。
- 改目标口径：优先改 `targets.py` 和建模窗口构建逻辑，确保前端解释同步更新。
