# 1_data_handling

## 目录定位

`1_data_handling` 是项目的数据资产层，负责维护当前主线使用的结构化、文本、图片和事件数据，并生成统一的数据盘点文件。`2_encoding_feature` 从本目录读取编码输入。

当前活跃时间范围固定为 `2022-04-17` 至 `2026-04-16`。三类主模态全部按自然日对齐，形成统一的日频主时间轴。

## 目录结构

```text
1_data_handling/
├─ README.md
├─ DATA_INVENTORY.json
├─ DATA_INVENTORY.md
├─ raw/
│  ├─ structured/
│  │  ├─ README.md
│  │  ├─ __init__.py
│  │  ├─ field_registry.py
│  │  ├─ structured_daily_merged.csv
│  │  ├─ structured_monthly_derived.csv
│  │  └─ structured_dataset_manifest.json
│  ├─ text/
│  │  ├─ README.md
│  │  ├─ text_documents_multisource_cleaned.jsonl.gz
│  │  ├─ text_daily_coverage.json
│  │  └─ text_daily_coverage.csv
│  ├─ image/
│  │  ├─ README.md
│  │  ├─ image_manifest_commons_cleaned.jsonl.gz
│  │  ├─ image_daily_coverage.json
│  │  ├─ image_daily_coverage.csv
│  │  ├─ image_collection_report.json
│  │  └─ thumbnails_webp/
│  └─ event/
│     ├─ README.md
│     └─ event_manifest.jsonl
└─ _collection_scripts/
   ├─ run_collection.py
   ├─ build_structured_datasets.py
   ├─ collect_daily_text_dataset.py
   ├─ collect_daily_image_dataset.py
   ├─ profile_dataset_inventory.py
   └─ collection_utils.py
```

## 当前数据集

| 数据集 | 文件 | 规模 | 作用 |
| --- | --- | --- | --- |
| 结构化日表 | `raw/structured/structured_daily_merged.csv` | `1461 × 15` | 日频主表，供结构化编码与前端市场面板使用 |
| 结构化月表 | `raw/structured/structured_monthly_derived.csv` | `49 × 17` | 由日表月末锚点派生，供月频兼容使用 |
| 文本主表 | `raw/text/text_documents_multisource_cleaned.jsonl.gz` | `8852` 条 | 文本编码输入 |
| 图片主表 | `raw/image/image_manifest_commons_cleaned.jsonl.gz` | `1461` 条 | 图像编码输入 |
| Event 清单 | `raw/event/event_manifest.jsonl` | `18` 条 | 事件补充模态 |
| 盘点文件 | `DATA_INVENTORY.md/json` | 字段级清单 | 汇总时间范围、字段、规模和存储占用 |

## 数据来源

### 结构化
- FRED：`Brent`、`WTI`、`USD_Index`、`EPU`
- Iacoviello GPR：`GPR`
- NBU：`UAH_per_USD`
- CBR：`RUB_per_USD`
- CBUAE：`AED_per_USD`

### 文本
- 多源公开新闻网页
- 开放文章来源和正文抽取结果

### 图片
- GDELT DOC 新闻配图
- Wikimedia Commons
- NASA GIBS
- Copernicus OGC
- Unsplash
- Pexels

### Event
- 重要事件节点的人工整理清单

## 数据流

```text
外部真实源
  -> _collection_scripts 采集与清洗
  -> raw/structured | raw/text | raw/image | raw/event
  -> DATA_INVENTORY
  -> 2_encoding_feature 读取并编码
```

数据层的工作重点有两点：
- 用统一 `date` 轴把结构化、文本和图片对齐
- 把数据面固定成可复核的主表和盘点文件，避免下游直接依赖抓取过程

## 关键文件说明

| 文件 | 描述性说明 |
| --- | --- |
| `DATA_INVENTORY.json` | 机器可读的数据盘点总表，记录当前活跃数据集的规模、时间覆盖、字段集合和存储占用。 |
| `DATA_INVENTORY.md` | 数据资产说明文档，与 JSON 盘点保持同一口径，记录数据集状态、字段含义和覆盖情况。 |
| `raw/structured/field_registry.py` | 结构化字段注册表，定义列来源、频率和建模属性。 |
| `raw/structured/structured_daily_merged.csv` | 结构化日频主表，用于结构化编码、标签构造、价格展示和前端市场面板。 |
| `raw/text/text_documents_multisource_cleaned.jsonl.gz` | 文本模态输入表，保留日期、标题、摘要和来源信息。 |
| `raw/image/image_manifest_commons_cleaned.jsonl.gz` | 图像模态清单文件，记录每日图片、缩略图路径、来源和采集状态。 |
| `_collection_scripts/run_collection.py` | 数据层统一刷新入口。 |
| `_collection_scripts/collection_utils.py` | 数据采集、重试、时间范围处理和公共清洗工具。 |

## 关键脚本

| 脚本 | 作用 | 常用参数 |
| --- | --- | --- |
| `run_collection.py` | 统一刷新结构化、文本、图片并重建盘点 | `--start-date`、`--end-date`、`--years-back`、`--storage-mode`、`--skip-*` |
| `build_structured_datasets.py` | 重建结构化日表和月表 | `--start-date`、`--end-date`、`--years-back` |
| `collect_daily_text_dataset.py` | 刷新文本主表与覆盖统计 | `--daily-limit`、`--window-days`、`--maxrecords`、`--storage-mode`、`--fetch-open-source-body` |
| `collect_daily_image_dataset.py` | 刷新图片主表、缩略图和覆盖统计 | `--daily-limit`、`--max-images`、`--max-wikimedia`、`--max-nasa`、`--max-copernicus` |
| `profile_dataset_inventory.py` | 重建盘点文件 | 一般直接运行即可 |

## 参数项

- 日期范围：默认按近 4 年回溯，也可以通过 `--start-date` / `--end-date` 显式指定
- 文本采集强度：可通过 `--daily-limit`、`--window-days`、`--maxrecords` 控制单日抓取量和回看窗口
- 图片采集强度：可通过 `--max-images` 以及各来源上限控制真实图片补采规模
- 存储模式：`light` 用压缩文本与 WebP 缩略图；`full` 适合本地完整测试

## 下游关系

- `2_encoding_feature` 直接读取本目录主表，生成三模态特征和时间窗口
- `3_modeling` 通过编码层结果训练基线模型和 TimeMixer 主线
- `4_decision_rl` 读取结构化日表与导出结果构建企业风险前端
