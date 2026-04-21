# 1_data_handling

这个目录保存项目当前主线使用的数据资产。当前有效时间范围固定为 `2022-04-17` 至 `2026-04-16`，共 `1461` 个自然日。结构化、文本、图片三类数据全部按 `date` 对齐，下游编码与建模都从这里读取。

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

## 文件作用

| 文件 | 作用 |
| --- | --- |
| `DATA_INVENTORY.json` | 机器可读的数据盘点，记录规模、覆盖、字段、来源统计和存储占用。 |
| `DATA_INVENTORY.md` | 人工阅读版数据盘点。 |
| `raw/structured/field_registry.py` | 结构化字段注册表，说明原始字段与编码阶段历史派生字段的来源、频率、填充口径和建模用途。 |
| `raw/structured/structured_daily_merged.csv` | 日频结构化主表，是结构化编码和建模的直接输入。 |
| `raw/structured/structured_monthly_derived.csv` | 由日表派生的月末锚点表。 |
| `raw/structured/structured_dataset_manifest.json` | 结构化构建摘要和字段级说明。 |
| `raw/text/text_documents_multisource_cleaned.jsonl.gz` | 文本主表，按记录保存真实外部文本。 |
| `raw/text/text_daily_coverage.json` | 文本日覆盖摘要。 |
| `raw/text/text_daily_coverage.csv` | 文本按日覆盖明细。 |
| `raw/image/image_manifest_commons_cleaned.jsonl.gz` | 图片主表，按记录保存真实外部图片元数据。 |
| `raw/image/thumbnails_webp/` | 图片缩略图目录，下游图像编码直接读取 `thumbnail_path`。 |
| `raw/image/image_daily_coverage.json` | 图片日覆盖摘要。 |
| `raw/image/image_daily_coverage.csv` | 图片按日覆盖明细。 |
| `raw/image/image_collection_report.json` | 图片来源统计、覆盖情况和存储摘要。 |
| `raw/event/event_manifest.jsonl` | 事件扩展数据，当前不进入主线训练。 |
| `_collection_scripts/` | 本地数据维护脚本目录，已在 `.gitignore` 中排除，不属于打包交付资产。 |

## 数据如何流动

1. `run_collection.py` 按给定日期范围刷新结构化、文本、图片数据。
2. `build_structured_datasets.py` 从官方日频源抓取并合并结构化序列，生成 `structured_daily_merged.csv`，再派生 `structured_monthly_derived.csv`。
3. `collect_daily_text_dataset.py` 按日抓取公开新闻与官方文本，写入 `text_documents_multisource_cleaned.jsonl.gz` 和覆盖文件。
4. `collect_daily_image_dataset.py` 按日抓取新闻图、遥感图和开放图库图片，写入图片 manifest 与 `thumbnails_webp/`。
5. `profile_dataset_inventory.py` 扫描当前数据资产，重建 `DATA_INVENTORY.json` 和 `DATA_INVENTORY.md`。
6. `2_encoding_feature` 只读取 `raw/structured`、`raw/text`、`raw/image` 中的主文件，不读取采集中间文件。

## 结构化数据

当前日频主表规模为 `1461` 行、`15` 列。下面表格描述的是当前日表实际保留字段。

| 字段 | 来源 | 含义 | 作用 |
| --- | --- | --- | --- |
| `date` | 自然日历 | 主时间键 | 与文本、图片按日对齐。 |
| `Brent` | FRED `DCOILBRENTEU` | Brent 现货价格 | 核心价格序列。 |
| `WTI` | FRED `DCOILWTICO` | WTI 现货价格 | 油价对照序列。 |
| `USD_Index` | FRED `DTWEXBGS` | 美元广义指数 | 宏观与汇率环境。 |
| `EPU` | FRED `USEPUINDXD` | 美国经济政策不确定性指数 | 不确定性输入。 |
| `GPR` | Iacoviello GPR daily | 地缘政治风险指数 | 风险环境输入。 |
| `UAH_per_USD` | NBU API | 乌克兰格里夫纳兑美元 | 地缘冲击相关汇率。 |
| `RUB_per_USD` | CBR XML | 俄罗斯卢布兑美元 | 俄相关汇率。 |
| `AED_per_USD` | CBUAE 官方口径 | 迪拉姆兑美元 | 中东汇率参考。 |
| `reference_brent` | 派生 | 当前参考 Brent 价格 | 残差任务和前端风险解释基准。 |
| `brent_step_return` | 派生 | 相邻日收益率 | 短期动量。 |
| `brent_step_volatility_7` | 派生 | 7 日收益波动 | 波动状态输入。 |
| `market_closed_flag` | 派生 | 非交易日或前向延续标记 | 区分市场关闭与真实报价更新。 |
| `target_brent_avg_next_7d` | 派生 | 未来 7 日 Brent 均价 | 数据层监督标签保留项。 |
| `target_brent_day7` | 派生 | 第 7 日 Brent 价格 | 数据层辅助标签保留项。 |

月表 `structured_monthly_derived.csv` 由日表直接派生，不单独引入额外源。

`field_registry.py` 中还登记了编码阶段会继续生成的历史派生字段，例如 `brent_return_3d`、`brent_volatility_30d`、`wti_brent_spread` 和 `brent_zscore_30d`。这些字段不直接写回当前 raw 日表，而是在 `2_encoding_feature/structured/process_structured.py` 中生成并进入结构化特征矩阵。

## 文本数据

文本主表为 `raw/text/text_documents_multisource_cleaned.jsonl.gz`，当前规模为 `8852` 条，覆盖 `1461/1461` 天。

### 来源

| 来源类型 | 条数 | 说明 |
| --- | ---: | --- |
| `news` | `8523` | 公开新闻与网页文本，主要通过 GDELT 发现链路获取。 |
| `official` | `319` | EIA Today in Energy 等官方能源文本。 |
| `official_public_daily` | `10` | NASA APOD 等公开日频官方文本。 |

### 主要字段

| 字段 | 含义 | 作用 |
| --- | --- | --- |
| `date` | 文本归属日期 | 与结构化日表对齐。 |
| `doc_id` | 文档唯一 ID | 去重与追踪。 |
| `title` | 标题 | 文本编码输入。 |
| `summary` | 摘要或片段 | 文本编码主输入。 |
| `text` | 可保存全文时的正文 | 开放来源可用于增强编码。 |
| `content_storage` | 存储策略 | 区分摘要保存与全文保存。 |
| `source_name` / `domain` | 来源名与域名 | 来源审计。 |
| `source_type` | 来源类别 | 区分新闻、官方等来源。 |
| `topic_bucket` / `topic_tags` | 主题桶与主题标签 | 主题解释与筛选。 |
| `url` / `canonical_url` | 原始链接与规范链接 | 溯源与去重。 |
| `image_url_hint` / `social_image_url` | 页面配图线索 | 图片采集辅助输入。 |
| `language` | 语种 | 下游文本编码过滤。 |
| `open_source` | 是否开放来源 | 对应文本保存策略。 |

## 图片数据

图片主表为 `raw/image/image_manifest_commons_cleaned.jsonl.gz`，当前规模为 `1461` 条，覆盖 `1461/1461` 天。训练与编码统一读取 WebP 缩略图，不读取原始大图。

### 来源

| 来源流 | 条数 | 类型 | 说明 |
| --- | ---: | --- | --- |
| `gdelt_doc_socialimage` | `686` | `news_image` | 新闻配图。 |
| `nasa_gibs` | `429` | `satellite_image` | NASA GIBS 遥感缩略图。 |
| `wikimedia` | `103` | `commons` | Wikimedia Commons 开放图片。 |
| `unsplash` | `91` | `stock_photo` | 免费素材图。 |
| `copernicus_ogc` | `89` | `satellite_image` | Copernicus OGC 遥感图。 |
| `pexels` | `63` | `stock_photo` | 免费素材图。 |

### 主要字段

| 字段 | 含义 | 作用 |
| --- | --- | --- |
| `date` | 图片归属日期 | 与结构化日表对齐。 |
| `image_id` | 图片唯一 ID | 去重与文件追踪。 |
| `source_stream` | 来源流 | 区分新闻图、遥感图、图库图。 |
| `source_type` | 图片类型 | 下游解释与分组。 |
| `source_name` / `domain` | 来源名与域名 | 来源审计。 |
| `image_url` | 原始图片 URL | 溯源。 |
| `thumbnail_path` | 本地 WebP 路径 | 图像编码直接输入。 |
| `width` / `height` | 缩略图尺寸 | 编码前校验。 |
| `license_short` / `license_url` / `credit` | 授权与署名 | 合规追踪。 |
| `article_url` / `page_url` | 页面链接 | 来源落点。 |
| `published_at` | 原始发布时间 | 日期审计。 |
| `roi_name` / `bbox` / `cloud_cover` | 遥感 ROI、范围、云量 | 遥感图解释。 |
| `query_keyword` | 采集关键词 | 采集策略审计。 |
| `collection_status` | 采集状态 | 清洗与质控。 |

## event 数据

`raw/event/event_manifest.jsonl` 当前保留 `18` 条事件记录，主要用于扩展研究，不进入当前主线训练。主要字段包括 `date`、`event_id`、`event_tags`、`country_focus`、`source_name`、`title`、`url`、`notes` 和 `text`。

## 采集与维护脚本

| 脚本 | 作用 | 常用调整项 |
| --- | --- | --- |
| `run_collection.py` | 数据刷新总入口 | 日期范围、是否刷新结构化/文本/图片、文本日限额、图片日限额、存储模式。 |
| `build_structured_datasets.py` | 构建结构化日表与月表 | 日期范围、结构化源适配、字段映射、派生规则。 |
| `collect_daily_text_dataset.py` | 采集公开文本 | 查询桶、窗口长度、每日保留条数、是否抓开放来源正文。 |
| `collect_daily_image_dataset.py` | 采集真实图片 | 新闻、遥感、图库各源上限与优先级；Wayback 开关。 |
| `profile_dataset_inventory.py` | 重建数据盘点 | 盘点输出口径、来源统计和磁盘统计。 |
| `collection_utils.py` | 通用工具 | HTTP 重试、超时、HTML 提取和 WebP 保存参数。 |

常用维护命令：

```powershell
python 1_data_handling/_collection_scripts/run_collection.py --start-date 2022-04-17 --end-date 2026-04-16 --storage-mode light
python 1_data_handling/_collection_scripts/profile_dataset_inventory.py
```

## 下游读取约定

- `2_encoding_feature` 只读取 `structured_daily_merged.csv`、`text_documents_multisource_cleaned.jsonl.gz`、`image_manifest_commons_cleaned.jsonl.gz` 和 `thumbnails_webp/`。
- 图像编码只读取 `thumbnail_path`。
- 文本编码优先拼接 `title`、`summary` 和可用 `text`。
- 建模阶段的 30 日残差标签在窗口构建阶段动态生成，不直接写回原始日表。
