# Text Data

本目录保存近 4 年真实外部文本数据，活跃范围为 `2022-04-17` 至 `2026-04-16`。

## 文件

| 文件 | 说明 |
| --- | --- |
| `text_documents_multisource_cleaned.jsonl.gz` | 文本主表，gzip JSONL。 |
| `text_daily_coverage.json` | 日覆盖统计。 |
| `text_daily_coverage.csv` | 日覆盖统计表。 |

## 当前规模

- 文本记录：`8852` 条。
- 日覆盖：`1461 / 1461` 天。
- 日期范围：`2022-04-17` 至 `2026-04-16`。

## 来源类型

| 来源类型 | 条数 | 说明 |
| --- | ---: | --- |
| `news` | `8523` | 新闻和公开网页文本，主要来自 GDELT 发现链路。 |
| `official` | `319` | EIA Today in Energy 等官方能源文本。 |
| `official_public_daily` | `10` | NASA APOD 官方公共日频文本。 |

## 主要字段

| 字段 | 含义 |
| --- | --- |
| `date` | 文档归属日期。 |
| `doc_id` | 文档唯一 ID。 |
| `title` | 标题。 |
| `summary` | 摘要或片段。 |
| `text` | 可保存全文时的正文。 |
| `content_storage` | 内容保存策略。 |
| `source_name` / `domain` | 来源名称和域名。 |
| `source_type` | 来源类别。 |
| `topic_bucket` / `topic_tags` | 主题桶和主题标签。 |
| `url` / `canonical_url` | 原始链接和规范化链接。 |
| `image_url_hint` / `social_image_url` | 页面配图线索。 |
| `language` | 语言。 |
| `open_source` | 开放来源标记。 |

下游文本编码会优先组合 `title`、`summary` 和可用 `text`。
