# raw/text

## 目录定位

本目录保存文本主表与按日覆盖统计，是文本编码模块的直接输入。

## 文件结构

| 文件 | 作用 |
| --- | --- |
| `text_documents_multisource_cleaned.jsonl.gz` | 文本主表，压缩存储 |
| `text_daily_coverage.json` | 按日覆盖统计，便于程序读取 |
| `text_daily_coverage.csv` | 按日覆盖统计，便于人工查看 |

## 主表字段

文本主表的核心字段包括：

- 标识与时间：`doc_id`、`date`、`year_month`
- 来源：`source_name`、`source_type`、`domain`、`source_country`、`open_source`
- 检索与主题：`search_query`、`query_bucket`、`topic_bucket`、`topic_tags`
- 内容：`title`、`summary`、`text`
- 质量与状态：`content_quality`、`content_storage`、`has_summary`、`has_text`、`body_char_count`
- 链接：`url`、`canonical_url`、`image_url_hint`、`social_image_url`
- 语言：`language`

## 数据特点

- 当前共 `8852` 条记录
- 时间范围覆盖 `2022-04-17` 至 `2026-04-16`
- 按日覆盖 `1461/1461`
- 同一天可有多条文本，后续编码阶段按日做平均池化并保留 `text_count`

## 数据流

```text
多源公开文本
  -> collect_daily_text_dataset.py
  -> text_documents_multisource_cleaned.jsonl.gz
  -> text_daily_coverage.json/csv
  -> 2_encoding_feature/text_bert
```
