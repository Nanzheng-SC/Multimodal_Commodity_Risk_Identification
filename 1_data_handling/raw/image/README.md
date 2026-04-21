# raw/image

## 目录定位

本目录保存图片主清单、按日覆盖统计、采集报告和 WebP 缩略图，是图像编码模块的直接输入。

## 文件结构

| 文件 | 作用 |
| --- | --- |
| `image_manifest_commons_cleaned.jsonl.gz` | 图片主清单 |
| `image_daily_coverage.json` | 按日覆盖统计 |
| `image_daily_coverage.csv` | 按日覆盖统计表格版 |
| `image_collection_report.json` | 图片来源、命中数和存储摘要 |
| `thumbnails_webp/` | 统一缩略图目录，供图像编码直接读取 |

## 主表字段

图片主清单核心字段包括：

- 标识与时间：`image_id`、`date`、`year_month`、`published_at`
- 来源：`source_name`、`source_stream`、`source_type`、`domain`、`open_source`
- 检索与场景：`query_keyword`、`country_focus`、`roi_name`、`bbox`、`scene_tags`
- 图像信息：`image_url`、`thumbnail_path`、`mime`、`thumbnail_mime`、`width`、`height`
- 版权与署名：`credit`、`artist`、`license_short`、`license_url`
- 质量与状态：`collection_status`、`content_storage`、`cloud_cover`
- 页面与关联：`page_url`、`article_url`、`source_doc_id`、`download_location`

## 数据特点

- 当前共 `1461` 条记录，对应 `1461` 个自然日，一天一图
- 覆盖 `2022-04-17` 至 `2026-04-16`
- 当前来源统计：
  - `gdelt_doc_socialimage`: `686`
  - `nasa_gibs`: `429`
  - `wikimedia`: `103`
  - `copernicus_ogc`: `89`
  - `unsplash`: `91`
  - `pexels`: `63`

## 数据流

```text
新闻图像 / 遥感 / 开放图库
  -> collect_daily_image_dataset.py
  -> image_manifest_commons_cleaned.jsonl.gz
  -> thumbnails_webp/
  -> 2_encoding_feature/image_clip
```
