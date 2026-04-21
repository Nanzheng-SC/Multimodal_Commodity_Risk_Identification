# Image Data

本目录保存近 4 年真实来源图片数据，活跃范围为 `2022-04-17` 至 `2026-04-16`。下游图像编码读取 WebP 缩略图，不读取全尺寸原图。

## 文件

| 文件或目录 | 说明 |
| --- | --- |
| `image_manifest_commons_cleaned.jsonl.gz` | 图片 manifest 主表，gzip JSONL。 |
| `image_daily_coverage.json` | 日覆盖统计。 |
| `image_daily_coverage.csv` | 日覆盖统计表。 |
| `image_collection_report.json` | 图片来源、覆盖和存储统计。 |
| `thumbnails_webp/` | WebP 缩略图目录。 |

## 当前规模

- 图片 manifest：`1461` 条。
- 日覆盖：`1461 / 1461` 天。
- WebP 缩略图：`1461` 张，约 `40.95 MB`。

## 来源构成

| 来源 | 条数 | 类型 | 说明 |
| --- | ---: | --- | --- |
| `gdelt_doc_socialimage` | `686` | `news_image` | 新闻配图。 |
| `nasa_gibs` | `429` | `satellite_image` | NASA GIBS 遥感缩略图。 |
| `wikimedia` | `103` | `commons` | Wikimedia Commons 开放图片。 |
| `unsplash` | `91` | `stock_photo` | 免费素材图片。 |
| `copernicus_ogc` | `89` | `satellite_image` | Copernicus OGC 遥感缩略图。 |
| `pexels` | `63` | `stock_photo` | 免费素材图片。 |

## 主要字段

| 字段 | 含义 |
| --- | --- |
| `date` | 图片归属日期。 |
| `image_id` | 图片唯一 ID。 |
| `source_stream` | 来源流。 |
| `source_type` | 图片类型。 |
| `source_name` / `domain` | 来源名称和域名。 |
| `image_url` | 原始图片 URL。 |
| `thumbnail_path` | 本地 WebP 缩略图路径。 |
| `width` / `height` | 缩略图尺寸。 |
| `license_short` / `license_url` / `credit` | 授权和署名信息。 |
| `article_url` / `page_url` | 新闻或图片页面 URL。 |
| `published_at` | 原始发布时间。 |
| `roi_name` / `bbox` / `cloud_cover` | 遥感 ROI、范围和云量。 |
| `query_keyword` | 检索关键词。 |
| `collection_status` | 采集状态。 |

图像编码统一读取 `thumbnail_path`，该设计能降低磁盘占用并保持模型输入稳定。
