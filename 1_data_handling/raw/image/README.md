# raw/image GitHub 上传说明

本目录上传图片主清单、覆盖统计、采集报告和正式缩略图。

## 上传保留

- `image_manifest_commons_cleaned.jsonl.gz`
- `image_daily_coverage.json`
- `image_daily_coverage.csv`
- `image_collection_report.json`
- `thumbnails_webp/`

## 本地生成或忽略

- 原始图片下载目录不上传
- 临时候选表、下载缓存和失败重试缓存不上传
- 新一轮补采得到的缩略图先本地核验，再决定是否进入仓库

## 本地重建入口

```powershell
python 1_data_handling/_collection_scripts/collect_daily_image_dataset.py --start-date 2022-04-17 --end-date 2026-04-16 --storage-mode light
```

## 上传前检查

```powershell
git status --short 1_data_handling/raw/image
python -m json.tool 1_data_handling/raw/image/image_daily_coverage.json > $null
```
