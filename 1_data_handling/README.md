# 1_data_handling GitHub 上传说明

本目录用于上传当前主线仍在使用的原始数据主表、覆盖统计和盘点文件。

## 上传保留

- `DATA_INVENTORY.md`
- `DATA_INVENTORY.json`
- `raw/structured/` 下的主表、字段注册表和数据清单
- `raw/text/` 下的压缩文本主表和覆盖统计
- `raw/image/` 下的压缩清单、覆盖统计、采集报告和 `thumbnails_webp/`
- `raw/event/event_manifest.jsonl`

## 本地生成或忽略

- `_collection_scripts/` 为本地维护脚本，不上传
- 临时候选文件、原始抓取缓存、原图目录和中间清洗副本不上传
- 重新采集后生成的新文件先本地核验，再决定是否纳入仓库

## 本地重建入口

```powershell
python 1_data_handling/_collection_scripts/run_collection.py --start-date 2022-04-17 --end-date 2026-04-16 --storage-mode light
python 1_data_handling/_collection_scripts/profile_dataset_inventory.py
```

## 上传前检查

```powershell
git status --short 1_data_handling
python -m json.tool 1_data_handling/DATA_INVENTORY.json > $null
```
