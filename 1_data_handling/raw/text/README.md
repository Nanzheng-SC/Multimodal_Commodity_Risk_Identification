# raw/text GitHub 上传说明

本目录上传压缩后的文本主表和按日覆盖统计。

## 上传保留

- `text_documents_multisource_cleaned.jsonl.gz`
- `text_daily_coverage.json`
- `text_daily_coverage.csv`

## 本地生成或忽略

- 未压缩文本副本、候选表、正文提取缓存和失败重试文件不上传
- 临时补采结果先合并到压缩主表，再决定是否进入仓库

## 本地重建入口

```powershell
python 1_data_handling/_collection_scripts/collect_daily_text_dataset.py --start-date 2022-04-17 --end-date 2026-04-16 --storage-mode light
```
