# raw/event GitHub 上传说明

本目录仅上传事件清单主文件。

## 上传保留

- `event_manifest.jsonl`

## 本地生成或忽略

- 事件扩展脚本、临时候选文件和实验性补充文件不上传

## 本地重建入口

```powershell
python 1_data_handling/_collection_scripts/run_collection.py --start-date 2022-04-17 --end-date 2026-04-16 --storage-mode light
```
