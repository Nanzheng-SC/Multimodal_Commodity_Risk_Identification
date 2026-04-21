# raw/structured GitHub 上传说明

本目录上传结构化主表、字段注册表和数据清单。

## 上传保留

- `structured_daily_merged.csv`
- `structured_monthly_derived.csv`
- `structured_dataset_manifest.json`
- `field_registry.py`
- `__init__.py`

## 本地生成或忽略

- 源接口缓存、临时合并表和调试导出文件不上传
- 日表和月表如需刷新，由本地维护脚本重建后再核验

## 本地重建入口

```powershell
python 1_data_handling/_collection_scripts/build_structured_datasets.py --start-date 2022-04-17 --end-date 2026-04-16
```
