# structured GitHub 上传说明

本目录上传结构化处理源码，不上传结构化特征数组。

## 上传保留

- `process_structured.py`
- `__init__.py`

## 本地生成或忽略

- 结构化编码结果写入 `2_encoding_feature/outputs/`，不上传
- 派生特征和标准化结果由本地脚本按需重建

## 本地重建入口

```powershell
python 2_encoding_feature/run_all_real_data_pipeline.py
```
