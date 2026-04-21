# structured 交付说明

本目录保留结构化处理源码，不包含结构化特征数组。

## 交付内容

- `process_structured.py`
- `__init__.py`

## 本地重建或附加内容

- 结构化编码结果写入 `2_encoding_feature/outputs/`
- 派生特征和标准化结果由本地脚本按需重建

## 本地重建入口

```powershell
python 2_encoding_feature/run_all_real_data_pipeline.py
```
