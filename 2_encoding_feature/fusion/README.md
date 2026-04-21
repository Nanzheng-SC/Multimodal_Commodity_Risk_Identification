# fusion GitHub 上传说明

本目录上传多模态融合源码和配置，不上传融合后特征文件。

## 上传保留

- `config.py`
- `modules.py`
- `selector.py`
- `timemixer_backend.py`
- `run_fusion_pipeline.py`

## 本地生成或忽略

- `outputs/daily/fusion_features/` 为本地生成结果，不上传
- 各类 seed 中间结果和试验缓存不上传

## 本地重建入口

```powershell
python 2_encoding_feature/fusion/run_fusion_pipeline.py --frequency daily
```
