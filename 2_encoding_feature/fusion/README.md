# fusion 交付说明

本目录保留多模态融合源码和配置，不包含融合后特征文件。

## 交付内容

- `config.py`
- `modules.py`
- `selector.py`
- `timemixer_backend.py`
- `run_fusion_pipeline.py`

## 本地重建或附加内容

- `outputs/daily/fusion_features/` 为本地生成结果
- 各类 seed 中间结果和试验缓存不纳入交付目录

## 本地重建入口

```powershell
python 2_encoding_feature/fusion/run_fusion_pipeline.py --frequency daily
```
