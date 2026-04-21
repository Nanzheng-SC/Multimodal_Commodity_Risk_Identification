# 2_encoding_feature GitHub 上传说明

本目录上传编码、融合和时间窗口构建源码，不上传大体积特征数组和窗口缓存。

## 上传保留

- `config.py`
- `run_all_real_data_pipeline.py`
- `structured/`
- `text_bert/`
- `image_clip/`
- `fusion/`
- `time_series/`

## 本地生成或忽略

- `outputs/` 全部由本地脚本重建，不上传
- 文本模型的大体积权重文件不上传
- 仅保留轻量分词器配置文件和源码

## 本地重建入口

```powershell
python 2_encoding_feature/run_all_real_data_pipeline.py
python 2_encoding_feature/time_series/run_time_series_pipeline.py --frequency daily --variants fusion,text,image,structured
```

## 上传前检查

```powershell
git status --short 2_encoding_feature
git check-ignore -v 2_encoding_feature/outputs/*
```
