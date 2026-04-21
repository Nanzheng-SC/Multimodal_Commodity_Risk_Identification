# time_series GitHub 上传说明

本目录上传时间窗口构建源码，不上传生成后的窗口数组。

## 上传保留

- 时间窗口构建脚本、配置和辅助模块

## 本地生成或忽略

- `time_series_horizon30` 和 `time_series_horizon30_mainline` 下的数组文件不上传
- 训练集、验证集、测试集窗口均由本地脚本重建

## 本地重建入口

```powershell
python 2_encoding_feature/time_series/run_time_series_pipeline.py --frequency daily --variants fusion,text,image,structured
```
