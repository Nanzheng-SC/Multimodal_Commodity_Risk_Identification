# time_series 交付说明

本目录保留时间窗口构建源码，不包含生成后的窗口数组。

## 交付内容

- 时间窗口构建脚本、配置和辅助模块

## 本地重建或附加内容

- `time_series_horizon30` 和 `time_series_horizon30_mainline` 下的数组文件不纳入交付目录
- 训练集、验证集、测试集窗口均由本地脚本重建

## 本地重建入口

```powershell
python 2_encoding_feature/time_series/run_time_series_pipeline.py --frequency daily --variants fusion,text,image,structured
```
