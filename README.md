# GitHub 上传说明

本仓库只保留适合直接同步到 GitHub 的源码、原始数据主表、正式导出结果和前端展示文件。

## 上传保留

- `1_data_handling/` 中的活跃数据主表、覆盖统计和盘点文件
- `2_encoding_feature/` 中的编码、融合和时间窗口构建源码
- `3_modeling/` 中的建模源码、正式指标表、正式图表和导出结果
- `4_decision_rl/` 中的前端页面、前端数据文件和业务说明文档
- `project_shared/` 中的共享日期、路径、目标和 IO 工具
- 根目录下与仓库同步有关的说明文件和 `.gitignore`

## 本地生成或忽略

- `2_encoding_feature/outputs/` 为本地重建产物，不上传
- `3_modeling/results/official/**/best_run/checkpoint.pth` 和 `checkpoints/` 为本地训练检查点，不上传
- `1_data_handling/_collection_scripts/` 为本地数据维护脚本，不上传
- `0_docs/` 为本地资料归档目录，不上传
- 本地虚拟环境、缓存、编辑器配置和临时文件不上传

## 本地重建入口

```powershell
python 1_data_handling/_collection_scripts/profile_dataset_inventory.py
python 2_encoding_feature/run_all_real_data_pipeline.py
python 2_encoding_feature/time_series/run_time_series_pipeline.py --frequency daily --variants fusion,text,image,structured
python 3_modeling/run_modeling_benchmarks.py
python 4_decision_rl/build_decision_dashboard_data.py
```

## 上传前检查

```powershell
git status --short
git check-ignore -v 2_encoding_feature/outputs/*
python -m json.tool 4_decision_rl/decision_dashboard_data.json > $null
```

## 目录说明

- `1_data_handling/`：上传活跃原始数据和盘点文件
- `2_encoding_feature/`：上传编码与窗口构建源码，不上传大体积特征数组
- `3_modeling/`：上传正式结果、导出图表和建模源码，不上传训练检查点
- `4_decision_rl/`：上传展示页面、静态数据和业务说明
- `project_shared/`：上传全流程共享配置和工具
