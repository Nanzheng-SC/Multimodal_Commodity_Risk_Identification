# 3_modeling GitHub 上传说明

本目录上传建模源码、正式指标表、正式图表和最终导出结果，不上传训练检查点和临时搜索目录。

## 上传保留

- `baselines/`
- `common/`
- `TimeMixer/`
- `results/export/`
- `results/official/` 中的正式指标、预测表和展示图

## 本地生成或忽略

- `results/_scratch/` 不上传
- `results/official/**/best_run/checkpoint.pth` 和 `checkpoints/` 不上传
- 临时搜索目录、调参缓存和中间实验残留不上传

## 本地重建入口

```powershell
python 3_modeling/run_modeling_benchmarks.py
python 3_modeling/TimeMixer/run_horizon30_late_gru_gate_mainline_tuning.py
```

## 上传前检查

```powershell
git status --short 3_modeling
python -m json.tool 3_modeling/results/export/daily_horizon30_late_gru_gate_mainline_final/EXPORT_SUMMARY.json > $null
```
