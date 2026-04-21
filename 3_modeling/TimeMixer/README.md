# TimeMixer GitHub 上传说明

本目录上传 TimeMixer 主线源码、正式结果表和导出图表说明，不上传本地训练检查点。

## 上传保留

- TimeMixer 模型、实验入口和辅助脚本源码
- 与正式主线对应的结果表和展示图

## 本地生成或忽略

- `best_run/checkpoint.pth` 与 `best_run/checkpoints/` 不上传
- 试验过程中的临时日志和调参缓存不上传

## 本地重建入口

```powershell
python 3_modeling/TimeMixer/run_horizon30_late_gru_gate_mainline_tuning.py
```
