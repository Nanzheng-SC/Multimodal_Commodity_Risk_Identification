# TimeMixer

本目录保存 TimeMixer 训练入口、模型实现和主线导出逻辑。当前最终主线是 `TimeMixer + fusion + late.gru_gate`，任务是日频未来 30 日 Brent 均价残差预测。

## 文件结构

| 文件或目录 | 作用 |
| --- | --- |
| `official_benchmark.py` | TimeMixer official 训练、验证、测试和结果落盘入口。 |
| `run_fusion_timemixer.py` | 命令行运行 TimeMixer 的轻量入口。 |
| `run_horizon30_late_gru_gate_mainline_tuning.py` | 当前主线调优、rolling 复核和 export 生成脚本。 |
| `build_modality_comparison.py` | 单模态、多模态与基线对比结果整理。 |
| `data_provider/` | 数据加载适配。 |
| `exp/` | 训练实验类。 |
| `layers/` | TimeMixer 网络层。 |
| `models/` | TimeMixer 模型定义。 |
| `utils/` | 训练、指标和辅助工具。 |

## 输入

```text
2_encoding_feature/outputs/daily/time_series_horizon30_mainline/late_gru_gate/
2_encoding_feature/outputs/daily/time_series_horizon30/{structured,text,image}/
```

## 输出

```text
3_modeling/results/official/daily_horizon30/
3_modeling/results/export/daily_horizon30_late_gru_gate_mainline_final/
```

official 目录保存各模型族的正式训练结果；export 目录保存当前主线的最终表、图和前端所需摘要。

## 可调整参数

`official_benchmark.py` 和主线调优脚本支持的核心参数包括：

| 参数 | 影响 |
| --- | --- |
| `window_length` | 控制可见历史长度。 |
| `learning_rate` | 控制收敛速度与稳定性。 |
| `batch_size` | 控制吞吐和显存占用。 |
| `max_epochs` / `patience` | 控制训练预算与早停。 |
| `loss` / `huber_delta` | 控制优化口径。 |
| `d_model` / `d_ff` / `e_layers` | 控制模型容量。 |
| `dropout` / `weight_decay` | 控制泛化稳定性。 |
| `down_sampling_layers` | 控制多尺度层数。 |
| `target_mode` | 当前主线使用 `residual`。 |
| `grad_clip` | 控制训练稳定性。 |

## 推荐运行

```powershell
python 3_modeling/TimeMixer/run_fusion_timemixer.py --mode official --input-variant fusion --frequency daily --window-length 90
python 3_modeling/TimeMixer/run_horizon30_late_gru_gate_mainline_tuning.py
```

如果只需要刷新前端，不需要重训，直接运行 `4_decision_rl/build_decision_dashboard_data.py` 即可。

## 当前主线解读

`TimeMixer + fusion + late.gru_gate` 在 fixed test 和 6-fold rolling 两个口径下都处于当前项目最佳位置：

- fixed test RMSE `15.4705`
- rolling score `6.0236`
- rolling RMSE mean `4.7957`
- fixed test direction accuracy `95.0%`

这说明 TimeMixer 的多尺度时间结构适合当前 Brent 30 日均价残差任务，而融合输入进一步提升了相对于单模态输入的解释力和稳定性。
