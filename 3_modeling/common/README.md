# common

本目录保存建模层的公共工具。它统一了窗口读取、指标计算、结果落盘和对比表生成逻辑，保证 Naive、HAR、LSTM 和 TimeMixer 的比较口径一致。

## 文件

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 包标记。 |
| `comparison.py` | 读取 official 结果、生成 leaderboard 与对比图。 |
| `metrics.py` | 统一计算 RMSE、MAE、MAPE、direction accuracy 等指标。 |
| `paths.py` | 建模层结果目录和窗口目录路径工具。 |
| `reporting.py` | 结果文件、预测曲线和摘要图落盘工具。 |
| `window_data.py` | 读取 train/valid/test 窗口、拼接 reference、构造 rolling 计划。 |

## 作用

`common/` 的存在保证了三个关键一致性：

- 所有模型读取的是同一份窗口切分
- 所有模型使用的是同一套指标定义
- 所有 official/export 结果使用的是同一套落盘格式

这也是为什么当前 `TimeMixer + fusion + late.gru_gate` 的优势可以直接与 Naive、HAR、LSTM、单模态和其他融合器结果横向比较。
