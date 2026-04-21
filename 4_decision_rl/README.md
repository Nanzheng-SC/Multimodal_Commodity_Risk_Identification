# 4_decision_rl GitHub 上传说明

本目录上传前端页面、前端数据文件和业务解释文档。

## 上传保留

- `enterprise_risk_dashboard.html`
- `decision_dashboard_data.json`
- `decision_dashboard_data.js`
- `FRONTEND_RISK_HEDGING_GUIDE.md`
- `build_decision_dashboard_data.py`

## 本地生成或忽略

- 前端不上传构建缓存和调试快照
- 前端数据文件由本地脚本根据正式导出结果重建

## 本地重建入口

```powershell
python 4_decision_rl/build_decision_dashboard_data.py
```

## 上传前检查

```powershell
git status --short 4_decision_rl
python -m json.tool 4_decision_rl/decision_dashboard_data.json > $null
```
