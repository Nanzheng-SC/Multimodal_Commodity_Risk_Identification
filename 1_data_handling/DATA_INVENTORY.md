# Dataset Inventory

## Structured
### daily
- rows: `1461`
- columns: `15`
- range: `2022-04-17` -> `2026-04-16`
- path: `E:\Work\2025FireFlower\1_data_handling\raw\structured\structured_daily_merged.csv`

### monthly_derived
- rows: `49`
- columns: `17`
- range: `2022-04-30` -> `2026-04-16`
- path: `E:\Work\2025FireFlower\1_data_handling\raw\structured\structured_monthly_derived.csv`

### registered_structured_fields
| field | group | source | native_frequency | modeling_enabled |
| --- | --- | --- | --- | --- |
| Brent | price_benchmarks | FRED | daily | True |
| WTI | price_benchmarks | FRED | daily | True |
| USD_Index | macro_fx_uncertainty | FRED | daily | True |
| EPU | macro_fx_uncertainty | FRED | daily_7day | True |
| GPR | macro_fx_uncertainty | Iacoviello_GPR | daily | True |
| UAH_per_USD | macro_fx_uncertainty | NBU | daily | True |
| RUB_per_USD | macro_fx_uncertainty | CBR | daily | True |
| AED_per_USD | macro_fx_uncertainty | CBUAE_USD_peg | daily | False |
| reference_brent | derived_features | derived | daily | False |
| brent_step_return | derived_features | derived | daily | True |
| brent_step_volatility_7 | derived_features | derived | daily | True |
| brent_return_3d | derived_features | derived | historical_daily | True |
| brent_return_7d | derived_features | derived | historical_daily | True |
| brent_return_14d | derived_features | derived | historical_daily | True |
| brent_return_30d | derived_features | derived | historical_daily | True |
| brent_volatility_14d | derived_features | derived | historical_daily | True |
| brent_volatility_30d | derived_features | derived | historical_daily | True |
| wti_brent_spread | derived_features | derived | daily | True |
| usd_index_return_7d | derived_features | derived | historical_daily | True |
| gpr_change_7d | derived_features | derived | historical_daily | True |
| epu_change_7d | derived_features | derived | historical_daily | True |
| brent_zscore_30d | derived_features | derived | historical_daily | True |
| high_volatility_regime | derived_features | derived | historical_daily | True |
| fast_uptrend_regime | derived_features | derived | historical_daily | True |
| market_closed_flag | derived_features | derived | daily | False |
| target_brent_avg_next_7d | labels | derived | daily | False |
| target_brent_day7 | labels | derived | daily | False |

## text
- rows: `8852`
- columns: `25`
- range: `2022-04-17` -> `2026-04-16`
- path: `E:\Work\2025FireFlower\1_data_handling\raw\text\text_documents_multisource_cleaned.jsonl.gz`
- covered_days: `1461` / `1461`
- uncovered_days: `0`

## image
- rows: `1461`
- columns: `35`
- range: `2022-04-17` -> `2026-04-16`
- path: `E:\Work\2025FireFlower\1_data_handling\raw\image\image_manifest_commons_cleaned.jsonl.gz`
- covered_days: `1461` / `1461`
- uncovered_days: `0`
- source_counts: `{'wikimedia': 103, 'gdelt_doc_socialimage': 686, 'nasa_gibs': 429, 'copernicus_ogc': 89, 'unsplash': 91, 'pexels': 63}`

## Event
- rows: `18`
- range: `2022-07-27` -> `2025-06-03`
- path: `E:\Work\2025FireFlower\1_data_handling\raw\event\event_manifest.jsonl`

## Storage
- text_archive_bytes: `1877905`
- image_manifest_bytes: `309900`
- thumbnail_file_count: `1461`
- thumbnail_total_bytes: `40952974`
