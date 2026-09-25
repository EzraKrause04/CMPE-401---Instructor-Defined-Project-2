| Variant | Change | Params | Seeds | Epochs (mean) | Test MAE °C (mean ± std) | Test RMSE °C | Val MAE °C | Δ val MAE vs L0 | Δ test MAE vs L0 | Train time (min) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0_baseline | official model on the benchmark protocol | 5,153 | 3 | 8.0 | 2.430 ± 0.027 | 3.105 ± 0.027 | 2.539 ± 0.023 | — | — | 1.5 |
| V1a_hidden64 | LSTM width 32 -> 64 | 18,497 | 3 | 7.7 | 2.506 ± 0.126 | 3.198 ± 0.174 | 2.602 ± 0.078 | +0.063 °C (+2.5 %) | +0.076 °C (+3.1 %) | 1.8 |
| V1b_hidden128 | LSTM width 32 -> 128 | 69,761 | 3 | 10.0 | 2.421 ± 0.039 | 3.077 ± 0.051 | 2.518 ± 0.062 | -0.021 °C (-0.8 %) | -0.009 °C (-0.4 %) | 2.8 |
| V2a_stacked_2x32 | two stacked LSTM(32) layers | 13,473 | 3 | 10.0 | 2.337 ± 0.066 | 2.951 ± 0.080 | 2.451 ± 0.043 | -0.088 °C (-3.5 %) | -0.093 °C (-3.8 %) | 2.7 |
| V3a_past360 | input history 120 h -> 60 h | 5,153 | 3 | 10.0 | 2.318 ± 0.036 | 2.954 ± 0.051 | 2.406 ± 0.020 | -0.133 °C (-5.2 %) | -0.113 °C (-4.6 %) | 1.4 |
| V3b_past1440 | input history 120 h -> 240 h | 5,153 | 3 | 9.0 | 2.316 ± 0.082 | 2.951 ± 0.113 | 2.449 ± 0.064 | -0.090 °C (-3.6 %) | -0.115 °C (-4.7 %) | 2.4 |
| V4a_epochs30 | epoch budget 10 -> 30 | 5,153 | 3 | 14.7 | 2.341 ± 0.182 | 2.993 ± 0.215 | 2.450 ± 0.159 | -0.089 °C (-3.5 %) | -0.090 °C (-3.7 %) | 2.7 |
| V4b_epochs30_plateau | 30 epochs + ReduceLROnPlateau | 5,153 | 3 | 29.7 | 1.966 ± 0.015 | 2.503 ± 0.015 | 2.157 ± 0.011 | -0.382 °C (-15.1 %) | -0.464 °C (-19.1 %) | 5.4 |
| V5_dropout02 | LSTM input dropout 0.2 | 5,153 | 3 | 6.0 | 2.729 ± 0.029 | 3.500 ± 0.050 | 2.828 ± 0.026 | +0.289 °C (+11.4 %) | +0.298 °C (+12.3 %) | 1.2 |
| V6_clean_wv | wind-speed -9999 sentinels -> 0 | 5,153 | 3 | 8.0 | 2.430 ± 0.027 | 3.105 ± 0.027 | 2.538 ± 0.024 | -0.001 °C (-0.1 %) | +0.000 °C (+0.0 %) | 1.5 |
| V7_time_features | + sin/cos hour-of-day and day-of-year | 5,665 | 3 | 7.0 | 2.168 ± 0.022 | 2.731 ± 0.026 | 2.317 ± 0.039 | -0.222 °C (-8.7 %) | -0.262 °C (-10.8 %) | 1.5 |
| V8_shuffle | shuffle training windows each epoch | 5,153 | 3 | 6.3 | 1.985 ± 0.026 | 2.540 ± 0.033 | 2.185 ± 0.018 | -0.354 °C (-13.9 %) | -0.445 °C (-18.3 %) | 1.2 |
| C1_shuffle_time_plateau | L0 + shuffle=True, time_features=True, epochs=30, reduce_lr=True | 5,665 | 3 | 6.0 | 1.916 ± 0.022 | 2.455 ± 0.027 | 2.096 ± 0.044 | -0.443 °C (-17.5 %) | -0.514 °C (-21.1 %) | 1.4 |
| C2_C1_past360 | L0 + shuffle=True, time_features=True, epochs=30, reduce_lr=True, past=360 | 5,665 | 3 | 6.0 | 1.934 ± 0.020 | 2.460 ± 0.020 | 2.111 ± 0.036 | -0.428 °C (-16.9 %) | -0.497 °C (-20.4 %) | 1.0 |
| *persistence* | T at the last input row (label row - 78) | — | — | — | 4.222 | 5.431 | 4.592 | +2.053 °C (+80.8 %) | +1.792 °C (+73.7 %) | — |
| *seasonal_24h* | T 24 h before the label row (label row - 144) | — | — | — | 2.497 | 3.257 | 2.676 | +0.137 °C (+5.4 %) | +0.066 °C (+2.7 %) | — |
| *ridge* | Ridge(alpha=1.0) on flattened 120x7 windows, 50000 random train windows (seed 0), wv sentinels zeroed | — | — | — | 2.112 | 2.700 | 2.224 | -0.315 °C (-12.4 %) | -0.319 °C (-13.1 %) | — |

Test = held-out last 50 % of the official validation rows; each model is the best-val epoch (EarlyStopping restore_best_weights). ± is the sample std over seeds; Δ compares mean MAE with L0_baseline. Naive baselines (italic) are deterministic and scored on the same label rows.

**Selection rule:** which variants count as improvements (and go into a combined model) is decided on **Δ val** only; test is reported for the final numbers and never used for choices.

Val caveat: 732 of 58,417 val windows (0 of 58,417 test) contain a wind-speed -9999 sentinel in their inputs. LSTM runs with clean_wv=False (all but V6_clean_wv, unless a combined model enables it) see them as-is, while ridge and clean_wv runs see them zeroed, so val MAE is not strictly like-for-like across those rows; test is unaffected.
