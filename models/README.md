# Trained models

Weights for the models reported in the [main README](../README.md), trained on a Colab T4 (TensorFlow 2.20 /
Keras 3.13). They are Keras `.weights.h5` files. Build the architecture with the repo's own code, then call
`load_weights`.

| Path | Model | Reported result |
|---|---|---|
| `transformer/T0_official.weights.h5` | Official FordA Transformer, current keras-io code | test accuracy 0.516 (chance) |
| `transformer/T1_page_version.weights.h5` | T0 with `channels_first` pooling (the version the keras.io page shows) | test accuracy 0.867 |
| `transformer/T2_no_encoder.weights.h5` | T1 without the four encoder blocks | test accuracy 0.873 |
| `lstm/L0_official.weights.h5` | Official Jena LSTM, best checkpoint on the official split | val MSE 0.1376 (2.51 °C MAE) |
| `lstm_benchmark/<variant>/seed{0,1,2}.weights.h5` | Every benchmark configuration, including the final **C1** model | see `results/lstm_benchmark/summary.md` |

T3 (the "fixed" Transformer) is not included. It stayed at chance, and its weights are 13 MB.

## Check that these are the reported models

```bash
python experiments/verify_models.py
```

This rebuilds every Transformer and LSTM benchmark model, loads its weights, and re-scores it on the test set.
It then compares each score with `results/`. On a Mac (CPU, TensorFlow 2.21 / Keras 3.15) all 45 match: the
Transformer accuracies to 4 decimals, and the LSTM test MAEs to within 2e-7 °C.

## Loading one model

```python
import sys; sys.path.insert(0, "experiments")
import jena_data as jd, lstm_benchmark as lb, transformer_official as to

# Transformer (T1): input (500, 1), 2 classes
model = to.build_official((500, 1), 2, pooling="channels_first", blocks=4)
model.load_weights("models/transformer/T1_page_version.weights.h5")

# LSTM benchmark model C1, seed 0: 120 hourly steps x 11 features (7 weather + 4 calendar)
data = jd.prepare(past=720, time_features=True, eval_past=jd.EVAL_PAST)
model = lb.build_model((720 // 6, data.x.shape[1]), units=[32], dropout=0.0)
model.load_weights("models/lstm_benchmark/C1_shuffle_time_plateau/seed0.weights.h5")
```

The configuration of every benchmark run is stored in its `results/lstm_benchmark/<variant>/seed<k>.json`
under `config`.
