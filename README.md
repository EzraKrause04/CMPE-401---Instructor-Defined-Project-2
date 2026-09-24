# CMPE 401 Instructor-defined Project 2: Time-Series Benchmark

Transformer classification (FordA) and LSTM forecasting (Jena Climate), reproduced from the official Keras examples. The LSTM gets controlled modifications on a leak-free benchmark protocol.

> **Status:** code complete; runs in progress on Colab (T4 GPU). This README becomes the benchmark report once results are in.

## Run it

**Colab (recommended):** open [`notebooks/colab_P2.ipynb`](notebooks/colab_P2.ipynb) in Colab, select a T4 GPU, and choose **Run all**. That takes about 2–3 h. Results are saved to Google Drive, and a restarted notebook skips finished runs.

**Locally (CPU):**

```bash
pip install -r requirements.txt
python experiments/transformer_official.py --run-name T0_official
python experiments/lstm_official.py
python experiments/naive_baselines.py
python experiments/lstm_benchmark.py --variant all --seeds 0 1 2
python experiments/aggregate_lstm.py
```

## Layout

```
reference/                 official keras-io sources, verbatim (Apache-2.0)
experiments/
  transformer_official.py  Task 1: FordA Transformer (current master); --pooling/--blocks for diagnostics
  transformer_probe.py     numerical checks that each encoder block computes x + constant
  transformer_fixed.py     appendix: input projection + positional embedding
  lstm_official.py         Task 1: Jena LSTM, official split and hyperparameters
  jena_data.py             leak-free train/val/test windows for the benchmark
  test_jena_windows.py     window/label alignment tests
  lstm_benchmark.py        Task 2: single-change variants x seeds (resumable)
  naive_baselines.py       persistence, same-time-yesterday, ridge regression
  aggregate_lstm.py        Task 3: summary table and figures
notebooks/colab_P2.ipynb   runs everything on a Colab GPU
results/                   committed metrics, histories and figures
```
