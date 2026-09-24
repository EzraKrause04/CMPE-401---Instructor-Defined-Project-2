"""Naive reference forecasts on the exact val/test label rows used by lstm_benchmark.py.

persistence    T at the last input row (label row - 78, i.e. "in 13 h it will be as now")
seasonal_24h   T 24 h before the label row (label row - 144)
ridge          Ridge(alpha=1) on the flattened 120x7 L0 input window, fit on <= 50k random train windows

The ridge inputs zero the wind-speed -9999 sentinels (as V6_clean_wv does): they occur only in
val rows, never in train or test, so this changes the ridge val score only, not its fit or test score.
The LSTM rows (except V6) keep them, so protocol.wv_sentinel_windows records how many val/test
windows have one in their inputs (aggregate_lstm.py prints it as a caveat under the table).

Usage:
    python experiments/naive_baselines.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jena_data as jd  # noqa: E402

OUT = jd.REPO / "results" / "lstm_benchmark" / "naive_baselines.json"


def metrics(pred: np.ndarray, y: np.ndarray, t_std: float, prefix: str) -> dict:
    err = pred.astype(np.float64) - y
    mse = float(np.mean(err**2))
    return {
        f"{prefix}_mse": mse,
        f"{prefix}_mae_C": float(np.mean(np.abs(err))) * t_std,
        f"{prefix}_rmse_C": float(np.sqrt(mse)) * t_std,
    }


def main():
    ap = argparse.ArgumentParser(description="Persistence, seasonal-24h and ridge baselines on benchmark windows.")
    ap.add_argument("--max-fit-windows", type=int, default=50_000, help="ridge training subsample")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4, help="BLAS threads")
    args = ap.parse_args()

    df = jd.load_frame()
    data = jd.prepare(df, eval_past=jd.EVAL_PAST)
    t_std = data.meta["t_std"]
    lag_last_input = data.future + data.step
    results = {}

    for name, lag, desc in [
        ("persistence", lag_last_input, f"T at the last input row (label row - {lag_last_input})"),
        ("seasonal_24h", 144, "T 24 h before the label row (label row - 144)"),
    ]:
        results[name] = {"description": desc}
        for split in ("val", "test"):
            _, labels = data.rows(split)
            results[name].update(metrics(data.target[labels - lag], data.labels(split), t_std, split))

    clean = jd.prepare(df, clean_wv=True, eval_past=jd.EVAL_PAST)
    rng = np.random.default_rng(args.seed)
    n_train = len(clean.starts["train"])
    idx = np.sort(rng.choice(n_train, size=min(args.max_fit_windows, n_train), replace=False))
    with threadpool_limits(args.threads):
        x, y = clean.arrays("train", idx)
        t0 = time.perf_counter()
        ridge = Ridge(alpha=args.alpha).fit(x.reshape(len(x), -1), y)
        fit_s = time.perf_counter() - t0
        results["ridge"] = {
            "description": f"Ridge(alpha={args.alpha}) on flattened {x.shape[1]}x{x.shape[2]} windows, "
            f"{len(idx)} random train windows (seed {args.seed}), wv sentinels zeroed",
            "n_fit_windows": int(len(idx)),
            "fit_time_s": fit_s,
        }
        for split in ("val", "test"):
            chunks = np.array_split(np.arange(len(clean.starts[split])), 8)
            pred = np.concatenate([ridge.predict(clean.arrays(split, c)[0].reshape(len(c), -1)) for c in chunks])
            results["ridge"].update(metrics(pred, clean.labels(split), t_std, split))

    wv = df["wv (m/s)"].to_numpy()
    summary = {
        "protocol": {
            **{k: data.meta[k] for k in ("past", "future", "step", "eval_past", "n_windows", "label_rows", "t_mean",
                                         "t_std")},
            "wv_sentinel_windows": {
                split: int((wv[data.rows(split)[0]] == jd.SENTINEL).any(axis=1).sum()) for split in ("val", "test")
            },
        },
        "baselines": results,
        "versions": {**jd.versions(), "scikit-learn": sklearn.__version__},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    for name, r in results.items():
        print(f"{name:13s} val MAE {r['val_mae_C']:.3f} C | test MAE {r['test_mae_C']:.3f} C, RMSE {r['test_rmse_C']:.3f} C")


if __name__ == "__main__":
    main()
