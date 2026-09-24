"""Controlled LSTM variants on the Jena benchmark protocol (data pipeline: jena_data.py).

Each registry variant changes exactly one setting of L0_baseline, which is the official model
(LSTM(32) -> Dense(1), past=720, Adam 1e-3, batch 256, 10 epochs, patience 5) moved onto the
benchmark protocol: early stopping restores the best-val weights, and that selected model is
scored on val and on a held-out test split that no training or selection decision sees.
Val/test label rows are identical for every variant (jd.EVAL_PAST), so test numbers are comparable.

Runs are seed-major (all variants for seed 0, then seed 1, ...) and resumable: a run whose
results/lstm_benchmark/<variant>/seed<k>.json exists from a full (non-smoke) run is skipped.

Usage:
    python experiments/lstm_benchmark.py --variant all --seeds 0 1 2
    python experiments/lstm_benchmark.py --variant L0_baseline V7_time_features --threads 4
    python experiments/lstm_benchmark.py --variant L0_baseline --variant V6_clean_wv --seeds 0 --seeds 1
    python experiments/lstm_benchmark.py --override '{"units": [64], "time_features": true}' --name C1_combined
    python experiments/lstm_benchmark.py --variant L0_baseline --seeds 0 --max-epochs 1 --max-train-windows 5000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jena_data as jd  # noqa: E402

RESULTS = jd.REPO / "results" / "lstm_benchmark"
ARTIFACTS = jd.REPO / "artifacts" / "lstm_benchmark"
STEP, FUTURE = 6, 72

BASE = {
    "units": [32],
    "past": 720,
    "lr": 1e-3,
    "batch": 256,
    "epochs": 10,
    "es_patience": 5,
    "dropout": 0.0,
    "clean_wv": False,
    "time_features": False,
    "reduce_lr": False,
    "shuffle": False,
}
VARIANTS = {
    "L0_baseline": ({}, "official model on the benchmark protocol"),
    "V1a_hidden64": ({"units": [64]}, "LSTM width 32 -> 64"),
    "V1b_hidden128": ({"units": [128]}, "LSTM width 32 -> 128"),
    "V2a_stacked_2x32": ({"units": [32, 32]}, "two stacked LSTM(32) layers"),
    "V3a_past360": ({"past": 360}, "input history 120 h -> 60 h"),
    "V3b_past1440": ({"past": 1440}, "input history 120 h -> 240 h"),
    "V4a_epochs30": ({"epochs": 30}, "epoch budget 10 -> 30"),
    "V4b_epochs30_plateau": ({"epochs": 30, "reduce_lr": True}, "30 epochs + ReduceLROnPlateau"),
    "V5_dropout02": ({"dropout": 0.2}, "LSTM input dropout 0.2"),
    "V6_clean_wv": ({"clean_wv": True}, "wind-speed -9999 sentinels -> 0"),
    "V7_time_features": ({"time_features": True}, "+ sin/cos hour-of-day and day-of-year"),
    "V8_shuffle": ({"shuffle": True}, "shuffle training windows each epoch"),
}


def build_model(input_shape: tuple[int, int], units: list[int], dropout: float):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for k, u in enumerate(units):
        x = keras.layers.LSTM(u, dropout=dropout, return_sequences=k < len(units) - 1)(x)
    return keras.Model(inputs, keras.layers.Dense(1)(x))


def score(model, dataset, y: np.ndarray, t_std: float, prefix: str) -> dict:
    err = model.predict(dataset, verbose=0)[:, 0].astype(np.float64) - y
    mse = float(np.mean(err**2))
    return {
        f"{prefix}_mse": mse,
        f"{prefix}_mae_C": float(np.mean(np.abs(err))) * t_std,
        f"{prefix}_rmse_C": float(np.sqrt(mse)) * t_std,
    }


def finished(name: str, seed: int) -> dict | None:
    """The saved JSON of a finished full (non-smoke) run, else None."""
    out = RESULTS / name / f"seed{seed}.json"
    if out.exists() and not (prev := json.loads(out.read_text())).get("smoke"):
        return prev
    return None


def check_overrides(overrides: dict) -> str | None:
    """Error message if an --override would break the model or the shared val/test label rows."""
    bad = set(overrides) - set(BASE)
    if bad:
        return f"unknown override keys {sorted(bad)}; allowed: {list(BASE)}"
    for k, v in overrides.items():
        ref = BASE[k]
        if k == "units":
            ok = isinstance(v, list) and v and all(isinstance(u, int) and not isinstance(u, bool) and u > 0 for u in v)
        elif isinstance(ref, bool):
            ok = isinstance(v, bool)
        elif isinstance(ref, int):
            ok = isinstance(v, int) and not isinstance(v, bool) and v > 0
        else:
            ok = isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
        if not ok:
            return f"override {k}={v!r} does not match the type of the L0 value {ref!r}"
    past = overrides.get("past", BASE["past"])
    if past % STEP or past > jd.EVAL_PAST:
        # val/test label rows are pinned by EVAL_PAST; a longer past would score on fewer, later rows
        return f"past={past} must be a multiple of {STEP} and <= {jd.EVAL_PAST}"
    if not 0 <= overrides.get("dropout", 0.0) < 1 or overrides.get("lr", 1.0) <= 0:
        return "dropout must be in [0, 1) and lr > 0"
    return None


def run(df: pd.DataFrame, name: str, overrides: dict, change: str, seed: int, args) -> None:
    out_dir = RESULTS / name
    out = out_dir / f"seed{seed}.json"
    smoke = bool(args.max_epochs or args.max_train_windows)
    if finished(name, seed):
        print(f"skip {name} seed{seed}: {out.relative_to(jd.REPO)} exists")
        return
    cfg = {**BASE, **overrides}
    epochs = min(cfg["epochs"], args.max_epochs) if args.max_epochs else cfg["epochs"]
    print(f"=== {name} seed{seed}: {change} | {json.dumps(overrides)}", flush=True)

    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    data = jd.prepare(
        df,
        past=cfg["past"],
        future=FUTURE,
        step=STEP,
        clean_wv=cfg["clean_wv"],
        time_features=cfg["time_features"],
        eval_past=jd.EVAL_PAST,
        max_train_windows=args.max_train_windows,
    )
    t_std = data.meta["t_std"]
    train = data.dataset("train", cfg["batch"], shuffle=cfg["shuffle"], seed=seed)
    val = data.dataset("val", cfg["batch"])
    test = data.dataset("test", cfg["batch"])

    model = build_model((cfg["past"] // STEP, data.x.shape[1]), cfg["units"], cfg["dropout"])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg["lr"]), loss="mse", metrics=["mae"])
    timer = jd.epoch_timer()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["es_patience"], restore_best_weights=True),
        timer,
    ]
    if cfg["reduce_lr"]:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5))

    t0 = time.perf_counter()
    # shuffling (if any) happens inside the tf.data pipeline; fit's own flag does not apply to datasets
    hist = model.fit(train, epochs=epochs, validation_data=val, callbacks=callbacks, shuffle=False, verbose=2)
    train_time = time.perf_counter() - t0
    val_loss = [float(v) for v in hist.history["val_loss"]]

    # restore_best_weights has already put the best-val epoch back into the model
    metrics = {**score(model, val, data.labels("val"), t_std, "val"), **score(model, test, data.labels("test"), t_std, "test")}
    ARTIFACTS.joinpath(name).mkdir(parents=True, exist_ok=True)
    weights = ARTIFACTS / name / f"seed{seed}.weights.h5"
    model.save_weights(weights)

    out_dir.mkdir(parents=True, exist_ok=True)
    h = pd.DataFrame(hist.history)
    h.insert(0, "epoch", range(1, len(h) + 1))
    h["seconds"] = timer.seconds
    h.to_csv(out_dir / f"history_seed{seed}.csv", index=False)
    summary = {
        "variant": name,
        "change": change,
        "overrides": overrides,
        "seed": seed,
        "config": {**cfg, "epochs_effective": epochs, "future": FUTURE, "step": STEP},
        "smoke": smoke,
        "max_epochs": args.max_epochs,
        "max_train_windows": args.max_train_windows,
        "threads": args.threads,
        "params": int(model.count_params()),
        "input_shape": [cfg["past"] // STEP, int(data.x.shape[1])],
        "features": data.meta["features"],
        "epochs_run": len(val_loss),
        "best_epoch": int(np.argmin(val_loss)) + 1,
        "early_stopped": len(val_loss) < epochs,
        "train_time_s": train_time,
        "seconds_per_epoch": float(np.mean(timer.seconds)),
        **metrics,
        "best_logged_val_loss": min(val_loss),
        "n_windows": data.meta["n_windows"],
        "label_rows": data.meta["label_rows"],
        "t_mean": data.meta["t_mean"],
        "t_std": t_std,
        "weights": str(weights.relative_to(jd.REPO)),
        "versions": jd.versions(),
    }
    out.write_text(json.dumps(summary, indent=2))  # written last, so a crashed run is simply redone
    print(
        f"--- {name} seed{seed}: {summary['params']} params, epochs {summary['epochs_run']} (best {summary['best_epoch']}), "
        f"{train_time:.0f}s | val MAE {metrics['val_mae_C']:.3f} C | test MAE {metrics['test_mae_C']:.3f} C, "
        f"RMSE {metrics['test_rmse_C']:.3f} C",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description="Run LSTM benchmark variants (seed-major, resumable).")
    ap.add_argument("--variant", nargs="+", action="extend", default=[],
                    help=f"repeatable; 'all' or any of: {', '.join(VARIANTS)}")
    ap.add_argument("--override", type=json.loads, default=None, help="JSON applied on top of L0_baseline (needs --name)")
    ap.add_argument("--name", default=None, help="result name for --override")
    # default None, not [0, 1, 2]: "extend" would append to a non-empty default
    ap.add_argument("--seeds", nargs="+", action="extend", type=int, default=None, help="repeatable (default: 0 1 2)")
    ap.add_argument("--threads", type=int, default=4, help="TF intra-op threads (CPU only)")
    ap.add_argument("--max-epochs", type=int, default=None, help="smoke tests only")
    ap.add_argument("--max-train-windows", type=int, default=None, help="smoke tests only: first N train windows")
    args = ap.parse_args()
    seeds = list(dict.fromkeys(args.seeds or [0, 1, 2]))

    names = list(VARIANTS) if "all" in args.variant else list(dict.fromkeys(args.variant))
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        ap.error(f"unknown variant(s) {unknown}; choose from {list(VARIANTS)}")
    jobs = [(n, *VARIANTS[n]) for n in names]
    if args.override is not None or args.name:
        if args.override is None or not args.name:
            ap.error("--override and --name must be given together")
        if args.name in VARIANTS:
            ap.error(f"--name {args.name!r} clashes with a registry variant")
        if not isinstance(args.override, dict):
            ap.error("--override must be a JSON object")
        if msg := check_overrides(args.override):
            ap.error(msg)
        jobs.append((args.name, args.override, "L0 + " + ", ".join(f"{k}={v}" for k, v in args.override.items())))
    if not jobs:
        ap.error("nothing to run: pass --variant and/or --override/--name")
    # fail now rather than hours into the queue if a name was already used with other settings
    for name, overrides, _ in jobs:
        for seed in seeds:
            if (prev := finished(name, seed)) and prev["overrides"] != overrides:
                ap.error(f"{name}/seed{seed}.json was run with overrides {prev['overrides']}, not {overrides}; "
                         "pick another --name")

    jd.setup_tf(args.threads)  # before any TF op runs
    df = jd.load_frame()
    for seed in seeds:
        for name, overrides, change in jobs:
            run(df, name, overrides, change, seed, args)


if __name__ == "__main__":
    main()
