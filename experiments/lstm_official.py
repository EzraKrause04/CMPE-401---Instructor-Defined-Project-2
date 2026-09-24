"""Official Keras example "Timeseries forecasting for weather prediction", minimally instrumented.

Training semantics are identical to reference/timeseries_weather_forecasting.py: 7 features,
71.5 % train split with train-only normalization, 120-step windows sampled every 6 rows,
LSTM(32) -> Dense(1), Adam(1e-3), MSE, 10 epochs, batch 256, EarlyStopping(val_loss, patience=5,
no restore_best_weights) and a best-only ModelCheckpoint, both on the official "validation" rows.

Additions: --seed/--threads, data cached once in data/, checkpoint under artifacts/, figures
saved instead of shown, and history.csv + summary.json reporting both the in-memory model after
fit (the LAST epoch, which the example uses for its plots) and the reloaded best checkpoint.

Outputs go to results/lstm/L0_official/ (checkpoint: artifacts/lstm/L0_official/); seeds other
than 0 get an _seed<k> suffix, and smoke runs (--max-rows or --epochs != 10) default to
artifacts/lstm/smoke/ so they can never overwrite the committed results.

Usage:
    python experiments/lstm_official.py --seed 0
    python experiments/lstm_official.py --max-rows 60000 --epochs 2   # smoke test only
"""

import argparse
import json
import sys
import time
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jena_data as jd  # noqa: E402

RESULTS = jd.REPO / "results" / "lstm"
ARTIFACTS = jd.REPO / "artifacts" / "lstm"

titles = [
    "Pressure",
    "Temperature",
    "Temperature in Kelvin",
    "Temperature (dew point)",
    "Relative Humidity",
    "Saturation vapor pressure",
    "Vapor pressure",
    "Vapor pressure deficit",
    "Specific humidity",
    "Water vapor concentration",
    "Airtight",
    "Wind speed",
    "Maximum wind speed",
    "Wind direction in degrees",
]
feature_keys = jd.FEATURE_KEYS
colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
date_time_key = "Date Time"


def show_raw_visualization(data, path):
    time_data = data[date_time_key]
    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k")
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        ax = t_data.plot(ax=axes[i // 2, i % 2], color=c, title="{} - {}".format(titles[i], key), rot=25)
        ax.legend([titles[i]])
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std


def visualize_loss(history, title, path):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path)
    plt.close()


def show_plot(plot_data, delta, title, path):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    if delta:
        future = delta
    else:
        future = 0

    plt.figure()
    plt.title(title)
    for i, val in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    plt.savefig(path)
    plt.close()


def evaluate(model, dataset, t_std) -> dict:
    """Sample-mean MSE / MAE over a non-shuffled dataset, normalized and in degC."""
    err = np.concatenate([model.predict_on_batch(x)[:, 0] - y.numpy()[:, 0] for x, y in dataset])
    mse, mae = float(np.mean(err**2)), float(np.mean(np.abs(err)))
    return {
        "n_windows": int(err.size),
        "val_mse": mse,
        "val_mae_norm": mae,
        "val_mae_C": mae * t_std,
        "val_rmse_C": float(np.sqrt(mse)) * t_std,
    }


def main():
    ap = argparse.ArgumentParser(description="Official Keras Jena LSTM example (instrumented).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4, help="TF intra-op threads (CPU only)")
    ap.add_argument("--epochs", type=int, default=10, help="official value is 10; lower only for smoke tests")
    ap.add_argument("--max-rows", type=int, default=None, help="smoke tests only: keep the first N rows")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: see module docstring)")
    args = ap.parse_args()
    jd.setup_tf(args.threads, args.seed)  # before any TF op runs

    smoke = bool(args.max_rows) or args.epochs != 10
    name = "L0_official" + (f"_seed{args.seed}" if args.seed else "")
    out_dir = args.out or (ARTIFACTS / "smoke" / name if smoke else RESULTS / name)
    ckpt_dir = out_dir if smoke else ARTIFACTS / out_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    df = jd.load_frame(args.max_rows)
    show_raw_visualization(df, out_dir / "raw_features.png")

    split_fraction = 0.715
    train_split = int(split_fraction * int(df.shape[0]))
    step = 6

    past = 720
    future = 72
    learning_rate = 0.001
    batch_size = 256
    epochs = args.epochs

    print("The selected parameters are:", ", ".join([titles[i] for i in [0, 1, 5, 7, 8, 10, 11]]))
    selected_features = [feature_keys[i] for i in [0, 1, 5, 7, 8, 10, 11]]
    features = df[selected_features]
    features.index = df[date_time_key]
    t_mean = float(features.values[:train_split, 1].mean())
    t_std = float(features.values[:train_split, 1].std())

    features = normalize(features.values, train_split)
    features = pd.DataFrame(features)

    train_data = features.loc[0 : train_split - 1]
    val_data = features.loc[train_split:]

    start = past + future
    end = start + train_split

    x_train = train_data[[i for i in range(7)]].values
    y_train = features.iloc[start:end][[1]]

    sequence_length = int(past / step)

    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train, y_train, sequence_length=sequence_length, sampling_rate=step, batch_size=batch_size
    )

    x_end = len(val_data) - past - future

    label_start = train_split + past + future

    x_val = val_data.iloc[:x_end][[i for i in range(7)]].values
    y_val = features.iloc[label_start:][[1]]

    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        x_val, y_val, sequence_length=sequence_length, sampling_rate=step, batch_size=batch_size
    )

    for batch in dataset_train.take(1):
        inputs, targets = batch

    print("Input shape:", inputs.numpy().shape)
    print("Target shape:", targets.numpy().shape)

    inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
    lstm_out = keras.layers.LSTM(32)(inputs)
    outputs = keras.layers.Dense(1)(lstm_out)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    model.summary()

    path_checkpoint = str(ckpt_dir / "model_checkpoint.weights.h5")
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )
    timer = jd.epoch_timer()

    t0 = time.perf_counter()
    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        callbacks=[es_callback, modelckpt_callback, timer],
    )
    train_time = time.perf_counter() - t0

    visualize_loss(history, "Training and Validation Loss", out_dir / "loss_curve.png")

    # the official example predicts with the in-memory model, i.e. the last epoch
    for k, (x, y) in enumerate(dataset_val.take(5), 1):
        show_plot(
            [x[0][:, 1].numpy(), y[0].numpy(), model.predict(x, verbose=0)[0]],
            12,
            "Single Step Prediction",
            out_dir / f"prediction_{k}.png",
        )

    last = evaluate(model, dataset_val, t_std)
    model.load_weights(path_checkpoint)
    best = evaluate(model, dataset_val, t_std)

    val_loss = [float(v) for v in history.history["val_loss"]]
    hist = pd.DataFrame(history.history)
    hist.insert(0, "epoch", range(1, len(hist) + 1))
    hist["seconds"] = timer.seconds
    hist.to_csv(out_dir / "history.csv", index=False)

    wv = df["wv (m/s)"].to_numpy()
    summary = {
        "model": "L0_official",
        "source": "reference/timeseries_weather_forecasting.py",
        "seed": args.seed,
        "threads": args.threads,
        "max_rows": args.max_rows,
        "smoke": smoke,
        "config": {
            "features": selected_features,
            "split_fraction": split_fraction,
            "train_split": train_split,
            "past": past,
            "future": future,
            "step": step,
            "sequence_length": sequence_length,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "early_stopping": "val_loss, min_delta=0, patience=5, restore_best_weights=False",
            "checkpoint": "val_loss, save_best_only",
        },
        "task": "inputs rows i, i+6, ..., i+714; label T at row i+792 = 78 rows (13 h) after the last input",
        "n_rows": int(len(df)),
        "n_train_windows": len(x_train) - (sequence_length - 1) * step,
        "n_val_windows": last["n_windows"],
        "t_mean": t_mean,
        "t_std": t_std,
        "wv_sentinels": {
            "train": int((wv[:train_split] == jd.SENTINEL).sum()),
            "val": int((wv[train_split:] == jd.SENTINEL).sum()),
        },
        "val_mse_per_epoch": val_loss,
        "epochs_run": len(val_loss),
        "early_stopped": len(val_loss) < epochs,
        "best_epoch": int(np.argmin(val_loss)) + 1,
        "seconds_per_epoch": timer.seconds,
        "mean_seconds_per_epoch": float(np.mean(timer.seconds)),
        "train_time_s": train_time,
        "last_epoch": {k: v for k, v in last.items() if k != "n_windows"},
        "best_checkpoint": {k: v for k, v in best.items() if k != "n_windows"},
        "versions": jd.versions(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("epochs_run", "best_epoch", "last_epoch", "best_checkpoint")}, indent=2))


if __name__ == "__main__":
    main()
