"""Official Keras "Timeseries classification with a Transformer model" (FordA), minimally instrumented.

Data loading, preprocessing, `transformer_encoder`, `build_model`, hyperparameters and the
training loop follow reference/timeseries_classification_transformer.py (keras-io master).
Additions only: seeding (before the shuffle), CPU thread caps, a local FordA cache, logging
(history.csv, summary.json, curves.png, model_summary.txt), resumable training and two switches:

  --pooling channels_first  reproduces the pre-2024 example rendered on keras.io
                            (keras-io #1733 / a52a1c5f changed it to channels_last)
  --blocks N                number of encoder blocks (0 = MLP head only)

An interrupted run relaunched with the same --run-name resumes after its last completed epoch
(weights, optimizer state and EarlyStopping state are restored; dropout/shuffle RNG streams
restart, which summary.json records as "resumed_at_epoch"). summary.json is written last, so
its presence means the run finished; a finished run is only replaced with --overwrite.

Usage:
    python experiments/transformer_official.py --run-name T0_official
    python experiments/transformer_official.py --run-name T1_page_version --pooling channels_first
    python experiments/transformer_official.py --run-name T2_no_encoder --pooling channels_first --blocks 0
    python experiments/transformer_official.py --run-name smoke --epochs 1 --max-train-samples 256
"""

import argparse
import json
import math
import os
import platform
import shutil
import sys
import time
import urllib.request
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"  # configure_cpu relies on TF; never fall back to torch/MPS
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "forda"
RESULTS_DIR = REPO / "results" / "transformer"
ARTIFACTS_DIR = REPO / "artifacts" / "transformer"
ROOT_URL = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
METRIC_NAMES = {"sparse_categorical_accuracy": "acc", "val_sparse_categorical_accuracy": "val_acc"}


def configure_cpu(threads: int) -> None:
    """Bound TensorFlow's CPU thread pools (must run before any TF op).

    A GPU is used when TensorFlow sees one (Colab); on the Mac TF is CPU-only anyway.
    """
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(min(2, threads))


def forda_path(split: str) -> Path:
    """Local copy of FordA_{split}.tsv, downloaded once from the URL used by the example."""
    path = DATA_DIR / f"FordA_{split}.tsv"
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        print(f"downloading {ROOT_URL}FordA_{split}.tsv -> {path}")
        urllib.request.urlretrieve(ROOT_URL + f"FordA_{split}.tsv", tmp)
        tmp.rename(path)
    return path


def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


def load_forda():
    """Official preprocessing; call keras.utils.set_random_seed first for a reproducible shuffle."""
    x_train, y_train = readucr(forda_path("TRAIN"))
    x_test, y_test = readucr(forda_path("TEST"))

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    n_classes = len(np.unique(y_train))

    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    return x_train, y_train, x_test, y_test, n_classes


# This implementation applies Layer Normalization before the residual connection
# to improve training stability by producing better-behaved gradients and often
# eliminating the need for learning rate warm-up.


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    n_classes=2,  # added: a module-level global in the reference
    pooling="channels_last",  # added: master = "channels_last", keras.io page = "channels_first"
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format=pooling)(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def build_official(input_shape, n_classes, pooling="channels_last", blocks=4):
    """build_model with the example's hyperparameters."""
    return build_model(
        input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=blocks,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
        n_classes=n_classes,
        pooling=pooling,
    )


def pooled_shape(model) -> list:
    gap = next(l for l in model.layers if isinstance(l, layers.GlobalAveragePooling1D))
    return list(gap.output.shape)


class EpochLog(keras.callbacks.Callback):
    """Rewrites history.csv after every epoch (so long runs can be monitored) and times epochs.

    `rows` are the earlier epochs of a resumed run; `session` counts launches of the run.
    """

    def __init__(self, path: Path, rows=(), session: int = 1):
        super().__init__()
        self.path, self.rows, self.session = path, list(rows), session

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        row = {"epoch": epoch + 1, **{METRIC_NAMES.get(k, k): float(v) for k, v in (logs or {}).items()}}
        row["epoch_time_s"] = time.perf_counter() - self.t0
        row["session"] = self.session
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)


class ResumableEarlyStopping(keras.callbacks.EarlyStopping):
    """EarlyStopping whose best/wait/best_weights are rebuilt from history + best checkpoint on resume."""

    def __init__(self, prior_rows, best_path: Path, **kwargs):
        super().__init__(**kwargs)
        self.prior_rows, self.best_path = prior_rows, best_path

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        if not self.prior_rows:
            return
        val = [r["val_loss"] for r in self.prior_rows]
        b = int(np.argmin(val))  # first minimum, as EarlyStopping's strict "<" keeps it
        self.best, self.best_epoch, self.wait = val[b], b, len(val) - 1 - b
        if self.restore_best_weights:
            best = keras.models.clone_model(self.model)  # a clone, so the resumed optimizer state is untouched
            best.load_weights(self.best_path)
            self.best_weights = best.get_weights()


def plot_curves(hist: pd.DataFrame, best_epoch: int, val_const: np.ndarray, title: str, path: Path) -> None:
    """Loss/accuracy curves; val_const[k] = val accuracy of always predicting class k."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    train_c, val_c, ink, muted = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984"
    marker = {"marker": "o", "ms": 4} if len(hist) <= 20 else {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, label in [(axes[0], "loss", "Loss"), (axes[1], "acc", "Accuracy")]:
        ax.plot(hist["epoch"], hist[key], color=train_c, lw=2, label="train", **marker)
        ax.plot(hist["epoch"], hist[f"val_{key}"], color=val_c, lw=2, label="val", **marker)
        ax.axvline(best_epoch, color=muted, lw=1, ls="--", label=f"best epoch ({best_epoch})")
        if key == "loss":
            ax.axhline(np.log(len(val_const)), color=muted, lw=1, ls=":", label=f"ln {len(val_const)} (uniform output)")
        else:
            lo, hi = float(val_const.min()), float(val_const.max())
            ax.axhspan(lo, hi, color=muted, alpha=0.18, lw=0, label=f"constant prediction on val ({lo:.3f}-{hi:.3f})")
        # A flat chance-level curve must not be autoscaled into apparent divergence.
        y0, y1 = ax.get_ylim()
        if y1 - y0 < 0.02:
            mid = (y0 + y1) / 2
            ax.set_ylim(mid - 0.01, mid + 0.01)
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.set_xlim(0.5, hist["epoch"].max() + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        ax.set_xlabel("epoch", color=ink)
        ax.set_title(label, color=ink, loc="left")
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def base_parser(description: str, default_run: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--run-name", default=default_run)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4, help="TF intra-op threads (inter-op = min(2, threads))")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--blocks", type=int, default=4, help="number of transformer_encoder blocks")
    p.add_argument("--max-train-samples", type=int, default=None, help="smoke tests only")
    p.add_argument("--verbose", type=int, default=2, help="Keras fit verbosity (2 = one line per epoch)")
    p.add_argument("--overwrite", action="store_true", help="start over even if the run finished or can resume")
    return p


def prepare(args):
    """Thread caps, seed (before the shuffle), data. Returns (x_train, y_train, x_test, y_test, n_classes)."""
    configure_cpu(args.threads)
    keras.utils.set_random_seed(args.seed)
    x_train, y_train, x_test, y_test, n_classes = load_forda()
    if args.max_train_samples:
        x_train, y_train = x_train[: args.max_train_samples], y_train[: args.max_train_samples]
    return x_train, y_train, x_test, y_test, n_classes


RESUME_IGNORED_ARGS = ("threads", "verbose", "overwrite", "epochs")


def start_run(args, out_dir: Path, art_dir: Path):
    """Fresh start or resume; clears stale final outputs. Returns (resume_epoch, prior history rows)."""
    backup_dir = art_dir / "backup"
    meta, args_file = backup_dir / "training_metadata.json", backup_dir / "run_args.json"
    if args.overwrite:
        shutil.rmtree(backup_dir, ignore_errors=True)
    elif (out_dir / "summary.json").exists():
        sys.exit(f"{out_dir / 'summary.json'} exists (finished run): pass --overwrite or use another --run-name")
    resume_epoch = int(json.loads(meta.read_text())["epoch"]) if meta.exists() else 0
    run_args = {k: v for k, v in vars(args).items() if k not in RESUME_IGNORED_ARGS}
    if resume_epoch and json.loads(args_file.read_text()) != run_args:
        sys.exit(f"{backup_dir} was written with different arguments: pass --overwrite or use another --run-name")
    # summary.json is written last, so its presence means "finished"; remove it until this run is.
    for f in (out_dir / "summary.json", out_dir / "curves.png", art_dir / "model.weights.h5"):
        f.unlink(missing_ok=True)
    prior = []
    if resume_epoch:
        prior = [r for r in pd.read_csv(out_dir / "history.csv").to_dict("records") if r["epoch"] <= resume_epoch]
    else:
        shutil.rmtree(backup_dir, ignore_errors=True)
        (art_dir / "best.weights.h5").unlink(missing_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    args_file.write_text(json.dumps(run_args))
    return resume_epoch, prior


def train_and_report(model, data, args, model_kind: str) -> dict:
    """Compile/fit/evaluate exactly as the example, then write results/ and artifacts/ outputs."""
    x_train, y_train, x_test, y_test, n_classes = data
    out_dir = RESULTS_DIR / args.run_name
    art_dir = ARTIFACTS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    resume_epoch, prior = start_run(args, out_dir, art_dir)
    t_start = time.perf_counter()

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"],
    )
    os.environ["COLUMNS"] = "200"  # without a TTY, rich crops the summary to 80 columns (drops "Param #")
    lines = []
    model.summary(line_length=120, print_fn=lambda s, **_: lines.append(s))
    summary_txt = "\n".join(lines)
    print(summary_txt)
    (out_dir / "model_summary.txt").write_text(summary_txt + "\n", encoding="utf-8")
    params = {
        "total": int(model.count_params()),
        "trainable": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
    }
    print(f"[{args.run_name}] params={params['total']:,} pooled feature shape={pooled_shape(model)}")
    init_path = art_dir / "init.weights.h5"
    if resume_epoch and init_path.exists():
        print(f"[{args.run_name}] resuming after epoch {resume_epoch} from {art_dir / 'backup'}")
    else:
        # Initial weights let the probe check which weights training actually moved.
        model.save_weights(init_path)

    best_path = art_dir / "best.weights.h5"
    session = max((int(r.get("session", 1)) for r in prior), default=0) + 1
    epoch_log = EpochLog(out_dir / "history.csv", prior, session)
    # The example's EarlyStopping(patience=10, restore_best_weights=True), plus per-epoch checkpoints.
    # History and the best checkpoint are saved before the backup that marks an epoch as done.
    callbacks = [
        epoch_log,
        ResumableEarlyStopping(prior, best_path, patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            str(best_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            initial_value_threshold=min((r["val_loss"] for r in prior), default=None),
        ),
        keras.callbacks.BackupAndRestore(str(art_dir / "backup"), double_checkpoint=True),
    ]
    t_fit = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=args.verbose,
    )
    fit_time = time.perf_counter() - t_fit
    model.save_weights(art_dir / "model.weights.h5")  # best-val_loss weights (EarlyStopping restores them at train end)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=args.verbose)
    probs = model.predict(x_test, batch_size=256, verbose=0)
    pred = probs.argmax(1)

    hist = pd.DataFrame(epoch_log.rows)
    n_fit = int(math.floor(len(x_train) * (1.0 - 0.2)))  # same split rule as Keras validation_split
    best = int(hist["val_loss"].idxmin())
    fit_major = int(np.bincount(y_train[:n_fit]).argmax())
    val_const = np.bincount(y_train[n_fit:], minlength=n_classes) / (len(x_train) - n_fit)
    test_const = np.bincount(y_test, minlength=n_classes) / len(y_test)
    times = hist["epoch_time_s"].to_numpy()
    summary = {
        "run_name": args.run_name,
        "model": model_kind,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "majority_class_test_acc": float(test_const[fit_major]),  # always predicting the fit split's majority class
        "majority_class_val_acc": float(val_const.max()),
        # [k] = accuracy of always predicting class k (= class fractions); any constant model scores one of these
        "constant_predictor_test_acc": test_const.tolist(),
        "constant_predictor_val_acc": val_const.tolist(),
        "test_pred_class_fractions": (np.bincount(pred, minlength=probs.shape[1]) / len(pred)).tolist(),
        "test_prob_class1_std": float(probs[:, 1].std()),  # ~0 => same output for every series
        "params": params,
        "pooled_feature_shape": pooled_shape(model),
        "jit_compile": bool(model.jit_compile),  # Keras "auto" disables XLA on CPU-only machines
        "epochs_run": int(len(hist)),
        "early_stopped": bool(len(hist) < args.epochs),
        "best_epoch": best + 1,
        "best_val_loss": float(hist.loc[best, "val_loss"]),
        "best_val_acc": float(hist.loc[best, "val_acc"]),
        "train_acc_at_best": float(hist.loc[best, "acc"]),
        "wall_time_s": fit_time,  # fit() of this launch only; see epoch_time_total_s for resumed runs
        "epoch_time_total_s": float(times.sum()),
        "total_time_s": time.perf_counter() - t_start,
        "seconds_per_epoch": float(times.mean()),
        "seconds_per_epoch_median": float(np.median(times)),
        "first_epoch_s": float(times[0]),
        "sessions": int(hist["session"].max()),
        "resumed_at_epoch": [int(e) for e in hist.loc[hist["session"].diff() > 0, "epoch"]],
        "n_train": n_fit,
        "n_val": int(len(x_train) - n_fit),
        "n_test": int(len(x_test)),
        "versions": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "backend": keras.backend.backend(),
            "gpus": [d.name for d in tf.config.list_logical_devices("GPU")],
        },
        "args": vars(args),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
    }
    title = f"{args.run_name}: test acc {test_acc:.3f} ({params['total']:,} params, pooled {pooled_shape(model)})"
    try:  # a plotting error must not cost a finished run its summary
        plot_curves(hist, best + 1, val_const, title, out_dir / "curves.png")
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: curves.png not written: {e!r}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("test_acc", "majority_class_test_acc", "epochs_run", "best_epoch", "seconds_per_epoch")}))
    print(f"wrote {out_dir} and {art_dir}")
    return summary


def main() -> None:
    p = base_parser(__doc__.splitlines()[0], "T0_official")
    p.add_argument("--pooling", choices=["channels_last", "channels_first"], default="channels_last")
    args = p.parse_args()
    data = prepare(args)
    model = build_official(data[0].shape[1:], data[4], pooling=args.pooling, blocks=args.blocks)
    train_and_report(model, data, args, model_kind="official")


if __name__ == "__main__":
    main()
