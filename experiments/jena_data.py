"""Jena Climate data pipeline for the LSTM benchmark protocol (plus shared runtime helpers).

Task definition (unchanged from the official Keras example): a window starting at row i
reads rows i, i+step, ..., i+past-step and predicts normalized T at row i+past+future, i.e.
future+step = 78 rows (13 h) after the last input sample.

Protocol differences from the official example:
  * chronological train / val / test split; train rows are identical to the official
    train split, the official "validation" rows are halved into val and test;
  * every input row and label row of a window lies inside its split (the official
    example lets the last 78 train labels fall in the validation rows);
  * optionally, val/test label rows start at a + eval_past + future for every variant, so
    runs with different `past` are scored on exactly the same label rows.
"""

import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
URI = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
CSV_NAME = "jena_climate_2009_2016.csv"
DATE_KEY = "Date Time"
FEATURE_KEYS = [
    "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)", "VPact (mbar)",
    "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)", "wd (deg)",
]
SELECTED = [FEATURE_KEYS[i] for i in (0, 1, 5, 7, 8, 10, 11)]  # official feature subset
TARGET = "T (degC)"
WV_KEYS = ["wv (m/s)", "max. wv (m/s)"]
SENTINEL = -9999.0
SPLIT_FRACTION = 0.715
EVAL_PAST = 1440  # longest context in the benchmark registry; pins val/test label rows across variants


def csv_path() -> Path:
    """Download the zip once into data/ and extract the CSV next to it."""
    import keras

    csv = DATA_DIR / CSV_NAME
    if not csv.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = keras.utils.get_file(fname=CSV_NAME + ".zip", origin=URI, cache_dir=str(DATA_DIR), cache_subdir=".")
        with ZipFile(zip_path) as z:
            z.extract(CSV_NAME, DATA_DIR)
    return csv


def load_frame(max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path())
    return df.iloc[:max_rows].reset_index(drop=True) if max_rows else df


def setup_tf(threads: int = 4, seed: int | None = None):
    """Capped CPU thread pools and an optional global seed. Call before any TF op runs.

    Uses a GPU when TensorFlow sees one (Colab); on the Mac TF is CPU-only anyway.
    """
    import keras
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(min(2, threads))
    if seed is not None:
        keras.utils.set_random_seed(seed)
    return tf


def versions() -> dict:
    import keras
    import tensorflow as tf

    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "gpus": [d.name for d in tf.config.list_logical_devices("GPU")],
    }


def split_bounds(n: int) -> dict[str, tuple[int, int]]:
    n_train = int(SPLIT_FRACTION * n)
    n_mid = n_train + (n - n_train) // 2
    return {"train": (0, n_train), "val": (n_train, n_mid), "test": (n_mid, n)}


def window_starts(a: int, b: int, past: int, future: int, first_label: int | None = None) -> np.ndarray:
    """Start rows of all windows whose inputs and label lie in [a, b), optionally with label >= first_label."""
    lo = a if first_label is None else max(a, first_label - past - future)
    return np.arange(lo, b - past - future, dtype=np.int64)


def window_rows(starts: np.ndarray, past: int, future: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    """(input rows [W, past // step], label rows [W]) for the given window starts."""
    return starts[:, None] + np.arange(0, past, step), starts + past + future


def calendar_features(dates: pd.Series) -> tuple[np.ndarray, list[str]]:
    dt = pd.to_datetime(dates, format="%d.%m.%Y %H:%M:%S")
    hour = (dt.dt.hour + dt.dt.minute / 60).to_numpy()
    day = dt.dt.dayofyear.to_numpy() - 1 + hour / 24
    angles = [2 * np.pi * hour / 24, 2 * np.pi * day / 365.2425]
    cols = np.stack([f(a) for a in angles for f in (np.sin, np.cos)], axis=1)
    return cols, ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]


@dataclass
class JenaWindows:
    x: np.ndarray  # [N, F] model inputs (normalized features, then raw time features)
    target: np.ndarray  # [N] normalized T
    starts: dict[str, np.ndarray]
    past: int
    future: int
    step: int
    meta: dict
    _tensors: tuple = field(default=None, repr=False)

    def rows(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        return window_rows(self.starts[split], self.past, self.future, self.step)

    def labels(self, split: str) -> np.ndarray:
        return self.target[self.starts[split] + self.past + self.future]

    def arrays(self, split: str, idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Materialized (inputs [W, T, F], labels [W]); pass idx to take a subset of windows."""
        starts = self.starts[split] if idx is None else self.starts[split][idx]
        inp, lab = window_rows(starts, self.past, self.future, self.step)
        return self.x[inp], self.target[lab]

    def dataset(self, split: str, batch_size: int = 256, shuffle: bool = False, seed: int = 0):
        """Batched tf.data pipeline that gathers windows on the fly (never materializes all windows)."""
        import tensorflow as tf

        if self._tensors is None:
            self._tensors = (tf.constant(self.x), tf.constant(self.target))
        x, y = self._tensors
        offsets = tf.range(0, self.past, self.step, dtype=tf.int64)
        label_offset = self.past + self.future
        starts = self.starts[split]
        ds = tf.data.Dataset.from_tensor_slices(starts)
        if shuffle:
            ds = ds.shuffle(len(starts), seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).map(
            lambda s: (tf.gather(x, s[:, None] + offsets), tf.gather(y, s[:, None] + label_offset)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        opts = tf.data.Options()
        if threads := tf.config.threading.get_intra_op_parallelism_threads():
            opts.threading.private_threadpool_size = threads
        return ds.with_options(opts).prefetch(tf.data.AUTOTUNE)


def prepare(
    df: pd.DataFrame | None = None,
    past: int = 720,
    future: int = 72,
    step: int = 6,
    clean_wv: bool = False,
    time_features: bool = False,
    eval_past: int | None = None,
    max_train_windows: int | None = None,
) -> JenaWindows:
    """Normalize with train-only stats and index the windows of each split."""
    df = load_frame() if df is None else df
    n = len(df)
    bounds = split_bounds(n)
    n_train = bounds["train"][1]
    raw = df[SELECTED].to_numpy(dtype=np.float64, copy=True)
    wv_sentinels = {k: int((df[k].to_numpy() == SENTINEL).sum()) for k in WV_KEYS}
    if clean_wv:
        for k in WV_KEYS:
            if k in SELECTED:
                col = raw[:, SELECTED.index(k)]
                col[col == SENTINEL] = 0.0
    mean, std = raw[:n_train].mean(axis=0), raw[:n_train].std(axis=0)
    x = (raw - mean) / std
    names = list(SELECTED)
    if time_features:
        extra, extra_names = calendar_features(df[DATE_KEY])
        x = np.concatenate([x, extra], axis=1)
        names += extra_names
    t = SELECTED.index(TARGET)

    starts = {}
    for split, (a, b) in bounds.items():
        first_label = None if split == "train" or eval_past is None else a + max(eval_past, past) + future
        starts[split] = window_starts(a, b, past, future, first_label)
    if max_train_windows:
        starts["train"] = starts["train"][:max_train_windows]

    meta = {
        "n_rows": n,
        "bounds": bounds,
        "features": names,
        "past": past,
        "future": future,
        "step": step,
        "clean_wv": clean_wv,
        "time_features": time_features,
        "eval_past": eval_past,
        "t_mean": float(mean[t]),
        "t_std": float(std[t]),
        "wv_sentinel_rows": wv_sentinels,
        "n_windows": {s: int(len(v)) for s, v in starts.items()},
        "label_rows": {s: [int(v[0] + past + future), int(v[-1] + past + future)] for s, v in starts.items() if len(v)},
    }
    return JenaWindows(x.astype(np.float32), x[:, t].astype(np.float32), starts, past, future, step, meta)


def epoch_timer():
    """Keras callback whose .seconds lists wall-clock time per epoch (train + validation)."""
    import keras

    class EpochTimer(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.seconds = []

        def on_epoch_begin(self, epoch, logs=None):
            self.t0 = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            self.seconds.append(time.perf_counter() - self.t0)

    return EpochTimer()
