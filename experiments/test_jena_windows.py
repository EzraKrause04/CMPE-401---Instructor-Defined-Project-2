"""Window/label alignment checks for jena_data on a synthetic series where value == row index.

Run: python experiments/test_jena_windows.py   (or pytest experiments/test_jena_windows.py)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jena_data as jd  # noqa: E402

jd.setup_tf(threads=2)  # CPU only, capped threads

N = 420_551  # real dataset length, so split sizes match the benchmark
PAST, FUTURE, STEP = 720, 72, 6


def synthetic_frame(n: int = N) -> pd.DataFrame:
    rows = np.arange(n, dtype=np.float64)
    df = pd.DataFrame({k: rows for k in jd.FEATURE_KEYS})
    start = pd.Timestamp("2009-01-01 00:10:00")
    df[jd.DATE_KEY] = (start + pd.to_timedelta(rows * 10, unit="min")).strftime("%d.%m.%Y %H:%M:%S")
    return df


def denorm_rows(w: jd.JenaWindows, arr: np.ndarray) -> np.ndarray:
    """Normalized T values back to row indices (T column == row index)."""
    return np.rint(arr.astype(np.float64) * w.meta["t_std"] + w.meta["t_mean"]).astype(np.int64)


def check_split(w: jd.JenaWindows, split: str):
    a, b = w.meta["bounds"][split]
    inp, lab = w.rows(split)
    assert inp.shape[1] == PAST // STEP == 120
    assert inp[0, 0] == a and inp[0, -1] == a + PAST - STEP
    assert lab[0] == a + PAST + FUTURE
    assert lab[-1] == b - 1, (split, lab[-1], b)  # the last usable label is the last row of the split
    assert inp[-1, 0] == b - 1 - PAST - FUTURE
    assert lab[0] - inp[0, -1] == FUTURE + STEP == 78  # 13 h after the last input sample
    # no window crosses a split boundary
    assert inp.min() >= a and inp.max() < b and lab.min() >= a and lab.max() < b
    # the values served by arrays() and by tf.data carry the same row indices
    x, y = w.arrays(split, np.array([0, len(lab) - 1]))
    t = jd.SELECTED.index(jd.TARGET)
    assert np.array_equal(denorm_rows(w, x[..., t]), inp[[0, -1]])
    assert np.array_equal(denorm_rows(w, y), lab[[0, -1]])


def test_splits_and_windows():
    w = jd.prepare(synthetic_frame(), PAST, FUTURE, STEP)
    b = w.meta["bounds"]
    assert b["train"] == (0, 300_693) and b["val"] == (300_693, 360_622) and b["test"] == (360_622, N)
    for split in ("train", "val", "test"):
        check_split(w, split)
    assert w.meta["n_windows"] == {"train": 299_901, "val": 59_137, "test": 59_137}


def test_eval_past_pins_label_rows():
    for past in (360, 720, 1440):
        w = jd.prepare(synthetic_frame(), past, FUTURE, STEP, eval_past=jd.EVAL_PAST)
        for split in ("val", "test"):
            a, b = w.meta["bounds"][split]
            inp, lab = w.rows(split)
            assert lab[0] == a + jd.EVAL_PAST + FUTURE and lab[-1] == b - 1
            assert inp.min() >= a and np.array_equal(lab - inp[:, -1], np.full(len(lab), FUTURE + STEP))
        assert w.rows("train")[1][0] == past + FUTURE  # train windows are never trimmed


def test_first_train_window_matches_official():
    import keras

    df = synthetic_frame()
    w = jd.prepare(df, PAST, FUTURE, STEP)
    # official example: x_train = train rows, y_train = rows [past+future, past+future+train_split)
    train_split = int(jd.SPLIT_FRACTION * N)
    rows = np.arange(N, dtype=np.float64)
    official = keras.utils.timeseries_dataset_from_array(
        rows[:train_split, None],
        rows[PAST + FUTURE : PAST + FUTURE + train_split, None],
        sequence_length=PAST // STEP,
        sampling_rate=STEP,
        batch_size=4,
    )
    ox, oy = next(iter(official))
    inp, lab = w.rows("train")
    assert np.array_equal(ox.numpy()[0, :, 0].astype(np.int64), inp[0])
    assert int(oy.numpy()[0, 0]) == lab[0] == 792
    # and the tf.data pipeline serves the same first window
    bx, by = next(iter(w.dataset("train", batch_size=4)))
    t = jd.SELECTED.index(jd.TARGET)
    assert np.array_equal(denorm_rows(w, bx.numpy()[0, :, t]), inp[0])
    assert denorm_rows(w, by.numpy()[0])[0] == 792
    # the official train set has 78 extra windows whose labels fall past train_split
    assert (train_split - (PAST // STEP - 1) * STEP) - len(lab) == FUTURE + STEP


def test_time_features_and_shuffle():
    w = jd.prepare(synthetic_frame(), PAST, FUTURE, STEP, time_features=True)
    assert w.x.shape[1] == 11 and w.meta["features"][-4:] == ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    assert np.abs(w.x[:, -4:]).max() <= 1.0 + 1e-6
    # hour_sin at 06:00 (row 35 = 00:10 + 350 min) is sin(pi/2)
    assert abs(w.x[35, -4] - 1.0) < 1e-6
    ws = jd.prepare(synthetic_frame(), PAST, FUTURE, STEP, max_train_windows=1000)
    order = np.concatenate([b[1].numpy()[:, 0] for b in ws.dataset("train", 100, shuffle=True, seed=0)])
    ref = ws.labels("train")
    assert not np.array_equal(order, ref) and np.allclose(np.sort(order), np.sort(ref))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
