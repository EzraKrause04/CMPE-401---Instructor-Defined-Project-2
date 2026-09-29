"""Reload the committed models in models/ and re-score them against results/.

Rebuilds each architecture with the same code that trained it, loads the saved
weights, and recomputes the test metric reported in results/: FordA test accuracy
for the Transformer runs and held-out test MAE (degC) for every LSTM benchmark
model. A match shows the weights in models/ are the ones behind the README.

Usage:
    python experiments/verify_models.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jena_data as jd  # noqa: E402

jd.setup_tf(threads=8)
import keras  # noqa: E402

import lstm_benchmark as lb  # noqa: E402
import transformer_official as to  # noqa: E402

MODELS = jd.REPO / "models"
RESULTS = jd.REPO / "results"
TRANSFORMER_RUNS = [("T0_official", "channels_last", 4), ("T1_page_version", "channels_first", 4),
                    ("T2_no_encoder", "channels_first", 0)]


def main() -> None:
    ok = True
    print("Transformer, FordA test accuracy")
    _, _, x_test, y_test, n_classes = to.load_forda()
    for run, pooling, blocks in TRANSFORMER_RUNS:
        keras.backend.clear_session()
        model = to.build_official(x_test.shape[1:], n_classes, pooling=pooling, blocks=blocks)
        model.load_weights(MODELS / "transformer" / f"{run}.weights.h5")
        acc = float((model.predict(x_test, batch_size=64, verbose=0).argmax(1) == y_test).mean())
        ref = json.loads((RESULTS / "transformer" / run / "summary.json").read_text())["test_acc"]
        match = abs(acc - ref) < 1e-3
        ok &= match
        print(f"  {run:16s} {acc:.4f} (reported {ref:.4f}) {'ok' if match else 'MISMATCH'}")

    print("LSTM benchmark, held-out test MAE (degC)")
    df, prepared, worst = jd.load_frame(), {}, 0.0
    for path in sorted((RESULTS / "lstm_benchmark").glob("*/seed*.json")):
        rec = json.loads(path.read_text())
        cfg, name, seed = rec["config"], path.parent.name, rec["seed"]
        key = (cfg["past"], cfg["clean_wv"], cfg["time_features"])
        if key not in prepared:
            prepared[key] = jd.prepare(df, past=cfg["past"], future=lb.FUTURE, step=lb.STEP, clean_wv=cfg["clean_wv"],
                                       time_features=cfg["time_features"], eval_past=jd.EVAL_PAST)
        data = prepared[key]
        keras.backend.clear_session()
        model = lb.build_model((cfg["past"] // lb.STEP, data.x.shape[1]), cfg["units"], cfg["dropout"])
        model.load_weights(MODELS / "lstm_benchmark" / name / f"seed{seed}.weights.h5")
        mae = lb.score(model, data.dataset("test", 256), data.labels("test"), data.meta["t_std"], "test")["test_mae_C"]
        diff = abs(mae - rec["test_mae_C"])
        worst = max(worst, diff)
        ok &= diff < 1e-4
        print(f"  {name:26s} seed{seed} {mae:.4f} (reported {rec['test_mae_C']:.4f})")
    print(f"largest LSTM difference: {worst:.1e} degC")
    print("ALL MATCH" if ok else "MISMATCH FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
