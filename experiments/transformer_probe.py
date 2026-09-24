"""Numerical evidence that the official FordA Transformer encoder reduces to x + constant.

With one input feature, every LayerNormalization(axis=-1) normalizes over a single value, so
it outputs exactly its beta and each encoder block computes x + const. This script measures:

  a) params / pooled shape for channels_last (keras-io master) vs channels_first (keras.io page)
  b) encoder output - input over time, and LayerNorm output - beta, for random weights with
     random non-zero betas/gammas and for trained runs whose weights exist; gradients that
     reach attention/feed-forward weights; which weights training moved (init vs final)
  c) across-sample spread of the pooled feature under each pooling, and whether the model
     equals "MLP head applied to GAP(x + C)"
  d) per-series mean/std of FordA (z-normalization)

Writes results/transformer/probe.json and probe_encoder_identity.png.

Usage:
    python experiments/transformer_probe.py
    python experiments/transformer_probe.py --runs T0_official T1_page_version --n-test 1320
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["KERAS_BACKEND"] = "tensorflow"  # before keras is imported (see transformer_official.py)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import roc_auc_score

from transformer_official import ARTIFACTS_DIR, RESULTS_DIR, build_official, configure_cpu, load_forda, pooled_shape

EPS = 1e-6  # LayerNormalization epsilon in the example


def ulp32(v: float) -> float:
    return float(np.spacing(np.float32(abs(v))))


def auc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None


def series_stats(x: np.ndarray) -> dict:
    x = x.reshape(len(x), -1).astype(np.float64)
    m, s = x.mean(1), x.std(1)
    return {
        "n_series": int(len(x)),
        "length": int(x.shape[1]),
        "mean_abs_max": float(np.abs(m).max()),
        "mean_abs_median": float(np.median(np.abs(m))),
        "std_min": float(s.min()),
        "std_max": float(s.max()),
        "std_mean": float(s.mean()),
        "std_ddof1_mean": float(x.std(1, ddof=1).mean()),  # ~1.0: normalized with the sample std
    }


def parts(model):
    lns = [l for l in model.layers if isinstance(l, layers.LayerNormalization)]
    gap = next(l for l in model.layers if isinstance(l, layers.GlobalAveragePooling1D))
    head = model.layers[model.layers.index(gap) + 1 :]
    return lns, gap, head


def weight_groups(model) -> dict:
    """Map each trainable variable to a readable group (layer type, LayerNorm gamma/beta split)."""
    groups = {}
    for layer in model.layers:
        for w in layer.trainable_weights:
            name = type(layer).__name__
            if isinstance(layer, layers.LayerNormalization):
                name += "." + w.name.split("/")[-1]
            groups[id(w)] = name
    return groups


def gradient_probe(model, x, y) -> dict:
    """Max |dLoss/dw| per weight group on one batch in training mode (dropout on), as in fit()."""
    groups = weight_groups(model)
    with tf.GradientTape() as tape:
        p = model(x, training=True)
        loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y, p))
    grads = tape.gradient(loss, model.trainable_variables)
    out = {}
    for w, g in zip(model.trainable_variables, grads):
        m = 0.0 if g is None else float(tf.reduce_max(tf.abs(g)))
        k = groups[id(w)]
        out[k] = max(out.get(k, 0.0), m)
    return out


def weight_change(model, init_path: Path) -> dict:
    """Max |w_final - w_init| per weight group (0 => training never moved that group)."""
    init = keras.models.clone_model(model)
    init.load_weights(init_path)
    groups = weight_groups(model)
    out = {}
    for w, w0 in zip(model.trainable_weights, init.trainable_weights):
        k = groups[id(w)]
        out[k] = max(out.get(k, 0.0), float(np.abs(w.numpy() - w0.numpy()).max()))
    return out


def probe_model(model, x, y, batch_size: int) -> dict:
    """Encoder-identity, pooled-feature and MLP-equivalence measurements for one model."""
    lns, gap, head = parts(model)
    x32 = x.astype(np.float32)
    feats = keras.Model(model.input, [gap.input, model.output] + [l.input for l in lns] + [l.output for l in lns])
    enc, probs, *ln_io = feats.predict(x32, batch_size=batch_size, verbose=0)
    ln_in, ln_out = ln_io[: len(lns)], ln_io[len(lns) :]
    betas = np.array([float(l.beta.numpy()[0]) for l in lns])
    gammas = np.array([float(l.gamma.numpy()[0]) for l in lns])
    C = float(betas.sum())

    r = (enc - x32)[..., 0].astype(np.float64)  # encoder output minus input, (N, 500)
    ln_dev = [float(np.abs(o - b).max()) for o, b in zip(ln_out, betas)]
    # Keras computes x*inv + (beta - mean*inv) with inv = gamma/sqrt(var + eps) and var = 0, so the
    # result is beta plus the rounding of two equal products of size |gamma * x| / sqrt(eps).
    ln_bound = [ulp32(np.abs(i).max() * abs(g) / np.sqrt(EPS)) + ulp32(b) for i, g, b in zip(ln_in, gammas, betas)]

    pooled_last = layers.GlobalAveragePooling1D(data_format="channels_last")(enc).numpy()[:, 0]
    series_mean = x.reshape(len(x), -1).astype(np.float64).mean(1)  # exact mean_t(x), what pooled_last - C estimates
    pooled_first = layers.GlobalAveragePooling1D(data_format="channels_first")(enc).numpy()
    mine = pooled_last if gap.data_format == "channels_last" else pooled_first

    # The claim "model == MLP head on GAP(x + C)": rebuild the classifier input without the encoder.
    h = gap(x32 + np.float32(C))
    for layer in head:
        h = layer(h)
    mlp_probs = np.asarray(h)

    out = {
        "pooling": gap.data_format,
        "params": int(model.count_params()),
        "pooled_feature_shape": pooled_shape(model),
        "n_samples": int(len(x)),
        "encoder": {
            "sum_layernorm_betas": C,
            "layernorm_betas": betas.tolist(),
            "layernorm_gammas": gammas.tolist(),
            "residual_mean": float(r.mean()),
            "residual_std_over_time_max": float(r.std(1).max()),
            "residual_mean_std_across_samples": float(r.mean(1).std()),
            "residual_minus_sum_betas_max_abs": float(np.abs(r - C).max()),
            "input_std_over_time_mean": float(x32[..., 0].std(1).mean()),
            "layernorm_out_minus_beta_max_abs": ln_dev,
            "layernorm_rounding_bound": ln_bound,
            "layernorm_input_max_abs": [float(np.abs(i).max()) for i in ln_in],
        },
        "pooled": {
            "channels_last_value_mean": float(pooled_last.mean()),
            "channels_last_std_across_samples": float(pooled_last.std()),
            "channels_last_range": float(np.ptp(pooled_last)),
            "channels_last_unique_float32_values": int(np.unique(pooled_last).size),
            "channels_last_ulp_at_value": ulp32(pooled_last.mean()),
            "channels_last_range_in_ulps": float(np.ptp(pooled_last) / ulp32(pooled_last.mean())),
            # AUC of the float32 pooled scalar: its few distinct values are rounding ties (see notes)
            "channels_last_label_auc": auc(y, pooled_last),
            "series_mean_float64_abs_max": float(np.abs(series_mean).max()),
            "series_mean_float64_label_auc": auc(y, series_mean),
            "channels_first_std_across_samples_mean": float(pooled_first.std(0).mean()),
            "channels_first_minus_x_plus_C_max_abs": float(np.abs(pooled_first - (x32[..., 0] + C)).max()),
            "this_model_pooled_std_across_samples_mean": float(mine.reshape(len(mine), -1).std(0).mean()),
        },
        "predictions": {
            "prob_class1_std_across_samples": float(probs[:, 1].std()),
            "pred_class_fractions": (np.bincount(probs.argmax(1), minlength=probs.shape[1]) / len(probs)).tolist(),
            "accuracy": float(np.mean(probs.argmax(1) == y)),
            "mlp_on_x_plus_C_max_abs_prob_diff": float(np.abs(mlp_probs - probs).max()),
            "mlp_on_x_plus_C_argmax_agreement": float(np.mean(mlp_probs.argmax(1) == probs.argmax(1))),
        },
    }
    return out, (x32[..., 0], enc[..., 0], C)


def randomize_layernorms(model, rng) -> None:
    """Non-default LayerNorm parameters, so 'output == beta' is not trivially 'output == 0'."""
    for l in parts(model)[0]:
        l.beta.assign(rng.normal(0.0, 1.0, l.beta.shape).astype("float32"))
        l.gamma.assign(rng.normal(1.0, 0.5, l.gamma.shape).astype("float32"))


def execution_modes(model, x) -> dict:
    """Encoder residual and LayerNorm deviation in graph mode (as fit/predict run) and eager mode, train/eval."""
    lns, gap, _ = parts(model)
    sub = keras.Model(model.input, [gap.input] + [l.output for l in lns])
    betas = np.array([float(l.beta.numpy()[0]) for l in lns])
    x32 = tf.constant(x.astype(np.float32))

    @tf.function(autograph=False)
    def graph(t, training):
        return sub(t, training=training)

    out = {}
    for mode, fn in [("graph", graph), ("eager", sub)]:
        for training in (False, True):
            enc, *ln_out = [o.numpy() for o in fn(x32, training=training)]
            out[f"{mode}_{'train' if training else 'eval'}"] = {
                "residual_std_over_time_max": float((enc - x32.numpy())[..., 0].std(1).max()),
                "layernorm_out_minus_beta_max_abs": float(max(np.abs(o - b).max() for o, b in zip(ln_out, betas))),
            }
    return out


def plot_identity(x, enc, C, title, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    t = np.arange(x.shape[1])
    axes[0].plot(t, x[0], color=blue, lw=1.2, label="encoder input x(t)")
    axes[0].plot(t, enc[0], color=orange, lw=1.2, label=f"encoder output (= x + {C:.3f})")
    axes[0].set_title("One FordA test series", loc="left", color=ink)
    axes[0].legend(frameon=False, fontsize=8)
    for i in range(min(5, len(x))):
        axes[1].plot(t, enc[i] - x[i], lw=1.2, color=blue if i == 0 else muted)
    axes[1].axhline(C, color=orange, lw=1, ls="--", label=f"sum of LayerNorm betas = {C:.4f}")
    span = max(1e-3, 50 * float(np.abs(enc[:5] - x[:5] - C).max()))
    axes[1].set_ylim(C - span, C + span)
    axes[1].set_title("Output - input for 5 series", loc="left", color=ink)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("time step", color=ink)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def trained_run(name: str, x, y, x_batch, y_batch, batch_size: int) -> dict:
    """Probe a trained transformer_official.py run from its summary.json + saved weights."""
    summary_path = RESULTS_DIR / name / "summary.json"
    weights = ARTIFACTS_DIR / name / "model.weights.h5"
    if not (summary_path.exists() and weights.exists()):
        return {"skipped": f"missing {summary_path.name} or {weights.name}"}
    summary = json.loads(summary_path.read_text())
    run_args = summary["args"]
    if summary.get("model") != "official" or run_args.get("blocks", 4) == 0:
        return {"skipped": "not an official-architecture run with encoder blocks"}
    model = build_official(x.shape[1:], 2, pooling=run_args["pooling"], blocks=run_args["blocks"])
    model.load_weights(weights)
    res, _ = probe_model(model, x, y, batch_size)
    res["gradient_max_abs"] = gradient_probe(model, x_batch, y_batch)
    res["execution_modes_train_batch"] = execution_modes(model, x_batch)
    init = ARTIFACTS_DIR / name / "init.weights.h5"
    if init.exists():
        res["weight_change_max_abs"] = weight_change(model, init)
    res["summary_test_acc"] = summary["test_acc"]
    res["weights"] = str(weights.relative_to(ARTIFACTS_DIR.parents[1]))
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", nargs="*", default=["T0_official", "T1_page_version"], help="trained runs to probe if present")
    p.add_argument("--n-test", type=int, default=None, help="probe only the first N test series (default: all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "probe.json")
    args = p.parse_args()

    configure_cpu(args.threads)
    tf.get_logger().setLevel("ERROR")  # per-model tf.function retracing warnings are expected here
    keras.utils.set_random_seed(args.seed)
    x_train, y_train, x_test, y_test, _ = load_forda()
    x, y = x_test[: args.n_test], y_test[: args.n_test]
    x_batch, y_batch = x_train[:64].astype(np.float32), y_train[:64]

    report = {"n_test_probed": int(len(x)), "seed": args.seed}
    report["a_architecture"] = {
        f"{pooling}_{blocks}_blocks": {"params": int(m.count_params()), "pooled_feature_shape": pooled_shape(m)}
        for pooling, blocks in [("channels_last", 4), ("channels_first", 4), ("channels_first", 0)]
        for m in [build_official(x.shape[1:], 2, pooling=pooling, blocks=blocks)]
    }
    report["a_architecture"]["expected"] = {"channels_last_4_blocks": 29258, "channels_first_4_blocks": 93130}
    print(json.dumps(report["a_architecture"], indent=1))

    rng = np.random.default_rng(args.seed)
    model = build_official(x.shape[1:], 2, pooling="channels_last", blocks=4)
    randomize_layernorms(model, rng)
    rand, (xs, enc, C) = probe_model(model, x, y, args.batch_size)
    rand["execution_modes_train_batch"] = execution_modes(model, x_batch)
    rand["gradient_max_abs"] = gradient_probe(model, x_batch, y_batch)
    report["b_random_init"] = rand
    print(json.dumps({k: rand[k] for k in ("encoder", "pooled", "gradient_max_abs")}, indent=1))

    report["b_trained"] = {}
    for name in args.runs:
        print(f"probing trained run {name}")
        report["b_trained"][name] = trained_run(name, x, y, x_batch, y_batch, args.batch_size)

    report["d_forda_series_stats"] = {"train": series_stats(x_train), "test": series_stats(x_test)}
    report["notes"] = [
        "LayerNormalization(axis=-1) over a size-1 feature axis: mean = x and var = 0, so the normalized "
        "value is 0 and the output is beta. Keras evaluates x*inv + (beta - mean*inv) with "
        "inv = gamma/sqrt(var + 1e-6); in graph execution (what fit/predict run, measured via "
        "tf.function) the output equals beta bitwise; in eager execution the two equal products of size "
        "|gamma*x|*1000 cancel only up to float32 rounding, so deviations up to "
        "'layernorm_rounding_bound' (one ulp of that product plus one ulp of beta) are rounding, not signal. "
        "See 'execution_modes_train_batch'.",
        "Hence each encoder block computes x + beta_attn + beta_ff and the stack computes x + C with "
        "C = sum of the LayerNorm betas. 'residual_std_over_time_max' is the largest per-series std over "
        "time of (output - input), i.e. float32 rounding of adding the betas to x (ulp ~1e-7 at |x| ~ 1); "
        "compare it with 'input_std_over_time_mean' (~1). Dropout (training mode) sits before the "
        "LayerNorms, so it does not change this.",
        "The derivative of a size-1 LayerNorm w.r.t. its input is analytically 0, so attention, "
        "feed-forward Conv1D and LayerNorm gamma receive zero gradient ('gradient_max_abs', eager) and "
        "only the betas and the MLP head learn. 'weight_change_max_abs' compares saved initial and final "
        "weights of a trained run: 0 for a group means fit() never moved it.",
        "channels_last pooling averages over time: pooled = C + mean_t(x). FordA series are z-normalized "
        "per series (d_forda_series_stats: |mean_t(x)| <~ 1e-8, std = 1 with ddof=1), so in exact arithmetic "
        "the pooled scalar is C for every series up to ~1e-8, versus ~1 per feature under channels_first. "
        "In float32 that residue is below one ulp of C ('channels_last_ulp_at_value', ~6e-8 for |C| ~ 0.5-1), "
        "so the observed across-sample spread ('channels_last_range_in_ulps', a few ulps over "
        "'channels_last_unique_float32_values' distinct values) is rounding in the 500-step float32 sum, "
        "not the series mean. 'channels_last_label_auc' is computed on those rounding ties and is not a "
        "measure of information content; 'series_mean_float64_label_auc' is the AUC of the exact float64 "
        "per-series mean, the quantity the pooled scalar would carry in exact arithmetic. The head would "
        "need weights ~1e8 to use a 1e-8 residue even then.",
        "channels_first pooling averages over the size-1 feature axis: pooled = x + C (all 500 steps), "
        "so the model is the MLP head applied to the raw signal plus a constant; "
        "'mlp_on_x_plus_C_*' compares the full model with exactly that computation (differences are "
        "float32 rounding of x + C).",
        "Random-init probe: default initializers except LayerNorm beta ~ N(0,1) and gamma ~ N(1,0.5), "
        "so 'output == beta' is not trivially 'output == 0'.",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    fig = args.out.with_name(args.out.stem + "_encoder_identity.png")
    plot_identity(xs, enc, C, "Official encoder (random weights, random LayerNorm betas): output = input + constant", fig)
    print(f"wrote {args.out} and {fig}")


if __name__ == "__main__":
    main()
