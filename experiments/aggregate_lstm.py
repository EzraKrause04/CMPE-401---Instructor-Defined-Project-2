"""Aggregate LSTM benchmark runs into a table and figures.

Reads results/lstm_benchmark/<variant>/seed*.json, history_seed0.csv and naive_baselines.json;
writes summary.csv, summary.md, test_mae.png and val_loss_curves.png next to them. Variants
without finished seeds are skipped, so this can be re-run while the queue is still going.

Usage:
    python experiments/aggregate_lstm.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "lstm_benchmark"
BASELINE = "L0_baseline"
NAIVE = ["persistence", "seasonal_24h", "ridge"]
METRICS = ["test_mae_C", "test_rmse_C", "test_mse", "val_mae_C", "val_rmse_C", "val_mse"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BLUE, REF = "#2a78d6", "#52514e"


def order_key(name: str) -> tuple:
    return (0 if name == BASELINE else 1 if name.startswith("V") else 2, name)


def load_runs() -> pd.DataFrame:
    rows = []
    for f in sorted(RESULTS.glob("*/seed*.json")):
        r = json.loads(f.read_text())
        rows.append({k: r.get(k) for k in ("variant", "change", "seed", "params", "epochs_run", "best_epoch",
                                            "train_time_s", "seconds_per_epoch", "smoke", *METRICS)})
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    # once a variant has a full run, its leftover smoke runs are dropped so they cannot skew the mean
    runs = runs.assign(smoke=runs["smoke"].eq(True))
    runs = runs[~(runs["smoke"] & ~runs.groupby("variant")["smoke"].transform("all"))]
    g = runs.groupby("variant")
    s = pd.DataFrame({
        "change": g["change"].first(),
        "n_seeds": g["seed"].count(),
        "seeds": g["seed"].apply(lambda v: " ".join(map(str, sorted(v)))),
        "params": g["params"].first(),
        "epochs_mean": g["epochs_run"].mean(),
        "best_epoch_mean": g["best_epoch"].mean(),
        "train_time_s_mean": g["train_time_s"].mean(),
        "seconds_per_epoch_mean": g["seconds_per_epoch"].mean(),
        "smoke": g["smoke"].any(),
    })
    for m in METRICS:
        s[f"{m}_mean"] = g[m].mean()
        s[f"{m}_std"] = g[m].std(ddof=1)  # NaN with a single seed
    if BASELINE in s.index:
        for split in ("val", "test"):
            ref = s.loc[BASELINE, f"{split}_mae_C_mean"]
            s[f"delta_{split}_mae_C"] = s[f"{split}_mae_C_mean"] - ref
            s[f"delta_{split}_mae_pct"] = 100 * s[f"delta_{split}_mae_C"] / ref
    s = s.loc[sorted(s.index, key=order_key)]
    s.insert(0, "kind", "lstm")
    return s


def load_naive() -> tuple[pd.DataFrame, dict]:
    f = RESULTS / "naive_baselines.json"
    if not f.exists():
        return pd.DataFrame(), {}
    j = json.loads(f.read_text())
    b = j["baselines"]
    rows = {
        n: {"kind": "naive", "change": b[n]["description"], **{f"{m}_mean": b[n][m] for m in METRICS}}
        for n in NAIVE
        if n in b
    }
    return pd.DataFrame.from_dict(rows, orient="index"), j["protocol"]


def pm(mean: float, std: float) -> str:
    return f"{mean:.3f}" if np.isnan(std) else f"{mean:.3f} ± {std:.3f}"


def delta_str(d: float, ref: float) -> str:
    return f"{d:+.3f} °C ({100 * d / ref:+.1f} %)"


def markdown(s: pd.DataFrame, naive: pd.DataFrame, protocol: dict) -> str:
    has_ref = BASELINE in s.index
    ref = {sp: s.loc[BASELINE, f"{sp}_mae_C_mean"] for sp in ("val", "test")} if has_ref else {}
    head = ("| Variant | Change | Params | Seeds | Epochs (mean) | Test MAE °C (mean ± std) | Test RMSE °C | "
            "Val MAE °C | Δ val MAE vs L0 | Δ test MAE vs L0 | Train time (min) |")
    lines = [head, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in s.iterrows():
        dv, dt = (("—", "—") if not has_ref or name == BASELINE else
                  (delta_str(r.delta_val_mae_C, ref["val"]), delta_str(r.delta_test_mae_C, ref["test"])))
        lines.append(
            f"| {name} | {r.change} | {int(r.params):,} | {r.n_seeds} | {r.epochs_mean:.1f} | "
            f"{pm(r.test_mae_C_mean, r.test_mae_C_std)} | {pm(r.test_rmse_C_mean, r.test_rmse_C_std)} | "
            f"{pm(r.val_mae_C_mean, r.val_mae_C_std)} | {dv} | {dt} | {r.train_time_s_mean / 60:.1f} |"
        )
    for name, r in naive.iterrows():
        dv, dt = ("—", "—") if not has_ref else (
            delta_str(r.val_mae_C_mean - ref["val"], ref["val"]),
            delta_str(r.test_mae_C_mean - ref["test"], ref["test"]),
        )
        lines.append(
            f"| *{name}* | {r.change} | — | — | — | {r.test_mae_C_mean:.3f} | {r.test_rmse_C_mean:.3f} | "
            f"{r.val_mae_C_mean:.3f} | {dv} | {dt} | — |"
        )
    notes = [
        "",
        "Test = held-out last 50 % of the official validation rows; each model is the best-val epoch "
        "(EarlyStopping restore_best_weights). ± is the sample std over seeds; Δ compares mean MAE with L0_baseline. "
        "Naive baselines (italic) are deterministic and scored on the same label rows.",
        "",
        "**Selection rule:** which variants count as improvements (and go into a combined model) is decided on "
        "**Δ val** only; test is reported for the final numbers and never used for choices.",
    ]
    if sw := protocol.get("wv_sentinel_windows"):
        nw = protocol["n_windows"]
        notes += ["", f"Val caveat: {sw['val']:,} of {nw['val']:,} val windows ({sw['test']:,} of {nw['test']:,} test) "
                      "contain a wind-speed -9999 sentinel in their inputs. LSTM runs with clean_wv=False (all but "
                      "V6_clean_wv, unless a combined model enables it) see them as-is, while ridge and clean_wv runs "
                      "see them zeroed, so val MAE is not strictly like-for-like across those rows; test is unaffected."]
    if s["smoke"].any():
        notes.append(f"\n**Smoke-test runs included** (shortened training): {', '.join(s.index[s['smoke']])}.")
    return "\n".join(lines + notes) + "\n"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_test_mae(s: pd.DataFrame, naive: pd.DataFrame, path: Path):
    """Absolute test MAE (zero-based bars, naive reference lines) and, if L0 exists, a Δ-vs-L0 panel
    where differences of a few hundredths of a degree are actually visible."""
    has_ref = BASELINE in s.index
    rows = 2 if has_ref else 1
    x = np.arange(len(s))
    colors = [REF if n == BASELINE else BLUE for n in s.index]
    err = s["test_mae_C_std"].fillna(0).to_numpy()
    bar = {"width": 0.7, "color": colors, "yerr": err, "capsize": 3, "error_kw": {"ecolor": INK, "elinewidth": 1}}
    fig, axes = plt.subplots(rows, 1, figsize=(max(6, 0.75 * len(s) + 3), 3.4 * rows + 0.8), dpi=150,
                             sharex=True, squeeze=False)
    ax = axes[0, 0]
    ax.bar(x, s["test_mae_C_mean"], **bar)
    for (name, r), ls in zip(naive.iterrows(), ["--", ":", "-."]):
        ax.axhline(r.test_mae_C_mean, color=MUTED, linestyle=ls, linewidth=1.2)
        # labels sit just above their line in a right-hand gutter, clear of the bars
        ax.text(len(s) - 0.35, r.test_mae_C_mean, f" {name} {r.test_mae_C_mean:.2f}", ha="left", va="bottom",
                fontsize=7, color=REF)
    ax.set_xlim(-0.6, len(s) + (1.2 if len(naive) else -0.4))
    ax.set_ylabel("Test MAE (°C)")
    ax.set_title("LSTM variants: test MAE, mean ± std over seeds (gray = L0 baseline)", fontsize=10, color=INK)
    style(ax)
    if has_ref:
        ax = axes[1, 0]
        d = s["delta_test_mae_C"].to_numpy()
        ax.bar(x, d, **bar)
        ax.axhline(0, color=MUTED, linewidth=1)
        for xi, (v, e) in enumerate(zip(d, err)):
            if s.index[xi] != BASELINE:
                up = v >= 0
                ax.annotate(f"{v:+.3f}", (xi, v + e if up else v - e), xytext=(0, 2 if up else -2),
                            textcoords="offset points", ha="center", va="bottom" if up else "top", fontsize=7, color=INK)
        ax.margins(y=0.15)
        ax.set_ylabel("Δ test MAE vs L0 (°C)\n(< 0 = better)")
        style(ax)
    axes[-1, 0].set_xticks(x, s.index, rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_val_curves(s: pd.DataFrame, path: Path):
    hist = {n: pd.read_csv(f) for n in s.index if (f := RESULTS / n / "history_seed0.csv").exists()}
    others = [n for n in hist if n != BASELINE]
    if not hist:
        return
    names = others or [BASELINE]
    cols = min(4, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), dpi=150, sharey=True, squeeze=False)
    for ax, n in zip(axes.flat, names):
        if BASELINE in hist and n != BASELINE:
            h0 = hist[BASELINE]
            ax.plot(h0["epoch"], h0["val_loss"], color=MUTED, linewidth=1.5, marker="s", markersize=3, label=BASELINE)
        h = hist[n]
        ax.plot(h["epoch"], h["val_loss"], color=BLUE, linewidth=2, marker="o", markersize=3, label=n)
        ax.set_title(n, fontsize=9, color=INK)
        n_epochs = max(len(h), len(hist.get(BASELINE, h)))
        ax.set_xlim(0.5, n_epochs + 0.5)
        ax.xaxis.set_major_locator(MultipleLocator(max(1, n_epochs // 6)))
        style(ax)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Val MSE (normalized)", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch", fontsize=8)
    handles = [
        plt.Line2D([], [], color=MUTED, linewidth=1.5, marker="s", markersize=3),
        plt.Line2D([], [], color=BLUE, linewidth=2, marker="o", markersize=3),
    ]
    fig.suptitle("Validation loss per epoch, seed 0 (each variant vs L0_baseline)", fontsize=10, color=INK)
    fig.legend(
        handles, [BASELINE, "variant (panel title)"],
        loc="upper center", bbox_to_anchor=(0.5, 0.97 - 0.01 * rows), ncol=2, fontsize=8, frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94 - 0.01 * rows))
    fig.savefig(path)
    plt.close(fig)


def main():
    runs = load_runs()
    if runs.empty:
        raise SystemExit(f"no seed*.json under {RESULTS}")
    s = summarize(runs)
    naive, protocol = load_naive()
    table = pd.concat([s, naive]).rename_axis("variant")
    table[["n_seeds", "params"]] = table[["n_seeds", "params"]].astype("Int64")  # concat with naive rows made them float
    table.to_csv(RESULTS / "summary.csv", float_format="%.5f")
    md = markdown(s, naive, protocol)
    (RESULTS / "summary.md").write_text(md)
    plot_test_mae(s, naive, RESULTS / "test_mae.png")
    plot_val_curves(s, RESULTS / "val_loss_curves.png")
    print(md)


if __name__ == "__main__":
    main()
