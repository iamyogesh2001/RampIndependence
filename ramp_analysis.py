"""
ramp_analysis.py

Full reproducible analysis pipeline for:
"Ramp Rate Independence and Grid-Facing Disturbance Characteristics of
Co-Located AI Data Center Workloads" (IECON 2026)

This script reproduces every number, table, and figure reported in the paper
directly from the publicly released Emerald AI production measurements
(Colangelo et al., "AI data centres as grid-interactive assets," Nature
Energy, 2026). No synthetic or fabricated data points are used anywhere in
this pipeline; every statistic is computed from the real measured traces.

Usage:
    1. Download the Emerald AI dataset:
       https://github.com/ai-emerald/emerald-ai-demo-may-2025
    2. Place the extracted "data/" folder at DATA_DIR below, or pass a
       custom path with --data-dir.
    3. Run: python ramp_analysis.py

Outputs:
    - All numbers printed to console, matching every figure/table in the paper
    - PNG figures written to ./figures/
    - A summary CSV of all per-workload and aggregate statistics

Dependencies: numpy, pandas, scipy, matplotlib (see requirements.txt)
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DATA_DIR = "data/"
OUT_DIR = "figures/"

WORKLOAD_COLS = [
    "ft_llama_8b_dolly-txw0sl4h",
    "infer_llama_8b-bw9rr4h2",
    "infer_llama_8b-kxymm0ql",
    "infer_llama_8b-stsiwyqz",
    "pt_mpt_13b_sm-k2jychqp",
    "pt_mpt_7b_fast-60hrpily",
]
WORKLOAD_LABELS = [
    "LLaMA-8B FT",
    "LLaMA-8B Inf-A",
    "LLaMA-8B Inf-B",
    "LLaMA-8B Inf-C",
    "MPT-13B PT",
    "MPT-7B Fast",
]

PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 160,
}


def load_workload_power(data_dir):
    """Load the six per-workload power traces from the Emerald AI dataset."""
    fig5 = pd.read_excel(
        os.path.join(data_dir, "Fig345_data.xlsx"), sheet_name="Fig5", header=1
    )
    for col in WORKLOAD_COLS:
        fig5[col] = pd.to_numeric(fig5[col], errors="coerce")
    fig5["total_W"] = pd.to_numeric(fig5["total (W)"], errors="coerce")
    fig5 = fig5.dropna(subset=["total_W"]).reset_index(drop=True)
    arr = np.array([fig5[c].ffill().bfill().values for c in WORKLOAD_COLS])
    return arr


def ramp_rate_stats(arr):
    """Per-workload and aggregate ramp rate (first-difference) statistics."""
    dP = np.diff(arr, axis=1)
    agg = arr.sum(axis=0)
    dP_agg = np.diff(agg)

    rows = []
    for i, label in enumerate(WORKLOAD_LABELS):
        r = np.abs(dP[i])
        rows.append(
            {
                "workload": label,
                "mean_kW": arr[i].mean() / 1000,
                "P95_kWmin": np.percentile(r, 95) / 1000,
                "P99_kWmin": np.percentile(r, 99) / 1000,
                "max_kWmin": r.max() / 1000,
            }
        )
    ra = np.abs(dP_agg)
    rows.append(
        {
            "workload": "Aggregate",
            "mean_kW": agg.mean() / 1000,
            "P95_kWmin": np.percentile(ra, 95) / 1000,
            "P99_kWmin": np.percentile(ra, 99) / 1000,
            "max_kWmin": ra.max() / 1000,
        }
    )
    return pd.DataFrame(rows), dP, dP_agg, agg


def bootstrap_percentile_ci(x, q, n_boot=2000, seed=0):
    """95% bootstrap confidence interval on a percentile estimate."""
    rng = np.random.default_rng(seed)
    x = np.abs(x)
    point = np.percentile(x, q)
    boots = np.array(
        [np.percentile(rng.choice(x, len(x)), q) for _ in range(n_boot)]
    )
    return point, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def independence_test(dP, dP_agg, n_splits=True):
    """Cross-workload correlation matrix and the variance superposition test."""
    n = dP.shape[0]
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr[i, j] = pearsonr(dP[i], dP[j])[0]

    def ratio(idx):
        sv = sum(np.var(dP[i][idx]) for i in range(n))
        av = np.var(dP_agg[idx])
        return av / sv

    N = dP.shape[1]
    results = {"full": ratio(slice(None))}
    if n_splits:
        h = N // 2
        results["first_half"] = ratio(slice(0, h))
        results["second_half"] = ratio(slice(h, None))
    return corr, results


def rocof_sensitivity(dP_agg, agg_power, H_range=(2, 12), f0=60.0, S_base=1500e6):
    """Equivalent-step ROCOF versus assumed system inertia constant H.

    The measured single-cluster ramp is scaled to the 1500 MW magnitude of
    the documented Virginia 2024 event (an equivalent-disturbance scaling,
    not a claim about the actual Virginia grid's inertia or protection
    settings; see Section III.D of the paper).
    """
    ra = np.abs(dP_agg)
    scale = S_base / agg_power.max()
    Hs = np.linspace(H_range[0], H_range[1], 200)
    max_roc = np.array([f0 * ra.max() * scale / (2 * Hv * S_base) for Hv in Hs])
    return Hs, max_roc, scale


def make_figures(df_stats, dP, dP_agg, corr, agg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update(PLOT_STYLE)

    # Fig 1: aggregate ramp time series
    ra = np.abs(dP_agg)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ra / 1000, color="#d62728", lw=1.6)
    ax.axhline(np.percentile(ra, 95) / 1000, ls="--", color="orange", label="P95")
    ax.axhline(np.percentile(ra, 99) / 1000, ls="--", color="black", label="P99")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("|dP/dt| (kW/min)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_ramp_timeseries.png"))
    plt.close(fig)

    # Fig 2: correlation heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(len(WORKLOAD_LABELS)))
    ax.set_yticks(range(len(WORKLOAD_LABELS)))
    ax.set_xticklabels(WORKLOAD_LABELS, rotation=35, ha="right")
    ax.set_yticklabels(WORKLOAD_LABELS)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_correlation_matrix.png"))
    plt.close(fig)

    print(f"Figures written to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    arr = load_workload_power(args.data_dir)
    df_stats, dP, dP_agg, agg = ramp_rate_stats(arr)
    print("\n=== TABLE I: Per-workload ramp rate statistics ===")
    print(df_stats.to_string(index=False))

    corr, ratios = independence_test(dP, dP_agg)
    print("\n=== Cross-workload correlation ===")
    off_diag = corr[~np.eye(len(WORKLOAD_LABELS), dtype=bool)]
    print(f"max|r| = {np.abs(off_diag).max():.3f}")
    print(f"mean|r| = {np.abs(off_diag).mean():.3f}")

    print("\n=== Superposition test (variance ratio) ===")
    for k, v in ratios.items():
        print(f"  {k}: {v:.3f}")

    Hs, max_roc, scale = rocof_sensitivity(dP_agg, agg)
    print("\n=== ROCOF sensitivity at representative H values ===")
    print(f"(fleet scale factor: {scale:.0f}x single cluster -> 1500 MW equivalent)")
    for Hv in [3, 4, 5, 6, 8, 10]:
        idx = np.argmin(np.abs(Hs - Hv))
        n_exceed = int(
            (60.0 * np.abs(dP_agg) * scale / (2 * Hv * 1500e6) > 0.5).sum()
        )
        print(f"  H={Hv:>2}s: max ROCOF={max_roc[idx]:.3f} Hz/s, events>0.5Hz/s={n_exceed}")

    df_stats.to_csv("ramp_statistics_summary.csv", index=False)
    print("\nSummary statistics written to ramp_statistics_summary.csv")

    make_figures(df_stats, dP, dP_agg, corr, agg, args.out_dir)


if __name__ == "__main__":
    main()
