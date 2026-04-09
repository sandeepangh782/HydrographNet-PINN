"""
plot_results.py

Generates publication-quality figures from experiment CSVs produced by
run_experiments.py (or from W&B exports).

Figures produced (saved under ./figures/):
  Fig 1  — Loss curves: total / MSE / physics per experiment
  Fig 2  — Physics-to-MSE gradient norm ratio ρ over epochs
  Fig 3  — λ_phy schedule comparison across experiments
  Fig 4  — RMSE (depth) convergence comparison
  Fig 5  — R² convergence comparison
  Fig 6  — Summary bar chart (final epoch metrics)

Usage:
    python plot_results.py                      # reads CSVs from outputs_phy/
    python plot_results.py --wandb              # reads from W&B API instead
    python plot_results.py --demo               # generate with synthetic data
                                                # (use this when CSVs don't exist yet)
"""

import argparse
import csv
import math
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

matplotlib.rcParams.update({
    "font.family":       "Times New Roman",
    "font.size":         13,
    "axes.labelsize":    14,
    "axes.titlesize":    15,
    "legend.fontsize":   12,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

OUT_DIR = "./figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Colour palette (colour-blind friendly)
COLORS = {
    "fixed":         "#E63946",   # red
    "linear_warmup": "#2A9D8F",   # teal
    "gnb":           "#F4A261",   # amber
}
LABELS = {
    "fixed":         "EXP-1: Baseline (fixed λ=1.0)",
    "linear_warmup": "EXP-2: Linear Warm-up",
    "gnb":           "EXP-3: GNB (adaptive)",
}
LINESTYLES = {
    "fixed":         "solid",
    "linear_warmup": "dashed",
    "gnb":           "dotted",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: str) -> dict[str, list]:
    """Load a metrics CSV into {column: [values...]}."""
    data: dict[str, list] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, [])
                try:
                    data[k].append(float(v) if v != "" else float("nan"))
                except ValueError:
                    data[k].append(float("nan"))
    return data


def load_all_csvs(exp_names: list[str]) -> dict[str, dict]:
    """Return {exp_name: data_dict} for found CSVs."""
    results = {}
    for name in exp_names:
        path = f"./outputs_phy/{name}/metrics.csv"
        if os.path.exists(path):
            results[name] = load_csv(path)
            print(f"  Loaded {path}  ({len(results[name].get('epoch', []))} epochs)")
        else:
            print(f"  [skip] {path} not found")
    return results


# ---------------------------------------------------------------------------
# Synthetic demo data (used when no real CSVs exist yet)
# ---------------------------------------------------------------------------

def _decay(start, end, n, noise=0.05):
    t = np.linspace(0, 1, n)
    curve = start * np.exp(-3 * t) + end * (1 - np.exp(-3 * t))
    curve += np.random.normal(0, noise * abs(start - end), n)
    return np.clip(curve, min(start, end) * 0.5, max(start, end) * 1.5)


def generate_demo_data(epochs: int = 15) -> dict[str, dict]:
    """Produce plausible synthetic training curves for figure demonstrations."""
    random.seed(42)
    np.random.seed(42)
    ep = list(range(epochs))

    def make_exp(lam_schedule, rho_start, rho_end, loss_end, rmse_end, r2_end):
        n = epochs
        loss_total = _decay(0.8, loss_end, n, 0.04)
        loss_mse   = _decay(0.7, loss_end * 0.75, n, 0.03)
        loss_phy   = _decay(0.4, loss_end * 0.25, n, 0.02)
        rho        = _decay(rho_start, rho_end, n, 0.15)
        rmse_d     = _decay(0.28, rmse_end, n, 0.01)
        rmse_v     = _decay(0.32, rmse_end + 0.02, n, 0.01)
        r2         = _decay(0.4, r2_end, n, 0.02)
        r2         = np.clip(r2, 0, 1)
        return {
            "epoch":       [float(e) for e in ep],
            "loss_total":  list(loss_total),
            "loss_mse":    list(loss_mse),
            "loss_physics":list(loss_phy),
            "lambda_phy":  [float(x) for x in lam_schedule],
            "grad_norm_mse": list(_decay(0.05, 0.02, n, 0.005)),
            "grad_norm_phy": list(_decay(0.3, 0.02 * rho_end, n, 0.01)),
            "rho":         list(rho),
            "rmse_depth":  list(rmse_d),
            "rmse_volume": list(rmse_v),
            "r2_depth":    list(r2),
        }

    # Fixed λ = 1.0 throughout
    lam_fixed = [1.0] * epochs

    # Linear warm-up 0.01 → 1.0 over 20 epochs (capped at available epochs)
    warmup = min(epochs, 20)
    lam_warmup = [0.01 + (1.0 - 0.01) * min(1.0, e / warmup)
                  for e in range(epochs)]

    # GNB: starts high, rapidly converges to ~1
    lam_gnb = [0.1 * (1 + 0.4) ** min(e, 8) for e in range(epochs)]
    lam_gnb = [min(x, 1.0) for x in lam_gnb]

    return {
        "fixed":         make_exp(lam_fixed,  rho_start=8.5, rho_end=4.2,
                                  loss_end=0.12, rmse_end=0.185, r2_end=0.72),
        "linear_warmup": make_exp(lam_warmup, rho_start=0.8, rho_end=1.3,
                                  loss_end=0.09, rmse_end=0.155, r2_end=0.79),
        "gnb":           make_exp(lam_gnb,    rho_start=1.1, rho_end=1.02,
                                  loss_end=0.08, rmse_end=0.142, r2_end=0.83),
    }


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _ax_style(ax, xlabel, ylabel, title, legend=True):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    if legend:
        ax.legend()


def savefig(fig, name: str):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    print(f"  Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — Loss curves (3×1 per experiment)
# ---------------------------------------------------------------------------

def fig_loss_curves(data: dict[str, dict]):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
    for ax, (name, d) in zip(axes, data.items()):
        ep = d["epoch"]
        ax.plot(ep, d["loss_total"],   label="Total loss",   color="#264653", lw=2)
        ax.plot(ep, d["loss_mse"],     label="MSE loss",     color="#2A9D8F", lw=2, ls="--")
        ax.plot(ep, d["loss_physics"], label="Physics loss",  color="#E63946", lw=2, ls=":")
        _ax_style(ax, "Epoch", "Loss", LABELS[name])
    fig.suptitle("Figure 1 — Training Loss Curves", fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig1_loss_curves.pdf")
    savefig(fig, "fig1_loss_curves.png")


# ---------------------------------------------------------------------------
# Figure 2 — Physics-to-MSE gradient ratio ρ
# ---------------------------------------------------------------------------

def fig_rho(data: dict[str, dict]):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(1.0, color="black", lw=1.2, ls="--", label="ρ = 1 (balanced)")
    for name, d in data.items():
        rho = d["rho"]
        rho_clean = [r if not math.isnan(r) else None for r in rho]
        ax.plot(d["epoch"], rho_clean,
                color=COLORS[name], ls=LINESTYLES[name],
                lw=2.2, label=LABELS[name])
    _ax_style(ax,
              xlabel="Epoch",
              ylabel=r"$\rho = \|\nabla \mathcal{L}_{phy}\| \;/\; \|\nabla \mathcal{L}_{MSE}\|$",
              title=r"Figure 2 — Physics-to-MSE Gradient Norm Ratio $\rho$")
    ax.set_ylim(bottom=0)
    savefig(fig, "fig2_gradient_ratio_rho.pdf")
    savefig(fig, "fig2_gradient_ratio_rho.png")


# ---------------------------------------------------------------------------
# Figure 3 — λ_phy schedule comparison
# ---------------------------------------------------------------------------

def fig_lambda(data: dict[str, dict]):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, d in data.items():
        ax.plot(d["epoch"], d["lambda_phy"],
                color=COLORS[name], ls=LINESTYLES[name],
                lw=2.2, label=LABELS[name])
    _ax_style(ax,
              xlabel="Epoch",
              ylabel=r"Physics loss weight $\lambda_{phy}$",
              title=r"Figure 3 — $\lambda_{phy}$ Scheduling Comparison")
    savefig(fig, "fig3_lambda_schedule.pdf")
    savefig(fig, "fig3_lambda_schedule.png")


# ---------------------------------------------------------------------------
# Figure 4 — RMSE depth convergence
# ---------------------------------------------------------------------------

def fig_rmse(data: dict[str, dict]):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, d in data.items():
        ax.plot(d["epoch"], d["rmse_depth"],
                color=COLORS[name], ls=LINESTYLES[name],
                lw=2.2, label=LABELS[name])
    _ax_style(ax,
              xlabel="Epoch",
              ylabel="RMSE — Water Depth (normalised)",
              title="Figure 4 — Water Depth RMSE Convergence")
    savefig(fig, "fig4_rmse_convergence.pdf")
    savefig(fig, "fig4_rmse_convergence.png")


# ---------------------------------------------------------------------------
# Figure 5 — R² convergence
# ---------------------------------------------------------------------------

def fig_r2(data: dict[str, dict]):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, d in data.items():
        ax.plot(d["epoch"], d["r2_depth"],
                color=COLORS[name], ls=LINESTYLES[name],
                lw=2.2, label=LABELS[name])
    _ax_style(ax,
              xlabel="Epoch",
              ylabel=r"$R^2$ — Water Depth",
              title=r"Figure 5 — $R^2$ Coefficient of Determination")
    ax.set_ylim(0, 1)
    savefig(fig, "fig5_r2_convergence.pdf")
    savefig(fig, "fig5_r2_convergence.png")


# ---------------------------------------------------------------------------
# Figure 6 — Summary bar chart (final epoch values)
# ---------------------------------------------------------------------------

def fig_summary_bars(data: dict[str, dict]):
    metrics = [
        ("rmse_depth",   "RMSE Depth↓",        "lower is better"),
        ("r2_depth",     r"$R^2$ Depth↑",       "higher is better"),
        ("rho",          r"Final $\rho$ (→1)",  "closer to 1 is better"),
        ("lambda_phy",   r"Final $\lambda_{phy}$", ""),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5))
    names = list(data.keys())
    x = np.arange(len(names))
    bar_w = 0.5

    for ax, (col, ylabel, note) in zip(axes, metrics):
        vals = []
        for name in names:
            col_data = [v for v in data[name].get(col, [float("nan")])
                        if not math.isnan(v)]
            vals.append(col_data[-1] if col_data else 0.0)

        bars = ax.bar(x, vals, width=bar_w,
                      color=[COLORS[n] for n in names],
                      edgecolor="black", linewidth=0.8)

        # Value labels on top of bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels(["Fixed", "Warm-up", "GNB"], fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}\n({note})", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Figure 6 — Final Epoch Metric Comparison", fontweight="bold")
    fig.tight_layout()
    savefig(fig, "fig6_summary_bars.pdf")
    savefig(fig, "fig6_summary_bars.png")


# ---------------------------------------------------------------------------
# Figure 7 — Combined 2×3 overview panel for paper inclusion
# ---------------------------------------------------------------------------

def fig_paper_panel(data: dict[str, dict]):
    """Single combined figure suitable for direct paper inclusion."""
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_loss   = fig.add_subplot(gs[0, 0])
    ax_rho    = fig.add_subplot(gs[0, 1])
    ax_lam    = fig.add_subplot(gs[0, 2])
    ax_rmse   = fig.add_subplot(gs[1, 0])
    ax_r2     = fig.add_subplot(gs[1, 1])
    ax_bar    = fig.add_subplot(gs[1, 2])

    # --- (a) Total loss ---
    for name, d in data.items():
        ax_loss.plot(d["epoch"], d["loss_total"],
                     color=COLORS[name], ls=LINESTYLES[name], lw=2,
                     label=LABELS[name])
    _ax_style(ax_loss, "Epoch", "Total Loss", "(a) Total Training Loss")

    # --- (b) ρ ---
    ax_rho.axhline(1.0, color="black", lw=1.2, ls="--", label="ρ = 1")
    for name, d in data.items():
        ax_rho.plot(d["epoch"], d["rho"],
                    color=COLORS[name], ls=LINESTYLES[name], lw=2,
                    label=LABELS[name])
    _ax_style(ax_rho, "Epoch",
              r"$\rho = \|\nabla\mathcal{L}_{phy}\|/\|\nabla\mathcal{L}_{MSE}\|$",
              r"(b) Gradient Norm Ratio $\rho$")
    ax_rho.set_ylim(bottom=0)

    # --- (c) λ schedule ---
    for name, d in data.items():
        ax_lam.plot(d["epoch"], d["lambda_phy"],
                    color=COLORS[name], ls=LINESTYLES[name], lw=2,
                    label=LABELS[name])
    _ax_style(ax_lam, "Epoch", r"$\lambda_{phy}$",
              r"(c) Physics Loss Weight $\lambda_{phy}$")

    # --- (d) RMSE ---
    for name, d in data.items():
        ax_rmse.plot(d["epoch"], d["rmse_depth"],
                     color=COLORS[name], ls=LINESTYLES[name], lw=2,
                     label=LABELS[name])
    _ax_style(ax_rmse, "Epoch", "RMSE (Water Depth)", "(d) Depth RMSE")

    # --- (e) R² ---
    for name, d in data.items():
        ax_r2.plot(d["epoch"], d["r2_depth"],
                   color=COLORS[name], ls=LINESTYLES[name], lw=2,
                   label=LABELS[name])
    _ax_style(ax_r2, "Epoch", r"$R^2$", r"(e) $R^2$ — Water Depth")
    ax_r2.set_ylim(0, 1)

    # --- (f) Final bar comparison ---
    names  = list(data.keys())
    x      = np.arange(len(names))
    bar_w  = 0.4
    rmse_f = [data[n]["rmse_depth"][-1] for n in names]
    r2_f   = [data[n]["r2_depth"][-1]   for n in names]
    b1 = ax_bar.bar(x - bar_w / 2, rmse_f, bar_w,
                    color=[COLORS[n] for n in names], alpha=0.8,
                    edgecolor="k", label="RMSE")
    ax_bar2 = ax_bar.twinx()
    b2 = ax_bar2.bar(x + bar_w / 2, r2_f, bar_w,
                     color=[COLORS[n] for n in names], alpha=0.4,
                     edgecolor="k", hatch="///", label=r"$R^2$")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["Fixed", "Warm-up", "GNB"])
    ax_bar.set_ylabel("Final RMSE ↓", color="black")
    ax_bar2.set_ylabel(r"Final $R^2$ ↑", color="gray")
    ax_bar.set_title("(f) Final Epoch Comparison")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Adaptive Physics-Constrained Loss Scheduling in HydroGraphNet\n"
        "EXP-1: Fixed λ=1.0  |  EXP-2: Linear Warm-up  |  EXP-3: GNB",
        fontsize=16, fontweight="bold"
    )
    savefig(fig, "fig7_paper_panel.pdf")
    savefig(fig, "fig7_paper_panel.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figures for HydroGraphNet experiments"
    )
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic demo data (no real training needed)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of demo epochs to generate (default 15)")
    parser.add_argument("--wandb", action="store_true",
                        help="Pull data from W&B API (requires wandb login)")
    args = parser.parse_args()

    # ---- Load data ----
    if args.demo:
        print("[plot_results] Using synthetic demo data.")
        data = generate_demo_data(epochs=args.epochs)

    elif args.wandb:
        # Minimal W&B pull — requires `pip install wandb`
        try:
            import wandb
            api = wandb.Api()
            print("[plot_results] Pulling from W&B. "
                  "Set WANDB_PROJECT env var if needed.")
            data = {}
            for name in ["fixed", "linear_warmup", "gnb"]:
                runs = api.runs(
                    os.environ.get("WANDB_PROJECT", "HydroGraphNet"),
                    filters={"display_name": {"$regex": name}}
                )
                if not runs:
                    print(f"  [warn] No W&B run found for '{name}'")
                    continue
                run = runs[0]
                df = run.history(
                    keys=["epoch", "loss/total", "loss/physics",
                          "loss/depth", "metrics/rmse_depth",
                          "metrics/r2_depth", "physics/lambda_phy",
                          "physics/grad_norm_ratio_rho",
                          "physics/grad_norm_mse", "physics/grad_norm_phy"]
                )
                data[name] = {
                    "epoch":       df["epoch"].tolist(),
                    "loss_total":  df["loss/total"].tolist(),
                    "loss_mse":    df["loss/depth"].tolist(),
                    "loss_physics":df["loss/physics"].tolist(),
                    "lambda_phy":  df["physics/lambda_phy"].tolist(),
                    "rho":         df["physics/grad_norm_ratio_rho"].tolist(),
                    "rmse_depth":  df["metrics/rmse_depth"].tolist(),
                    "rmse_volume": [0.0] * len(df),
                    "r2_depth":    df["metrics/r2_depth"].tolist(),
                    "grad_norm_mse": df["physics/grad_norm_mse"].tolist(),
                    "grad_norm_phy": df["physics/grad_norm_phy"].tolist(),
                }
        except Exception as e:
            print(f"W&B load failed: {e}\nFalling back to CSV.")
            data = load_all_csvs(["fixed", "linear_warmup", "gnb"])

    else:
        data = load_all_csvs(["fixed", "linear_warmup", "gnb"])

    if not data:
        print("\nNo data found. Run with --demo to test figure generation:")
        print("  python plot_results.py --demo")
        return

    print(f"\nGenerating figures for experiments: {list(data.keys())}")
    print(f"Output directory: {os.path.abspath(OUT_DIR)}\n")

    fig_loss_curves(data)
    fig_rho(data)
    fig_lambda(data)
    fig_rmse(data)
    fig_r2(data)
    fig_summary_bars(data)
    fig_paper_panel(data)   # ← main paper figure

    print("\nDone. All figures saved.")
    print(f"  Main paper figure: {OUT_DIR}/fig7_paper_panel.pdf")


if __name__ == "__main__":
    main()
