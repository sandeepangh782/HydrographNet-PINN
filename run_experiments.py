"""
run_experiments.py

Runs three training experiments sequentially and saves per-epoch metrics as
CSV files for paper figure generation (see plot_results.py).

Experiments defined in RESEARCH_REPORT.md §6.4:
  EXP-1  fixed        λ_phy = 1.0  throughout training
  EXP-2  linear_warmup λ_phy: 0.01 → 1.0 over T_warmup epochs
  EXP-3  gnb          gradient-norm-balanced λ_phy

Each experiment:
  • creates its own output directory under outputs_phy/<exp_name>/
  • writes a CSV with columns: epoch, loss_total, loss_mse, loss_physics,
    lambda_phy, grad_norm_mse, grad_norm_phy, rho, rmse_depth, rmse_volume, r2
  • runs for `--epochs` (default 15 — enough for clear trend figures on CPU)

Usage:
    python run_experiments.py                        # all three, 15 epochs
    python run_experiments.py --epochs 5             # quick smoke test
    python run_experiments.py --only fixed           # single experiment
    python run_experiments.py --only linear_warmup gnb
"""

import argparse
import csv
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "fixed": {
        "desc": "Baseline — fixed λ_phy = 1.0",
        "overrides": [
            "physics_loss_schedule=fixed",
            "physics_loss_weight=1.0",
            "wandb_mode=online",
        ],
    },
    "linear_warmup": {
        "desc": "Strategy 1 — Linear warm-up (λ: 0.01 → 1.0 over 20 epochs)",
        "overrides": [
            "physics_loss_schedule=linear_warmup",
            "physics_lambda_init=0.01",
            "physics_lambda_max=1.0",
            "physics_warmup_epochs=20",
            "wandb_mode=online",
        ],
    },
    "gnb": {
        "desc": "Strategy 2 — Gradient Norm Balancing",
        "overrides": [
            "physics_loss_schedule=gnb",
            "physics_lambda_init=0.1",
            "physics_lambda_max=5.0",
            "wandb_mode=online",
        ],
    },
}


def run_one(exp_name: str, cfg: dict, epochs: int, n_samples: int) -> int:
    """Launch train.py for a single experiment via subprocess."""
    out_dir = f"./outputs_phy/{exp_name}"
    os.makedirs(out_dir, exist_ok=True)

    overrides = cfg["overrides"] + [
        f"epochs={epochs}",
        f"num_training_samples={n_samples}",
        f"hydra.run.dir={out_dir}",
        f"ckpt_path={out_dir}/checkpoints",
    ]
    cmd = [sys.executable, "train.py"] + overrides

    print(f"\n{'='*60}")
    print(f"  EXP: {exp_name}  —  {cfg['desc']}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"\n[{exp_name}] finished in {elapsed/60:.1f} min — {status}")
    return result.returncode


def collect_csvs(exp_names: list[str]) -> dict[str, str]:
    """Return {exp_name: csv_path} for all experiments that produced a log."""
    paths = {}
    for name in exp_names:
        csv_path = f"./outputs_phy/{name}/metrics.csv"
        if os.path.exists(csv_path):
            paths[name] = csv_path
        else:
            print(f"[warn] No metrics CSV found for experiment '{name}' "
                  f"(expected at {csv_path})")
    return paths


# ---------------------------------------------------------------------------
# CSV writer hook — written into the output directory by train.py
# via a Hydra callback.  We provide a standalone helper here so the
# existing train.py log file can be post-processed too.
# ---------------------------------------------------------------------------

CSV_HEADER = [
    "epoch", "loss_total", "loss_mse", "loss_physics",
    "lambda_phy", "grad_norm_mse", "grad_norm_phy", "rho",
    "rmse_depth", "rmse_volume", "r2_depth",
]


def parse_train_log(log_path: str, csv_out: str):
    """
    Parse the plain-text train.log written by PythonLogger and convert it
    to a structured CSV.

    Expected log line format (written by the enhanced console logging):
      Epoch {e} | Loss: {loss:.4e} | RMSE (depth/vol): {rd:.4f}/{rv:.4f} |
      R²: {r2:.3f} | λ_phy: {lam:.4f} | ρ: {rho:.3f} | LR: ...
    """
    import re

    pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)\s*\|"
        r".*?Loss:\s*(?P<loss>[0-9e.+\-]+)\s*\|"
        r".*?RMSE \(depth/vol\):\s*(?P<rmse_d>[0-9.]+)/(?P<rmse_v>[0-9.]+)\s*\|"
        r".*?R²:\s*(?P<r2>[0-9.\-]+)\s*\|"
        r".*?λ_phy:\s*(?P<lam>[0-9.e+\-]+)\s*\|"
        r".*?ρ:\s*(?P<rho>[0-9.e+\-]+)"
    )

    rows = []
    try:
        with open(log_path) as fh:
            for line in fh:
                m = pattern.search(line)
                if m:
                    rows.append({
                        "epoch":       int(m.group("epoch")),
                        "loss_total":  float(m.group("loss")),
                        "loss_mse":    "",   # not in log line — filled from wandb
                        "loss_physics": "",
                        "lambda_phy":  float(m.group("lam")),
                        "grad_norm_mse": "",
                        "grad_norm_phy": "",
                        "rho":         float(m.group("rho")),
                        "rmse_depth":  float(m.group("rmse_d")),
                        "rmse_volume": float(m.group("rmse_v")),
                        "r2_depth":    float(m.group("r2")),
                    })
    except FileNotFoundError:
        print(f"[warn] Log file not found: {log_path}")
        return

    if not rows:
        print(f"[warn] No matching log lines found in {log_path}")
        return

    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[parse_train_log] wrote {len(rows)} rows → {csv_out}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run HydroGraphNet adaptive-loss experiments"
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Training epochs per experiment (default: 15)"
    )
    parser.add_argument(
        "--samples", type=int, default=50,
        help="Number of training hydrograph samples (default: 50, fast on CPU)"
    )
    parser.add_argument(
        "--only", nargs="+",
        choices=list(EXPERIMENTS.keys()),
        default=list(EXPERIMENTS.keys()),
        help="Run only the specified experiment(s)"
    )
    parser.add_argument(
        "--parse-logs-only", action="store_true",
        help="Skip training; just parse existing train.log files into CSVs"
    )
    args = parser.parse_args()

    selected = {k: EXPERIMENTS[k] for k in args.only}

    if args.parse_logs_only:
        # Post-process existing logs without retraining
        for name in selected:
            log = f"./outputs_phy/{name}/train.log"
            csv_out = f"./outputs_phy/{name}/metrics.csv"
            os.makedirs(f"./outputs_phy/{name}", exist_ok=True)
            parse_train_log(log, csv_out)
        print("\nLog parsing complete. Run plot_results.py to generate figures.")
        return

    failed = []
    for name, cfg in selected.items():
        rc = run_one(name, cfg, epochs=args.epochs, n_samples=args.samples)
        if rc != 0:
            failed.append(name)
        # Parse the log immediately after each run
        log = f"./outputs_phy/{name}/train.log"
        csv_out = f"./outputs_phy/{name}/metrics.csv"
        parse_train_log(log, csv_out)

    print("\n" + "="*60)
    print("All experiments finished.")
    if failed:
        print(f"  FAILED: {failed}")
    else:
        print("  All passed.")
    print("\nNext step: python plot_results.py")
    print("="*60)


if __name__ == "__main__":
    main()
