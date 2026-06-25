from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List

import pandas as pd


@dataclass(frozen=True)
class Scenario:
    name: str
    family: str
    description: str
    proxy_correlation: float
    target_c_index: float
    n_features: int
    budgets: List[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a compact scenario sweep comparing ScoreNet against a Cox PH baseline "
            "across proxy correlation, target C-index, feature count, and ScoreNet budget."
        )
    )
    parser.add_argument("--n-samples", type=int, default=1500)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123,315,2,4])
    parser.add_argument("--nuisance-noise-scale", type=float, default=0.15)
    parser.add_argument("--score-max-iters", type=int, default=100000)
    parser.add_argument("--score-patience-iters", type=int, default=100000)
    parser.add_argument("--score-eval-every", type=int, default=50)
    parser.add_argument("--score-learning-rate", type=float, default=0.001)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--score-n-bins", type=int, default=25)
    parser.add_argument("--score-margin", type=float, default=1.5)
    parser.add_argument("--score-weight-values", type=float, nargs="+", default=[ 1, 2, 3,4,5])
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--cox-max-iters", type=int, default=5000)
    parser.add_argument("--cox-patience-iters", type=int, default=3000)
    parser.add_argument("--cox-eval-every", type=int, default=50)
    parser.add_argument("--cox-learning-rates", type=float, nargs="+", default=[0.01])
    parser.add_argument("--cox-ridge-grid", type=float, nargs="+", default=[0.0, 0.01,0.1])
    parser.add_argument("--output-root", type=str, default="outputs/scorenet_vs_cox_sweeps")
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def default_scenarios() -> List[Scenario]:
    return [
        Scenario(
            name="baseline",
            family="budget",
            description="Baseline data setting with a ScoreNet budget sweep.",
            proxy_correlation=0.35,
            target_c_index=0.80,
            n_features=100,
            budgets=[5, 10, 20, 25],
        ),
        Scenario(
            name="corr_0p65",
            family="correlation",
            description="Higher proxy correlation (0.65) with ScoreNet budget fixed at 10.",
            proxy_correlation=0.65,
            target_c_index=0.80,
            n_features=100,
            budgets=[10],
        ),
        Scenario(
            name="corr_0p85",
            family="correlation",
            description="Very high proxy correlation (0.85) with ScoreNet budget fixed at 10.",
            proxy_correlation=0.85,
            target_c_index=0.80,
            n_features=100,
            budgets=[10],
        ),
        Scenario(
            name="cindex_0p70",
            family="target_c_index",
            description="Lower target discrimination (0.70) with ScoreNet budget fixed at 10.",
            proxy_correlation=0.35,
            target_c_index=0.70,
            n_features=100,
            budgets=[10],
        ),
        Scenario(
            name="cindex_0p90",
            family="target_c_index",
            description="Higher target discrimination (0.90) with ScoreNet budget fixed at 10.",
            proxy_correlation=0.35,
            target_c_index=0.90,
            n_features=100,
            budgets=[10],
        ),
        Scenario(
            name="features_30",
            family="feature_count",
            description="Lower-dimensional raw feature table (30 raw features) with ScoreNet budget fixed at 10.",
            proxy_correlation=0.35,
            target_c_index=0.80,
            n_features=30,
            budgets=[10],
        ),
        Scenario(
            name="features_300",
            family="feature_count",
            description="Higher-dimensional raw feature table (300 raw features) with ScoreNet budget fixed at 10.",
            proxy_correlation=0.35,
            target_c_index=0.80,
            n_features=300,
            budgets=[10],
        ),
    ]



def run_cmd(cmd: List[str], cwd: Path) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd))



def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())



def build_score_command(
    cwd: Path,
    out_dir: Path,
    args: argparse.Namespace,
    scenario: Scenario,
    budget: int,
    seed: int,
) -> List[str]:
    cmd = [
        sys.executable,
        str(cwd / "main_fun.py"),
        "--n-samples", str(args.n_samples),
        "--n-features", str(scenario.n_features),
        "--test-size", str(args.test_size),
        "--seed", str(seed),
        "--max-selected-features", str(budget),
        "--max-iters", str(args.score_max_iters),
        "--patience-iters", str(args.score_patience_iters),
        "--eval-every", str(args.score_eval_every),
        "--learning-rate", str(args.score_learning_rate),
        "--batch-size", str(args.score_batch_size),
        "--n-bins", str(args.score_n_bins),
        "--proxy-correlation", str(scenario.proxy_correlation),
        "--target-c-index", str(scenario.target_c_index),
        "--nuisance-noise-scale", str(args.nuisance_noise_scale),
        "--torch-num-threads", str(args.torch_num_threads),
        "--margin", str(args.score_margin),
        "--output-dir", str(out_dir),
        "--no-plot",
        "--weight-values",
    ] + [str(v) for v in args.score_weight_values]
    return cmd



def build_cox_command(
    cwd: Path,
    out_dir: Path,
    args: argparse.Namespace,
    scenario: Scenario,
    seed: int,
) -> List[str]:
    cmd = [
        sys.executable,
        str(cwd / "cox_baseline.py"),
        "--n-samples", str(args.n_samples),
        "--n-features", str(scenario.n_features),
        "--test-size", str(args.test_size),
        "--seed", str(seed),
        "--max-iters", str(args.cox_max_iters),
        "--patience-iters", str(args.cox_patience_iters),
        "--eval-every", str(args.cox_eval_every),
        "--n-bins", str(args.score_n_bins),
        "--proxy-correlation", str(scenario.proxy_correlation),
        "--target-c-index", str(scenario.target_c_index),
        "--nuisance-noise-scale", str(args.nuisance_noise_scale),
        "--torch-num-threads", str(args.torch_num_threads),
        "--output-dir", str(out_dir),
        "--no-plot",
        "--learning-rates",
    ] + [str(v) for v in args.cox_learning_rates] + [
        "--ridge-grid",
    ] + [str(v) for v in args.cox_ridge_grid]
    return cmd



def summarize_group(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row["n_runs"] = int(len(group))
        for metric in metric_cols:
            values = [float(v) for v in group[metric].tolist()]
            row[f"{metric}_mean"] = float(mean(values))
            row[f"{metric}_std"] = float(pstdev(values)) if len(values) > 1 else 0.0
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(group_cols).reset_index(drop=True)
    return result



def write_markdown_report(
    args: argparse.Namespace,
    scenarios: Iterable[Scenario],
    raw_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    family_df: pd.DataFrame,
    out_path: Path,
) -> None:
    scenario_lookup = {scenario.name: scenario for scenario in scenarios}
    lines: List[str] = []
    lines.append("# ScoreNet vs Cox scenario sweep")
    lines.append("")
    lines.append("This report compares the deterministic ScoreNet rule-table scorer against the Cox PH baseline.")
    lines.append("")
    lines.append("## Experimental setup")
    lines.append("")
    lines.append(f"- Samples per run: {args.n_samples}")
    lines.append(f"- Seeds: {args.seeds}")
    lines.append(f"- ScoreNet training schedule: max_iters={args.score_max_iters}, patience={args.score_patience_iters}, eval_every={args.score_eval_every}")
    lines.append(f"- Cox training schedule: max_iters={args.cox_max_iters}, patience={args.cox_patience_iters}, eval_every={args.cox_eval_every}")
    lines.append(f"- Baseline nuisance noise scale: {args.nuisance_noise_scale}")
    lines.append("")

    lines.append("## Mean comparison by scenario and ScoreNet budget")
    lines.append("")
    lines.append("| Family | Scenario | Budget | Runs | ScoreNet test C-index | ScoreNet test C-index tie-out | Cox test C-index | Delta (ScoreNet - Cox) | ScoreNet test margin loss | Cox test margin loss | Oracle test C-index |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in agg_df.iterrows():
        lines.append(
            "| {family} | {scenario_name} | {budget} | {n_runs} | {score_test_c:.4f} +/- {score_test_c_std:.4f} | {score_test_c_tie_out:.4f} +/- {score_test_c_tie_out_std:.4f} | {cox_test_c:.4f} +/- {cox_test_c_std:.4f} | {delta_test_c:.4f} +/- {delta_test_c_std:.4f} | {score_loss:.4f} +/- {score_loss_std:.4f} | {cox_loss:.4f} +/- {cox_loss_std:.4f} | {oracle:.4f} +/- {oracle_std:.4f} |".format(
                family=row["family"],
                scenario_name=row["scenario_name"],
                budget=int(row["budget"]),
                n_runs=int(row["n_runs"]),
                score_test_c=row["scorenet_test_c_index_mean"],
                score_test_c_std=row["scorenet_test_c_index_std"],
                score_test_c_tie_out=row["scorenet_test_c_index_tie_out_mean"],
                score_test_c_tie_out_std=row["scorenet_test_c_index_tie_out_std"],
                cox_test_c=row["cox_test_c_index_mean"],
                cox_test_c_std=row["cox_test_c_index_std"],
                delta_test_c=row["delta_test_c_index_mean"],
                delta_test_c_std=row["delta_test_c_index_std"],
                score_loss=row["scorenet_test_margin_ranking_loss_mean"],
                score_loss_std=row["scorenet_test_margin_ranking_loss_std"],
                cox_loss=row["cox_test_margin_ranking_loss_mean"],
                cox_loss_std=row["cox_test_margin_ranking_loss_std"],
                oracle=row["oracle_test_c_index_mean"],
                oracle_std=row["oracle_test_c_index_std"],
            )
        )
    lines.append("")

    lines.append("## Family-level takeaways")
    lines.append("")
    for _, row in family_df.iterrows():
        lines.append(
            "- **{family}**: mean delta test C-index = {delta_mean:.4f} +/- {delta_std:.4f}; mean ScoreNet test C-index = {score_mean:.4f}; mean Cox test C-index = {cox_mean:.4f}.".format(
                family=row["family"],
                delta_mean=row["delta_test_c_index_mean"],
                delta_std=row["delta_test_c_index_std"],
                score_mean=row["scorenet_test_c_index_mean"],
                cox_mean=row["cox_test_c_index_mean"],
            )
        )
    lines.append("")

    lines.append("## Scenario definitions")
    lines.append("")
    for scenario_name in sorted(raw_df["scenario_name"].unique().tolist()):
        scenario = scenario_lookup[scenario_name]
        lines.append(
            f"- **{scenario.name}** ({scenario.family}): {scenario.description} "
            f"[proxy_correlation={scenario.proxy_correlation}, target_c_index={scenario.target_c_index}, n_features={scenario.n_features}, budgets={scenario.budgets}]"
        )
    lines.append("")

    best_row = agg_df.sort_values(["delta_test_c_index_mean", "scorenet_test_c_index_mean"], ascending=[False, False]).iloc[0]
    worst_row = agg_df.sort_values(["delta_test_c_index_mean", "scorenet_test_c_index_mean"], ascending=[True, False]).iloc[0]
    lines.append("## Extremes")
    lines.append("")
    lines.append(
        "- Best ScoreNet-vs-Cox mean test C-index delta: "
        f"{best_row['scenario_name']} at budget {int(best_row['budget'])} with delta {best_row['delta_test_c_index_mean']:.4f}."
    )
    lines.append(
        "- Worst ScoreNet-vs-Cox mean test C-index delta: "
        f"{worst_row['scenario_name']} at budget {int(worst_row['budget'])} with delta {worst_row['delta_test_c_index_mean']:.4f}."
    )
    out_path.write_text("\n".join(lines))



def main() -> Dict[str, Any]:
    args = parse_args()
    cwd = Path(__file__).resolve().parent
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and not args.keep_existing:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    scenarios = default_scenarios()
    run_rows: List[Dict[str, Any]] = []

    for scenario in scenarios:
        for seed in args.seeds:
            seed_dir = output_root / scenario.name / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            cox_dir = seed_dir / "cox"
            if not (cox_dir / "run_summary.json").exists():
                run_cmd(build_cox_command(cwd, cox_dir, args, scenario, seed), cwd=cwd)
            cox_summary = load_json(cox_dir / "run_summary.json")
            cox_test_pred = pd.read_csv(cox_dir / "test_predictions.csv")

            for budget in scenario.budgets:
                score_dir = seed_dir / f"scorenet_budget_{budget}"
                if not (score_dir / "run_summary.json").exists():
                    run_cmd(build_score_command(cwd, score_dir, args, scenario, budget, seed), cwd=cwd)
                score_summary = load_json(score_dir / "run_summary.json")
                score_test_pred = pd.read_csv(score_dir / "test_predictions.csv")

                same_test_split = False
                if len(score_test_pred) == len(cox_test_pred):
                    same_test_split = (
                        score_test_pred[["time", "status", "true_log_hazard", "guanrank_target"]]
                        .round(10)
                        .equals(cox_test_pred[["time", "status", "true_log_hazard", "guanrank_target"]].round(10))
                    )

                row = {
                    "family": scenario.family,
                    "scenario_name": scenario.name,
                    "description": scenario.description,
                    "seed": int(seed),
                    "budget": int(budget),
                    "n_samples": int(args.n_samples),
                    "n_features": int(scenario.n_features),
                    "proxy_correlation": float(scenario.proxy_correlation),
                    "target_c_index": float(scenario.target_c_index),
                    "oracle_test_c_index": float(cox_summary["metrics"]["oracle_test_c_index"]),
                    "same_test_split_verified": bool(same_test_split),
                    "scorenet_test_c_index": float(score_summary["metrics"]["test_c_index_raw"]),
                    "scorenet_test_c_index_tie_out": float(score_summary["metrics"]["test_c_index_tie_out"]),
                    "scorenet_val_c_index": float(score_summary["metrics"]["val_c_index_raw"]),
                    "scorenet_train_c_index": float(score_summary["metrics"]["train_c_index_raw"]),
                    "scorenet_test_margin_ranking_loss": float(score_summary["metrics"]["test_margin_ranking_loss"]),
                    "scorenet_val_margin_ranking_loss": float(score_summary["metrics"]["val_margin_ranking_loss"]),
                    "scorenet_train_margin_ranking_loss": float(score_summary["metrics"]["train_margin_ranking_loss"]),
                    "scorenet_best_step": int(score_summary["training"]["best_step"]),
                    "scorenet_steps_trained": int(score_summary["training"]["steps_trained"]),
                    "scorenet_selected_rules": int(score_summary["interpretable_model"]["n_selected_rules"]),
                    "cox_test_c_index": float(cox_summary["metrics"]["test_c_index_raw"]),
                    "cox_val_c_index": float(cox_summary["metrics"]["val_c_index_raw"]),
                    "cox_train_c_index": float(cox_summary["metrics"]["train_c_index_raw"]),
                    "cox_test_margin_ranking_loss": float(cox_summary["metrics"]["test_margin_ranking_loss"]),
                    "cox_val_margin_ranking_loss": float(cox_summary["metrics"]["val_margin_ranking_loss"]),
                    "cox_train_margin_ranking_loss": float(cox_summary["metrics"]["train_margin_ranking_loss"]),
                    "cox_best_step": int(cox_summary["training"]["best_step"]),
                    "cox_steps_trained": int(cox_summary["training"]["steps_trained"]),
                    "delta_test_c_index": float(score_summary["metrics"]["test_c_index_raw"] - cox_summary["metrics"]["test_c_index_raw"]),
                    "delta_test_margin_ranking_loss": float(score_summary["metrics"]["test_margin_ranking_loss"] - cox_summary["metrics"]["test_margin_ranking_loss"]),
                }
                run_rows.append(row)

    raw_df = pd.DataFrame(run_rows).sort_values(["family", "scenario_name", "budget", "seed"]).reset_index(drop=True)
    raw_path = output_root / "scenario_runs.csv"
    raw_df.to_csv(raw_path, index=False)

    agg_metric_cols = [
        "scorenet_test_c_index",
        "scorenet_test_c_index_tie_out",
        "cox_test_c_index",
        "delta_test_c_index",
        "scorenet_test_margin_ranking_loss",
        "cox_test_margin_ranking_loss",
        "delta_test_margin_ranking_loss",
        "oracle_test_c_index",
        "scorenet_selected_rules",
        "same_test_split_verified",
    ]
    agg_df = summarize_group(raw_df, ["family", "scenario_name", "budget"], agg_metric_cols)
    agg_path = output_root / "scenario_summary.csv"
    agg_df.to_csv(agg_path, index=False)

    family_df = summarize_group(raw_df, ["family"], [
        "scorenet_test_c_index",
        "cox_test_c_index",
        "delta_test_c_index",
        "scorenet_test_margin_ranking_loss",
        "cox_test_margin_ranking_loss",
        "delta_test_margin_ranking_loss",
    ])
    family_path = output_root / "family_summary.csv"
    family_df.to_csv(family_path, index=False)

    summary_payload: Dict[str, Any] = {
        "config": vars(args),
        "n_total_runs": int(len(raw_df)),
        "n_unique_scenarios": int(len(scenarios)),
        "scenario_names": [scenario.name for scenario in scenarios],
        "mean_delta_test_c_index": float(raw_df["delta_test_c_index"].mean()),
        "mean_scorenet_test_c_index": float(raw_df["scorenet_test_c_index"].mean()),
        "mean_scorenet_test_c_index_tie_out": float(raw_df["scorenet_test_c_index_tie_out"].mean()),
        "mean_cox_test_c_index": float(raw_df["cox_test_c_index"].mean()),
        "all_same_test_split_verified": bool(raw_df["same_test_split_verified"].all()),
    }
    (output_root / "summary.json").write_text(json.dumps(summary_payload, indent=2))
    write_markdown_report(args, scenarios, raw_df, agg_df, family_df, output_root / "summary.md")
    return summary_payload


if __name__ == "__main__":
    main()
