from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from data_preparation import prepare_data_splits_for_scorenet
from guanrank import guanrank_labels
from metrics import concordance_index
from ranking_utils import deterministic_margin_ranking_loss
from survival_simulation import generate_survival_data


@dataclass
class CoxFitResult:
    ridge_penalty: float
    learning_rate: float
    best_step: int
    best_train_loss: float
    best_val_c_index: float
    steps_trained: int
    coefficients: np.ndarray
    history: List[Dict]


class CoxPHLinear(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.zeros(n_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.matmul(self.beta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Cox PH baseline on the same synthetic dataset as ScoreNet.")
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--n-features", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--val-size-from-trainval", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--censor-rate", type=float, default=0.3)
    parser.add_argument("--target-c-index", type=float, default=0.8)
    parser.add_argument("--proxy-correlation", type=float, default=0.35)
    parser.add_argument("--nuisance-noise-scale", type=float, default=0.15)
    parser.add_argument("--max-iters", type=int, default=20000)
    parser.add_argument("--patience-iters", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.01])
    parser.add_argument("--ridge-grid", type=float, nargs="+", default=[0.0, 1e-4, 1e-3, 1e-2, 1e-1])
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="outputs/cox_all_features")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def set_threads(n_threads: int) -> None:
    if n_threads > 0:
        torch.set_num_threads(n_threads)
        try:
            torch.set_num_interop_threads(n_threads)
        except RuntimeError:
            pass


def _cox_neg_partial_log_likelihood(
    linear_predictor: torch.Tensor,
    event_time: torch.Tensor,
    event_observed: torch.Tensor,
) -> torch.Tensor:
    eta = linear_predictor.reshape(-1)
    time = event_time.reshape(-1)
    event = event_observed.reshape(-1) > 0.5

    # Sort by descending observed time so the risk set becomes a prefix.
    order = torch.argsort(time, descending=True)
    eta = eta[order]
    event = event[order]

    log_risk_cumsum = torch.logcumsumexp(eta, dim=0)
    event_eta = eta[event]
    event_log_risk = log_risk_cumsum[event]

    if event_eta.numel() == 0:
        raise ValueError("Cox loss is undefined when there are no observed events in the split")

    return -(event_eta - event_log_risk).mean()


def _collect_predictions(model: CoxPHLinear, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        pred = model(x).reshape(-1).detach().cpu().numpy()
    return pred




def fit_single_configuration(
    x_train: torch.Tensor,
    train_time: torch.Tensor,
    train_status: torch.Tensor,
    x_val: torch.Tensor,
    val_time: np.ndarray,
    val_status: np.ndarray,
    ridge_penalty: float,
    learning_rate: float,
    max_iters: int,
    patience_iters: int,
    eval_every: int,
) -> CoxFitResult:
    model = CoxPHLinear(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_c_index = float("-inf")
    best_train_loss = float("inf")
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    steps_trained = 0
    history: List[Dict] = []

    for step in range(1, max_iters + 1):
        steps_trained = step
        optimizer.zero_grad(set_to_none=True)
        risk = model(x_train)
        loss = _cox_neg_partial_log_likelihood(risk, train_time, train_status)
        if ridge_penalty > 0.0:
            loss = loss + 0.5 * ridge_penalty * torch.sum(model.beta ** 2)
        loss.backward()
        optimizer.step()

        should_eval = (step == 1) or (step % eval_every == 0) or (step == max_iters)
        if not should_eval:
            continue

        with torch.no_grad():
            train_risk = model(x_train)
            train_loss_value = float(_cox_neg_partial_log_likelihood(train_risk, train_time, train_status).item())
            val_risk_np = model(x_val).reshape(-1).detach().cpu().numpy()
            val_c_index = float(concordance_index(val_time, -val_risk_np, val_status))
            beta_norm = float(torch.linalg.vector_norm(model.beta).item())
        row = {
            "step": step,
            "train_loss": train_loss_value,
            "val_c_index": val_c_index,
            "beta_norm": beta_norm,
            "ridge_penalty": float(ridge_penalty),
            "learning_rate": float(learning_rate),
        }
        history.append(row)

        improved = (val_c_index > best_val_c_index + 1e-8) or (
            abs(val_c_index - best_val_c_index) <= 1e-8 and train_loss_value < best_train_loss - 1e-8
        )
        if improved:
            best_val_c_index = val_c_index
            best_train_loss = train_loss_value
            best_step = step
            best_state = copy.deepcopy(model.state_dict())

        if best_step > 0 and (step - best_step) >= patience_iters:
            break

    model.load_state_dict(best_state)
    coefficients = model.beta.detach().cpu().numpy().copy()
    return CoxFitResult(
        ridge_penalty=float(ridge_penalty),
        learning_rate=float(learning_rate),
        best_step=int(best_step),
        best_train_loss=float(best_train_loss),
        best_val_c_index=float(best_val_c_index),
        steps_trained=int(steps_trained),
        coefficients=coefficients,
        history=history,
    )


def fit_cox_with_grid_search(
    x_train: torch.Tensor,
    train_time: torch.Tensor,
    train_status: torch.Tensor,
    x_val: torch.Tensor,
    val_time: np.ndarray,
    val_status: np.ndarray,
    ridge_grid: Iterable[float],
    learning_rates: Iterable[float],
    max_iters: int,
    patience_iters: int,
    eval_every: int,
) -> Dict:
    all_runs: List[Dict] = []
    best_result: Optional[CoxFitResult] = None

    for ridge_penalty in ridge_grid:
        for learning_rate in learning_rates:
            result = fit_single_configuration(
                x_train=x_train,
                train_time=train_time,
                train_status=train_status,
                x_val=x_val,
                val_time=val_time,
                val_status=val_status,
                ridge_penalty=float(ridge_penalty),
                learning_rate=float(learning_rate),
                max_iters=max_iters,
                patience_iters=patience_iters,
                eval_every=eval_every,
            )
            all_runs.append(
                {
                    "ridge_penalty": result.ridge_penalty,
                    "learning_rate": result.learning_rate,
                    "best_step": result.best_step,
                    "best_train_loss": result.best_train_loss,
                    "best_val_c_index": result.best_val_c_index,
                    "steps_trained": result.steps_trained,
                }
            )
            if best_result is None:
                best_result = result
            else:
                better = (result.best_val_c_index > best_result.best_val_c_index + 1e-8) or (
                    abs(result.best_val_c_index - best_result.best_val_c_index) <= 1e-8
                    and result.best_train_loss < best_result.best_train_loss - 1e-8
                )
                if better:
                    best_result = result

    assert best_result is not None
    return {
        "best_result": best_result,
        "grid_results": all_runs,
    }


def coefficient_table(feature_names: List[str], coefficients: np.ndarray, top_k: int = 25) -> List[Dict]:
    rows = []
    for name, coef in zip(feature_names, coefficients):
        rows.append(
            {
                "feature": str(name),
                "coefficient": float(coef),
                "abs_coefficient": float(abs(coef)),
                "source_root": str(name).split("_proxy_")[0],
            }
        )
    rows.sort(key=lambda r: r["abs_coefficient"], reverse=True)
    return rows[:top_k]


def write_history_csv(history: List[Dict], path: Path) -> None:
    if not history:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main() -> Dict:
    args = parse_args()
    set_threads(args.torch_num_threads)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_trainval_raw, df_test_raw, simulation_metadata = generate_survival_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        censor_rate=args.censor_rate,
        target_c_index=args.target_c_index,
        test_size=args.test_size,
        seed=args.seed,
        return_metadata=True,
        proxy_correlation=args.proxy_correlation,
        nuisance_noise_scale=args.nuisance_noise_scale,
    )
    df_train_raw, df_val_raw = train_test_split(
        df_trainval_raw,
        test_size=args.val_size_from_trainval,
        random_state=args.seed,
        stratify=df_trainval_raw["status"],
    )
    df_train_raw = df_train_raw.reset_index(drop=True)
    df_val_raw = df_val_raw.reset_index(drop=True)
    df_test_raw = df_test_raw.reset_index(drop=True)

    bundle = prepare_data_splits_for_scorenet(df_train_raw, df_val_raw, df_test_raw, n_bins=args.n_bins)
    x_train = bundle["X_train"]
    x_val = bundle["X_val"]
    x_test = bundle["X_test"]
    train_time = bundle["train_time"]
    train_status = bundle["train_status"]
    val_time = bundle["val_time"].numpy()
    val_status = bundle["val_status"].numpy()
    test_time = bundle["test_time"].numpy()
    test_status = bundle["test_status"].numpy()
    train_time_np = bundle["train_time"].numpy()
    train_status_np = bundle["train_status"].numpy()
    y_train = bundle["y_train"].numpy().reshape(-1)
    y_val = bundle["y_val"].numpy().reshape(-1)
    y_test = bundle["y_test"].numpy().reshape(-1)

    search = fit_cox_with_grid_search(
        x_train=x_train,
        train_time=train_time,
        train_status=train_status,
        x_val=x_val,
        val_time=val_time,
        val_status=val_status,
        ridge_grid=args.ridge_grid,
        learning_rates=args.learning_rates,
        max_iters=args.max_iters,
        patience_iters=args.patience_iters,
        eval_every=args.eval_every,
    )
    best_result: CoxFitResult = search["best_result"]

    model = CoxPHLinear(x_train.shape[1])
    with torch.no_grad():
        model.beta.copy_(torch.tensor(best_result.coefficients, dtype=torch.float32))

    train_risk = _collect_predictions(model, x_train)
    val_risk = _collect_predictions(model, x_val)
    test_risk = _collect_predictions(model, x_test)

    train_rank_loss = float(
        deterministic_margin_ranking_loss(
            torch.tensor(train_risk, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            margin=0.5,
            seed=0,
        ).item()
    )
    val_rank_loss = float(
        deterministic_margin_ranking_loss(
            torch.tensor(val_risk, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
            margin=0.5,
            seed=0,
        ).item()
    )
    test_rank_loss = float(
        deterministic_margin_ranking_loss(
            torch.tensor(test_risk, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
            margin=0.5,
            seed=0,
        ).item()
    )

    oracle_test_c = float(concordance_index(test_time, -df_test_raw["true_log_hazard"].to_numpy(), test_status))
    score_feature_names = bundle["all_feature_names"]
    top_coeffs = coefficient_table(score_feature_names, best_result.coefficients, top_k=25)

    summary = {
        "config": vars(args),
        "training": {
            "selection_metric": "validation_c_index_raw",
            "best_step": best_result.best_step,
            "steps_trained": best_result.steps_trained,
            "best_ridge_penalty": best_result.ridge_penalty,
            "best_learning_rate": best_result.learning_rate,
        },
        "metrics": {
            "train_c_index_raw": float(concordance_index(train_time_np, -train_risk, train_status_np)),
            "val_c_index_raw": float(concordance_index(val_time, -val_risk, val_status)),
            "test_c_index_raw": float(concordance_index(test_time, -test_risk, test_status)),
            "oracle_test_c_index": oracle_test_c,
            "train_margin_ranking_loss": train_rank_loss,
            "val_margin_ranking_loss": val_rank_loss,
            "test_margin_ranking_loss": test_rank_loss,
        },
        "grid_results": search["grid_results"],
        "top_coefficients": top_coeffs,
        "simulation_metadata": simulation_metadata,
    }

    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "grid_results.json").write_text(json.dumps(search["grid_results"], indent=2))
    (output_dir / "top_coefficients.json").write_text(json.dumps(top_coeffs, indent=2))
    write_history_csv(best_result.history, output_dir / "training_history.csv")

    predictions = pd.DataFrame(
        {
            "time": df_test_raw["time"].values,
            "status": df_test_raw["status"].values,
            "true_log_hazard": df_test_raw["true_log_hazard"].values,
            "guanrank_target": y_test,
            "cox_risk_score": test_risk,
        }
    )
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    return summary


if __name__ == "__main__":
    main()
