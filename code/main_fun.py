from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from data_preparation import prepare_data_splits_for_scorenet
from metrics import concordance_index
from ranking_utils import deterministic_margin_ranking_loss, sampled_margin_ranking_loss
from scorenet import DeterministicRuleTableModel, ScoreNet, StratifiedSampler
from survival_simulation import generate_survival_data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ScoreNet with pairwise margin ranking loss against GuanRank ordering.")
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--n-features", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--val-size-from-trainval", type=float, default=0.25)
    parser.add_argument("--max-selected-features", type=int, default=10)
    parser.add_argument("--weight-values", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128*2)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--amplifier", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0)
    parser.add_argument("--max-iters", type=int, default=None, help="Maximum number of optimizer steps.")
    parser.add_argument("--patience-iters", type=int, default=None, help="Early stopping patience measured in optimizer steps.")
    parser.add_argument("--eval-every", type=int, default=50, help="Run validation every N optimizer steps.")
    parser.add_argument("--max-epochs", type=int, default=50000, help="Legacy alias for --max-iters.")
    parser.add_argument("--patience", type=int, default=50000, help="Legacy alias for --patience-iters.")
    parser.add_argument("--censor-rate", type=float, default=0.3)
    parser.add_argument("--target-c-index", type=float, default=0.8)
    parser.add_argument("--proxy-correlation", type=float, default=0.35)
    parser.add_argument("--nuisance-noise-scale", type=float, default=0.15)
    parser.add_argument("--output-dir", type=str, default="../results")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--eval-loss-seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()



def resolve_training_schedule(args: argparse.Namespace) -> argparse.Namespace:
    max_iters = args.max_iters if args.max_iters is not None else args.max_epochs
    patience_iters = args.patience_iters if args.patience_iters is not None else args.patience

    if max_iters is None:
        max_iters = 100000
    if patience_iters is None:
        patience_iters = 100000

    args.max_iters = int(max_iters)
    args.patience_iters = int(patience_iters)
    args.eval_every = int(args.eval_every)

    if args.max_iters <= 0:
        raise ValueError("max-iters must be positive")
    if args.patience_iters <= 0:
        raise ValueError("patience-iters must be positive")
    if args.eval_every <= 0:
        raise ValueError("eval-every must be positive")
    return args



def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def save_history_csv(history: List[Dict], path: Path) -> None:
    if not history:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)



def save_training_curve(history: List[Dict], path: Path) -> None:
    if not history:
        return
    steps = [row["step"] for row in history]
    train_rank = [row["train_margin_ranking_loss"] for row in history]
    val_rank = [row["val_margin_ranking_loss"] for row in history]
    val_c = [row["val_c_index_raw"] for row in history]

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, train_rank, label="Recent train margin ranking loss")
    plt.plot(steps, val_rank, label="Validation margin ranking loss")
    plt.plot(steps, val_c, label="Validation C-index (raw)")
    plt.xlabel("Optimizer step")
    plt.ylabel("Value")
    plt.title("ScoreNet training history with pairwise margin ranking loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()



def summarize_selected_rules(selected_rules: List[Dict], signal_features: List[str]) -> Dict:
    signal_features = list(signal_features)
    signal_roots_sorted = sorted(signal_features, key=len, reverse=True)

    def canonical_root(feature_name: str) -> str:
        base = str(feature_name).split("_proxy_")[0]
        for root in signal_roots_sorted:
            if base == root or base.startswith(f"{root}_"):
                return root
        return base

    def is_signal_related(feature_name: str) -> bool:
        return canonical_root(feature_name) in signal_features

    selected_only = [rule for rule in selected_rules if int(rule["selected"]) == 1]
    selected_nonzero = [rule for rule in selected_only if abs(float(rule["coefficient"])) > 1e-8]
    unique_roots = sorted({canonical_root(rule["source_feature"]) for rule in selected_only})
    signal_roots = sorted({root for root in unique_roots if root in signal_features})
    return {
        "n_selected": len(selected_only),
        "n_nonzero_selected": len(selected_nonzero),
        "n_signal_related_selected": int(sum(is_signal_related(rule["source_feature"]) for rule in selected_only)),
        "signal_related_fraction": float(sum(is_signal_related(rule["source_feature"]) for rule in selected_only) / max(len(selected_only), 1)),
        "n_unique_source_roots": int(len(unique_roots)),
        "n_unique_signal_roots": int(len(signal_roots)),
        "unique_signal_roots": signal_roots,
    }



def build_deterministic_rule_model(model: ScoreNet, data_bundle: Dict) -> DeterministicRuleTableModel:
    return model.model.deterministic_rule_table(
        all_feature_names=data_bundle["all_feature_names"],
        cont_columns=data_bundle["cont_columns"],
        cont_feature_bins=data_bundle["cont_feature_bins"],
        cont_feature_mins=data_bundle["cont_feature_mins"],
        cont_feature_maxs=data_bundle["cont_feature_maxs"],
        binary_feature_names=data_bundle["binary_feature_names"],
        binary_feature_metadata=data_bundle.get("binary_feature_metadata"),
        raw_feature_names=data_bundle.get("raw_feature_names"),
    )



def evaluate_split(
    model: ScoreNet,
    x: torch.Tensor | pd.DataFrame | np.ndarray,
    y: torch.Tensor,
    time: np.ndarray,
    status: np.ndarray,
    device: torch.device,
    margin: float,
    eval_loss_seed: int,
    deterministic_model: DeterministicRuleTableModel | None = None,
    input_processed: bool | None = True,
) -> Dict:
    model.eval()
    with torch.no_grad():
        if deterministic_model is None:
            if not isinstance(x, torch.Tensor):
                raise TypeError("Tensor features are required when deterministic_model is not provided")
            x_device = x.to(device)
            raw_scores = model.predict(x_device).reshape(-1)
        else:
            raw_np = deterministic_model.predict(x, input_processed=input_processed)
            raw_scores = torch.tensor(raw_np, dtype=torch.float32, device=device).reshape(-1)
        rank_loss = deterministic_margin_ranking_loss(raw_scores, y.to(device).reshape(-1), margin=margin, seed=eval_loss_seed)

    raw_np = raw_scores.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy().reshape(-1)

    return {
        "raw_scores": raw_np,
        "guanrank_targets": y_np,
        "margin_ranking_loss": float(rank_loss.detach().cpu().item()),
        "c_index_raw": float(concordance_index(time, -raw_np, status)),
        "c_index_tie_out": float(concordance_index(time, -raw_np, status, tie_out=True)),
        "raw_score_min": float(raw_np.min()),
        "raw_score_max": float(raw_np.max()),
    }



def write_analysis_markdown(summary: Dict, output_path: Path) -> None:
    metrics = summary["metrics"]
    training = summary["training"]
    config = summary["config"]
    lines = []
    lines.append("# ScoreNet analysis")
    lines.append("")
    lines.append("## Training objective")
    lines.append("")
    lines.append(
        "The model now optimizes the raw discrete ScoreNet score directly with a sampled pairwise MarginRankingLoss "
        "against GuanRank-derived ordering targets. There is no sigmoid calibration layer and no MSE term in ScoreNet training."
    )
    lines.append(
        "Final train/validation/test evaluation exports the deterministic interpretable rule-table score built from the modal "
        "cutoff, coefficient, and selection assignments rather than calling the neural-network prediction helper."
    )
    lines.append("")
    lines.append("## Training schedule")
    lines.append("")
    lines.append(f"- Maximum optimizer steps: {config['max_iters']}")
    lines.append(f"- Early stopping patience: {config['patience_iters']} steps")
    lines.append(f"- Validation frequency: every {config['eval_every']} steps")
    lines.append(f"- Steps trained: {training['steps_trained']}")
    lines.append(f"- Best validation step: {training['best_step']}")
    lines.append(f"- Checkpoint selection metric: {training['selection_metric']}")
    lines.append("")
    lines.append("## Key metrics")
    lines.append("")
    lines.append(f"- Best validation C-index (raw): {metrics['best_val_c_index_raw']:.4f}")
    lines.append(f"- Best historical validation margin ranking loss: {metrics['best_val_margin_ranking_loss']:.6f}")
    lines.append(f"- Selected-model validation margin ranking loss: {metrics['selected_model_val_margin_ranking_loss']:.6f}")
    lines.append(f"- Train margin ranking loss: {metrics['train_margin_ranking_loss']:.6f}")
    lines.append(f"- Validation margin ranking loss: {metrics['val_margin_ranking_loss']:.6f}")
    lines.append(f"- Test margin ranking loss: {metrics['test_margin_ranking_loss']:.6f}")
    lines.append(f"- Train C-index (raw): {metrics['train_c_index_raw']:.4f}")
    lines.append(f"- Validation C-index (raw): {metrics['val_c_index_raw']:.4f}")
    lines.append(f"- Test C-index (raw): {metrics['test_c_index_raw']:.4f}")
    lines.append(f"- Test C-index tie-out: {metrics['test_c_index_tie_out']:.4f}")
    lines.append("")
    lines.append("## Score ranges")
    lines.append("")
    lines.append(f"- Train raw score range: [{metrics['train_raw_score_min']:.4f}, {metrics['train_raw_score_max']:.4f}]")
    lines.append(f"- Validation raw score range: [{metrics['val_raw_score_min']:.4f}, {metrics['val_raw_score_max']:.4f}]")
    lines.append(f"- Test raw score range: [{metrics['test_raw_score_min']:.4f}, {metrics['test_raw_score_max']:.4f}]")
    lines.append("")
    lines.append("## Feature recovery")
    lines.append("")
    fr = summary["feature_recovery"]
    lines.append(f"- Selected rules: {fr['n_selected']}")
    lines.append(f"- Unique source roots among selected rules: {fr['n_unique_source_roots']}")
    lines.append(f"- Unique informative roots recovered: {fr['n_unique_signal_roots']}")
    lines.append(f"- Informative roots recovered: {', '.join(fr['unique_signal_roots']) if fr['unique_signal_roots'] else 'None'}")
    lines.append("")
    output_path.write_text("\n".join(lines))



def infinite_loader(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch



def main() -> Dict:
    args = resolve_training_schedule(parse_args())
    set_seed(args.seed)

    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
        try:
            torch.set_num_interop_threads(args.torch_num_threads)
        except RuntimeError:
            pass

    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Generating simulation data with {args.n_samples} samples and {args.n_features} features...")
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

    print("Creating validation split stratified by censor status...")
    df_train_raw, df_val_raw = train_test_split(
        df_trainval_raw,
        test_size=args.val_size_from_trainval,
        random_state=args.seed,
        stratify=df_trainval_raw["status"],
    )
    df_train_raw = df_train_raw.reset_index(drop=True)
    df_val_raw = df_val_raw.reset_index(drop=True)
    df_test_raw = df_test_raw.reset_index(drop=True)

    print(f"Split sizes -> train: {len(df_train_raw)}, val: {len(df_val_raw)}, test: {len(df_test_raw)}")
    print(
        "Event rates -> "
        f"train: {df_train_raw['status'].mean():.4f}, "
        f"val: {df_val_raw['status'].mean():.4f}, "
        f"test: {df_test_raw['status'].mean():.4f}"
    )

    data_bundle = prepare_data_splits_for_scorenet(
        df_train_raw,
        df_val_raw,
        df_test_raw,
        n_bins=args.n_bins,
    )

    x_train = data_bundle["X_train"]
    y_train = data_bundle["y_train"]
    x_val = data_bundle["X_val"]
    y_val = data_bundle["y_val"]
    x_test = data_bundle["X_test"]
    y_test = data_bundle["y_test"]

    val_time = data_bundle["val_time"].numpy()
    val_status = data_bundle["val_status"].numpy()
    test_time = data_bundle["test_time"].numpy()
    test_status = data_bundle["test_status"].numpy()
    train_time = data_bundle["train_time"].numpy()
    train_status = data_bundle["train_status"].numpy()

    train_dataset = TensorDataset(x_train, y_train)
    train_sampler = StratifiedSampler(
        class_vector=torch.tensor(df_train_raw["status"].values, dtype=torch.long),
        batch_size=args.batch_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        drop_last=False,
    )
    train_stream = infinite_loader(train_loader)

    model = ScoreNet(
        n_inputs=x_train.shape[1],
        learning_rate=args.learning_rate,
        amplifier=args.amplifier,
        Contfeature_bins=data_bundle["cont_feature_bins"],
        ContFeat_index=data_bundle["cont_feat_index"],
        BinaryFeat_index=data_bundle["binary_feat_index"],
        IntWeight=np.asarray(args.weight_values, dtype=np.float32),
        temp=0.1,
        Max_selected_feature=args.max_selected_features,
        MAX_EPOCHS=args.max_iters,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    y_val_device = y_val.to(device)

    history: List[Dict] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_margin_loss = float("inf")
    best_val_c_raw = float("-inf")
    best_selected_val_margin_loss = float("inf")
    best_step = 0
    steps_trained = 0
    recent_batch_losses: List[float] = []

    print("\nStarting ScoreNet training with raw pairwise MarginRankingLoss...")
    print(
        f"Training schedule -> max_iters={args.max_iters}, patience_iters={args.patience_iters}, "
        f"eval_every={args.eval_every}, margin={args.margin}"
    )

    for step in range(1, args.max_iters + 1):
        steps_trained = step
        model.train()
        x_batch, y_batch = next(train_stream)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        raw_scores = model.forward(x_batch, epoch=step - 1).reshape(-1)
        loss = sampled_margin_ranking_loss(raw_scores, y_batch.reshape(-1), margin=args.margin)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        recent_batch_losses.append(float(loss.item()))

        should_evaluate = (step == 1) or (step % args.eval_every == 0) or (step == args.max_iters)
        if not should_evaluate:
            continue

        model.eval()
        val_rule_model = build_deterministic_rule_model(model, data_bundle)
        with torch.no_grad():
            val_raw_np = val_rule_model.predict(x_val, input_processed=True)
            val_raw = torch.tensor(val_raw_np, dtype=torch.float32, device=device).reshape(-1)
            val_margin_loss = float(
                deterministic_margin_ranking_loss(
                    val_raw,
                    y_val_device.reshape(-1),
                    margin=args.margin,
                    seed=args.eval_loss_seed,
                ).item()
            )

        val_c_raw = float(concordance_index(val_time, -val_raw_np, val_status))
        val_c_raw_tie_out = float(concordance_index(val_time, -val_raw_np, val_status,tie_out=True))

        train_margin_loss = float(np.mean(recent_batch_losses)) if recent_batch_losses else float("nan")

        history_row = {
            "step": step,
            "train_margin_ranking_loss": train_margin_loss,
            "val_margin_ranking_loss": val_margin_loss,
            "val_c_index_raw": val_c_raw,
            "temp_feature_cut": model.anneal_temp_feature_cut(step - 1),
            "temp_sparse": model.anneal_temp_sparse(step - 1),
            "temp_coeff": model.anneal_temp_coeff(step - 1),
        }
        history.append(history_row)
        recent_batch_losses = []

        print(
            f"Step {step:06d} | train_margin_loss={train_margin_loss:.6f} | val_margin_loss={val_margin_loss:.6f} "
            f"| val_c_raw={val_c_raw:.4f}"
            f"| val_c_raw_tie_out={val_c_raw_tie_out:.4f}"

        )

        if val_margin_loss < best_val_margin_loss:
            best_val_margin_loss = val_margin_loss

        improved_selection_metric = (val_c_raw > best_val_c_raw + 1e-8) or (
            abs(val_c_raw - best_val_c_raw) <= 1e-8 and val_margin_loss < best_selected_val_margin_loss - 1e-8
        )
        if improved_selection_metric and step>40000:
            best_val_c_raw = val_c_raw
            best_selected_val_margin_loss = val_margin_loss
            best_step = step
            best_state = copy.deepcopy(model.state_dict())

        if best_step > 0 and (step - best_step) >= args.patience_iters:
            print(
                f"Early stopping triggered at step {step}: no validation C-index improvement for "
                f"{step - best_step} steps."
            )
            break

    model.load_state_dict(best_state)
    checkpoint_path = output_dir / "best_model.pt"
    torch.save(best_state, checkpoint_path)

    rule_model = build_deterministic_rule_model(model, data_bundle)
    train_eval = evaluate_split(
        model,
        df_train_raw,
        y_train,
        train_time,
        train_status,
        device,
        args.margin,
        args.eval_loss_seed,
        deterministic_model=rule_model,
        input_processed=False,
    )
    val_eval = evaluate_split(
        model,
        df_val_raw,
        y_val,
        val_time,
        val_status,
        device,
        args.margin,
        args.eval_loss_seed,
        deterministic_model=rule_model,
        input_processed=False,
    )
    test_eval = evaluate_split(
        model,
        df_test_raw,
        y_test,
        test_time,
        test_status,
        device,
        args.margin,
        args.eval_loss_seed,
        deterministic_model=rule_model,
        input_processed=False,
    )

    print("\nTraining complete.")
    print(f"Best validation C-index (raw): {best_val_c_raw:.4f} at step {best_step}")
    print(f"Best historical validation margin ranking loss: {best_val_margin_loss:.6f}")
    print(f"Selected-model validation margin ranking loss: {val_eval['margin_ranking_loss']:.6f}")
    print(f"Train margin ranking loss: {train_eval['margin_ranking_loss']:.6f}")
    print(f"Validation margin ranking loss: {val_eval['margin_ranking_loss']:.6f}")
    print(f"Test margin ranking loss: {test_eval['margin_ranking_loss']:.6f}")
    print(f"Train C-index (raw): {train_eval['c_index_raw']:.4f}")
    print(f"Validation C-index (raw): {val_eval['c_index_raw']:.4f}")
    print(f"Test C-index (raw): {test_eval['c_index_raw']:.4f}")
    print(f"Test C-index tie-out: {test_eval['c_index_tie_out']:.4f}")

    rule_table = rule_model.to_records(selected_only=False)
    selected_rules = rule_model.to_records(selected_only=True)
    feature_recovery = summarize_selected_rules(selected_rules, simulation_metadata["signal_features"])
    interpretable_model_summary = {
        "score_source": "deterministic_rule_table",
        "bias": float(rule_model.bias),
        "n_rules": int(len(rule_model)),
        "n_selected_rules": int(len(rule_model.selected_rules())),
        "supports_processed_features": True,
        "supports_raw_features": True,
    }

    summary = {
        "config": vars(args),
        "training": {
            "steps_trained": int(steps_trained),
            "best_step": int(best_step),
            "selection_metric": "validation_c_index_raw",
            "checkpoint_path": str(checkpoint_path),
            "eval_every": int(args.eval_every),
        },
        "metrics": {
            "best_val_c_index_raw": float(best_val_c_raw),
            "best_val_margin_ranking_loss": float(best_val_margin_loss),
            "selected_model_val_margin_ranking_loss": float(val_eval["margin_ranking_loss"]),
            "train_margin_ranking_loss": float(train_eval["margin_ranking_loss"]),
            "val_margin_ranking_loss": float(val_eval["margin_ranking_loss"]),
            "test_margin_ranking_loss": float(test_eval["margin_ranking_loss"]),
            "train_c_index_raw": float(train_eval["c_index_raw"]),
            "val_c_index_raw": float(val_eval["c_index_raw"]),
            "test_c_index_raw": float(test_eval["c_index_raw"]),
            "train_c_index_tie_out": float(train_eval["c_index_tie_out"]),
            "val_c_index_tie_out": float(val_eval["c_index_tie_out"]),
            "test_c_index_tie_out": float(test_eval["c_index_tie_out"]),
            "train_raw_score_min": float(train_eval["raw_score_min"]),
            "train_raw_score_max": float(train_eval["raw_score_max"]),
            "val_raw_score_min": float(val_eval["raw_score_min"]),
            "val_raw_score_max": float(val_eval["raw_score_max"]),
            "test_raw_score_min": float(test_eval["raw_score_min"]),
            "test_raw_score_max": float(test_eval["raw_score_max"]),
            "deterministic_rule_bias": float(rule_model.bias),
        },
        "interpretable_model": interpretable_model_summary,
        "feature_recovery": feature_recovery,
        "selected_rules": selected_rules,
        "simulation_metadata": simulation_metadata,
    }

    deterministic_rule_table_payload = {
        **interpretable_model_summary,
        "rules": rule_table,
    }

    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "simulation_metadata.json").write_text(json.dumps(simulation_metadata, indent=2))
    (output_dir / "selected_rules.json").write_text(json.dumps(selected_rules, indent=2))
    (output_dir / "deterministic_rule_table.json").write_text(json.dumps(deterministic_rule_table_payload, indent=2))
    save_history_csv(history, output_dir / "training_history.csv")
    if not args.no_plot:
        save_training_curve(history, output_dir / "training_curve.png")

    predictions = pd.DataFrame(
        {
            "time": df_test_raw["time"].values,
            "status": df_test_raw["status"].values,
            "true_log_hazard": df_test_raw["true_log_hazard"].values,
            "guanrank_target": test_eval["guanrank_targets"],
            "predicted_raw_score": test_eval["raw_scores"],
            "predicted_raw_score_source": "deterministic_rule_table",
            "prediction_input_representation": "raw_unprocessed_features",
        }
    )
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    write_analysis_markdown(summary, output_dir / "analysis.md")

    print("\nSelected hard rules:")
    for rule in selected_rules:
        print(f"- {rule['rule']} -> coefficient {rule['coefficient']:.1f}")

    print(f"\nArtifacts written to: {output_dir.resolve()}")
    return summary


if __name__ == "__main__":
    main()
