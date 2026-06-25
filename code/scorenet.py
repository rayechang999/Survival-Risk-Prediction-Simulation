from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


EPSILON = np.finfo(np.float32).tiny



@dataclass(frozen=True)
class SelectedRule:
    extended_feature_index: int
    feature_type: str
    source_feature: str
    rule: str
    coefficient: float
    selected: int
    processed_feature_name: str = ""
    comparator: str = ""
    cutoff_value_processed: Optional[float] = None
    cutoff_value_raw: Optional[float] = None
    category_value: Optional[str] = None
    cutoff_mode_index: Optional[int] = None
    weight_mode_index: Optional[int] = None


class DeterministicRuleTableModel(Sequence[SelectedRule]):
    """Interpretable deterministic scorer induced by the modal discrete assignments.

    The scorer can evaluate either the processed ScoreNet design matrix
    (scaled continuous features plus one-hot encoded categorical features) or the
    original/raw feature table. Continuous rules use the modal cutoff bin chosen
    by the learned cutoff distribution. Coefficients use the modal discrete
    weight for each branch. Feature selection uses the deterministic top-k set
    induced by the learned selection logits.
    """

    def __init__(
        self,
        rules: Sequence[SelectedRule],
        bias: float,
        processed_feature_names: Sequence[str],
        raw_feature_names: Sequence[str],
    ):
        self.rules = list(rules)
        self.bias = float(bias)
        self.processed_feature_names = list(processed_feature_names)
        self.raw_feature_names = list(raw_feature_names)
        self._processed_feature_set = set(self.processed_feature_names)
        self._raw_feature_set = set(self.raw_feature_names)

    def __len__(self) -> int:
        return len(self.rules)

    def __getitem__(self, index):
        return self.rules[index]

    def __iter__(self):
        return iter(self.rules)

    def selected_rules(self) -> List[SelectedRule]:
        return [rule for rule in self.rules if int(rule.selected) == 1]

    def to_records(self, selected_only: bool = False) -> List[Dict]:
        source = self.selected_rules() if selected_only else self.rules
        return [rule.__dict__.copy() for rule in source]

    def _infer_input_processed(self, df: pd.DataFrame) -> bool:
        has_all_processed = self._processed_feature_set.issubset(df.columns)
        has_all_raw = self._raw_feature_set.issubset(df.columns)
        if has_all_processed and not has_all_raw:
            return True
        if has_all_raw and not has_all_processed:
            return False
        if has_all_processed and has_all_raw:
            processed_only_columns = self._processed_feature_set - self._raw_feature_set
            if processed_only_columns:
                return any(col in df.columns for col in processed_only_columns)
            raise ValueError(
                "Processed and raw feature names overlap completely, so the input type is ambiguous. "
                "Provide input_processed explicitly."
            )
        raise ValueError(
            "Unable to infer whether the provided features are processed or raw. "
            "Provide input_processed explicitly and ensure the required columns are present."
        )

    def _coerce_input(
        self,
        x: pd.DataFrame | np.ndarray | torch.Tensor,
        input_processed: Optional[bool],
        feature_names: Optional[Sequence[str]],
    ) -> tuple[pd.DataFrame, bool]:
        if isinstance(x, pd.DataFrame):
            df = x.copy()
            inferred_processed = self._infer_input_processed(df) if input_processed is None else bool(input_processed)
            return df, inferred_processed

        if isinstance(x, torch.Tensor):
            array = x.detach().cpu().numpy()
        else:
            array = np.asarray(x)

        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError("Input must be a 2D table of features")

        if input_processed is None:
            if feature_names is not None:
                names = list(feature_names)
                if self._processed_feature_set.issubset(names):
                    input_processed = True
                elif self._raw_feature_set.issubset(names):
                    input_processed = False
            if input_processed is None:
                processed_width = len(self.processed_feature_names)
                raw_width = len(self.raw_feature_names)
                if array.shape[1] == processed_width and array.shape[1] == raw_width:
                    raise ValueError(
                        "Processed and raw feature arrays have the same width, so the input type is ambiguous. "
                        "Provide input_processed explicitly or pass feature_names."
                    )
                if array.shape[1] == processed_width:
                    input_processed = True
                elif array.shape[1] == raw_width:
                    input_processed = False
                else:
                    raise ValueError(
                        "Unable to infer whether the provided array contains processed or raw features. "
                        "Provide input_processed explicitly or pass feature_names."
                    )

        if feature_names is None:
            feature_names = self.processed_feature_names if bool(input_processed) else self.raw_feature_names
        if len(feature_names) != array.shape[1]:
            raise ValueError("feature_names length must match the number of input columns")
        return pd.DataFrame(array, columns=list(feature_names)), bool(input_processed)

    def _binary_indicator(self, values: pd.Series) -> np.ndarray:
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any():
            text = values.astype(str).str.strip().str.lower()
            return text.isin(["1", "1.0", "true", "yes"]).to_numpy(dtype=np.float32)
        return (numeric.to_numpy(dtype=np.float64) >= 0.5).astype(np.float32)

    def _rule_indicator(self, rule: SelectedRule, df: pd.DataFrame, input_processed: bool) -> np.ndarray:
        if rule.feature_type == "continuous_positive_branch":
            column = rule.processed_feature_name if input_processed and rule.processed_feature_name else rule.source_feature
            if column not in df.columns:
                raise KeyError(f"Missing continuous feature column: {column}")
            threshold = rule.cutoff_value_processed if input_processed else rule.cutoff_value_raw
            if threshold is None:
                raise ValueError(f"Rule {rule.rule} is missing the required cutoff value")
            values = pd.to_numeric(df[column], errors="raise").to_numpy(dtype=np.float64)
            return (values >= float(threshold)).astype(np.float32)

        if rule.feature_type == "continuous_negative_branch":
            column = rule.processed_feature_name if input_processed and rule.processed_feature_name else rule.source_feature
            if column not in df.columns:
                raise KeyError(f"Missing continuous feature column: {column}")
            threshold = rule.cutoff_value_processed if input_processed else rule.cutoff_value_raw
            if threshold is None:
                raise ValueError(f"Rule {rule.rule} is missing the required cutoff value")
            values = pd.to_numeric(df[column], errors="raise").to_numpy(dtype=np.float64)
            return (values <= float(threshold)).astype(np.float32)

        if rule.feature_type == "binary_one_hot_branch":
            processed_column = rule.processed_feature_name or rule.source_feature
            if input_processed:
                if processed_column not in df.columns:
                    raise KeyError(f"Missing binary feature column: {processed_column}")
                return self._binary_indicator(df[processed_column])

            if rule.source_feature in df.columns and rule.category_value is not None:
                return (df[rule.source_feature].astype(str).to_numpy() == str(rule.category_value)).astype(np.float32)

            if processed_column in df.columns:
                return self._binary_indicator(df[processed_column])
            raise KeyError(
                f"Missing raw categorical feature {rule.source_feature} and processed column {processed_column}"
            )

        raise ValueError(f"Unsupported rule type: {rule.feature_type}")

    def raw_scores(
        self,
        x: pd.DataFrame | np.ndarray | torch.Tensor,
        input_processed: Optional[bool] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        df, resolved_input_processed = self._coerce_input(
            x=x,
            input_processed=input_processed,
            feature_names=feature_names,
        )
        scores = np.full(len(df), self.bias, dtype=np.float32)
        for rule in self.selected_rules():
            indicator = self._rule_indicator(rule, df=df, input_processed=resolved_input_processed)
            scores = scores + indicator.astype(np.float32) * np.float32(rule.coefficient)
        return scores.astype(np.float32)


    def predict(
        self,
        x: pd.DataFrame | np.ndarray | torch.Tensor,
        input_processed: Optional[bool] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        return self.raw_scores(x=x, input_processed=input_processed, feature_names=feature_names)


def differentiable_top_k(
    scores: torch.Tensor,
    k: int,
    tau: float,
    hard: bool,
    dim: int = -1,
) -> torch.Tensor:
    """Sample a differentiable k-hot mask using the iterative Gumbel top-k trick."""
    if scores.ndim == 0:
        raise ValueError("scores must have at least one dimension")

    n_features = scores.shape[dim]
    k = int(max(1, min(k, n_features)))

    gumbel_noise = torch.distributions.gumbel.Gumbel(
        torch.zeros_like(scores),
        torch.ones_like(scores),
    ).sample()
    perturbed_scores = scores + gumbel_noise

    khot = torch.zeros_like(perturbed_scores)
    onehot_approx = torch.zeros_like(perturbed_scores)
    for _ in range(k):
        available_mass = torch.clamp(1.0 - onehot_approx, min=EPSILON)
        perturbed_scores = perturbed_scores + torch.log(available_mass)
        onehot_approx = F.softmax(perturbed_scores / tau, dim=dim)
        khot = khot + onehot_approx

    if not hard:
        return khot

    hard_mask = torch.zeros_like(khot)
    _, topk_index = torch.topk(khot, k, dim=dim)
    hard_mask.scatter_(dim, topk_index, 1.0)
    return hard_mask - khot.detach() + khot


class ScoreNetHead(nn.Module):
    """
        1. Feature binarization module
        2. Feature weighting module
        3. Feature selection module
        4. Score aggregation
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        amplifier: float = 0.5,
        Contfeature_bins: Optional[np.ndarray] = None,
        ContFeat_index: Optional[Sequence[int]] = None,
        BinaryFeat_index: Optional[Sequence[int]] = None,
        IntWeight: Optional[Sequence[float]] = None,
        temp: Optional[float] = None,
        Max_selected_feature: Optional[int] = None,
    ):
        super().__init__()

        contfeature_bins = np.asarray(Contfeature_bins, dtype=np.float32)
        if contfeature_bins.ndim != 2:
            raise ValueError("Contfeature_bins must be a 2D array of candidate cutoffs")

        cont_feat_index = list(ContFeat_index or [])
        binary_feat_index = list(BinaryFeat_index or [])
        int_weight_values = np.asarray(IntWeight if IntWeight is not None else [0.0, 1.0], dtype=np.float32)
        if int_weight_values.ndim != 1:
            raise ValueError("IntWeight must be a 1D array of discrete coefficient choices")
        if Max_selected_feature is None:
            raise ValueError("Max_selected_feature must be provided")

        self.input_dim = input_dim
        self.amplifier = float(amplifier)
        self.temperature_hint = temp
        self.max_selected_feature = int(Max_selected_feature)
        self.cont_feat_index = cont_feat_index
        self.binary_feat_index = binary_feat_index
        self.num_extended_features = 2 * len(self.cont_feat_index) + len(self.binary_feat_index)

        self.register_buffer("continuous_cutoff_values", torch.from_numpy(contfeature_bins).to(torch.float32))
        self.register_buffer("integer_weight_values", torch.from_numpy(int_weight_values).to(torch.float32))

        self.cutoff_choice_logits = nn.Parameter(torch.zeros(contfeature_bins.shape[0], contfeature_bins.shape[1]))
        self.feature_selection_logits = nn.Parameter(torch.zeros(1, self.num_extended_features))
        self.weight_choice_logits = nn.Parameter(torch.zeros(self.num_extended_features + 1, len(int_weight_values)))

        self.Amplifier = self.amplifier
        self.Contfeature_bins = contfeature_bins
        self.Binarized_cut_off = self.continuous_cutoff_values
        self.IntWeight = self.integer_weight_values
        self.ContFeat_index = self.cont_feat_index
        self.BinaryFeat_index = self.binary_feat_index
        self.Temp = self.temperature_hint
        self.Max_selected_feature = self.max_selected_feature
        self.ContFeat_Cut_off_logits = self.cutoff_choice_logits
        self.Sparse_Weight = self.feature_selection_logits
        self.Integer_Weight_logits = self.weight_choice_logits


    def sample_binarization_assignment(self, tau: float, hard: bool = True) -> torch.Tensor:
        if len(self.cont_feat_index) == 0:
            return self.cutoff_choice_logits.new_zeros((0, 0))
        if hard:
            return F.gumbel_softmax(self.cutoff_choice_logits, tau=tau, hard=True, dim=-1)
        return F.gumbel_softmax(self.cutoff_choice_logits, tau=tau, hard=False, dim=-1)

    def build_continuous_binary_bank(
        self,
        x: torch.Tensor,
        cutoff_assignment: torch.Tensor,
    ) -> torch.Tensor:
        if len(self.cont_feat_index) == 0:
            return x.new_zeros((x.shape[0], 0))

        cont_feat = x[:, self.cont_feat_index]
        selected_cutoffs = torch.sum(cutoff_assignment * self.continuous_cutoff_values, dim=-1)

        amp = max(self.amplifier, EPSILON)
        positive_surrogate = nn.Sigmoid()((cont_feat - selected_cutoffs) / amp)
        negative_surrogate = nn.Sigmoid()((selected_cutoffs - cont_feat) / amp)
        
        positive_hard = torch.heaviside(cont_feat - selected_cutoffs, torch.ones_like(cont_feat))
        negative_hard = torch.heaviside(selected_cutoffs - cont_feat, torch.ones_like(cont_feat))

        positive_branch = positive_surrogate - positive_surrogate.detach() + positive_hard.detach()
        negative_branch = negative_surrogate - negative_surrogate.detach() + negative_hard.detach()
        return torch.cat((positive_branch, negative_branch), dim=-1)

    def feature_binarization_module(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        cutoff_assignment = self.sample_binarization_assignment(tau=tau, hard=True)
        continuous_bank = self.build_continuous_binary_bank(x, cutoff_assignment)
        if len(self.binary_feat_index) == 0:
            return continuous_bank
        original_binary_bank = x[:, self.binary_feat_index]
        return torch.cat((continuous_bank, original_binary_bank), dim=-1)

    def sample_weight_values(self, tau: float, hard: bool = True) -> torch.Tensor:
        onehot = F.gumbel_softmax(self.weight_choice_logits, tau=tau, hard=hard, dim=-1)
        return torch.sum(onehot * self.integer_weight_values.view(1, -1), dim=-1)

    def feature_weighting_module(self, feature_bank: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
        weight_vector = self.sample_weight_values(tau=tau, hard=True)
        feature_weights = weight_vector[:-1]
        bias_weight = weight_vector[-1]
        weighted_bank = feature_bank * feature_weights.view(1, -1)
        return weighted_bank, bias_weight

    def sample_selection_mask(self, tau: float, hard: bool = False) -> torch.Tensor:
        return differentiable_top_k(
            scores=self.feature_selection_logits,
            k=self.max_selected_feature,
            tau=tau,
            hard=hard,
            dim=-1,
        ).reshape(-1)

    def feature_selection_module(self, weighted_bank: torch.Tensor, tau: float) -> torch.Tensor:
        selection_mask = self.sample_selection_mask(tau=tau, hard=False)
        return weighted_bank * selection_mask.view(1, -1)

    def score_aggregation(self, selected_weighted_bank: torch.Tensor, bias_weight: torch.Tensor) -> torch.Tensor:
        return selected_weighted_bank.sum(dim=-1, keepdim=True) + bias_weight.view(1, 1)

    # ------------------------------------------------------------------
    # Forward / inference
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, temp_feature_cut: float, temp_sparse: float, temp_coeff: float) -> torch.Tensor:
        feature_bank = self.feature_binarization_module(x, tau=temp_feature_cut)
        weighted_bank, bias_weight = self.feature_weighting_module(feature_bank, tau=temp_coeff)
        selected_weighted_bank = self.feature_selection_module(weighted_bank, tau=temp_sparse)
        return self.score_aggregation(selected_weighted_bank, bias_weight)

    def mode_cutoff_assignment(self) -> torch.Tensor:
        if self.cutoff_choice_logits.numel() == 0:
            return torch.zeros_like(self.cutoff_choice_logits)
        probabilities = torch.softmax(self.cutoff_choice_logits, dim=-1)
        return self.argmax_onehot(probabilities)

    def mode_weight_assignment(self) -> torch.Tensor:
        probabilities = torch.softmax(self.weight_choice_logits, dim=-1)
        return self.argmax_onehot(probabilities)

    def mode_selection_mask(self) -> torch.Tensor:
        n_features = self.feature_selection_logits.shape[-1]
        if n_features == 0:
            return self.feature_selection_logits.reshape(-1)
        k = int(max(1, min(self.max_selected_feature, n_features)))
        tie_break = -torch.arange(
            n_features,
            device=self.feature_selection_logits.device,
            dtype=self.feature_selection_logits.dtype,
        ) * 1e-12
        stable_scores = self.feature_selection_logits.reshape(-1) + tie_break
        mask = torch.zeros_like(stable_scores)
        topk_index = torch.topk(stable_scores, k).indices
        mask.scatter_(0, topk_index, 1.0)
        return mask

    def hard_cutoff_assignment(self) -> torch.Tensor:
        return self.mode_cutoff_assignment()

    def hard_weight_values(self) -> torch.Tensor:
        onehot = self.mode_weight_assignment()
        return torch.sum(onehot * self.integer_weight_values.view(1, -1), dim=-1)

    def hard_selection_mask(self) -> torch.Tensor:
        return self.mode_selection_mask()

    def build_feature_bank_for_prediction(self, x: torch.Tensor) -> torch.Tensor:
        cutoff_assignment = self.hard_cutoff_assignment()
        continuous_bank = self.build_continuous_binary_bank(x, cutoff_assignment)
        if len(self.binary_feat_index) == 0:
            return continuous_bank
        original_binary_bank = x[:, self.binary_feat_index]
        return torch.cat((continuous_bank, original_binary_bank), dim=-1)

    def predict_scores(self, x: torch.Tensor) -> torch.Tensor:
        feature_bank = self.build_feature_bank_for_prediction(x)
        weight_vector = self.hard_weight_values()
        feature_weights = weight_vector[:-1]
        bias_weight = weight_vector[-1]
        weighted_bank = feature_bank * feature_weights.view(1, -1)
        selected_weighted_bank = weighted_bank * self.hard_selection_mask().view(1, -1)
        return self.score_aggregation(selected_weighted_bank, bias_weight).detach()



    def argmax_onehot(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0:
            return torch.zeros_like(logits)
        index = logits.max(-1, keepdim=True)[1]
        return torch.zeros_like(logits).scatter_(-1, index, 1.0)

 
    def deterministic_rule_table(
        self,
        all_feature_names: Sequence[str],
        cont_columns: Sequence[str],
        cont_feature_bins: np.ndarray,
        cont_feature_mins: np.ndarray,
        cont_feature_maxs: np.ndarray,
        binary_feature_names: Sequence[str],
        binary_feature_metadata: Optional[Sequence[Dict[str, str]]] = None,
        raw_feature_names: Optional[Sequence[str]] = None,
    ) -> DeterministicRuleTableModel:
        """Build a deterministic interpretable scorer from the modal discrete assignments.

        The returned object is iterable like the previous rule list, but it also
        acts as a standalone scoring model via `.predict(...)` / `.raw_scores(...)`.
        It accepts either the processed ScoreNet design matrix or the original/raw
        input features.
        """
        rules: List[SelectedRule] = []
        selection_mask = self.hard_selection_mask().detach().cpu().numpy().astype(int)
        mode_weight_assignment = self.mode_weight_assignment().detach().cpu().numpy()
        weight_mode_index = mode_weight_assignment.argmax(axis=-1)
        weight_vector = np.sum(mode_weight_assignment * self.integer_weight_values.detach().cpu().numpy().reshape(1, -1), axis=-1)
        feature_weights = weight_vector[:-1]
        bias_weight = float(weight_vector[-1])
        cutoff_assignment = self.mode_cutoff_assignment().detach().cpu().numpy()
        cutoff_index = cutoff_assignment.argmax(axis=-1) if cutoff_assignment.size else np.zeros((0,), dtype=np.int64)
        cont_feature_bins = np.asarray(cont_feature_bins, dtype=np.float32)
        cont_feature_mins = np.asarray(cont_feature_mins, dtype=np.float32)
        cont_feature_maxs = np.asarray(cont_feature_maxs, dtype=np.float32)

        n_cont = len(cont_columns)
        n_binary = len(binary_feature_names)
        expected_extended = 2 * n_cont + n_binary
        if expected_extended != len(feature_weights):
            raise ValueError("Feature metadata does not match the learned extended feature bank")

        binary_metadata_list = list(binary_feature_metadata or [])
        if binary_metadata_list and len(binary_metadata_list) != n_binary:
            raise ValueError("binary_feature_metadata does not match the binary feature block")
        if not binary_metadata_list:
            binary_metadata_list = [
                {
                    "processed_feature_name": feat_name,
                    "source_feature": feat_name,
                    "source_value": None,
                }
                for feat_name in binary_feature_names
            ]

        if raw_feature_names is None:
            derived_raw_names = list(cont_columns)
            for meta in binary_metadata_list:
                source_feature = str(meta.get("source_feature", meta.get("processed_feature_name", "")))
                if source_feature and source_feature not in derived_raw_names:
                    derived_raw_names.append(source_feature)
            raw_feature_names = derived_raw_names

        for idx in range(expected_extended):
            coeff = float(feature_weights[idx])
            selected = int(selection_mask[idx])
            weight_idx = int(weight_mode_index[idx])
            if idx < n_cont:
                feat_name = cont_columns[idx]
                processed_cutoff = float(cont_feature_bins[idx, cutoff_index[idx]])
                raw_cutoff = float(
                    cont_feature_mins[idx]
                    + processed_cutoff * (cont_feature_maxs[idx] - cont_feature_mins[idx])
                )
                rule = f"{feat_name} >= {raw_cutoff:.4f}"
                feature_type = "continuous_positive_branch"
                source_feature = feat_name
                processed_feature_name = feat_name
                comparator = ">="
                category_value = None
                cutoff_mode_index = int(cutoff_index[idx])
                cutoff_value_processed = processed_cutoff
                cutoff_value_raw = raw_cutoff
            elif idx < 2 * n_cont:
                j = idx - n_cont
                feat_name = cont_columns[j]
                processed_cutoff = float(cont_feature_bins[j, cutoff_index[j]])
                raw_cutoff = float(
                    cont_feature_mins[j]
                    + processed_cutoff * (cont_feature_maxs[j] - cont_feature_mins[j])
                )
                rule = f"{feat_name} <= {raw_cutoff:.4f}"
                feature_type = "continuous_negative_branch"
                source_feature = feat_name
                processed_feature_name = feat_name
                comparator = "<="
                category_value = None
                cutoff_mode_index = int(cutoff_index[j])
                cutoff_value_processed = processed_cutoff
                cutoff_value_raw = raw_cutoff
            else:
                j = idx - 2 * n_cont
                feat_name = binary_feature_names[j]
                meta = binary_metadata_list[j]
                processed_feature_name = str(meta.get("processed_feature_name", feat_name))
                source_feature = str(meta.get("source_feature", feat_name))
                source_value = meta.get("source_value")
                category_value = None if source_value is None else str(source_value)
                if category_value is None:
                    rule = f"{processed_feature_name} == 1"
                else:
                    rule = f"{source_feature} == {category_value}"
                feature_type = "binary_one_hot_branch"
                comparator = "=="
                cutoff_mode_index = None
                cutoff_value_processed = None
                cutoff_value_raw = None

            rules.append(
                SelectedRule(
                    extended_feature_index=idx,
                    feature_type=feature_type,
                    source_feature=source_feature,
                    rule=rule,
                    coefficient=coeff,
                    selected=selected,
                    processed_feature_name=processed_feature_name,
                    comparator=comparator,
                    cutoff_value_processed=cutoff_value_processed,
                    cutoff_value_raw=cutoff_value_raw,
                    category_value=category_value,
                    cutoff_mode_index=cutoff_mode_index,
                    weight_mode_index=weight_idx,
                )
            )
        return DeterministicRuleTableModel(
            rules=rules,
            bias=bias_weight,
            processed_feature_names=list(all_feature_names),
            raw_feature_names=list(raw_feature_names),
        )


class ScoreNet(nn.Module):
    """
    The discrete ScoreNet head directly produces the raw interpretable score used
    for both optimization and inference.
    """

    def __init__(
        self,
        n_inputs: int = 1,
        learning_rate: float = 0.05,
        amplifier: float = 0.05,
        Contfeature_bins: Optional[np.ndarray] = None,
        ContFeat_index: Optional[Sequence[int]] = None,
        IntWeight: Optional[Sequence[float]] = None,
        temp: Optional[float] = None,
        BinaryFeat_index: Optional[Sequence[int]] = None,
        Max_selected_feature: Optional[int] = None,
        MAX_EPOCHS: Optional[int] = None,
    ):
        super().__init__()

        self.learning_rate_model = float(learning_rate)
        self.Contfeature_bins = Contfeature_bins
        self.ContFeat_index = list(ContFeat_index or [])
        self.IntWeight = np.asarray(IntWeight if IntWeight is not None else [0.0, 1.0], dtype=np.float32)
        self.temp = temp
        self.n_inputs = int(n_inputs)
        self.amplifier = float(amplifier)
        self.BinaryFeat_index = list(BinaryFeat_index or [])
        self.max_schedule_steps = int(MAX_EPOCHS if MAX_EPOCHS is not None else 1)
        self.max_epochs = self.max_schedule_steps

        self.model = ScoreNetHead(
            input_dim=self.n_inputs,
            amplifier=self.amplifier,
            Contfeature_bins=self.Contfeature_bins,
            ContFeat_index=self.ContFeat_index,
            BinaryFeat_index=self.BinaryFeat_index,
            IntWeight=self.IntWeight,
            temp=self.temp,
            Max_selected_feature=Max_selected_feature,
        )

        self.N = self.max_schedule_steps
        self.K_temp_feature_cut = 10
        self.K_temp_sparse = 10
        self.K_temp_coeff = 10

    def anneal_temp_feature_cut(self, current_step: int) -> float:
        return max(2 * (1 / self.K_temp_feature_cut) ** (current_step / max(self.N, 1)), 1 / self.K_temp_feature_cut)

    def anneal_temp_sparse(self, current_step: int) -> float:
        return max(2 * (1 / self.K_temp_sparse) ** (current_step / max(self.N, 1)), 1 / self.K_temp_sparse)

    def anneal_temp_coeff(self, current_step: int) -> float:
        return max(2 * (1 / self.K_temp_coeff) ** (current_step / max(self.N, 1)), 1 / self.K_temp_coeff)

    def current_temperatures(self, step: int) -> tuple[float, float, float]:
        return (
            self.anneal_temp_feature_cut(step),
            self.anneal_temp_sparse(step),
            self.anneal_temp_coeff(step),
        )

    def raw_scores(
        self,
        x: torch.Tensor,
        epoch: Optional[int] = None,
        temp_feature_cut: Optional[float] = None,
        temp_sparse: Optional[float] = None,
        temp_coeff: Optional[float] = None,
    ) -> torch.Tensor:
        if temp_feature_cut is None or temp_sparse is None or temp_coeff is None:
            step = int(epoch if epoch is not None else 0)
            temp_feature_cut, temp_sparse, temp_coeff = self.current_temperatures(step)
        return self.model(x, temp_feature_cut=temp_feature_cut, temp_sparse=temp_sparse, temp_coeff=temp_coeff)

    def forward(
        self,
        x: torch.Tensor,
        epoch: Optional[int] = None,
        temp_feature_cut: Optional[float] = None,
        temp_sparse: Optional[float] = None,
        temp_coeff: Optional[float] = None,
    ) -> torch.Tensor:
        return self.raw_scores(
            x,
            epoch=epoch,
            temp_feature_cut=temp_feature_cut,
            temp_sparse=temp_sparse,
            temp_coeff=temp_coeff,
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.predict_scores(x)


class Sampler(object):
    def __init__(self, data_source):
        del data_source

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class StratifiedSampler(Sampler):
    def __init__(self, class_vector: torch.Tensor, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.class_vector = class_vector.reshape(-1)
        self.batch_size = int(batch_size)
        self.n_splits = max(int(np.ceil(self.class_vector.numel() / self.batch_size)), 1)

    def gen_sample_array(self):
        from sklearn.model_selection import StratifiedShuffleSplit

        x_dummy = torch.randn(self.class_vector.size(0), 2).numpy()
        y = self.class_vector.detach().cpu().numpy()
        splitter = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5, random_state=0)
        try:
            train_index, test_index = next(splitter.split(x_dummy, y))
            return np.hstack([train_index, test_index])
        except Exception:
            return np.arange(len(y))

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


