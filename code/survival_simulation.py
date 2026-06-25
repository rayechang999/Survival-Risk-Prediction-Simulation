from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split

from metrics import concordance_index


CLINICAL_CONFIGS = [
    ("Age_at_Diagnosis", 40, 90, True, False, None),
    ("Tumor_Diameter_mm", 5, 100, False, False, None),
    ("Albumin_Level_gL", 30, 50, False, False, None),
    ("Lymph_Nodes_Positive", 0, 25, True, False, None),
    ("Hemoglobin_Level", 10, 18, False, False, None),
    ("Systolic_BP", 100, 190, True, False, None),
    ("LDH_U_L", 100, 600, True, False, None),
    ("Cancer_Stage", None, None, False, True, ["Stage_I", "Stage_II", "Stage_III"]),
    ("Treatment_Group", None, None, False, True, ["Drug_A", "Drug_B", "Placebo"]),
    ("Histology_Type", None, None, False, True, ["Adeno", "Squamous", "Small_Cell"]),
]

# Seven continuous source features are informative through monotone sigmoid contributions.
DEFAULT_SIGMOID_SIGNAL_SPECS = [
    {
        "feature": "Age_at_Diagnosis",
        "cutoff_raw": 72.0,
        "slope_norm": 31.0,
        "weight": 2.10,
        "direction": "increasing",
        "description": "higher age raises risk with a sigmoid transition around 72 years",
    },
    {
        "feature": "Tumor_Diameter_mm",
        "cutoff_raw": 58.0,
        "slope_norm": 22.0,
        "weight": 1.00,
        "direction": "increasing",
        "description": "larger tumors raise risk with a sigmoid transition around 58 mm",
    },
    {
        "feature": "Albumin_Level_gL",
        "cutoff_raw": 38.5,
        "slope_norm": 14.0,
        "weight": 2,
        "direction": "decreasing",
        "description": "lower albumin raises risk with a sigmoid transition around 38.5 g/L",
    },
    {
        "feature": "Lymph_Nodes_Positive",
        "cutoff_raw": 8.0,
        "slope_norm": 42.0,
        "weight": 3,
        "direction": "increasing",
        "description": "more positive nodes raise risk with a sigmoid transition around 8 nodes",
    },
    {
        "feature": "Hemoglobin_Level",
        "cutoff_raw": 12.4,
        "slope_norm": 12.0,
        "weight": 0.90,
        "direction": "decreasing",
        "description": "lower hemoglobin raises risk with a sigmoid transition around 12.4 g/dL",
    },
    {
        "feature": "Systolic_BP",
        "cutoff_raw": 145.0,
        "slope_norm": 50.0,
        "weight": 2.80,
        "direction": "increasing",
        "description": "higher systolic blood pressure raises risk with a sigmoid transition around 145 mmHg",
    },
    {
        "feature": "LDH_U_L",
        "cutoff_raw": 285.0,
        "slope_norm": 11.0,
        "weight": 1.00,
        "direction": "increasing",
        "description": "higher LDH raises risk with a sigmoid transition around 285 U/L",
    },
]

DEFAULT_CATEGORICAL_SIGNAL_SPECS = [
    {
        "feature": "Cancer_Stage",
        "class_weights": {"Stage_I": 0.0, "Stage_II": 0.75, "Stage_III": 1.55},
        "description": "more advanced cancer stage increases risk",
    },
    {
        "feature": "Treatment_Group",
        "class_weights": {"Drug_A": 0.0, "Drug_B": 0.35, "Placebo": 0.95},
        "description": "placebo has the highest risk, Drug_A the lowest",
    },
    {
        "feature": "Histology_Type",
        "class_weights": {"Adeno": 0.0, "Squamous": 0.55, "Small_Cell": 1.10},
        "description": "small-cell histology has the highest risk",
    },
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _config_lookup() -> Dict[str, Tuple]:
    return {cfg[0]: cfg for cfg in CLINICAL_CONFIGS}


def _generate_core_features(
    n_samples: int,
    rng: np.random.Generator,
    n_informative: int,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    informative_configs = CLINICAL_CONFIGS[:n_informative]
    informative_names = [cfg[0] for cfg in informative_configs]
    x_latent = rng.normal(0.0, 1.0, (n_samples, n_informative))

    df_core = pd.DataFrame()
    for i, cfg in enumerate(informative_configs):
        name, v_min, v_max, is_int, is_cat, cats = cfg
        u = norm.cdf(x_latent[:, i])
        if is_cat:
            bins = np.linspace(0.0, 1.0, len(cats) + 1)
            indices = np.digitize(u, bins) - 1
            indices = np.clip(indices, 0, len(cats) - 1)
            df_core[name] = [cats[idx] for idx in indices]
        else:
            values = v_min + u * (v_max - v_min)
            if is_int:
                values = np.round(values)
            df_core[name] = values
    return df_core, x_latent, informative_names


def _generate_proxy_features(
    df_core: pd.DataFrame,
    x_latent: np.ndarray,
    informative_names: List[str],
    n_features: int,
    rng: np.random.Generator,
    proxy_correlation: float,
) -> pd.DataFrame:
    n_informative = len(informative_names)
    if n_features < n_informative:
        raise ValueError("n_features must be at least the number of informative features")

    n_proxy_total = n_features - n_informative
    proxy_counts = np.full(n_informative, n_proxy_total // n_informative, dtype=int)
    proxy_counts[: (n_proxy_total % n_informative)] += 1

    all_frames = [df_core]
    for i, n_proxies_for_feature in enumerate(proxy_counts):
        name_parent = informative_names[i]
        cfg_parent = CLINICAL_CONFIGS[i]
        for j in range(int(n_proxies_for_feature)):
            noise = rng.normal(0.0, 1.0, len(df_core))
            proxy_latent = (
                proxy_correlation * x_latent[:, i]
                + np.sqrt(max(1e-8, 1.0 - proxy_correlation ** 2)) * noise
            )
            u_proxy = norm.cdf(proxy_latent)
            proxy_name = f"{name_parent}_proxy_{j + 1}"

            if cfg_parent[4]:
                cats = cfg_parent[5]
                bins = np.linspace(0.0, 1.0, len(cats) + 1)
                indices = np.digitize(u_proxy, bins) - 1
                indices = np.clip(indices, 0, len(cats) - 1)
                all_frames.append(pd.DataFrame({proxy_name: [cats[idx] for idx in indices]}))
            else:
                v_min, v_max, is_int = cfg_parent[1:4]
                values = v_min + u_proxy * (v_max - v_min)
                if is_int:
                    values = np.round(values)
                all_frames.append(pd.DataFrame({proxy_name: values}))

    return pd.concat(all_frames, axis=1)


def _build_mixed_signal(
    df_core: pd.DataFrame,
    rng: np.random.Generator,
    sigmoid_specs: List[Dict],
    categorical_specs: List[Dict],
    nuisance_noise_scale: float,
) -> Tuple[np.ndarray, Dict]:
    cfg_lookup = _config_lookup()
    raw_signal = np.zeros(len(df_core), dtype=np.float64)
    component_matrix = {}
    sigmoid_specs_enriched: List[Dict] = []
    categorical_specs_enriched: List[Dict] = []

    for spec in sigmoid_specs:
        feature = spec["feature"]
        _, v_min, v_max, _, is_cat, _ = cfg_lookup[feature]
        if is_cat:
            raise ValueError(f"Sigmoid feature must be continuous, got categorical feature: {feature}")

        x = df_core[feature].to_numpy(dtype=np.float64)
        x_norm = (x - v_min) / max(v_max - v_min, 1e-8)
        cutoff_raw = float(spec["cutoff_raw"])
        cutoff_norm = (cutoff_raw - v_min) / max(v_max - v_min, 1e-8)
        slope_norm = float(spec["slope_norm"])
        direction = str(spec["direction"]).lower()
        weight = float(spec["weight"])

        if direction == "increasing":
            component = _sigmoid(slope_norm * (x_norm - cutoff_norm))
        elif direction == "decreasing":
            component = _sigmoid(slope_norm * (cutoff_norm - x_norm))
        else:
            raise ValueError(f"Unsupported sigmoid direction: {direction}")

        raw_signal += weight * component
        component_matrix[feature] = component
        sigmoid_specs_enriched.append(
            {
                **spec,
                "feature_min": float(v_min),
                "feature_max": float(v_max),
                "cutoff_norm": float(cutoff_norm),
            }
        )

    for spec in categorical_specs:
        feature = spec["feature"]
        _, _, _, _, is_cat, cats = cfg_lookup[feature]
        if not is_cat:
            raise ValueError(f"Categorical signal feature must be categorical, got continuous feature: {feature}")
        class_weights = {str(k): float(v) for k, v in spec["class_weights"].items()}
        if len(class_weights) > 3:
            raise ValueError(f"Categorical feature {feature} exceeds the requested maximum of 3 classes")
        if set(class_weights) - set(cats):
            raise ValueError(f"Categorical weights for {feature} include unknown levels")

        values = df_core[feature].astype(str).to_numpy()
        component = np.array([class_weights.get(v, 0.0) for v in values], dtype=np.float64)
        raw_signal += component
        component_matrix[feature] = component
        categorical_specs_enriched.append(
            {
                **spec,
                "class_weights": class_weights,
                "categories": list(cats),
            }
        )

    raw_signal += nuisance_noise_scale * rng.normal(0.0, 1.0, len(df_core))
    raw_signal = (raw_signal - raw_signal.mean()) / max(raw_signal.std(), 1e-8)

    metadata = {
        "signal_type": "mixed_continuous_sigmoid_and_categorical",
        "n_signal_features": len(sigmoid_specs) + len(categorical_specs),
        "signal_features": [spec["feature"] for spec in sigmoid_specs]
        + [spec["feature"] for spec in categorical_specs],
        "sigmoid_signal_features": [spec["feature"] for spec in sigmoid_specs],
        "linear_signal_features": [],
        "categorical_signal_features": [spec["feature"] for spec in categorical_specs],
        "n_sigmoid_signal_features": len(sigmoid_specs),
        "n_linear_signal_features": 0,
        "n_categorical_signal_features": len(categorical_specs),
        "sigmoid_signal_specs": sigmoid_specs_enriched,
        "linear_signal_specs": [],
        "categorical_signal_specs": categorical_specs_enriched,
        "component_matrix": component_matrix,
        "nuisance_noise_scale": float(nuisance_noise_scale),
    }
    return raw_signal.astype(np.float64), metadata


def _calibrate_censoring(
    event_time: np.ndarray,
    censor_rate: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float]:
    u_censor = rng.uniform(0.0, 1.0, len(event_time))
    low = 1e-6
    high = float(np.quantile(event_time, 0.98) * 5.0 + 1e-6)
    best_upper = high
    best_gap = float("inf")

    for _ in range(40):
        mid = 0.5 * (low + high)
        censor_time = u_censor * mid
        status = (event_time <= censor_time).astype(int)
        actual_censor_rate = 1.0 - float(status.mean())
        gap = abs(actual_censor_rate - censor_rate)
        if gap < best_gap:
            best_gap = gap
            best_upper = mid
        if actual_censor_rate > censor_rate:
            low = mid
        else:
            high = mid

    censor_time = u_censor * best_upper
    status = (event_time <= censor_time).astype(int)
    actual_censor_rate = 1.0 - float(status.mean())
    return censor_time, status, actual_censor_rate


def _simulate_outcomes_from_signal(
    raw_signal: np.ndarray,
    censor_rate: float,
    target_c_index: float,
    rng: np.random.Generator,
    lambda_val: float = 0.01,
    rho: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    scale = max(0.10, 1.50 * (target_c_index - 0.5) / 0.30)
    log_hazard = scale * raw_signal

    u_event = rng.uniform(0.0, 1.0, len(raw_signal))
    event_time = (-np.log(u_event) / (lambda_val * np.exp(log_hazard))) ** (1.0 / rho)
    censor_time, status, actual_censor_rate = _calibrate_censoring(event_time, censor_rate, rng)
    observed_time = np.minimum(event_time, censor_time)
    actual_c_index = concordance_index(observed_time, -log_hazard, status)

    metadata = {
        "log_hazard_scale": float(scale),
        "lambda_val": float(lambda_val),
        "rho": float(rho),
        "actual_censor_rate": float(actual_censor_rate),
        "actual_oracle_c_index": float(actual_c_index),
    }
    return observed_time, status, event_time, log_hazard, metadata


def generate_survival_data(
    n_samples: int = 3000,
    n_features: int = 100,
    n_informative: int = 10,
    censor_rate: float = 0.3,
    target_c_index: float = 0.8,
    test_size: float = 0.3,
    seed: int = 42,
    return_metadata: bool = False,
    proxy_correlation: float = 0.35,
    nuisance_noise_scale: float = 0.15,
):
    """Generate a mixed-signal synthetic survival data set for ScoreNet.

    The 10 truly informative source variables are:
      - 7 continuous features with monotone sigmoid risk contributions
      - 3 categorical features with additive class-specific risk contributions

    The remaining features are mildly correlated proxy variables generated from the same latent
    factors. The defaults match the requested experimental setup: 3000 samples, 100 total raw
    features, and a 30% test split.
    """
    if n_informative != 10:
        raise ValueError("This mixed simulation is defined for exactly 10 informative source features")
    if n_informative > len(CLINICAL_CONFIGS):
        raise ValueError(f"n_informative={n_informative} exceeds available configs={len(CLINICAL_CONFIGS)}")

    rng = np.random.default_rng(seed)

    df_core, x_latent, informative_names = _generate_core_features(
        n_samples=n_samples,
        rng=rng,
        n_informative=n_informative,
    )
    df = _generate_proxy_features(
        df_core=df_core,
        x_latent=x_latent,
        informative_names=informative_names,
        n_features=n_features,
        rng=rng,
        proxy_correlation=proxy_correlation,
    )

    raw_signal, signal_metadata = _build_mixed_signal(
        df_core=df_core,
        rng=rng,
        sigmoid_specs=[spec.copy() for spec in DEFAULT_SIGMOID_SIGNAL_SPECS],
        categorical_specs=[spec.copy() for spec in DEFAULT_CATEGORICAL_SIGNAL_SPECS],
        nuisance_noise_scale=nuisance_noise_scale,
    )
    observed_time, status, event_time, log_hazard, outcome_metadata = _simulate_outcomes_from_signal(
        raw_signal=raw_signal,
        censor_rate=censor_rate,
        target_c_index=target_c_index,
        rng=rng,
    )

    df = pd.concat(
        [
            df.reset_index(drop=True),
            pd.DataFrame(
                {
                    "time": observed_time,
                    "status": status,
                    "id": np.arange(n_samples),
                    "true_log_hazard": log_hazard,
                }
            ),
        ],
        axis=1,
    )

    print("--- Simulation Metadata ---")
    print("Signal form: 7 sigmoid continuous + 3 categorical source features")
    print(f"Target Censor Rate: {censor_rate:.2%} | Actual: {outcome_metadata['actual_censor_rate']:.2%}")
    print(f"Target C-index: {target_c_index:.4f} | Actual (Oracle): {outcome_metadata['actual_oracle_c_index']:.4f}")
    print(f"Proxy correlation strength: {proxy_correlation:.2f}")
    print("Sigmoid signal features:")
    for spec in signal_metadata["sigmoid_signal_specs"]:
        direction_text = ">=" if spec["direction"].lower() == "increasing" else "<="
        print(
            f"- {spec['feature']} with sigmoid transition near {spec['cutoff_raw']} "
            f"({direction_text}), slope={spec['slope_norm']}, weight={spec['weight']}"
        )
    print("Categorical signal features:")
    for spec in signal_metadata["categorical_signal_specs"]:
        print(f"- {spec['feature']} class weights: {spec['class_weights']}")

    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["status"],
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    metadata = {
        "simulation_mode": "mixed_sigmoid_categorical_signal",
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_informative_generated": int(n_informative),
        "proxy_correlation": float(proxy_correlation),
        "target_censor_rate": float(censor_rate),
        "target_c_index": float(target_c_index),
        "actual_censor_rate": float(outcome_metadata["actual_censor_rate"]),
        "actual_oracle_c_index": float(outcome_metadata["actual_oracle_c_index"]),
        "log_hazard_scale": float(outcome_metadata["log_hazard_scale"]),
        "signal_type": signal_metadata["signal_type"],
        "n_signal_features": int(signal_metadata["n_signal_features"]),
        "signal_features": signal_metadata["signal_features"],
        "sigmoid_signal_features": signal_metadata["sigmoid_signal_features"],
        "linear_signal_features": signal_metadata["linear_signal_features"],
        "categorical_signal_features": signal_metadata["categorical_signal_features"],
        "n_sigmoid_signal_features": int(signal_metadata["n_sigmoid_signal_features"]),
        "n_linear_signal_features": int(signal_metadata["n_linear_signal_features"]),
        "n_categorical_signal_features": int(signal_metadata["n_categorical_signal_features"]),
        "sigmoid_signal_specs": signal_metadata["sigmoid_signal_specs"],
        "linear_signal_specs": signal_metadata["linear_signal_specs"],
        "categorical_signal_specs": signal_metadata["categorical_signal_specs"],
        "nuisance_noise_scale": float(signal_metadata["nuisance_noise_scale"]),
        "full_dataframe_shape": [int(df.shape[0]), int(df.shape[1])],
        "event_time_summary": {
            "min": float(np.min(event_time)),
            "median": float(np.median(event_time)),
            "max": float(np.max(event_time)),
        },
    }

    if return_metadata:
        return df_train, df_test, metadata
    return df_train, df_test


if __name__ == "__main__":
    train_data, test_data, meta = generate_survival_data(
        n_samples=3000,
        n_features=100,
        censor_rate=0.3,
        target_c_index=0.8,
        test_size=0.3,
        seed=42,
        return_metadata=True,
    )
    print("\nTrain preview:")
    preview_cols = [
        "Age_at_Diagnosis",
        "Tumor_Diameter_mm",
        "Albumin_Level_gL",
        "Lymph_Nodes_Positive",
        "Hemoglobin_Level",
        "Systolic_BP",
        "LDH_U_L",
        "Cancer_Stage",
        "Treatment_Group",
        "Histology_Type",
        "time",
        "status",
    ]
    print(train_data[preview_cols].head())
    print("\nMetadata JSON:")
    print(json.dumps(meta, indent=2))
