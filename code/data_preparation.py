from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from guanrank import guanrank_labels


PROTECTED_COLS = ["time", "status", "id", "true_log_hazard"]


def _fit_feature_processor(df_train: pd.DataFrame, n_bins: int = 10):
    feature_cols = [c for c in df_train.columns if c not in PROTECTED_COLS]
    cont_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_train[c])]
    cat_cols = [c for c in feature_cols if c not in cont_cols]

    scaler = MinMaxScaler()
    scaler.fit(df_train[cont_cols])

    x_train_cont = pd.DataFrame(
        scaler.transform(df_train[cont_cols]),
        columns=cont_cols,
        index=df_train.index,
    )

    quantile_grid = np.linspace(0.05, 0.95, n_bins)
    cont_feature_bins = np.stack(
        [np.quantile(x_train_cont[col].values, quantile_grid) for col in cont_cols],
        axis=0,
    ).astype(np.float32)

    category_levels = {}
    for col in cat_cols:
        category_levels[col] = sorted(df_train[col].dropna().astype(str).unique().tolist())

    return {
        "feature_cols": feature_cols,
        "cont_cols": cont_cols,
        "cat_cols": cat_cols,
        "scaler": scaler,
        "cont_feature_bins": cont_feature_bins,
        "category_levels": category_levels,
    }


def _encode_split(df_split: pd.DataFrame, processor) -> pd.DataFrame:
    cont_cols = processor["cont_cols"]
    cat_cols = processor["cat_cols"]
    scaler = processor["scaler"]
    category_levels = processor["category_levels"]

    x_cont = pd.DataFrame(
        scaler.transform(df_split[cont_cols]),
        columns=cont_cols,
        index=df_split.index,
    )

    if cat_cols:
        encoded_parts = []
        for col in cat_cols:
            categories = category_levels[col]
            values = pd.Categorical(df_split[col].astype(str), categories=categories)
            dummies = pd.get_dummies(values, prefix=col, prefix_sep="_", dtype=int)
            expected_columns = [f"{col}_{cat}" for cat in categories]
            dummies = dummies.reindex(columns=expected_columns, fill_value=0)
            dummies.index = df_split.index
            encoded_parts.append(dummies)
        x_bin = pd.concat(encoded_parts, axis=1) if encoded_parts else pd.DataFrame(index=df_split.index)
    else:
        x_bin = pd.DataFrame(index=df_split.index)

    x_final = pd.concat([x_cont, x_bin], axis=1)
    return x_final


def _build_bundle(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, processor):
    df_train = df_train.reset_index(drop=True).copy()
    df_val = df_val.reset_index(drop=True).copy()
    df_test = df_test.reset_index(drop=True).copy()

    x_train_final = _encode_split(df_train, processor).reset_index(drop=True)
    x_val_final = _encode_split(df_val, processor).reset_index(drop=True)
    x_test_final = _encode_split(df_test, processor).reset_index(drop=True)

    y_train, train_expected_time = guanrank_labels(df_train["time"].values, df_train["status"].values)
    y_val, val_expected_time = guanrank_labels(df_val["time"].values, df_val["status"].values)
    y_test, test_expected_time = guanrank_labels(df_test["time"].values, df_test["status"].values)

    cont_cols = processor["cont_cols"]
    cat_cols = processor["cat_cols"]
    category_levels = processor["category_levels"]
    binary_feature_names = [c for c in x_train_final.columns if c not in cont_cols]
    binary_feature_metadata = []
    for col in cat_cols:
        for category in category_levels[col]:
            binary_feature_metadata.append(
                {
                    "processed_feature_name": f"{col}_{category}",
                    "source_feature": col,
                    "source_value": str(category),
                }
            )
    cont_feat_index = list(range(len(cont_cols)))
    binary_feat_index = list(range(len(cont_cols), x_train_final.shape[1]))
    binary_mapping = {i: col for i, col in zip(binary_feat_index, binary_feature_names)}

    print("\n--- Feature Index Mapping Summary ---")
    if cont_feat_index:
        print(
            f"Continuous Features: Indices {cont_feat_index[0]} to {cont_feat_index[-1]} "
            f"(Count: {len(cont_feat_index)})"
        )
    else:
        print("Continuous Features: None")
    if binary_feat_index:
        print(
            f"Binary Features:     Indices {binary_feat_index[0]} to {binary_feat_index[-1]} "
            f"(Count: {len(binary_feat_index)})"
        )
    else:
        print("Binary Features: None")

    bundle = {
        "X_train": torch.tensor(x_train_final.values, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.float32),
        "train_time": torch.tensor(df_train["time"].values, dtype=torch.float32),
        "train_status": torch.tensor(df_train["status"].values, dtype=torch.float32),
        "X_val": torch.tensor(x_val_final.values, dtype=torch.float32),
        "y_val": torch.tensor(y_val, dtype=torch.float32),
        "val_time": torch.tensor(df_val["time"].values, dtype=torch.float32),
        "val_status": torch.tensor(df_val["status"].values, dtype=torch.float32),
        "X_test": torch.tensor(x_test_final.values, dtype=torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.float32),
        "test_time": torch.tensor(df_test["time"].values, dtype=torch.float32),
        "test_status": torch.tensor(df_test["status"].values, dtype=torch.float32),
        # Compatibility aliases expected by the original main_fun.py
        "y_test_time": torch.tensor(df_test["time"].values, dtype=torch.float32),
        "y_test_status": torch.tensor(df_test["status"].values, dtype=torch.float32),
        "X_train_df": x_train_final,
        "X_val_df": x_val_final,
        "X_test_df": x_test_final,
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
        "cont_feat_index": cont_feat_index,
        "binary_feat_index": binary_feat_index,
        "binary_mapping": binary_mapping,
        "cont_feature_bins": processor["cont_feature_bins"],
        "cont_columns": cont_cols,
        "binary_feature_names": binary_feature_names,
        "binary_feature_metadata": binary_feature_metadata,
        "raw_feature_names": cont_cols + cat_cols,
        "all_feature_names": list(x_train_final.columns),
        "cont_feature_mins": processor["scaler"].data_min_.astype(np.float32),
        "cont_feature_maxs": processor["scaler"].data_max_.astype(np.float32),
        "train_expected_time": torch.tensor(train_expected_time, dtype=torch.float32),
        "val_expected_time": torch.tensor(val_expected_time, dtype=torch.float32),
        "test_expected_time": torch.tensor(test_expected_time, dtype=torch.float32),
        "status_rates": {
            "train": float(df_train["status"].mean()),
            "val": float(df_val["status"].mean()),
            "test": float(df_test["status"].mean()),
        },
    }
    return bundle


def prepare_data_splits_for_scorenet(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, n_bins: int = 10):
    """Prepare train/validation/test splits for ScoreNet.

    Output layout is:
        [scaled continuous block | one-hot categorical block]
    """
    print("Applying GuanRank transformation to train/validation/test splits...")
    processor = _fit_feature_processor(df_train.reset_index(drop=True), n_bins=n_bins)
    return _build_bundle(df_train, df_val, df_test, processor)


def prepare_data_for_scorenet(df_train: pd.DataFrame, df_test: pd.DataFrame, n_bins: int = 10):
    """Compatibility wrapper for code that expects only train/test output."""
    df_train = df_train.reset_index(drop=True).copy()
    df_test = df_test.reset_index(drop=True).copy()
    processor = _fit_feature_processor(df_train, n_bins=n_bins)

    x_train_final = _encode_split(df_train, processor).reset_index(drop=True)
    x_test_final = _encode_split(df_test, processor).reset_index(drop=True)

    y_train, train_expected_time = guanrank_labels(df_train["time"].values, df_train["status"].values)
    y_test, test_expected_time = guanrank_labels(df_test["time"].values, df_test["status"].values)

    cont_cols = processor["cont_cols"]
    cat_cols = processor["cat_cols"]
    category_levels = processor["category_levels"]
    binary_feature_names = [c for c in x_train_final.columns if c not in cont_cols]
    binary_feature_metadata = []
    for col in cat_cols:
        for category in category_levels[col]:
            binary_feature_metadata.append(
                {
                    "processed_feature_name": f"{col}_{category}",
                    "source_feature": col,
                    "source_value": str(category),
                }
            )
    cont_feat_index = list(range(len(cont_cols)))
    binary_feat_index = list(range(len(cont_cols), x_train_final.shape[1]))
    binary_mapping = {i: col for i, col in zip(binary_feat_index, binary_feature_names)}

    return {
        "X_train": torch.tensor(x_train_final.values, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.float32),
        "X_test": torch.tensor(x_test_final.values, dtype=torch.float32),
        "test_time": torch.tensor(df_test["time"].values, dtype=torch.float32),
        "test_status": torch.tensor(df_test["status"].values, dtype=torch.float32),
        "X_train_df": x_train_final,
        "X_test_df": x_test_final,
        "df_train": df_train,
        "df_test": df_test,
        "cont_feat_index": cont_feat_index,
        "binary_feat_index": binary_feat_index,
        "binary_mapping": binary_mapping,
        "cont_feature_bins": processor["cont_feature_bins"],
        "cont_columns": cont_cols,
        "binary_feature_names": binary_feature_names,
        "binary_feature_metadata": binary_feature_metadata,
        "raw_feature_names": cont_cols + cat_cols,
        "all_feature_names": list(x_train_final.columns),
        "cont_feature_mins": processor["scaler"].data_min_.astype(np.float32),
        "cont_feature_maxs": processor["scaler"].data_max_.astype(np.float32),
        "train_expected_time": torch.tensor(train_expected_time, dtype=torch.float32),
        "test_expected_time": torch.tensor(test_expected_time, dtype=torch.float32),
        "y_test_time": torch.tensor(df_test["time"].values, dtype=torch.float32),
        "y_test_status": torch.tensor(df_test["status"].values, dtype=torch.float32),
    }
