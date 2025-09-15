import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ----------------------------------------------------------------
# Helper Functions (from your original script)
# ----------------------------------------------------------------

def log_message(message, level="INFO"):
    """
    Print a timestamped log message with specified level.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def compute_pca_features(df, feature_columns, target_column,
                         id_column="material_id",
                         n_components=10,
                         include_target_in_pca=True,
                         target_weight=1.0,
                         random_state=42):
    """
    Compute PCA features based on the provided dataframe.
    """
    np.random.seed(random_state)
    log_message(f"Computing PCA using {len(feature_columns)} features for target '{target_column}' ...")

    X = df[feature_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    log_message(f"Feature matrix shape after scaling: {X_scaled.shape}")

    if include_target_in_pca:
        log_message(f"Including target '{target_column}' with weight {target_weight}...")
        y = df[target_column].values.reshape(-1, 1)
        y_scaled = StandardScaler().fit_transform(y)
        X_augmented = np.hstack([X_scaled, target_weight * y_scaled])
        log_message(f"Augmented feature matrix shape: {X_augmented.shape}")
    else:
        X_augmented = X_scaled

    n_components_actual = min(n_components, X_augmented.shape[1])
    log_message(f"Performing PCA with {n_components_actual} components")
    pca = PCA(n_components=n_components_actual, random_state=random_state)
    X_pca = pca.fit_transform(X_augmented)
    log_message(f"Explained variance ratio (sum): {pca.explained_variance_ratio_.sum():.3f}")

    pca_columns = [f"PC_{i+1}" for i in range(X_pca.shape[1])]
    pca_df = pd.DataFrame(X_pca, columns=pca_columns)
    pca_df[id_column] = df[id_column].values

    return pca_df

def pca_based_sampler(pca_df, n_bins=10, alpha=1.0,
                      random_state=42, id_column="material_id",
                      max_pca_components=None):
    """
    Perform sampling on the provided PCA features dataframe.
    """
    log_message(f"Sampling from PCA features using random seed {random_state}")
    np.random.seed(random_state)

    pca_columns = [col for col in pca_df.columns if col.startswith("PC_")]
    if max_pca_components is not None:
        pca_columns = pca_columns[:int(max_pca_components)]

    X_pca = pca_df[pca_columns].values
    log_message(f"Sampling using {len(pca_columns)} PCA components on {len(pca_df)} samples")

    bins = []
    for i in range(X_pca.shape[1]):
        component = X_pca[:, i]
        bin_edges = np.linspace(component.min(), component.max(), n_bins + 1)
        bin_indices = np.digitize(component, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        bins.append(bin_indices)
    bins = np.stack(bins, axis=1)

    bin_keys = ["_".join(map(str, b)) for b in bins]
    bin_to_indices = {}
    for idx, key in enumerate(bin_keys):
        bin_to_indices.setdefault(key, []).append(idx)
    log_message(f"Found {len(bin_to_indices)} unique bins during sampling.")

    selected_indices = []
    for key, indices in bin_to_indices.items():
        n_samples_bin = max(1, int(len(indices) * alpha))
        sampled = np.random.choice(indices, size=n_samples_bin, replace=False)
        selected_indices.extend(sampled)
    log_message(f"Total selected samples: {len(selected_indices)}")

    selected_ids = pca_df.iloc[selected_indices][id_column].tolist()
    return selected_ids

# ----------------------------------------------------------------
# Main Function to Generate SISSO Input
# ----------------------------------------------------------------
def create_sisso_input_from_selected_features(
    input_csv_path: str,
    target_column: str,
    suffix: str = "0",
    id_column: str = "material_id",
    output_folder: str = "sisso_run",
    pca_n_components: int = 20,
    pca_include_target: bool = True,
    pca_target_weight: float = 3.0,
    sampler_n_bins: int = 5,
    sampler_alpha: float = 0.01,
    sampler_max_pca_components: int = 3,
    random_seed: int = 42,
):
    """
    Generates a SISSO-ready sampled dataset from a CSV with pre-selected features.
    """
    log_message("="*60)
    log_message("STARTING SISSO SAMPLE GENERATION FROM PRE-SELECTED FEATURES")
    log_message(f"Input Dataset: {input_csv_path}")
    log_message(f"Target Column: {target_column}")
    log_message("="*60)

    # 1. Load the data
    if not os.path.exists(input_csv_path):
        log_message(f"Input data file not found: {input_csv_path}", "ERROR")
        return

    try:
        df = pd.read_csv(input_csv_path)
        log_message(f"Successfully loaded dataset with shape: {df.shape}")
    except Exception as e:
        log_message(f"Error loading dataset: {e}", "ERROR")
        return
    output_folder = f"{suffix}_{output_folder}"
    os.makedirs(output_folder, exist_ok=True)

    # 2. Identify feature columns (all columns except id and target)
    feature_columns = [col for col in df.columns if col not in [id_column, target_column]]
    log_message(f"Identified {len(feature_columns)} feature columns.")

    # 3. Use PCA on the features to obtain sampled material_ids.
    pca_df = compute_pca_features(
        df=df,
        feature_columns=feature_columns,
        target_column=target_column,
        id_column=id_column,
        n_components=pca_n_components,
        include_target_in_pca=pca_include_target,
        target_weight=pca_target_weight,
        random_state=random_seed
    )

    sampled_ids = pca_based_sampler(
        pca_df=pca_df,
        n_bins=sampler_n_bins,
        alpha=sampler_alpha,
        random_state=random_seed,
        id_column=id_column,
        max_pca_components=sampler_max_pca_components
    )
    log_message(f"Sampled {len(sampled_ids)} material IDs.")

    # 4. Filter the original dataframe to keep only the sampled rows.
    final_df = df[df[id_column].isin(sampled_ids)].copy()
    log_message(f"Number of rows after sampling: {len(final_df)}")
    log_message(f"Target distribution in sampled dataset:\n{final_df[target_column].describe()}", "INFO")

    # 5. Prepare SISSO input format
    # Drop the material_id and rename the target column
    final_df = final_df.drop(columns=[id_column])
    final_df = final_df.rename(columns={target_column: "target"})

    # Create the feature mapping dictionary and rename columns
    mapping_dict = {}
    rename_mapping = {}
    # Use the original feature_columns list to preserve order
    for i, orig_col in enumerate(feature_columns):
        new_name = f"feature_{i+1}"
        mapping_dict[new_name] = orig_col
        rename_mapping[orig_col] = new_name
    
    final_df = final_df.rename(columns=rename_mapping)
    
    # Reorder columns to have target first, then features
    final_df = final_df[["target"] + list(mapping_dict.keys())]

    # Add a sample_id column
    final_df.reset_index(drop=True, inplace=True)
    final_df.index.name = "sample_id"
    final_df.reset_index(inplace=True)

    # 6. Save the output files
    sampled_output_path = os.path.join(output_folder, "SISSO_sampled_target_{suffix}.csv")
    mapping_json_path = os.path.join(output_folder, "feature_mapping.json")

    try:
        with open(mapping_json_path, "w") as jf:
            json.dump(mapping_dict, jf, indent=4)
        log_message(f"Feature mapping saved to {mapping_json_path}")
    except Exception as e:
        log_message(f"Error saving feature mapping: {e}", "ERROR")

    try:
        with open(sampled_output_path, "w") as outf:
            outf.write("#") # SISSO comment character
            final_df.to_csv(outf, index=False, header=True)
        log_message(f"SISSO-ready sampled data saved to {sampled_output_path}\n")
    except Exception as e:
        log_message(f"Error saving SISSO sampled data: {e}", "ERROR")

    log_message("="*60)
    log_message("SISSO SAMPLE PREPARATION COMPLETE")
    log_message("Next step: Configure and run your SISSO calculation using the generated files.")
    log_message("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare SISSO input from a CSV with pre-selected features.")
    parser.add_argument('--input-csv', type=str, required=True,
                        help='Path to the input CSV file containing id, target, and features.')
    parser.add_argument('--target-column', type=str, required=True,
                        help='The name of the target column in the CSV file.')
    parser.add_argument('--suffix', type=str, default='0',
                        help='The suffix for the data file.')
    parser.add_argument('--id-column', type=str, default='material_id',
                        help='The name of the material identifier column.')
    parser.add_argument('--output-folder', type=str, default='sisso_run',
                        help='Folder where the output files will be saved.')
    parser.add_argument('--sampler-alpha', type=float, default=0.01,
                        help='Fraction of samples to select per bin (at least one).')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    # Add other parameters if you want to control them from the command line
    
    args = parser.parse_args()

    create_sisso_input_from_selected_features(
        input_csv_path=args.input_csv,
        target_column=args.target_column,
        suffix=args.suffix,
        id_column=args.id_column,
        output_folder=args.output_folder,
        sampler_alpha=args.sampler_alpha,
        random_seed=args.random_seed
    )