import json
import time
import pandas as pd
from pathlib import Path
from pymatgen.core.structure import Structure

# Import all the required featurizers from mattervial
from mattervial.featurizers.composition import RoostModelFeaturizer
from mattervial.featurizers.structure import MVLFeaturizer, l_OFM_v1, l_MM_v1

# =============================================================================
# Helper Function for Generalized Featurization
# =============================================================================

def compute_and_save_all_features(
    df: pd.DataFrame, 
    filename: Path, 
    featurizers: dict,
    structure_col: str = 'pymatgen_structure',
    composition_col: str = 'formula',
    id_col: str = 'material_id'
):
    """
    Computes features from multiple mattervial featurizers and saves them to a single CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing structures, formulas, and material IDs.
        filename (Path): Path for the output CSV file.
        featurizers (dict): A dictionary containing initialized featurizer objects.
        structure_col (str): Column name for pymatgen Structure objects.
        composition_col (str): Column name for chemical formula strings.
        id_col (str): Column name for unique material identifiers.
    """
    # --- 1. Validate and Prepare Input Data ---
    required_cols = [structure_col, composition_col, id_col]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain all required columns: {required_cols}")

    # Filter out rows with null structures or formulas
    valid_mask = df[structure_col].notnull() & df[composition_col].notnull()
    if not valid_mask.any():
        print(f"⚠️  Warning: No valid structures or compositions found. Skipping file generation for {filename.name}.")
        return
        
    df_valid = df.loc[valid_mask].copy()
    
    structures_series = df_valid[structure_col]
    compositions_series = df_valid[composition_col]
    material_ids = df_valid[id_col].tolist()
    
    print(f"Featurizing {len(df_valid)} valid entries...")
    
    # --- 2. Compute All Features ---
    feature_dfs = []

    # Structure-based featurizers
    for name, featurizer in featurizers.items():
        if name in ['l_ofm', 'l_mm', 'mvl']:
            print(f"Computing {name.upper()} features...")
            features = featurizer.get_features(structures_series)
            feature_dfs.append(pd.DataFrame(features))

    # Composition-based featurizers
    for name, featurizer in featurizers.items():
        if 'roost' in name:
            print(f"Computing {name.upper()} features...")
            features = featurizer.get_features(compositions_series)
            feature_dfs.append(pd.DataFrame(features))

    # --- 3. Combine and Save Results ---
    print("Combining all features...")
    all_features_df = pd.concat(feature_dfs, axis=1)
    
    # Add the material IDs as the first column
    all_features_df.insert(0, id_col, material_ids)
    
    print(f"Generated combined feature matrix with shape: {all_features_df.shape}")
    
    # Ensure the output directory exists and save the file
    filename.parent.mkdir(parents=True, exist_ok=True)
    all_features_df.to_csv(filename, index=False)
    print(f"✅ Successfully saved combined features to '{filename}'")


# =============================================================================
# Main Execution Block
# =============================================================================

if __name__ == '__main__':
    # --- 1. Global Configurations ---
    
    # Instantiate all featurizers once to be reused for all datasets
    print("Initializing mattervial featurizers...")
    all_featurizers = {
        "l_ofm": l_OFM_v1,
        "l_mm": l_MM_v1,
        "mvl": MVLFeaturizer(),
        "roost_mpgap": RoostModelFeaturizer(model_type='mpgap'),
        "roost_oqmd": RoostModelFeaturizer(model_type='oqmd_eform'),
    }
    print("Featurizers initialized.")
    
    # Define a central directory to store all generated feature files
    output_dir = Path('./mattervial_noorb_featurized_data')

    # --- 2. Define Dataset Processing Tasks ---
    tasks = [
        # {
        #     "csv_path": "./featurize_with_MM/m2ax_223.csv",
        #     "structure_col": "structure",
        #     "dataset_name": "m2ax_223"
        # },
        # {
        #     "csv_path": "tholander_nitrides_12815.csv",
        #     "structure_col": "final_structure_json",
        #     "dataset_name": "tholander_nitrides_12815"
        # },
        # {
        #     "csv_path": "boltztrap_mp_8924.csv",
        #     "structure_col": "structure_json",
        #     "dataset_name": "boltztrap_mp_8924"
        # },
        # {
        #     "csv_path": "./featurize_with_MM/double_perovskites_gap_1306.csv",
        #     "structure_col": "structure",
        #     "dataset_name": "double_perovskites_gap_1306"
        # },
        {
            "csv_path": "./featurize_with_MM/double_perovskites_gap_1306_optimized.csv",
            "structure_col": "structure",
            "dataset_name": "double_perovskites_gap_1306_optimized"
        }
    ]

    # --- 3. Loop Through and Process Each Task ---
    for task in tasks:
        print("\n" + "="*80)
        print(f"🚀 STARTING TASK: {task['dataset_name']}")
        print("="*80 + "\n")
        
        start_time = time.perf_counter()
        
        try:
            # Step A: Load data
            print(f"Loading data from '{task['csv_path']}'...")
            csv_path = Path(task['csv_path'])
            if not csv_path.exists():
                raise FileNotFoundError(f"Source file not found at '{csv_path}'")
            df = pd.read_csv(csv_path)

            # Step B: Parse structure objects
            print(f"Parsing structure data from '{task['structure_col']}' column...")
            df['pymatgen_structure'] = df[task['structure_col']].apply(
                lambda s: Structure.from_dict(json.loads(s)) if pd.notna(s) else None
            )

            # Step C: Generate formula strings for composition-based featurizers
            print("Generating chemical formulas from structures...")
            df['formula'] = df['pymatgen_structure'].apply(
                lambda s: s.composition.reduced_formula if pd.notna(s) else None
            )

            # Step D: Create a unique 'material_id' for each row
            df['material_id'] = [f"{task['dataset_name']}_{i}" for i in df.index]
            print("Data loading and preparation complete.")

            # Step E: Compute all features and save the results
            output_filename = output_dir / f"all_features_{task['dataset_name']}.csv"
            compute_and_save_all_features(
                df=df,
                filename=output_filename,
                featurizers=all_featurizers
            )

        except FileNotFoundError as e:
            print(f"\n❌ ERROR: {e}")
            print("Please ensure the CSV file is in the correct directory. Skipping task.")
            continue
        except Exception as e:
            print(f"\n❌ An unexpected error occurred during task '{task['dataset_name']}': {e}")
            print("Skipping task.")
            continue

        end_time = time.perf_counter()
        duration = end_time - start_time
        
        print("\n" + "-"*80)
        print(f"🎉 COMPLETED TASK: {task['dataset_name']}")
        print(f"🕒 Total wall time for this task: {duration:.2f} seconds")
        print("-" * 80 + "\n")