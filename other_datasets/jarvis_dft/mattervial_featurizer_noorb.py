import json
import time
import pandas as pd
from pathlib import Path
from pymatgen.core.structure import Structure
import argparse
import sys
import numpy as np

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
# New Function to Combine Chunks
# =============================================================================

def combine_chunk_files(dataset_name: str, n_chunks: int, output_dir: Path):
    """
    Finds, validates, and combines featurized chunk files into a single CSV.

    Args:
        dataset_name (str): The base name of the dataset.
        n_chunks (int): The total number of chunks that were created.
        output_dir (Path): The directory where chunk files are stored.
    """
    print("\n" + "="*80)
    print(f"🤝 STARTING COMBINATION TASK FOR: {dataset_name}")
    print("="*80 + "\n")

    chunk_files = []
    missing_files = False
    for i in range(1, n_chunks + 1):
        chunk_file = output_dir / f"all_features_{dataset_name}_chunk_{i}_of_{n_chunks}.csv"
        if chunk_file.exists():
            chunk_files.append(chunk_file)
        else:
            print(f"❌ ERROR: Missing chunk file: {chunk_file}")
            missing_files = True
    
    if missing_files:
        print("\nCannot combine files because one or more chunks are missing. Aborting.")
        sys.exit(1)

    if not chunk_files:
        print("No chunk files found to combine. Aborting.")
        sys.exit(1)

    print(f"Found {len(chunk_files)} chunk files. Reading and concatenating...")
    
    try:
        df_list = [pd.read_csv(f) for f in chunk_files]
        combined_df = pd.concat(df_list, ignore_index=True)

        final_filename = output_dir / f"all_features_{dataset_name}.csv"
        combined_df.to_csv(final_filename, index=False)
        
        print(f"\n🎉 Successfully combined all chunks into '{final_filename}'")
        print(f"Combined DataFrame has shape: {combined_df.shape}")
        
        # Optional: Clean up chunk files after combining
        # for f in chunk_files:
        #     f.unlink()
        # print("Cleaned up individual chunk files.")

    except Exception as e:
        print(f"\n❌ An error occurred during the combination process: {e}")
        sys.exit(1)



# =============================================================================
# Main Execution Block
# =============================================================================

if __name__ == '__main__':
    # --- 0. Setup Command-Line Argument Parser ---
    parser = argparse.ArgumentParser(description="Featurize crystal structures in chunks using mattervial.")
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to process (must match a key in the 'tasks' list).")
    parser.add_argument("--n_chunks", type=int, default=1, help="Total number of chunks to split the data into.")
    parser.add_argument("--chunk", type=int, default=1, help="The specific chunk number to process (from 1 to n_chunks).")
    parser.add_argument("--combine_chunks", action='store_true', help="If specified, combine all existing chunks for the dataset instead of featurizing.")
    
    args = parser.parse_args()

    # --- 1. Global Configurations ---
    print("Initializing mattervial featurizers...")
    all_featurizers = {
        "l_ofm": l_OFM_v1,
        "l_mm": l_MM_v1,
        "mvl": MVLFeaturizer(),
        "roost_mpgap": RoostModelFeaturizer(model_type='mpgap'),
        "roost_oqmd": RoostModelFeaturizer(model_type='oqmd_eform'),
    }
    print("Featurizers initialized.")
    
    output_dir = Path('./mv_noorb_featurized_data')

    # --- 2. Define Dataset Processing Tasks ---
    tasks = {
        "jarvis_dft_3d": {
            "csv_path": "./jarvis_dft_3d_with_structures.csv",
            "structure_col": "structure",
            "id_col": "jid"  # <<< CHANGE 1: Specify your ID column here
        }
    }

    # --- 3. Execute Task Based on Arguments ---
    
    if args.dataset_name not in tasks:
        print(f"❌ ERROR: Dataset '{args.dataset_name}' not found in the defined tasks.")
        sys.exit(1)
    task = tasks[args.dataset_name]
    task["dataset_name"] = args.dataset_name

    if args.combine_chunks:
        if args.n_chunks <= 1:
            print("❌ ERROR: '--n_chunks' must be greater than 1 when combining.")
            sys.exit(1)
        combine_chunk_files(task['dataset_name'], args.n_chunks, output_dir)
        sys.exit(0)

    print("\n" + "="*80)
    print(f"🚀 STARTING TASK: {task['dataset_name']} (Chunk {args.chunk} of {args.n_chunks})")
    print("="*80 + "\n")
    
    start_time = time.perf_counter()
    
    try:
        # Step A: Load data
        print(f"Loading data from '{task['csv_path']}'...")
        csv_path = Path(task['csv_path'])
        if not csv_path.exists():
            raise FileNotFoundError(f"Source file not found at '{csv_path}'")
        df = pd.read_csv(csv_path)

        # Step B: Slice the DataFrame for the current chunk
        n_total = len(df)
        if args.chunk > args.n_chunks or args.chunk < 1:
            raise ValueError(f"Chunk number must be between 1 and {args.n_chunks}.")
        
        split_indices = np.array_split(range(n_total), args.n_chunks)
        chunk_indices = split_indices[args.chunk - 1]
        
        if len(chunk_indices) == 0:
            print(f"⚠️  Warning: Chunk {args.chunk} is empty. Exiting.")
            sys.exit(0)

        df_chunk = df.iloc[chunk_indices].copy() # Use .copy() to avoid SettingWithCopyWarning
        print(f"Processing {len(df_chunk)} rows from index {chunk_indices[0]} to {chunk_indices[-1]}")

        # Step C: Parse structure objects
        print(f"Parsing structure data from '{task['structure_col']}' column...")
        df_chunk['pymatgen_structure'] = df_chunk[task['structure_col']].apply(
            lambda s: Structure.from_dict(json.loads(s)) if pd.notna(s) else None
        )

        # Step D: Generate formula strings
        print("Generating chemical formulas from structures...")
        df_chunk['formula'] = df_chunk['pymatgen_structure'].apply(
            lambda s: s.composition.reduced_formula if pd.notna(s) else None
        )

        # <<< CHANGE 2: REMOVE the line that creates 'material_id'
        # df_chunk['material_id'] = [f"{task['dataset_name']}_{i}" for i in df_chunk.index]
        print("Data loading and preparation complete for this chunk.")

        # <<< CHANGE 3: Get the ID column name from the task config
        # Use .get() to fall back to a default if 'id_col' is not specified
        id_column_name = task.get('id_col', 'material_id')

        # Step F: Compute all features and save the results for the chunk
        output_filename = output_dir / f"all_features_{task['dataset_name']}_chunk_{args.chunk}_of_{args.n_chunks}.csv"
        compute_and_save_all_features(
            df=df_chunk,
            filename=output_filename,
            featurizers=all_featurizers,
            id_col=id_column_name # Pass the correct column name here
        )

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Please ensure the CSV file is in the correct directory. Aborting task.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during task '{task['dataset_name']}': {e}")
        print("Aborting task.")

    end_time = time.perf_counter()
    duration = end_time - start_time
    
    print("\n" + "-"*80)
    print(f"🎉 COMPLETED TASK: {task['dataset_name']} (Chunk {args.chunk} of {args.n_chunks})")
    print(f"🕒 Total wall time for this task: {duration:.2f} seconds")
    print("-" * 80 + "\n")