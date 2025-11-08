import json
import time  # Import the time module for timing
import pandas as pd
from pathlib import Path
from pymatgen.core.structure import Structure
from mattervial.featurizers.structure import ORBFeaturizer

# =============================================================================
# Helper Function for Featurization
# =============================================================================

def compute_and_save_orb_features(
    df: pd.DataFrame, 
    filename: Path, 
    featurizer: ORBFeaturizer,
    structure_col: str = 'pymatgen_structure',
    id_col: str = 'material_id'
):
    """
    Computes ORB features for structures in a DataFrame and saves them to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing the structures and material IDs.
        filename (Path): Path for the output CSV file.
        featurizer (ORBFeaturizer): An instance of the ORBFeaturizer.
        structure_col (str): The name of the column with pymatgen Structure objects.
        id_col (str): The name of the column with unique material identifiers.
    """
    # --- 1. Validate Input DataFrame ---
    if structure_col not in df.columns or id_col not in df.columns:
        raise ValueError(f"Input DataFrame must contain '{structure_col}' and '{id_col}' columns.")

    # Filter out rows where the structure is null
    valid_mask = df[structure_col].notnull()
    if not valid_mask.any():
        print(f"⚠️  Warning: No valid structures found in the '{structure_col}' column. Skipping file generation for {filename.name}.")
        return
        
    df_valid = df.loc[valid_mask].copy()
    
    # --- 2. Compute ORB Features ---
    print(f"Computing ORB features for {len(df_valid)} structures...")
    valid_structures = df_valid[structure_col].tolist()
    
    # get_features computes the ORB representation for a list of structures
    features_orb = featurizer.get_features(valid_structures)
    
    # --- 3. Format and Save Results ---
    features_orb_df = pd.DataFrame(features_orb)
    
    # Add the material IDs as the first column for easy reference
    material_ids = df_valid[id_col].tolist()
    features_orb_df.insert(0, id_col, material_ids)
    
    print(f"Generated feature matrix with shape: {features_orb_df.shape}")
    
    # Ensure the output directory exists
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the featurized data to a new CSV file
    features_orb_df.to_csv(filename, index=False)
    print(f"✅ Successfully saved features to '{filename}'")


# =============================================================================
# Main Execution Block
# =============================================================================

# =============================================================================
# Main Execution Block
# =============================================================================

if __name__ == '__main__':
    # --- 1. Global Configurations ---
    orb_featurizer = ORBFeaturizer()
    output_dir = Path('./orb_featurized_data')

    # --- 2. Define Dataset Processing Tasks ---
    tasks = [
        {
            "csv_path": "./jarvis_dft_3d_with_structures.csv",
            "structure_col": "structure",
            "dataset_name": "jarvis_dft_3d",
            "id_col": "jid"  # <<< CHANGE 1: Specify your ID column here
        }
    ]

    # --- 3. Loop Through and Process Each Task ---
    for task in tasks:
        print("\n" + "="*80)
        print(f"🚀 STARTING TASK: {task['dataset_name']}")
        print("="*80 + "\n")
        
        start_time = time.perf_counter()
        
        try:
            # Step A: Load the dataset
            print(f"Loading data from '{task['csv_path']}'...")
            csv_path = Path(task['csv_path'])
            if not csv_path.exists():
                raise FileNotFoundError(f"Source file not found at '{csv_path}'")
            
            df = pd.read_csv(csv_path)

            # Step B: Parse structures
            print(f"Parsing structure data from '{task['structure_col']}' column...")
            df['pymatgen_structure'] = df[task['structure_col']].apply(
                lambda s: Structure.from_dict(json.loads(s)) if pd.notna(s) else None
            )

            # <<< CHANGE 2: REMOVE the line that creates 'material_id'
            # df['material_id'] = [f"{task['dataset_name']}_{i}" for i in df.index]
            print("Data loading and preparation complete.")

            # Step D: Compute features and save
            output_filename = output_dir / f"orb_features_{task['dataset_name']}.csv"
            
            # <<< CHANGE 3: Tell the function which ID column to use
            # We use .get() to provide a default value if 'id_col' isn't in the task
            id_column_name = task.get('id_col', 'material_id') 
            
            compute_and_save_orb_features(
                df=df,
                filename=output_filename,
                featurizer=orb_featurizer,
                id_col=id_column_name # Pass the correct column name here
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