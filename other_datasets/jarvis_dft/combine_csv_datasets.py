import pandas as pd
import os

def combine_csv_features_by_id():
    """
    Combines corresponding feature CSVs from two directories by merging them on
    a common identifier (like 'jid'). It then merges in all non-duplicate
    columns from a third (original dataset) directory.
    """
    # Define the directories
    mattervial_dir = 'mv_noorb_featurized_data/'
    orb_dir = 'mv_orb_featurized_data/'
    original_datasets_dir = './'
    output_dir = 'combined_mv_features/'

    # --- MODIFICATION START ---
    # Create a mapping from the dataset name to its original filename.
    # You MUST edit this dictionary with your actual filenames.
    # The 'key' is the name derived from the feature files (e.g., 'matpedia').
    # The 'value' is the exact name of the corresponding original CSV file.
    original_file_map = {
        'jarvis_dft_3d': 'jarvis_dft_3d_with_structures.csv',
        # Add other mappings here, for example:
        # 'mp_all': 'materials_project_all.csv',
    }
    # --- MODIFICATION END ---

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Get the list of files from the mattervial directory
    try:
        mattervial_files = os.listdir(mattervial_dir)
    except FileNotFoundError:
        print(f"Error: Directory not found - {mattervial_dir}")
        print("Please ensure you are running this script from the parent directory.")
        return

    for mattervial_file in mattervial_files:
        if mattervial_file.startswith('all_features_') and mattervial_file.endswith('.csv'):
            dataset_name = mattervial_file.replace('all_features_', '').replace('.csv', '')
            orb_file = f'orb_features_{dataset_name}.csv'
            
            original_dataset_file = original_file_map.get(dataset_name)
            if not original_dataset_file:
                print(f"Warning: No mapping found for '{dataset_name}' in original_file_map. Skipping.")
                continue

            # Define file paths
            mattervial_path = os.path.join(mattervial_dir, mattervial_file)
            orb_path = os.path.join(orb_dir, orb_file)
            original_dataset_path = os.path.join(original_datasets_dir, original_dataset_file)
            output_path = os.path.join(output_dir, f'{dataset_name}_mattervial_features.csv')

            # Check if all corresponding files exist
            if not all(os.path.exists(p) for p in [mattervial_path, orb_path, original_dataset_path]):
                print(f"Warning: One or more source files for '{dataset_name}' not found. Skipping.")
                continue

            try:
                # Read the CSV files
                mattervial_df = pd.read_csv(mattervial_path)
                orb_df = pd.read_csv(orb_path)
                original_df = pd.read_csv(original_dataset_path)

                # --- MERGE LOGIC START ---

                # 1. Identify the common merge key in all DataFrames
                id_candidates = ['jid', 'material_id', 'formula']
                merge_key = None
                for key in id_candidates:
                    if key in mattervial_df.columns and key in orb_df.columns and key in original_df.columns:
                        merge_key = key
                        break
                
                if not merge_key:
                    print(f"Error: Could not find a common ID ({', '.join(id_candidates)}) in all three files for dataset '{dataset_name}'. Skipping.")
                    continue
                
                print(f"Info: Merging '{dataset_name}' on common key: '{merge_key}'")

                # 2. Merge the first two feature sets ('mattervial' and 'orb')
                # An 'inner' merge ensures we only keep materials present in BOTH feature sets.
                merged_df = pd.merge(mattervial_df, orb_df, on=merge_key, how='inner')

                # 3. Identify new columns from the original dataset to add
                existing_columns = set(merged_df.columns)
                # Ensure the merge key itself is not considered a "new" column
                columns_to_add = [col for col in original_df.columns if col not in existing_columns]

                if columns_to_add:
                    # Include the merge key in the subset of the original_df for the merge
                    original_subset_df = original_df[[merge_key] + columns_to_add]
                    
                    # 4. Merge the result with the new columns from the original dataset
                    # A 'left' merge ensures we keep all materials from the first merge
                    # and just add the extra info from the original file where it exists.
                    final_df = pd.merge(merged_df, original_subset_df, on=merge_key, how='left')
                    print(f"Info: Added {len(columns_to_add)} new columns from {original_dataset_file}.")
                else:
                    final_df = merged_df # No new columns to add
                    print(f"Info: No new unique columns to add from {original_dataset_file}.")

                # --- MERGE LOGIC END ---

                # Save the final merged dataframe
                final_df.to_csv(output_path, index=False)
                print(f"Successfully merged and saved to '{os.path.basename(output_path)}'")
                print("-" * 30)

            except Exception as e:
                print(f"An unexpected error occurred while processing '{dataset_name}': {e}")

if __name__ == '__main__':
    combine_csv_features_by_id()