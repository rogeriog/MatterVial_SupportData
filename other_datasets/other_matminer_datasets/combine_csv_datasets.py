import pandas as pd
import os

def combine_csv_features():
    """
    Combines corresponding feature CSVs from two directories and appends the
    'structure' column from a third directory. Assumes a one-to-one row
    correspondence and concatenates files horizontally by index.
    """
    # Define the directories
    mattervial_dir = 'mv_noorb_featurized_data/'
    orb_dir = 'mv_orb_featurized_data/'
    original_datasets_dir = './'  # Adjusted path based on user feedback
    output_dir = 'combined_mv_features/'

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Get the list of files from the mattervial directory
    try:
        mattervial_files = os.listdir(mattervial_dir)
    except FileNotFoundError:
        print(f"Error: Directory not found - {mattervial_dir}")
        print("Please ensure you are running this script from the parent directory that contains all data folders.")
        return

    for mattervial_file in mattervial_files:
        if mattervial_file.startswith('all_features_') and mattervial_file.endswith('.csv'):
            # Construct the corresponding feature and dataset filenames
            dataset_name = mattervial_file.replace('all_features_', '').replace('.csv', '')
            orb_file = f'orb_features_{dataset_name}.csv'
            original_dataset_file = f'{dataset_name}.csv'

            # Define file paths
            mattervial_path = os.path.join(mattervial_dir, mattervial_file)
            orb_path = os.path.join(orb_dir, orb_file)
            original_dataset_path = os.path.join(original_datasets_dir, original_dataset_file)
            output_path = os.path.join(output_dir, f'{dataset_name}_mattervial_features.csv')

            # Check if all corresponding files exist
            if not os.path.exists(orb_path):
                print(f"Warning: Corresponding orb file not found for {mattervial_file}. Skipping.")
                continue
            if not os.path.exists(original_dataset_path):
                print(f"Warning: Corresponding original dataset file not found: {original_dataset_file}. Skipping.")
                continue

            try:
                # Read the CSV files into pandas DataFrames
                mattervial_df = pd.read_csv(mattervial_path)
                orb_df = pd.read_csv(orb_path)
                original_df = pd.read_csv(original_dataset_path)

                # --- NEW LOGIC: Concatenate by index, adding target and standardizing ID ---

                # Standardize ID column in the primary dataframe (mattervial_df) which provides the ID
                if 'material_id' not in mattervial_df.columns and 'formula' in mattervial_df.columns:
                    mattervial_df.rename(columns={'formula': 'material_id'}, inplace=True)
                    print(f"Info: Renamed 'formula' to 'material_id' in memory for {mattervial_file}")

                # Identify and drop the ID column from the second dataframe to avoid duplication
                id_col_orb = None
                if 'material_id' in orb_df.columns:
                    id_col_orb = 'material_id'
                elif 'formula' in orb_df.columns:
                    id_col_orb = 'formula'

                if id_col_orb:
                    orb_features_df = orb_df.drop(columns=[id_col_orb])
                else:
                    # If no id column is found, use the whole dataframe but warn the user.
                    print(f"Warning: No 'material_id' or 'formula' column found in {orb_file} to drop. Concatenating all its columns.")
                    orb_features_df = orb_df

                # Prepare a list of dataframes to concatenate, starting with the first two feature sets
                dfs_to_concat = [mattervial_df, orb_features_df]

                # Select 'structure' and 'target' columns from the original dataset
                cols_to_add_from_original = []
                if 'structure' in original_df.columns:
                    cols_to_add_from_original.append('structure')
                else:
                    print(f"Warning: 'structure' column not found in {original_dataset_file}.")

                if 'target' in original_df.columns:
                    cols_to_add_from_original.append('target')
                else:
                    print(f"Warning: 'target' column not found in {original_dataset_file}.")

                if cols_to_add_from_original:
                    original_cols_df = original_df[cols_to_add_from_original]
                    dfs_to_concat.append(original_cols_df)

                # Concatenate all parts horizontally (side-by-side)
                final_df = pd.concat(dfs_to_concat, axis=1)

                # Save the final concatenated dataframe to the new file
                final_df.to_csv(output_path, index=False)
                print(f"Successfully concatenated and saved to '{os.path.basename(output_path)}'")

            except Exception as e:
                print(f"An unexpected error occurred while processing files for {dataset_name}: {e}")

if __name__ == '__main__':
    combine_csv_features()

