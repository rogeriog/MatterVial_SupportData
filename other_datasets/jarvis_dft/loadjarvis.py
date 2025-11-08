import numpy as np
import pandas as pd
from jarvis.db.figshare import data
from pymatgen.core import Structure
import json

# 1. Load the JARVIS dataset
print("Loading JARVIS dft_3d dataset...")
dft_3d = data(dataset='dft_3d')
print("Loading complete.")

# 2. Create and clean the DataFrame
df = pd.DataFrame(dft_3d)
df.replace('na', np.nan, inplace=True)

# 3. Define the conversion function
def to_pymatgen_structure_json(atoms_dict):
    """
    Converts a JARVIS atoms dictionary to a JSON serialized pymatgen Structure.
    Returns None if the input is not a valid dictionary.
    """
    if not isinstance(atoms_dict, dict):
        return None
    
    try:
        # Create a pymatgen Structure object
        structure = Structure(
            lattice=atoms_dict['lattice_mat'],
            species=atoms_dict['elements'],
            coords=atoms_dict['coords'],
            coords_are_cartesian=atoms_dict.get('cartesian', True)
        )
        # Serialize the structure to a JSON string
        return json.dumps(structure.as_dict())
    except Exception as e:
        # Return None if any error occurs during conversion
        print(f"Skipping an entry due to an error: {e}")
        return None

# 4. Apply the function to create the 'structure' column
print("\nConverting atomic data to pymatgen structures...")
df['structure'] = df['atoms'].apply(to_pymatgen_structure_json)

# Optional: Drop the original 'atoms' column to save space
df.drop('atoms', axis=1, inplace=True)
print("Conversion complete.")

# 5. Save the final DataFrame to a CSV file
output_filename = 'jarvis_dft_3d_with_structures.csv'
df.to_csv(output_filename, index=False)
print(f"\nSuccessfully saved the full dataset to '{output_filename}'")

# --- Verification ---
print("\n--- Head of the final DataFrame ---")
# Display key columns of the first 5 rows to verify the result
print(df[['jid', 'formula', 'structure']].head())

print("\n--- DataFrame Info ---")
df.info()