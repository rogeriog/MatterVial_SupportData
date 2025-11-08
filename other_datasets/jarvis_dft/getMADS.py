import pandas as pd
import numpy as np
import sys

def calculate_mad(file_path='jarvis_dft_3d_with_structures.csv'):
    """
    Reads a CSV file and calculates the Mean Absolute Deviation (MAD) for
    pre-defined properties.

    Args:
        file_path (str): The path to the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please make sure the CSV file is in the same directory as the script.")
        return

    property_map = {
        'Formation energy': 'formation_energy_peratom',
        'Bandgap (OPT)': 'optb88vdw_bandgap',
        'Total energy': 'optb88vdw_total_energy',
        'Ehull': 'ehull',
        'Bandgap (MBJ)': 'mbj_bandgap',
        'Kv': 'bulk_modulus_kv',
        'Gv': 'shear_modulus_gv',
        'Mag. mom': 'magmom_oszicar',
        'SLME (%)': 'slme',
        'Spillage': 'spillage',
        'Kpoint-length': 'kpoint_length_unit',
        'Plane-wave cutoff': 'encut',
        'єx (OPT)': 'epsx',
        'єy (OPT)': 'epsy',
        'єz (OPT)': 'epsz',
        'єx (MBJ)': 'mepsx',
        'єy (MBJ)': 'mepsy',
        'єz (MBJ)': 'mepsz',
        'є (DFPT:elec+ionic)': 'dfpt_piezo_max_dielectric',
        'Max. piezoelectric strain coeff (dij)': 'dfpt_piezo_max_dij',
        'Max. piezo. stress coeff (eij)': 'dfpt_piezo_max_eij',
        'Exfoliation energy': 'exfoliation_energy',
        'Max. EFG': 'max_efg',
        'avg. me': 'avg_elec_mass',
        'avg. mh': 'avg_hole_mass',
        'n-Seebeck': 'n-Seebeck',
        'n-PF': 'n-powerfact',
        'p-Seebeck': 'p-Seebeck',
        'p-PF': 'p-powerfact'
    }

    print("Calculated MAD for each property:")
    for prop, col in property_map.items():
        if col in df.columns:
            # Convert column to numeric, coercing errors to NaN
            series = pd.to_numeric(df[col], errors='coerce')
            # Drop rows with NaN values
            series = series.dropna()
            
            if not series.empty:
                # Calculate the mean
                mean_val = series.mean()
                # Calculate the Mean Absolute Deviation (MAD)
                mad = (series - mean_val).abs().mean()
                print(f"{prop}: {mad:.4f}")
            else:
                print(f"{prop}: No valid data to calculate MAD.")
        else:
            print(f"Column '{col}' for property '{prop}' not found in the dataframe.")

def count_property_entries(csv_filepath: str):
    """
    Loads a CSV and counts non-empty entries for a predefined list of properties.

    Args:
        csv_filepath (str): The path to the input CSV file.
    """
    # This list contains the exact column names for the properties of interest.
    property_columns = [
        'formation_energy_peratom', 'optb88vdw_bandgap', 'optb88vdw_total_energy',
        'ehull', 'mbj_bandgap', 'bulk_modulus_kv', 'shear_modulus_gv',
        'magmom_oszicar', 'slme', 'spillage', 'kpoint_length_unit', 'encut',
        'epsx', 'epsy', 'epsz', 'mepsx', 'mepsy', 'mepsz',
        'dfpt_piezo_max_dielectric', 'dfpt_piezo_max_dij', 'dfpt_piezo_max_eij',
        'exfoliation_energy', 'max_efg', 'avg_elec_mass', 'avg_hole_mass',
        'n-Seebeck', 'n-powerfact', 'p-Seebeck', 'p-powerfact'
    ]

    try:
        print(f"🔎 Loading data from '{csv_filepath}'...")
        df = pd.read_csv(csv_filepath, low_memory=False)
        print("✅ Successfully loaded the dataset.")
    except FileNotFoundError:
        print(f"❌ Error: The file '{csv_filepath}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        sys.exit(1)

    total_rows = len(df)
    if total_rows == 0:
        print("⚠️ The CSV file is empty.")
        return

    print(f"\nTotal number of materials in dataset: {total_rows}")
    print("-" * 60)
    print("Entry Count for Specified Properties:")
    print("-" * 60)

    # Iterate through the predefined list of property columns
    for column_name in property_columns:
        if column_name in df.columns:
            # Count non-NaN values in the specific column
            count = df[column_name].count()
            completeness = (count / total_rows) * 100
            print(f"{column_name:<40} | {count:<8} ({completeness:.1f}%)")
        else:
            # Report if a specific property column is missing from the file
            print(f"{column_name:<40} | Not found in CSV")

if __name__ == '__main__':
    calculate_mad()
    DATASET_PATH = "jarvis_dft_3d_with_structures.csv"
    
    count_property_entries(DATASET_PATH)