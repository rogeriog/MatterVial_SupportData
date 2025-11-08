import pandas as pd
import sys
import os
import re

def extract_test_mae(file_path: str) -> float | None:
    """
    (CORRECTED & VERBOSE)
    Parses a results text file to find and extract the average Test MAE from the
    final "Average Metrics" section by finding the LAST matching line in the file.

    Args:
        file_path (str): The path to the results .txt file.

    Returns:
        The Test MAE as a float, or None if not found.
    """
    print(f"    🔎 Attempting to parse MAE from: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            print(f"    ✅ Successfully opened file.")
            
            last_mae_line_found = None
            # Read the entire file and find the last line that matches
            for line in f:
                if re.search(r'^\s*Test\s+-\s+MAE:', line):
                    # This line is a candidate. If we find more, this will be overwritten.
                    last_mae_line_found = line.strip()
            
            # After checking all lines, process the last one we found
            if last_mae_line_found:
                print(f"    ℹ️  Parsing the LAST MAE line found in the file: '{last_mae_line_found}'")
                try:
                    parts = last_mae_line_found.split()
                    
                    # Find the index of the 'MAE:' label
                    try:
                        mae_index = parts.index('MAE:')
                    except ValueError:
                        mae_index = parts.index('MAE') # Handle cases without colon

                    # The value is the next item in the list
                    raw_value = parts[mae_index + 1]
                    cleaned_value = raw_value.replace(',', '')
                    mae_value = float(cleaned_value)
                    
                    print(f"    ➡️  Extracted MAE value: {mae_value}")
                    return mae_value
                except (ValueError, IndexError) as e:
                    print(f"    ❌ ERROR: Found the line, but could not parse the value. Error: {e}")
                    return None
            else:
                # If the loop finished and we never found a matching line
                print(f"    ⚠️  Warning: Finished reading file, but no 'Test  - MAE:' line was found at all.")
                return None
            
    except FileNotFoundError:
        print(f"    ❌ ERROR: Results file not found at this path.")
        return None
    except Exception as e:
        print(f"    ❌ ERROR: An unexpected error occurred: {e}")
        return None


def analyze_properties_and_results(data_csv_path: str, results_dir: str, features_suffix: str):
    """
    (VERBOSE MODE)
    Calculates MAD for properties, finds the corresponding Test MAE from result
    files, and computes the MAD/MAE ratio, printing all intermediate steps.
    """
    print(f"--- Loading Source Data for MAD Calculation ---")
    try:
        df = pd.read_csv(data_csv_path)
        print(f"✅ Successfully loaded source data from '{data_csv_path}' ({len(df)} rows).\n")
    except FileNotFoundError:
        print(f"❌ FATAL ERROR: The data file '{data_csv_path}' was not found. Exiting.")
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

    final_results = []

    for prop_name, col_name in property_map.items():
        print(f"\n{'='*80}\nProcessing Property: '{prop_name}' (Column: '{col_name}')\n{'='*80}")
        
        print("\n[Step 1: Calculating MAD from source CSV]")
        mad_value = None
        if col_name in df.columns:
            series = pd.to_numeric(df[col_name], errors='coerce').dropna()
            if not series.empty:
                print(f"  - Found {len(series)} valid numeric entries for '{col_name}'.")
                mean_val = series.mean()
                mad_value = (series - mean_val).abs().mean()
                print(f"  - Calculated Mean: {mean_val:.4f}")
                print(f"  - Calculated MAD:  {mad_value:.4f}")
            else:
                print(f"  - ⚠️  Warning: Column '{col_name}' exists but has no valid numeric data.")
        else:
            print(f"  - ❌ ERROR: Column '{col_name}' not found in the dataframe.")

        print("\n[Step 2: Extracting Test MAE from results file]")
        results_filename = f"results_jarvis_dft_3d_{col_name}_{features_suffix}.txt"
        results_filepath = os.path.join(results_dir, results_filename)
        mae_value = extract_test_mae(results_filepath)

        final_results.append({
            "name": prop_name,
            "mad": mad_value,
            "mae": mae_value
        })

    print(f"\n\n{'='*85}\n✅ FINAL SUMMARY TABLE\n{'='*85}")
    print(f"{'Property':<40} | {'MAD':<10} | {'Test MAE':<10} | {'MAD/MAE Ratio':<15}")
    print("-" * 85)
    
    for result in final_results:
        mad_str = f"{result['mad']:.4f}" if result['mad'] is not None else "N/A"
        mae_str = f"{result['mae']:.4f}" if result['mae'] is not None else "Not Found"
        
        ratio_str = "N/A"
        if result['mad'] is not None and result['mae'] is not None:
            if result['mae'] != 0:
                ratio = result['mad'] / result['mae']
                ratio_str = f"{ratio:.4f}"
            else:
                ratio_str = "Inf (MAE is 0)"

        print(f"{result['name']:<40} | {mad_str:<10} | {mae_str:<10} | {ratio_str:<15}")


if __name__ == '__main__':
    # --- CONFIGURATION ---
    DATA_CSV_PATH = "jarvis_dft_3d_with_structures.csv"
    RESULTS_DIR = "combined_mv_features/"
    FEATURES_SUFFIX = "roost_all_megnet_all_mvl_all_orb_v3"
    
    analyze_properties_and_results(
        data_csv_path=DATA_CSV_PATH,
        results_dir=RESULTS_DIR,
        features_suffix=FEATURES_SUFFIX
    )