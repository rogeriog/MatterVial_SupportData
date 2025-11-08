import sys, os
import traceback
import glob, re
import argparse
import os
import pandas as pd
import json
import numpy as np
import traceback

MATBENCH_SEED = 18012019
# --- Define Feature Group Patterns ---
### These are the expected features for default DeBreuck2020 Featurizer.
MATMINER_PREFIXES = [
    'AtomicOrbitals|', 'AtomicPackingEfficiency|', 'BandCenter|', 'ElementFraction|',
    'ElementProperty|', 'IonProperty|', 'Miedema|', 'Stoichiometry|',
    'TMetalFraction|', 'ValenceOrbital|', 'YangSolidSolution|',
    'ElectronegativityDiff|', 'OxidationStates|', 'DensityFeatures|',
    'GlobalSymmetryFeatures|', 'CoulombMatrix|', 'SineCoulombMatrix|',
    'BondFractions|', 'StructuralHeterogeneity|', 'MaximumPackingEfficiency|',
    'ChemicalOrdering|', 'XRDPowderPattern|', 'RadialDistributionFunction|',
    'AGNIFingerPrint|', 'AverageBondAngle|', 'AverageBondLength|',
    'BondOrientationParameter|', 'ChemEnvSiteFingerprint|', 'CoordinationNumber|',
    'CrystalNNFingerprint|', 'GaussianSymmFunc|', 'GeneralizedRDF|',
    'LocalPropertyDifference|', 'OPSiteFingerprint|', 'VoronoiFingerprint|'
]
# Escape special characters like '|'
MATMINER_PATTERN = r'^(' + '|'.join([re.escape(p) for p in MATMINER_PREFIXES]) + ')'

# Updated: Added 'sisso' pattern, removed matbench_set dependency logic
FEATURE_GROUP_PATTERNS = {
        'matminer': MATMINER_PATTERN,
        'sisso': r'^SISSO_',
        'sisso_v2': r'^SISSOv2_',
        'sissolvl2': r'^SISSOlvl2_',
        'sisso_mb_dielectric':  r'^SISSO_matbench_dielectric_',
        'sisso_mb_phonons':     r'^SISSO_matbench_phonons_',
        'sisso_mb_perovskites': r'^SISSO_matbench_perovskites_',
        'sisso_mb_expt_is_metal': r'^SISSO_matbench_expt_is_metal_',
        'sisso_mb_steels':      r'^SISSO_matbench_steels_',
        'sisso_mb_jdft2d':      r'^SISSO_matbench_jdft2d_',
        'sisso_mb_log_gvrh':    r'^SISSO_matbench_log_gvrh_',
        'sisso_mb_mp_e_form':   r'^SISSO_matbench_mp_e_form_',
        'sisso_noemd_hse_pbe_diff':   r'^SISSO_noemd_hse_pbe_diff_',
        'sisso_noemd_shg':            r'^SISSO_noemd_shg_',
        'sisso_mb_log_kvrh':    r'^SISSO_matbench_log_kvrh_',
        'sisso_mb_glass':       r'^SISSO_matbench_glass_',
        'sisso_mb_mp_is_metal': r'^SISSO_matbench_mp_is_metal_',
        'sisso_mb_mp_gap':      r'^SISSO_matbench_mp_gap_',
        'sisso_mb_expt_gap':    r'^SISSO_matbench_expt_gap_',
        'roost_mpgap_lo': r'^ROOST_mpgap_LayerOutput_',
        'roost_mpgap_lmp': r'^ROOST_mpgap_LayerMaterialPooling_',
        'roost_oqmd_lo': r'^ROOST_oqmd_eform_LayerOutput_',
        'roost_oqmd_lmp': r'^ROOST_oqmd_eform_LayerMaterialPooling_',
        'megnet_mm': r'^MEGNet_MatMinerEncoded_v1_',
        'megnet_ofm': r'^MEGNet_OFMEncoded_v1_',
        'mvl32': r'^MVL32_',
        'mvl16': r'^MVL16_',
        # Aliases for convenience
        'roost_lmp': r'^ROOST_.*_LayerMaterialPooling_',
        'roost_lo': r'^ROOST_.*_LayerOutput_',
        'roost_mpgap': r'^ROOST_mpgap_',
        'roost_oqmd': r'^ROOST_oqmd_eform_',
        'roost_all': r'^ROOST_',
        'megnet_all': r'^MEGNet_',
        'mvl_all': r'^MVL(16|32)_',
        'orb_v3': r'^ORB_v3_',
        'cogn_fold0': r'^coGN_.*_fold0',
        'cogn_fold1': r'^coGN_.*_fold1',
        'cogn_fold2': r'^coGN_.*_fold2',
        'cogn_fold3': r'^coGN_.*_fold3',
        'cogn_fold4': r'^coGN_.*_fold4',
        'ofm': r'^OFM:', # for reconstructed OFM features
        'sisso_residuals': r'^SISSOresiduals_',
        'sissolvl3': r'SISSOlvl3_',
        'soap_mean': r'^mean_SOAP_',  
        'soap_std': r'^std_dev_SOAP_',  
        'soap_all': r'^(mean_SOAP_|std_dev_SOAP_)',  
        'rosa': r'^ROSA\|',  
        'g': r'^G\|',
    }
# --- End Feature Group Patterns ---

## Best settings obtained from https://github.com/ml-evs/modnet-matbench
BEST_SETTINGS={
    "matbench_expt_gap":
    {
    "increase_bs": False,
    "num_neurons": [[256], [128], [16], [16]],
    "n_feat": 100,
    "lr": 0.007,
    "epochs": 400,
    "verbose": 0,
    "act": "elu",
    "batch_size": 64,
    "loss": "mae",
    },
    "matbench_expt_is_metal":
    {
    "increase_bs":False, 
    "num_neurons": [[128], [32], [32], [16]],
    "n_feat": 120,
    "lr": 0.005, #0.005
    "epochs": 100,
    "verbose": 0,
    "act": "elu",
    "batch_size": 64,
    "num_classes": {'target':2},
    "loss": "categorical_crossentropy",
    },
    "matbench_glass": {
    "increase_bs":True,
    "num_neurons": [[128], [64], [16], []],
    "n_feat": 150,
    "lr": 0.002,
    "epochs": 200,
    "verbose": 0,
    "act": "elu",
    "batch_size": 64,
    "num_classes": {'target':2},
    "loss": "categorical_crossentropy",
    "metrics": ["categorical_accuracy"]
    },
    "matbench_log_gvrh": {
    "increase_bs":False,
    "num_neurons": [[256], [64], [64], [32]],
    "n_feat": 350,
    "lr": 0.005,
    "epochs": 200,
    "verbose": 0,
    "act": "elu",
    "batch_size": 64,
    "loss": "mae",
    },
    "matbench_log_kvrh": {
    "increase_bs":False,
    "num_neurons": [[256], [64], [64], [32]],
    "n_feat": 350,
    "lr": 0.005,
    "epochs": 200,
    "verbose": 0,
    "act": "elu",
    "batch_size": 64,
    "loss": "mae",
    },
    "matbench_steels": {
    "increase_bs":False,
    "num_neurons": [[64], [32], [8], [8]],
    "lr": 0.005,
    "epochs": 500,
    "verbose": 0,
    "act": "elu",
    "batch_size": 32,
    "loss": "mae",
    "xscale": "standard",
    },
    "matbench_phonons": {
    "increase_bs":True,
    "num_neurons": [[512], [128], [64], [64]],
    "n_feat": 280,
    "lr": 0.005,
    "epochs": 800,
    "act": "elu",
    "batch_size": 64,
    "loss": "mae",
    },
    "matbench_jdft2d": {
        "increase_bs":False,
        "num_neurons": [[256], [64], [32], [32]],
        "n_feat": 100,
        "lr": 0.01,
        "epochs": 1000,
        "act": "relu",
        "batch_size": 32,
        "loss": "mae",
    },    
    "matbench_dielectric": {
    "increase_bs": False,
    "num_neurons": [[128], [32], [8], [8]],
    "n_feat": 512,
    "lr": 0.005,
    "epochs": 1000,
    "verbose": 0,
    "act": "relu",
    "batch_size": 64,
    "loss": "mae",
    },
}

def setup_threading():
    import tensorflow as tf
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # tf.keras.mixed_precision.set_global_policy("mixed_float16")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)    
    # LIMIT NUMBER OF THREADS, IF NOT IT MAY GO CRAZY
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # import tensorflow as tf
    # tf.config.threading.set_intra_op_parallelism_threads(24)
    # tf.config.threading.set_inter_op_parallelism_threads(1)
# ============================================================================
# SECTION: DOWNLOAD MATBENCH DATA SETS AS SPLITS IN CSV
# ============================================================================
def download_matbench_sets_as_csv():
    import os
    import json
    import sys
    from matbench.bench import MatbenchBenchmark
    import pandas as pd

    # Define the tasks to run
    task_list = [
        # 'matbench_dielectric',
        # # 'matbench_expt_gap',
        # 'matbench_expt_is_metal',
        # 'matbench_glass',
        # 'matbench_jdft2d',
        # 'matbench_log_gvrh',
        # 'matbench_log_kvrh',
        'matbench_mp_e_form',
        # 'matbench_mp_gap',
        # 'matbench_mp_is_metal',
        # 'matbench_perovskites',
        # 'matbench_phonons',
        # 'matbench_steels'
    ]

    # Initialize benchmark, note that autoload is set to False so we load later.
    mb = MatbenchBenchmark(autoload=False, subset=task_list)

    # Optionally, get the pymatgen Structure type (if available)
    try:
        from pymatgen import Structure
    except ImportError:
        Structure = None

    # Iterate over each task and its folds.
    for task in mb.tasks:
        task_name = task.dataset_name
        # Create main task directory
        task_dir = os.path.join("data", task_name)
        os.makedirs(task_dir, exist_ok=True)
        
        for fold_number in task.folds:
            print(f"Task: {task_name}, Fold: {fold_number}")
            
            # Load data for this task (if not already loaded)
            task.load()
            
            # Create fold directory
            fold_dir = os.path.join(task_dir, f"fold_{fold_number}")
            os.makedirs(fold_dir, exist_ok=True)
            
            # Get training/validation data and testing data with targets.
            train_inputs, train_outputs = task.get_train_and_val_data(fold_number)
            test_inputs, test_outputs = task.get_test_data(fold_number, include_target=True)
            
            print("Train inputs:", train_inputs.head())
            print("Train outputs:", train_outputs.head())
            print("Test inputs:", test_inputs.head())
            print("Test outputs:", test_outputs.head())
            
            # Rename output targets to "target". We assume one column per output DataFrame.
            train_outputs = train_outputs.to_frame(name='target')
            test_outputs = test_outputs.to_frame(name='target')
            
            # --- Process train_inputs ---
            if isinstance(train_inputs, pd.Series):
                train_inputs = train_inputs.to_frame()

            # Rename columns based on type:
            for col in train_inputs.columns.copy():
                first_val = train_inputs[col].iloc[0]
                if isinstance(first_val, str):
                    train_inputs.rename(columns={col: 'composition'}, inplace=True)
                elif Structure is not None and isinstance(first_val, Structure):
                    train_inputs.rename(columns={col: 'structure'}, inplace=True)
            
            # If there is a "structure" column, convert pymatgen Structure to JSON
            if 'structure' in train_inputs.columns:
                train_inputs['structure'] = train_inputs['structure'].apply(
                    lambda x: json.dumps(x.as_dict()) if hasattr(x, "as_dict") else x
                )
                print("Processed train_inputs structure column:")
                print(train_inputs['structure'].head(), train_inputs['structure'].iloc[0],
                      type(train_inputs['structure'].iloc[0]))
            
            # --- Process test_inputs ---
            if isinstance(test_inputs, pd.Series):
                test_inputs = test_inputs.to_frame()

            for col in test_inputs.columns.copy():
                first_val = test_inputs[col].iloc[0]
                if isinstance(first_val, str):
                    test_inputs.rename(columns={col: 'composition'}, inplace=True)
                elif Structure is not None and isinstance(first_val, Structure):
                    test_inputs.rename(columns={col: 'structure'}, inplace=True)
            
            if 'structure' in test_inputs.columns:
                test_inputs['structure'] = test_inputs['structure'].apply(
                    lambda x: json.dumps(x.as_dict()) if hasattr(x, "as_dict") else x
                )
                print("Processed test_inputs structure column:")
                print(test_inputs['structure'].head(), test_inputs['structure'].iloc[0],
                      type(test_inputs['structure'].iloc[0]))
            
            # Save training and test data in fold directories as CSV, converting index to mbid column
            train_inputs.reset_index(names=['mbid']).to_csv(os.path.join(fold_dir, "train_inputs.csv"), index=False)
            train_outputs.reset_index(names=['mbid']).to_csv(os.path.join(fold_dir, "train_outputs.csv"), index=False)
            test_inputs.reset_index(names=['mbid']).to_csv(os.path.join(fold_dir, "test_inputs.csv"), index=False)
            test_outputs.reset_index(names=['mbid']).to_csv(os.path.join(fold_dir, "test_outputs.csv"), index=False)
            
            print(f"Saved data for {task_name} fold {fold_number}")
            print(f"Dataset Shape - Train Inputs: {train_inputs.shape}, Train Outputs: {train_outputs.shape}")
            print(f"Dataset Shape - Test Inputs: {test_inputs.shape}, Test Outputs: {test_outputs.shape}")
            print("-" * 50)
            print("\n")        
    print("All tasks processed.")



def process_chunk(chunk_df, output_csv_path, chunk_number, n_jobs, simplified_featurization=False):
    """
    Process a single chunk: create a MODData object, featurize and save results.
    This helper function is designed to be called in a separate process.
    """
    from modnet.preprocessing import MODData
    from modnet.featurizers.presets.debreuck_2020 import DeBreuck2020Featurizer

    # Create a new instance of the featurizer inside the process
    featurizer = DeBreuck2020Featurizer()
    if simplified_featurization:
        from modnet.featurizers.presets.debreuck_2020 import CompositionOnlyFeaturizer
        featurizer = CompositionOnlyFeaturizer()
        # print("No site or oxidcomp featurization!!!")
        # featurizer.site_featurizers = ()
        # featurizer.oxid_composition_featurizers  = ()
        # print("No structure featurization! ")
        # from matminer.featurizers.structure import (
        #         BondFractions,
        #         DensityFeatures,
        #         GlobalSymmetryFeatures,
        #     )
        # featurizer.structure_featurizers = (
        #                 # DensityFeatures(),
        #                 # GlobalSymmetryFeatures(),
        #                 # BondFractions()
        #                 )
    # --- START of new code block ---
    # Filter out the problematic RadialDistributionFunction from the list of structure featurizers
    # This creates a new tuple of featurizers that excludes any instance of RadialDistributionFunction
    # from matminer.featurizers.structure import RadialDistributionFunction
    # original_featurizers = featurizer.structure_featurizers
    # featurizer.structure_featurizers = tuple(
    #     f for f in original_featurizers if not isinstance(f, RadialDistributionFunction)
    # )
    # print(f"  INFO: Removed RadialDistributionFunction. {len(featurizer.structure_featurizers)} of {len(original_featurizers)} structure featurizers remain.")
    # # --- END of new code block ---
    # print(featurizer.structure_featurizers)
    
    # Extract columns from chunk_df (which already has a reset index)
    structures_filtered = chunk_df['pymatgen_structure']
    targets_filtered = chunk_df['target']
    ids_filtered = chunk_df['material_id']
    key_cols = chunk_df[['material_id', 'structure', 'target']].copy()

    try:
        # Create MODData for the chunk
        chunk_moddata = MODData(
            materials=structures_filtered.tolist(),   # list of Structure objects
            targets=targets_filtered.tolist(),          # list or array of target values
            structure_ids=ids_filtered.tolist(),        # list of IDs
            target_names=['target']
        )
    except Exception as e:
        print(f"  ERROR: Failed to create MODData object for chunk {chunk_number}: {e}")
        return False

    # Featurize the chunk
    try:
        chunk_moddata.featurizer = featurizer
        chunk_moddata.featurize(n_jobs=n_jobs)
        chunk_features_df = chunk_moddata.get_featurized_df()
        chunk_features_df = chunk_features_df.set_index(pd.Index(ids_filtered, name='material_id'))
    except Exception as e:
        print(f"  ERROR: Featurization failed for chunk {chunk_number}: {e}")
        traceback.print_exc()  # Print the traceback
        return False

    # Combine features with key columns
    try:
        key_cols.set_index('material_id', inplace=True)
        if not chunk_features_df.index.equals(key_cols.index):
            print("  ERROR: Index mismatch between features and key columns for chunk {chunk_number}.")
            return False
        chunk_final_df = pd.concat([key_cols, chunk_features_df], axis=1)
        chunk_final_df.reset_index(inplace=True)
    except Exception as e:
        print(f"  ERROR: Combining features failed for chunk {chunk_number}: {e}")
        return False

    # Save the featurized chunk to CSV
    try:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        chunk_final_df.to_csv(output_csv_path, index=False)
        print(f"  Successfully saved chunk {chunk_number} to '{output_csv_path}'.")
    except Exception as e:
        print(f"  ERROR: Failed to save chunk {chunk_number} to CSV: {e}")
        return False

    # Clean up (optional)
    del chunk_moddata, chunk_features_df, chunk_final_df
    return True


def modnet_featurize_MM2020Struct(matbench_set=None,
                                csv_path=None,
                                n_jobs=24,
                                split_dataset=None,
                                start_chunk_index=None,
                                end_chunk_index=None,
                                simplified_featurization=False,
                                chunk_timeout=200):  # timeout (seconds) per chunk
    """
    Featurizes structures using MODNet's DeBreuck2020Featurizer.
    Loads data from pre-generated CSV files for fold_0 of the specified matbench_set.
    If split_dataset > 1, processes the data in chunks and saves each
    featurized chunk immediately to a separate CSV file for fault tolerance.
    Allows processing of a specific range of chunks using start_chunk_index and end_chunk_index.
    A timeout is set for each chunk to prevent long-running processes.
    Args:
        matbench_set (str): The name of the matbench dataset (e.g., 'matbench_mp_e_form').
        n_jobs (int): Number of parallel jobs for featurization.
        csv_path (str, optional): Path to the CSV file containing the data to featurize.
        split_dataset (int, optional): Number of chunks to split processing into.
                                       Defaults to None (process as one chunk).
        start_chunk_index (int, optional): Starting index of the chunk to process (1-based index).
                                          If None, starts from the first chunk.
        end_chunk_index (int, optional): Ending index of the chunk to process (inclusive, 1-based index).
                                        If None, processes up to the last chunk.
        simplified_featurization (bool, optional): If True, skips site-level featurization. For problematic structures only.
        chunk_timeout (int, optional): Timeout for each chunk processing in seconds.
    """
    from concurrent.futures import ProcessPoolExecutor, TimeoutError
    import concurrent.futures
    from pymatgen.core import Structure
    import os, json, warnings
    import numpy as np
    import pandas as pd
    warnings.filterwarnings("ignore")
    print(f"\n--- Starting MODNet Structure Featurization for: {matbench_set} ---")

    # Load CSV data (or combine from fold_0) as in your original code...
    if csv_path is not None:
        df_orig = pd.read_csv(csv_path)
        print(f"Loading data from provided CSV: {csv_path}")
        if 'mbid' in df_orig.columns:
            df_orig.rename(columns={'mbid': 'material_id'}, inplace=True)
        if 'target' not in df_orig.columns:
            print("Warning: No target column found. Creating dummy target column.")
            df_orig['target'] = 1
        output_base = os.path.splitext(csv_path)[0] + '_featurizedMM2020Struct'
    else:
        raise ValueError("Either matbench_set or csv_path must be provided")

    # Preprocessing: convert structure strings/dicts to Pymatgen Structures
    if 'structure' not in df_orig.columns:
        raise ValueError("Input CSVs must have a 'structure' column.")
    if 'target' not in df_orig.columns:
        raise ValueError("Input CSVs must have a 'target' column.")
    if 'material_id' not in df_orig.columns:
        raise ValueError("Input CSVs must have a 'material_id' column.")

    def to_structure(val):
        if pd.isna(val):
            return None
        if isinstance(val, str):
            try:
                d = json.loads(val)
                return Structure.from_dict(d)
            except Exception:
                return None
        elif isinstance(val, dict):
            return Structure.from_dict(val)
        elif isinstance(val, Structure):
            return val
        return None

    print("Converting structure representations...")
    df_orig['_pymatgen_structure_temp'] = df_orig['structure'].apply(to_structure)
    valid_mask = df_orig['_pymatgen_structure_temp'].notna()
    if not valid_mask.all():
        print(f"Warning: {sum(~valid_mask)} entries could not be converted to Pymatgen Structures and will be skipped.")
        df_filtered = df_orig[valid_mask].copy().reset_index(drop=True)
    else:
        df_filtered = df_orig.copy().reset_index(drop=True)
    df_filtered.rename(columns={'_pymatgen_structure_temp': 'pymatgen_structure'}, inplace=True)
    if df_filtered.empty:
        print("Error: No valid structures found after filtering. Cannot featurize.")
        return

    print(f"Processing {len(df_filtered)} valid structures.")

    # --- Set Up Chunk Processing ---
    num_chunks = split_dataset if split_dataset and split_dataset > 1 else 1
    if num_chunks > len(df_filtered):
        print(f"Warning: Number of chunks ({num_chunks}) is greater than the number of samples ({len(df_filtered)}). Adjusting chunk count.")
        num_chunks = len(df_filtered)
    chunk_indices_list = np.array_split(df_filtered.index, num_chunks)

    start_chunk = start_chunk_index if start_chunk_index is not None else 1
    end_chunk = end_chunk_index if end_chunk_index is not None else num_chunks
    if start_chunk < 1:
        start_chunk = 1
    if end_chunk > num_chunks:
        end_chunk = num_chunks
    if start_chunk > end_chunk:
        raise ValueError(f"Start chunk index ({start_chunk}) cannot be greater than end chunk index ({end_chunk}).")
    print(f"Starting featurization across chunks {start_chunk}–{end_chunk} of {num_chunks} total chunks...")

    # We use a ProcessPoolExecutor to process each chunk with a timeout.
    with ProcessPoolExecutor(max_workers=1) as executor:
        # Loop over the desired chunk range
        for i in range(start_chunk - 1, end_chunk):
            chunk_number = i + 1  # user facing chunk number (1-based)
            chunk_indices = chunk_indices_list[i]
            if len(chunk_indices) == 0:
                continue

            # Prepare output file name
            if num_chunks == 1:
                output_csv_path = output_base + '.csv'
            else:
                output_csv_path = f"{output_base}_chunk_{chunk_number}_of_{num_chunks}.csv"

            if os.path.exists(output_csv_path):
                print(f"Output file '{output_csv_path}' already exists. Skipping chunk {chunk_number}.")
                continue

            # Prepare the chunk-specific DataFrame (make a copy so it can be pickled)
            chunk_df = df_filtered.loc[chunk_indices].copy().reset_index(drop=True)

            print(f"\nProcessing chunk {chunk_number}/{num_chunks} ({len(chunk_df)} samples)...")
            future = executor.submit(process_chunk, chunk_df, output_csv_path, chunk_number, n_jobs, simplified_featurization=simplified_featurization)
            try:
                # Wait for the process to finish, with the given timeout.
                result = future.result(timeout=chunk_timeout)
                if not result:
                    print(f"Chunk {chunk_number} failed during processing. Skipping it.")
            except TimeoutError:
                print(f"Chunk {chunk_number} timed out after {chunk_timeout} seconds. Skipping it.")
            except Exception as e:
                print(f"Unexpected error processing chunk {chunk_number}: {e}")

    print(f"\n--- Finished MODNet Structure Featurization for: {matbench_set} ---")
    if num_chunks > 1:
        print(f"Featurized data saved incrementally in chunks {start_chunk}–{end_chunk} of {num_chunks} total chunks.")
    elif os.path.exists(output_base + '.csv'):
        print(f"Featurized data saved in a single file: {output_base + '.csv'}")
    else:
        print("No output files were generated (possibly due to errors or skipped chunks).")


# ============================================================================
# SECTION: MODNET MATMINER 2020 COMPOSITION ONLY FEATURIZATION SAVE AS CSV (Adapted)
# ============================================================================
def modnet_featurize_MM2020Comp(matbench_set=None, csv_path=None, n_jobs=24, split_dataset=None, start_chunk_index=None, end_chunk_index=None):
    """
    Featurizes compositions using MODNet's CompositionOnlyFeaturizer.
    Loads data from pre-generated CSV files for fold_0 of the specified matbench_set.
    If split_dataset > 1, processes the data in chunks and saves each
    featurized chunk immediately to a separate CSV file for fault tolerance.
    Allows processing of a specific range of chunks using start_chunk_index and end_chunk_index.

    Args:
        matbench_set (str): The name of the matbench dataset (e.g., 'matbench_expt_gap').
        n_jobs (int): Number of parallel jobs for featurization.
        csv_path (str, optional): Path to the CSV file containing the data to featurize.
        split_dataset (int, optional): Number of chunks to split processing into.
                                       Defaults to None (process as one chunk).
        start_chunk_index (int, optional): Starting index of the chunk to process (1-based index).
                                          If None, starts from the first chunk.
        end_chunk_index (int, optional): Ending index of the chunk to process (inclusive, 1-based index).
                                        If None, processes up to the last chunk.
    """
    import os
    import warnings
    import pandas as pd
    import numpy as np
    import json
    from modnet.preprocessing import MODData
    from pymatgen.core import Composition, Structure # Import Structure for fallback conversion
    from modnet.featurizers.presets.debreuck_2020 import CompositionOnlyFeaturizer

    warnings.filterwarnings("ignore")
    print(f"\n--- Starting MODNet Composition Featurization for: {matbench_set} ---")

    if matbench_set is None and csv_path is None:
        raise ValueError("Either matbench_set or csv_path must be provided")

    if csv_path is not None:
        df_orig = pd.read_csv(csv_path)
        print(f"Loading data from provided CSV: {csv_path}")
        if 'mbid' in df_orig.columns:
            df_orig.rename(columns={'mbid': 'material_id'}, inplace=True)
        ## if no target column, create one
        if 'target' not in df_orig.columns:
            print("Warning: No target column found. Creating dummy target column.")
            df_orig['target'] = 1
        output_base = os.path.splitext(csv_path)[0] # Base name for output files
    else:
        print(f"Loading data from pre-generated CSVs in fold_0.")
        print(f"Processing chunks and saving incrementally: {'Yes (' + str(split_dataset) + ')' if split_dataset and split_dataset > 1 else 'No'}")

        fold_number = 0 # Use fold 0
        fold_dir = f'fold_{fold_number}'
        data_dir = f'data/{matbench_set}'
        output_base = os.path.join(data_dir, f'{matbench_set}_featurizedMM2020Comp') # Base name for output files

        # Define paths to the pre-generated CSV files for fold 0
        train_inputs_path = os.path.join(data_dir, fold_dir, 'train_inputs.csv')
        train_outputs_path = os.path.join(data_dir, fold_dir, 'train_outputs.csv')
        test_inputs_path = os.path.join(data_dir, fold_dir, 'test_inputs.csv')
        test_outputs_path = os.path.join(data_dir, fold_dir, 'test_outputs.csv')

        # --- Load and Combine Data ---
        required_files = [train_inputs_path, train_outputs_path, test_inputs_path, test_outputs_path]
        if not all(os.path.exists(p) for p in required_files):
            missing = [p for p in required_files if not os.path.exists(p)]
            print(f"Error: Required input CSV files not found for {matbench_set}, fold {fold_number}:")
            for p in missing: print(f"  - {p}")
            print("Please ensure the DOWNLOAD_MATBENCH action has been run successfully.")
            return

        try:
            print("Loading CSV data...")
            df_train_in = pd.read_csv(train_inputs_path)
            df_train_out = pd.read_csv(train_outputs_path)
            df_test_in = pd.read_csv(test_inputs_path)
            df_test_out = pd.read_csv(test_outputs_path)

            # Rename mbid to material_id if present
            for df in [df_train_in, df_train_out, df_test_in, df_test_out]:
                if 'mbid' in df.columns:
                    df.rename(columns={'mbid': 'material_id'}, inplace=True)

            # Merge inputs and outputs based on 'material_id'
            if 'material_id' not in df_train_in.columns or 'material_id' not in df_train_out.columns:
                raise ValueError("'material_id' column missing in train input/output CSVs.")
            if 'material_id' not in df_test_in.columns or 'material_id' not in df_test_out.columns:
                raise ValueError("'material_id' column missing in test input/output CSVs.")

            df_train = pd.merge(df_train_in, df_train_out, on='material_id', how='inner')
            df_test = pd.merge(df_test_in, df_test_out, on='material_id', how='inner')

            # Combine train and test sets
            df_orig = pd.concat([df_train, df_test], ignore_index=True)
            print(f"Loaded and combined data shape: {df_orig.shape}")
            print("Combined data columns:", df_orig.columns.tolist())

        except Exception as e:
            print(f"Error loading or merging CSV data: {e}")
            return

    # --- Pre-processing: Identify input column, convert to compositions, filter ---
    if 'target' not in df_orig.columns: raise ValueError("Combined DataFrame must contain 'target'.")
    if 'material_id' not in df_orig.columns: raise ValueError("Combined DataFrame must contain 'material_id'.")

    mat_col = None
    col_type = None
    if 'composition' in df_orig.columns:
        mat_col = 'composition'
        col_type = 'composition'
        print("Using 'composition' column for input.")
    elif 'structure' in df_orig.columns:
        mat_col = 'structure'
        col_type = 'structure'
        print("Using 'structure' column for input (will extract compositions).")
    else:
        raise ValueError("Combined DataFrame must contain 'composition' or 'structure' column.")

    def to_composition(val, input_type):
        if pd.isna(val): return None
        if input_type == 'composition':
            try: return Composition(val)
            except: return None
        elif input_type == 'structure':
            # Handle both JSON string and dict representations
            struct = None
            try:
                if isinstance(val, str):
                    struct_dict = json.loads(val)
                    struct = Structure.from_dict(struct_dict)
                elif isinstance(val, dict): # Should not happen from CSV, but for robustness
                     struct = Structure.from_dict(val)
                # Pymatgen Structure objects won't be directly in CSV, this is unlikely
                # elif isinstance(val, Structure):
                #      struct = val
                if struct:
                    return struct.composition
                else:
                    return None
            except Exception as e:
                # print(f"Warning: Could not convert structure to composition: {val}, Error: {e}") # Optional debug
                return None
        return None

    print("Converting materials to Composition objects...")
    # Use a temporary column name again
    df_orig['_pymatgen_composition_temp'] = df_orig[mat_col].apply(lambda x: to_composition(x, col_type))
    valid_mask = df_orig['_pymatgen_composition_temp'].notna()

    if not valid_mask.all():
        print(f"Warning: {sum(~valid_mask)} entries could not be converted to Compositions and will be skipped.")
        df_filtered = df_orig[valid_mask].copy().reset_index(drop=True)
    else:
        df_filtered = df_orig.copy().reset_index(drop=True)

    # Rename temp column
    df_filtered.rename(columns={'_pymatgen_composition_temp': 'pymatgen_composition'}, inplace=True)


    if df_filtered.empty:
        print("Error: No valid compositions found after filtering. Cannot featurize.")
        return

    print(f"Processing {len(df_filtered)} valid compositions.")
    compositions_filtered = df_filtered['pymatgen_composition']
    targets_filtered = df_filtered['target']
    ids_filtered = df_filtered['material_id']
    # Keep essential original columns
    key_cols_to_keep = ['material_id', mat_col, 'target']
    key_cols_filtered = df_filtered[key_cols_to_keep].copy()
    # Rename original material column if it was structure for clarity in final output
    if mat_col == 'structure':
         key_cols_filtered.rename(columns={'structure': 'composition_source_structure'}, inplace=True)

    # --- Featurization (Chunked or Single, Save Incrementally) ---
    featurizer = CompositionOnlyFeaturizer()

    num_chunks = split_dataset if split_dataset and split_dataset > 1 else 1
    if num_chunks > len(df_filtered):
         print(f"Warning: Number of chunks ({num_chunks}) is greater than the number of samples ({len(df_filtered)}). Setting chunks to number of samples.")
         num_chunks = len(df_filtered)

    chunk_indices_list = np.array_split(df_filtered.index, num_chunks)

    # Handle chunk index range
    start_chunk = start_chunk_index if start_chunk_index is not None else 1
    end_chunk = end_chunk_index if end_chunk_index is not None else num_chunks

    if start_chunk < 1: start_chunk = 1
    if end_chunk > num_chunks: end_chunk = num_chunks
    if start_chunk > end_chunk:
        raise ValueError(f"Start chunk index ({start_chunk}) cannot be greater than end chunk index ({end_chunk}).")


    print(f"Starting featurization across chunks {start_chunk}-{end_chunk} of {num_chunks} total chunks...")
    for i in range(start_chunk - 1, end_chunk): # Loop over desired chunk range (0-indexed)
        chunk_index = i # Current chunk index (0-based)
        chunk_number = i + 1 # Current chunk number (1-based, for user messages)
        chunk_indices = chunk_indices_list[chunk_index]

        if len(chunk_indices) == 0: continue # Skip empty chunks
        print(f"\n  Processing chunk {chunk_number}/{num_chunks} ({len(chunk_indices)} samples)...")

        # Define output filename for this chunk *before* processing
        if num_chunks == 1:
            output_csv_path = output_base + '.csv'
        else:
            output_csv_path = f"{output_base}_chunk_{chunk_number}_of_{num_chunks}.csv"

        # Check if this chunk's output already exists
        if os.path.exists(output_csv_path):
            print(f"  Output file already exists: '{output_csv_path}'. Skipping chunk {chunk_number}.")
            continue # Skip to the next chunk

        # Select data for the current chunk using .loc with the indices
        chunk_compositions = compositions_filtered.loc[chunk_indices]
        chunk_targets = targets_filtered.loc[chunk_indices]
        chunk_ids = ids_filtered.loc[chunk_indices]
        chunk_key_cols = key_cols_filtered.loc[chunk_indices].copy() # Get corresponding key columns

        # Create MODData for the chunk
        print("    Creating MODData object for chunk...")
        try:
            chunk_moddata = MODData(
                materials=chunk_compositions.tolist(), # Pass compositions here
                targets=chunk_targets.tolist(),
                structure_ids=chunk_ids.tolist(),      # Use material_id as the unique ID
                target_names=['target']
            )
        except Exception as e:
            print(f"    ERROR: Failed to create MODData object for chunk {chunk_number}: {e}")
            print(f"    Skipping this chunk.")
            continue

        # Featurize the chunk
        print("    Featurizing chunk...")
        chunk_moddata.featurizer = featurizer
        try:
            chunk_moddata.featurize(n_jobs=n_jobs)
            chunk_features_df = chunk_moddata.get_featurized_df()
            print(f"    Featurization successful. Features shape: {chunk_features_df.shape}")
            # Set index to material_id for merging
            chunk_features_df = chunk_features_df.set_index(pd.Index(chunk_ids, name='material_id'))

        except Exception as e:
            print(f"    ERROR: Featurization failed for chunk {chunk_number}: {e}")
            print(f"    Skipping saving for this chunk.")
            del chunk_moddata
            continue # Proceed to the next chunk

        # --- Combine features with key columns for this chunk ---
        print("    Combining features with key columns...")
        chunk_key_cols.set_index('material_id', inplace=True)

        if not chunk_features_df.index.equals(chunk_key_cols.index):
             print("    Error: Index mismatch between features and key columns after featurization.")
             print(f"    Skipping save for chunk {chunk_number}.")
             del chunk_moddata, chunk_features_df
             continue
        else:
             chunk_final_df = pd.concat([chunk_key_cols, chunk_features_df], axis=1)
             chunk_final_df.reset_index(inplace=True) # Put material_id back as column


        # --- Save the featurized chunk immediately ---
        print(f"    Saving featurized chunk {chunk_number}/{num_chunks} to '{output_csv_path}'...")
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            chunk_final_df.to_csv(output_csv_path, index=False)
            print(f"    Successfully saved chunk {chunk_number}.")
        except Exception as e:
            print(f"    ERROR: Failed to save chunk {chunk_number} to CSV: {e}")

        del chunk_moddata, chunk_features_df, chunk_final_df # Help garbage collection

    print(f"\n--- Finished MODNet Composition Featurization process for: {matbench_set} ---")
    if num_chunks > 1:
        print(f"Featurized data saved incrementally in chunks {start_chunk}-{end_chunk} of {num_chunks} total chunks.")
    elif os.path.exists(output_base + '.csv'):
        print(f"Featurized data saved to single file: {output_base + '.csv'}")
    else:
        print(f"No output files were generated (possibly due to errors or skipped chunks.")

# SECTION: COMBINE CHUNKED CSV FILES
# ============================================================================
def combine_chunked_csvs(chunk_file_pattern, output_csv_path=None, unique_id_col='material_id'):
    """
    Finds CSV files matching a pattern, combines them into a single DataFrame,
    optionally removes duplicates based on a unique ID, sorts the result by
    the unique ID using natural sort order, and saves the result.

    Args:
        chunk_file_pattern (str): A file path pattern (e.g., "data/set/*_chunk_*.csv")
                                   to find the chunked CSV files. It can also be a path
                                   to a single chunk file (with '_chunk_' and '_of_' in its
                                   name), in which case the pattern will be inferred.
        output_csv_path (str, optional): The destination path for the combined CSV file.
                                   If None, an output path is automatically inferred by removing
                                   the "_chunk_..._of_..." part from one of the chunk filenames.
        unique_id_col (str): The column name used for identifying and removing
                             duplicate entries and for sorting. Defaults to 'material_id'.
                             Set to None to disable duplicate removal and sorting by ID.
    """
    # --- Helper Functions ---
    def natural_sort_key_files(s):
        """
        Sorts filenames containing numbers naturally (e.g., chunk_10 before chunk_100).
        Used ONLY for sorting the input file list.
        """
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    def _natural_sort_key_single_id(material_id):
        """
        Extracts the numeric part of a material ID for natural sorting.
        Handles IDs like 'mb-123', 'mp-456', 'id789', or just numbers.
        Used via map for sorting the DataFrame column.
        """
        # Attempt to convert to string first to handle potential numeric types robustly
        material_id_str = str(material_id)
        # Regex to find common prefixes (mb-, mp-, id) followed by digits, or just digits
        match = re.search(r'(?:mb-|mp-|id)?(\d+)', material_id_str)
        if match:
            return int(match.group(1))
        # Fallback: If it doesn't match common patterns but is purely numeric string
        if material_id_str.isdigit():
             return int(material_id_str)
        # Final fallback: return the original string representation for comparison
        return material_id_str

    def natural_sort_key_df_column(id_series):
        """
        Applies the single ID natural sort key logic to a pandas Series.
        Passed to the `key` argument of sort_values.
        """
        return id_series.map(_natural_sort_key_single_id)
    # --- End Helper Functions ---        
    print(f"\n--- Starting Combine Chunks ---")
    print(f"Input pattern/path: {chunk_file_pattern}")
    print(f"Unique ID column for deduplication/sorting: {unique_id_col}")

    # Find matching files using glob.
    chunk_files = glob.glob(chunk_file_pattern)

    # If no files were found directly, attempt to infer a glob pattern
    if not chunk_files and os.path.isfile(chunk_file_pattern) and "_chunk_" in chunk_file_pattern and "_of_" in chunk_file_pattern:
        print("No files matched pattern directly, attempting to infer pattern from single file path...")
        base_pattern = re.sub(r"_chunk_\d+_of_\d+\.csv$", "_chunk_*_of_*.csv", chunk_file_pattern)
        if base_pattern != chunk_file_pattern:
            print(f"Inferred pattern: {base_pattern}")
            chunk_files = glob.glob(base_pattern)
        else:
            print("Could not infer a suitable pattern from the provided file path.")

    if not chunk_files:
        print(f"Error: No chunk files found matching pattern or inferred pattern from: {chunk_file_pattern}")
        print("--- Combine Chunks Failed ---")
        return

    # Sort input files using natural sorting for filenames.
    chunk_files = sorted(chunk_files, key=natural_sort_key_files) # Use file-specific key

    # Infer output path if needed
    if output_csv_path is None:
        candidate = os.path.basename(chunk_files[0])
        inferred_candidate = re.sub(r'(.*)_chunk_.*(\.csv)$', r'\1\2', candidate)
        output_dir = os.path.dirname(chunk_files[0])
        output_csv_path = os.path.join(output_dir, inferred_candidate)
        print(f"Inferred output path: {output_csv_path}")
    else:
        print(f"Output file: {output_csv_path}")

    print(f"Found {len(chunk_files)} chunk files to combine:")
    
    num_to_show = 5
    if len(chunk_files) <= 2 * num_to_show:
        for f in chunk_files:
            print(f"  {f}")
    else:
        for f in chunk_files[:num_to_show]:
            print(f"  {f}")
        print("  ...")
        for f in chunk_files[-num_to_show:]:
            print(f"  {f}")

    # Read each chunk file into dataframes.
    df_list = []
    total_rows_read = 0
    for i, f in enumerate(chunk_files):
        try:
            df_chunk = pd.read_csv(f)
            if not df_chunk.empty:
                df_list.append(df_chunk)
                total_rows_read += len(df_chunk)
            else:
                print(f"  Warning: Skipping empty file {f}")
        except pd.errors.EmptyDataError:
            print(f"  Warning: Skipping empty file {f} (pandas EmptyDataError)")
        except Exception as e:
            print(f"  Warning: Could not read or parse file {f}: {e}. Skipping.")
        if (i + 1) % 50 == 0:
            print(f"  Read {i+1}/{len(chunk_files)} files...")

    if not df_list:
        print("Error: No valid data loaded from any chunk files. Cannot combine.")
        print("--- Combine Chunks Failed ---")
        return

    print(f"\nRead a total of {total_rows_read} rows across {len(df_list)} non-empty files.")
    print("Concatenating dataframes...")
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Combined dataframe shape before deduplication: {combined_df.shape}")

    # # Perform KNN imputation for missing values
    # if combined_df.isnull().any().any():
    #     from sklearn.impute import KNNImputer
    #     print("Performing KNN imputation for missing values...")
    #     # Separate ID column if it exists
    #     id_col = None
    #     if unique_id_col and unique_id_col in combined_df.columns:
    #         id_col = combined_df[unique_id_col].copy()
    #         combined_df = combined_df.drop(columns=[unique_id_col])
        
    #     # Initialize and fit KNN imputer
    #     imputer = KNNImputer(n_neighbors=5)
    #     numeric_cols = combined_df.select_dtypes(include=['float64', 'int64']).columns
    #     if len(numeric_cols) > 0:
    #         combined_df[numeric_cols] = imputer.fit_transform(combined_df[numeric_cols])
        
    #     # Restore ID column if it was separated
    #     if id_col is not None:
    #         combined_df[unique_id_col] = id_col
        
    #     print("KNN imputation completed")
    # else:
    #     print("No missing values found, skipping imputation")

    # Remove duplicates based on the unique ID column if applicable.
    if unique_id_col and unique_id_col in combined_df.columns:
        initial_rows = combined_df.shape[0]
        combined_df.drop_duplicates(subset=[unique_id_col], keep='first', inplace=True)
        final_rows = combined_df.shape[0]
        if final_rows < initial_rows:
            print(f"Removed {initial_rows - final_rows} duplicate rows based on '{unique_id_col}'.")
            print(f"Combined dataframe shape after deduplication: {combined_df.shape}")
        else:
            print(f"No duplicate rows found based on '{unique_id_col}'.")
    elif unique_id_col:
        print(f"Warning: Unique ID column '{unique_id_col}' not found in combined data. Skipping deduplication.")

    # Sort the combined DataFrame by unique_id_col using natural sort order
    if unique_id_col and unique_id_col in combined_df.columns:
        print(f"Sorting combined data by '{unique_id_col}' using natural sort order...")
        try:
            combined_df.sort_values(
                by=unique_id_col,
                key=natural_sort_key_df_column,
                inplace=True,
                ignore_index=True
            )
            print(f"Data sorted. Final shape remains: {combined_df.shape}")
        except Exception as e:
            print(f"Warning: Failed to sort data by '{unique_id_col}'. Error: {e}")
            print("Proceeding with unsorted data.")
    elif unique_id_col:
        pass

    # Save the combined DataFrame to CSV.
    print(f"\nSaving combined data to: {output_csv_path}")
    try:
        output_dir = os.path.dirname(output_csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        combined_df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved combined file with {len(combined_df)} rows.")
    except Exception as e:
        print(f"Error: Failed to save combined file to {output_csv_path}: {e}")
        print("--- Combine Chunks Failed ---")
        return

    print("--- Finished Combine Chunks ---")

# ============================================================================
# SECTION: SISSO FORMULA CALCULATIONS
# ============================================================================
def process_data_for_sisso(input_csv, target_col="target", discard_columns=None, robust_scaler_path='robust_scaler_params.json'):
    """
    Process a CSV file to create a SISSO-friendly version using pre-saved scaling parameters.

    Parameters:
      input_csv       - (str) Path to the input CSV file.
      target_col      - (str) Name of the target column (default "target").
      discard_columns - (list) List of columns to discard if present.
                        (default ["composition", "material_id"])
      robust_scaler_path - (str) Path to the JSON file containing the pre-fitted RobustScaler parameters.

    The function will:
      1. Drop unwanted ("discard") columns.
      2. Separate the target column from the features.
      3. Run KNN imputation (if any NaN's are found in the features).
      4. Scale the features using the pre-saved RobustScaler parameters.
      5. Rename all (feature and flag) columns to simple letter labels
         (A, B, C, …, AA, AB, ...), saving the mapping as a JSON file.
      6. Insert a sample ID and the target column as the first two columns.
      7. Reset the index and write the complete DataFrame to a CSV file.
         The output CSV file is named the same as the input file plus the suffix _sisso.csv.
    """
    import os
    import json
    import pandas as pd
    from sklearn.impute import KNNImputer

    if discard_columns is None:
        discard_columns = ["composition", "material_id", "mbid", "structure"]

    print("Loading data from CSV...")
    df = pd.read_csv(input_csv)
    base_name = os.path.splitext(input_csv)[0]
    
    # Drop unwanted columns if they exist.
    for col in discard_columns:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
            print(f"Dropped column: {col}")

    # Check that the target column exists.
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the CSV file.")

    # Separate the target column.
    y = df[target_col]
    df.drop(columns=[target_col], inplace=True)
    # ----- ADAPTATION FOR CLASSIFICATION TARGETS -----
    if y.dtype == bool:
        print("Detected boolean target. Converting True/False to 1/0.")
        y = y.astype(int)
    elif y.dtype == object:
        # If not already numeric, try converting;
        # if there are still NaNs, we assume it is categorical and create a mapping.
        y_numeric = pd.to_numeric(y, errors='coerce')
        if y_numeric.isna().any():
            unique_vals = sorted(y.dropna().unique())
            mapping = {val: i for i, val in enumerate(unique_vals)}
            print(f"Detected categorical target. Applying mapping: {mapping}")
            y = y.map(mapping)
        else:
            y = y_numeric
    # ---------------------------------------------------
    # ------------------- KNN Imputation (only if missing values exist) -------------------
    if df.isna().sum().sum() > 0:
        print("Missing values detected. Running KNN imputation...")
        # Record the missing flags.
        missing_flags = df.isna()

        # Perform KNN imputation.
        knn_imputer = KNNImputer(n_neighbors=5, weights="uniform")
        df_imputed = pd.DataFrame(knn_imputer.fit_transform(df),
                                  columns=df.columns,
                                  index=df.index)
        df = df_imputed.copy()

        # For every column that originally had missing data, add a corresponding flag column.
        for col in df.columns:
            if missing_flags[col].any():
                df[f"missing_{col}"] = missing_flags[col].astype(int)
    else:
        print("No missing values detected. Skipping imputation.")

    # ------------------- Load Scaling Parameters -------------------
    print(f"Loading scaling parameters from '{robust_scaler_path}'...")
    try:
        with open(robust_scaler_path, 'r') as f:
            scaling_params = json.load(f)
        print(f"Scaling parameters loaded successfully from '{robust_scaler_path}'.")
    except Exception as e:
        raise FileNotFoundError(f"Error loading scaling parameters from '{robust_scaler_path}': {e}")

    # Identify the common features between the current DataFrame and the saved scaling parameters
    common_features = [
        f for f in df.columns if f in scaling_params and not f.startswith('missing_')
    ]

    # Apply scaling only to the common features
    for feature in common_features:
        median, scale_val = scaling_params[feature]['median'], scaling_params[feature]['iqr']
        df[feature] = (df[feature] - median) / scale_val

    # ------------------- Rename Columns -------------------
    # Store original column names (features and any additional "missing_" flags).
    original_columns = df.columns.tolist()

    # Create new column names using letters.
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    new_columns = []
    counter = 0
    while len(new_columns) < len(original_columns):
        suffix = str(counter) if counter > 0 else ''
        new_columns.extend([letter + suffix for letter in alphabet])
        counter += 1
    new_columns = new_columns[:len(original_columns)]

    # Create and save the mapping dictionary.
    mapping_dict = dict(zip(new_columns, original_columns))
    mapping_filename = f"{base_name}_sisso_column_mapping.json"
    with open(mapping_filename, "w") as f:
        json.dump(mapping_dict, f, indent=4)
    print(f"Column mapping saved as '{mapping_filename}'.")

    # Apply the new column names.
    df.columns = new_columns

    # ------------------- Insert Target and Sample ID, Finalize DataFrame -------------------
    # Insert the target column (it will be the second column, after sample_id).
    df.insert(0, target_col, y)
    # Insert sample ids as the very first column.
    df.insert(0, "sample_id", range(len(df)))

    # Reset index.
    df.reset_index(drop=True, inplace=True)

    # Save final DataFrame as CSV (with a '#' at the beginning of the header line).
    output_csv = base_name + "_sissoformatted.csv"
    with open(output_csv, "w") as f:
        f.write("#")  # Write '#' at the beginning of the header.
        df.to_csv(f, index=False)

    print(f"Processed data saved as '{output_csv}'.")
 

# ============================================================================
# SECTION: PREPARE SISSO FORMULA CALCULATIONS
# ============================================================================
def call_setup_sisso_hp_optimization(csv_filepath):
    import sys; sys.path.append(".")
    from setup_sisso_hp_optimization import setup_hyperparameters
    # Define the calculation type
    import pandas as pd
    df = pd.read_csv(csv_filepath)
    print(df)
    print(df['target'])
    # Check if target column exists and determine calc_type based on unique values
    target_values = df['target'].nunique()
    calc_type = "classification" if target_values <= 5 else "regression"
    
    
    setup_hyperparameters(
        data_file=csv_filepath,
        property_key="target",
        stage=1,
        desc_dim=2,
        max_rung=1,
        ns_list=[10],
        nr_list=[3],
        calc_type=calc_type,
    )
############ NOW YOU SHOULD RUN THE SCRIPT: ./prefix_run.sh

def generate_sisso_formula_file(prefix):
    """
    Generate a SISSO formula file using a prefix.
    
    The filepaths are constructed as follows based on the provided prefix:
      - Mapping file: {prefix}_sisso_column_mapping.json
      - SIS_summary file: {prefix}_sisso/stage1/ns_10_nr_3/feature_space/SIS_summary.txt
      - Output file: {prefix}_SISSO_mapped_features.txt

    The function reads the mapping file (which maps simple labels to original column names),
    then processes the SIS_summary file to translate each feature expression (by replacing tokens
    like "A7" with a Python-compatible expression referencing the original column, i.e. df["col_name"]).
    The translated expressions are printed and saved into the output file.
    """
    import os
    import re
    import json
    # Construct file paths based on the prefix.
    mapping_filepath = f"{prefix}_sisso_column_mapping.json"
    sis_summary_filepath = f"{prefix}_sissoformatted/stage1/ns_10_nr_3/feature_space/SIS_summary.txt"
    output_formula_filepath = f"{prefix}_SISSO_mapped_features.txt"

    # Step 1. Read the column mapping.
    if not os.path.exists(mapping_filepath):
        raise FileNotFoundError(f"Mapping file '{mapping_filepath}' not found.")
    with open(mapping_filepath, "r") as f:
        col_mapping = json.load(f)
    # For safety, convert mapping keys to uppercase:
    col_mapping = {k.upper(): v for k, v in col_mapping.items()}

    # Helper function: translate a token like 'A7' into a column access expression.
    def translate_token(token):
        token = token.upper()
        if token not in col_mapping:
            raise ValueError(f"Token '{token}' not found in column mapping.")
        original_col = col_mapping[token]
        return f'df["{original_col}"]'

    # Step 2. Read the SIS_summary file and extract expressions.
    if not os.path.exists(sis_summary_filepath):
        raise FileNotFoundError(f"SIS_summary file '{sis_summary_filepath}' not found.")
        
    expressions = {}  # holds mapping from feature id to raw expression
    with open(sis_summary_filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Expect lines in the format: FEAT_ID   Score   Feature_Expression
            parts = line.split(None, 2)
            if len(parts) >= 3:
                feat_id, score, feature_expr = parts
                expressions[feat_id] = feature_expr

    # Step 3. Convert a feature expression to an evaluable Python expression.
    def convert_expression(expr):
        # Optionally remove outer parentheses.
        expr = expr.strip()
        if expr.startswith("(") and expr.endswith(")"):
            expr = expr[1:-1]
        # Use regex to match tokens: one or more letters optionally followed by digits.
        token_pattern = re.compile(r"([A-Z]+\d*)")
        def repl(match):
            token = match.group(0)
            return translate_token(token)
        new_expr = token_pattern.sub(repl, expr)
        return new_expr

    # Step 4. Process and translate all expressions.
    translated_expr = {}
    print("Translated feature expressions:")
    for feat_id, expr in expressions.items():
        # For absolute value notation, replace "|" with abs() calls.
        if expr.count("|") >= 2:
            expr = expr.replace("|", "abs(", 1)
            expr = expr.replace("|", ")", 1)
        py_expr = convert_expression(expr)
        translated_expr[feat_id] = py_expr
        print(f"Feature {int(feat_id)+1}: {py_expr}")

    # Step 5. Save translated expressions to the output file.
    with open(output_formula_filepath, "w") as f:
        f.write("# Translated Feature Expressions\n")
        f.write("# Format: Feature_ID: Python Expression\n")
        f.write("#" + "-"*50 + "\n")
        for feat_id, expr in translated_expr.items():
            f.write(f"Feature {int(feat_id)+1}: {expr}\n")
    print(f"\nTranslated feature expressions saved to '{output_formula_filepath}'.")
# ============================================================================
def get_mattervial_features(matbench_set=None, csv_path=None, n_jobs=24, split_dataset=None, start_chunk_index=None, end_chunk_index=None, mvl_recalc=False):
    """
    Calculates MatterVial features (SISSO formulas + pGNN-based) for a dataset,
    processing in chunks for scalability. Saves each chunk incrementally.
    Handles datasets with only composition or structure+composition.

    Args:
        matbench_set (str, optional): The name of the matbench dataset (e.g., 'matbench_mp_e_form').
                                      Loads pre-generated CSVs if provided.
        csv_path (str, optional): Path to the CSV file containing the data (e.g., pre-featurized
                                  with Matminer). Required if matbench_set is None.
        n_jobs (int): Number of parallel jobs potentially used by featurizers (though
                      MatterVial featurizers might not support it directly). Defaults to 24.
        split_dataset (int, optional): Number of chunks to split processing into.
                                       Defaults to None (process as one chunk).
        start_chunk_index (int, optional): Starting index of the chunk to process (1-based index).
                                          If None, starts from the first chunk.
        end_chunk_index (int, optional): Ending index of the chunk to process (inclusive, 1-based index).
                                        If None, processes up to the last chunk.
        mvl_recalc (bool, optional): If True, only execute MVL16 and MVL32 featurizers. Defaults to False.

    Output:
        Saves featurized data incrementally in chunked CSV files. A separate step
        (e.g., using combine_chunked_csvs) is needed to merge them.
        The output filenames will be like '{base}_mattervial_chunk_N_of_M.csv'.
    """
    import os
    import pandas as pd
    import json
    import warnings
    import numpy as np
    from pymatgen.core import Structure, Composition
    warnings.filterwarnings("ignore")
    print(f"\n--- Starting GET_MATTERVIAL_FEATURES (Chunked) ---")

    if matbench_set is None and csv_path is None:
        raise ValueError("Either matbench_set or csv_path must be provided")

    # --- Lazy Import MatterVial Featurizers ---
    try:
        if not mvl_recalc:
            from mattervial.featurizers import get_sisso_features
            from mattervial.featurizers.structure import l_MM_v1, l_OFM_v1, mvl32, mvl16
            from mattervial.featurizers.composition import RoostModelFeaturizer
        else:
            from mattervial.featurizers.structure import mvl32, mvl16
        print("Successfully imported MatterVial featurizers.")
    except ImportError as e:
        print(f"Error: Failed to import MatterVial featurizers: {e}")
        print("Please ensure the 'mattervial' package is installed.")
        return
    except Exception as e:
         print(f"An unexpected error occurred during MatterVial import: {e}")
         return

    # --- Load Input Data ---
    print(f"Loading data from provided CSV: {csv_path}")
    try:
        df_orig = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {csv_path}")
        return
    except Exception as e:
        print(f"Error loading CSV {csv_path}: {e}")
        return

    if 'mbid' in df_orig.columns:
        df_orig.rename(columns={'mbid': 'material_id'}, inplace=True)
    output_base = os.path.splitext(csv_path)[0] + '_mattervial'

    # --- Pre-processing: Identify input, Convert to Pymatgen Objects, Filter ---
    if 'material_id' not in df_orig.columns: raise ValueError("Combined DataFrame must contain 'material_id'.")

    mat_col = None
    col_type = None # 'structure' or 'composition'
    has_structure_col = 'structure' in df_orig.columns
    has_composition_col = 'composition' in df_orig.columns

    if has_structure_col:
        mat_col = 'structure'
        col_type = 'structure'
        print("Input type: 'structure' column found.")
    elif has_composition_col:
        mat_col = 'composition'
        col_type = 'composition'
        print("Input type: 'composition' column found (no 'structure' column).")
    else:
        raise ValueError("DataFrame must contain either a 'structure' or 'composition' column.")

    # Define conversion functions
    def to_structure(val):
        if pd.isna(val): return None
        if isinstance(val, str):
            try: d = json.loads(val); return Structure.from_dict(d)
            except Exception: return None
        return None

    def to_composition(val, input_type):
        if pd.isna(val): return None
        if input_type == 'composition':
            try: return Composition(str(val))
            except Exception: return None
        elif input_type == 'structure':
            struct = to_structure(val)
            if struct: return struct.composition
            else: return None
        return None

    # Apply conversion and filtering
    pymatgen_objects = None
    valid_mask = None
    temp_col_name = f'_pymatgen_{col_type}_temp'

    print(f"Converting '{mat_col}' column to Pymatgen {col_type.capitalize()} objects...")
    if col_type == 'structure':
        df_orig[temp_col_name] = df_orig[mat_col].apply(to_structure)
        valid_mask = df_orig[temp_col_name].notna()
        pymatgen_objects = df_orig.loc[valid_mask, temp_col_name]
    else:
        df_orig[temp_col_name] = df_orig[mat_col].apply(lambda x: to_composition(x, 'composition'))
        valid_mask = df_orig[temp_col_name].notna()
        pymatgen_objects = df_orig.loc[valid_mask, temp_col_name]

    if not valid_mask.all():
        print(f"Warning: {sum(~valid_mask)} entries could not be converted and will be skipped.")
        df_filtered = df_orig[valid_mask].copy().reset_index(drop=True)
    else:
        df_filtered = df_orig.copy().reset_index(drop=True)

    if df_filtered.empty:
        print("Error: No valid structures or compositions found after filtering. Cannot featurize.")
        return

    df_filtered.drop(columns=[temp_col_name], inplace=True, errors='ignore')
    pymatgen_objects = pymatgen_objects.reset_index(drop=True)

    print(f"Processing {len(df_filtered)} valid entries.")
    ids_filtered = df_filtered['material_id']

    # --- Chunking Setup ---
    num_chunks = split_dataset if split_dataset and split_dataset > 1 else 1
    if num_chunks > len(df_filtered):
         print(f"Warning: Number of chunks ({num_chunks}) > samples ({len(df_filtered)}). Setting chunks = samples.")
         num_chunks = len(df_filtered)

    chunk_indices_list = np.array_split(df_filtered.index, num_chunks)

    start_chunk = start_chunk_index if start_chunk_index is not None else 1
    end_chunk = end_chunk_index if end_chunk_index is not None else num_chunks
    if start_chunk < 1: start_chunk = 1
    if end_chunk > num_chunks: end_chunk = num_chunks
    if start_chunk > end_chunk:
        raise ValueError(f"Start chunk index ({start_chunk}) cannot be > end chunk index ({end_chunk}).")

    # --- Initialize Roost Featurizers (outside loop) ---
    roost_gap, roost_eform = None, None
    if not mvl_recalc:
        try:
            roost_gap = RoostModelFeaturizer(model_type='mpgap')
            roost_eform = RoostModelFeaturizer(model_type='oqmd_eform')
            print("Initialized Roost featurizers.")
        except Exception as e:
            print(f"Warning: Could not initialize Roost featurizers: {e}. Roost features will be skipped.")
            roost_gap, roost_eform = None, None

    # --- Process Chunks ---
    print(f"Starting MatterVial featurization across chunks {start_chunk}-{end_chunk} of {num_chunks} total chunks...")
    for i in range(start_chunk - 1, end_chunk):
        chunk_index = i
        chunk_number = i + 1
        chunk_indices = chunk_indices_list[chunk_index]

        if len(chunk_indices) == 0: continue
        print(f"\n  Processing chunk {chunk_number}/{num_chunks} ({len(chunk_indices)} samples)...")

        if num_chunks == 1:
            output_csv_path = output_base + '.csv'
        else:
            output_csv_path = f"{output_base}_chunk_{chunk_number}_of_{num_chunks}.csv"

        if os.path.exists(output_csv_path):
            print(f"  Output file already exists: '{output_csv_path}'. Skipping chunk {chunk_number}.")
            continue

        chunk_df = df_filtered.loc[chunk_indices].copy()
        chunk_ids = ids_filtered.loc[chunk_indices].tolist()
        chunk_pmg_objects = pymatgen_objects.loc[chunk_indices]

        all_chunk_features = []

        if not mvl_recalc:
            # 1. SISSO Features (works for both structure and composition)
            print("    Featurizing with SISSO formulas...")
            try:
                df_sisso_chunk = get_sisso_features(chunk_df, type="SISSO_FORMULAS_v1")
                df_sisso_chunk = df_sisso_chunk.set_index(pd.Index(chunk_ids, name='material_id'))
                all_chunk_features.append(df_sisso_chunk)
                print(f"    SISSO features shape: {df_sisso_chunk.shape}")
            except Exception as e:
                print(f"    ERROR: SISSO featurization failed for chunk {chunk_number}: {e}")

            # 2. Roost Features (works for both structure and composition)
            if roost_gap:
                try:
                    print("      Featurizing with Roost MP GAP...")
                    df_roost_gap_chunk = roost_gap.get_features(chunk_pmg_objects)
                    df_roost_gap_chunk = df_roost_gap_chunk.set_index(pd.Index(chunk_ids, name='material_id'))
                    all_chunk_features.append(df_roost_gap_chunk)
                    print(f"      Roost MP GAP features shape: {df_roost_gap_chunk.shape}")
                except Exception as e:
                    print(f"      ERROR: Roost MP GAP featurization failed: {e}")
            if roost_eform:
                 try:
                    print("      Featurizing with Roost OQMD Eform...")
                    df_roost_eform_chunk = roost_eform.get_features(chunk_pmg_objects)
                    df_roost_eform_chunk = df_roost_eform_chunk.set_index(pd.Index(chunk_ids, name='material_id'))
                    all_chunk_features.append(df_roost_eform_chunk)
                    print(f"      Roost OQMD Eform features shape: {df_roost_eform_chunk.shape}")
                 except Exception as e:
                    print(f"      ERROR: Roost OQMD Eform featurization failed: {e}")

        # 3. Structure-Based Features (only if input is structure)
        if col_type == 'structure':
            print("    Featurizing with structure-based MatterVial featurizers...")
            chunk_structures = chunk_pmg_objects

            if mvl_recalc:
                featurizers_struct = {"mvl32": mvl32, "mvl16": mvl16}
            else:
                featurizers_struct = {
                    "l_MM_v1": l_MM_v1, "l_OFM_v1": l_OFM_v1,
                    "mvl32": mvl32, "mvl16": mvl16
                }

            for name, featurizer in featurizers_struct.items():
                try:
                    print(f"      Featurizing with {name}...")
                    df_feat_chunk = featurizer.get_features(chunk_structures)
                    df_feat_chunk = df_feat_chunk.set_index(pd.Index(chunk_ids, name='material_id'))
                    all_chunk_features.append(df_feat_chunk)
                    print(f"      {name} features shape: {df_feat_chunk.shape}")
                except Exception as e:
                     print(f"      ERROR: {name} featurization failed: {e}")

        # --- Combine original chunk data with new features ---
        print("    Combining original data with new features for the chunk...")
        chunk_df.set_index('material_id', inplace=True)

        if not all_chunk_features:
            print("    Warning: No MatterVial features were successfully generated for this chunk.")
            chunk_final_df = chunk_df
        else:
            try:
                chunk_final_df = pd.concat([chunk_df] + all_chunk_features, axis=1)
                chunk_final_df = chunk_final_df.loc[:, ~chunk_final_df.columns.duplicated()]
            except Exception as e:
                 print(f"    ERROR: Failed to concatenate features for chunk {chunk_number}: {e}")
                 print(f"    Skipping saving for this chunk.")
                 del chunk_df, all_chunk_features
                 if 'df_sisso_chunk' in locals(): del df_sisso_chunk
                 if 'df_roost_gap_chunk' in locals(): del df_roost_gap_chunk
                 if 'df_roost_eform_chunk' in locals(): del df_roost_eform_chunk
                 if 'df_feat_chunk' in locals(): del df_feat_chunk
                 continue

        chunk_final_df.reset_index(inplace=True)

        # --- Save the featurized chunk ---
        print(f"    Saving featurized chunk {chunk_number}/{num_chunks} to '{output_csv_path}'...")
        try:
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            chunk_final_df.to_csv(output_csv_path, index=False)
            print(f"    Successfully saved chunk {chunk_number} with shape {chunk_final_df.shape}.")
        except Exception as e:
            print(f"    ERROR: Failed to save chunk {chunk_number} to CSV: {e}")

        # --- Clean up memory for the chunk ---
        try:
            del chunk_df, chunk_ids, chunk_pmg_objects, all_chunk_features, chunk_final_df
            if 'df_sisso_chunk' in locals(): del df_sisso_chunk
            if 'df_roost_gap_chunk' in locals(): del df_roost_gap_chunk
            if 'df_roost_eform_chunk' in locals(): del df_roost_eform_chunk
            if 'df_feat_chunk' in locals(): del df_feat_chunk
        except Exception as e:
            print(f"    ERROR: Failed to clean up memory for chunk {chunk_number}: {e}")
    

# ============================================================================
# SECTION: ROOST FEATURES CODE (DEPRECATED, GET DIRECTLY FROM pGNN)
# ============================================================================
# def get_roost_features(data_path):
#     # ----- Begin ROOST FEATURES SECTION -----
#     import os
#     import types
#     import torch
#     import pandas as pd
#     from torch.utils.data import DataLoader
#     from roost.roost.data import CompositionData, collate_batch
#     from roost.roost.model import Roost

#     # Define a hook factory to capture activations.
#     def get_hook(name, storage):
#         def hook(module, input, output):
#             if isinstance(output, types.GeneratorType):
#                 output_list = list(output)
#                 output_clean = [o.detach() if hasattr(o, "detach") else o for o in output_list]
#             else:
#                 output_clean = output.detach() if hasattr(output, "detach") else output
#             storage[name] = output_clean
#         return hook
    
#     def get_output_path(file_path):
#         dirname = os.path.dirname(file_path)
#         basename = os.path.basename(file_path)
#         name, ext = os.path.splitext(basename)
#         return os.path.join(dirname, f"{name}_roost{ext}")

#     output_path = get_output_path(data_path)

#     # Task settings and featurizer embedding file.
#     task_dict = {"target": "regression"}
    
#     # Create a temporary CompositionData to obtain element embedding length.
#     print("Using FEA_PATH:", FEA_PATH)
#     print("Loading OQMD model from:", OQMD_CHECKPOINT_PATH)
#     print("Loading MPGAP model from:", MPGAP_CHECKPOINT_PATH)
#     tmp_dataset = CompositionData(data_path=data_path, fea_path=FEA_PATH, task_dict=task_dict)
#     elem_emb_len = tmp_dataset.elem_emb_len
#     n_targets = [2]
#     robust = False
#     elem_fea_len = 64
#     n_graph = 3

#     # Instantiate two Roost models with shared architecture.
#     model_mpgap = Roost(
#         task_dict=task_dict,
#         robust=robust,
#         n_targets=n_targets,
#         elem_emb_len=elem_emb_len,
#         elem_fea_len=elem_fea_len,
#         n_graph=n_graph,
#         elem_heads=3,
#         elem_gate=[256],
#         elem_msg=[256],
#         cry_heads=3,
#         cry_gate=[256],
#         cry_msg=[256],
#         trunk_hidden=[1024, 512],
#         out_hidden=[256, 128, 64],
#         device="gpu"
#     )
#     checkpoint_mpgap = torch.load(MPGAP_CHECKPOINT_PATH, map_location=torch.device("cpu"))
#     state_dict = checkpoint_mpgap["state_dict"] if "state_dict" in checkpoint_mpgap else checkpoint_mpgap
#     model_mpgap.load_state_dict(state_dict)
#     model_mpgap.eval()
#     print("Model mpgap loaded successfully!")

#     model_oqmd = Roost(
#         task_dict=task_dict,
#         robust=robust,
#         n_targets=n_targets,
#         elem_emb_len=elem_emb_len,
#         elem_fea_len=elem_fea_len,
#         n_graph=n_graph,
#         elem_heads=3,
#         elem_gate=[256],
#         elem_msg=[256],
#         cry_heads=3,
#         cry_gate=[256],
#         cry_msg=[256],
#         trunk_hidden=[1024, 512],
#         out_hidden=[256, 128, 64],
#         device="gpu"
#     )
    
#     checkpoint_oqmd = torch.load(OQMD_CHECKPOINT_PATH, map_location=torch.device("cpu"))
#     state_dict = checkpoint_oqmd["state_dict"] if "state_dict" in checkpoint_oqmd else checkpoint_oqmd
#     model_oqmd.load_state_dict(state_dict)
#     model_oqmd.eval()
#     print("Model oqmd loaded successfully!")

#     # Prepare dictionaries for hook activations.
#     activations_mpgap = {}
#     activations_oqmd = {}
#     for name, module in model_mpgap.named_modules():
#         module.register_forward_hook(get_hook(name, activations_mpgap))
#     for name, module in model_oqmd.named_modules():
#         module.register_forward_hook(get_hook(name, activations_oqmd))

#     # Function to process the dataset.
#     def process_dataset(data_file, output_file):
#         print(f"\nProcessing dataset: {data_file}")
#         df_orig = pd.read_csv(data_file)
#         print(f"Loaded original data with {len(df_orig)} rows.")
        
#         dataset = CompositionData(data_path=data_file, fea_path=FEA_PATH, task_dict=task_dict)
#         loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
#         features_list = []
        
#         with torch.no_grad():
#             for idx, batch in enumerate(loader):
#                 inputs, targets, comp_id, comp_formula = batch
#                 elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx, extra = inputs
#                 record = {
#                     "comp_id": comp_id[0] if isinstance(comp_id, (list, tuple)) else comp_id,
#                     "comp_formula": comp_formula[0] if isinstance(comp_formula, (list, tuple)) else comp_formula,
#                 }
                
#                 # Inference on mpgap model.
#                 activations_mpgap.clear()
#                 _ = model_mpgap(elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx, extra)
#                 if ('output_nns.0.acts.1' in activations_mpgap) and ('material_nn' in activations_mpgap):
#                     output_nn_tensor = activations_mpgap['output_nns.0.acts.1'][0]
#                     material_nn_tensor = activations_mpgap['material_nn'][0]
#                     output_nn_act = output_nn_tensor.cpu().numpy()
#                     material_nn_act = material_nn_tensor.cpu().numpy()
#                 else:
#                     output_nn_act = None
#                     material_nn_act = None
                
#                 if output_nn_act is not None:
#                     for i, val in enumerate(output_nn_act):
#                         record[f"ROOST_mpgap_LayerOutput_#{i+1:02d}"] = val
#                 if material_nn_act is not None:
#                     for i, val in enumerate(material_nn_act):
#                         record[f"ROOST_mpgap_LayerMaterialPooling_#{i+1:02d}"] = val
                
#                 # Inference on oqmd model.
#                 activations_oqmd.clear()
#                 _ = model_oqmd(elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx, extra)
#                 if ('output_nns.0.acts.1' in activations_oqmd) and ('material_nn' in activations_oqmd):
#                     output_nn_tensor = activations_oqmd['output_nns.0.acts.1'][0]
#                     material_nn_tensor = activations_oqmd['material_nn'][0]
#                     output_nn_act = output_nn_tensor.cpu().numpy()
#                     material_nn_act = material_nn_tensor.cpu().numpy()
#                 else:
#                     output_nn_act = None
#                     material_nn_act = None
                
#                 if output_nn_act is not None:
#                     for i, val in enumerate(output_nn_act):
#                         record[f"ROOST_oqmd_eform_LayerOutput_#{i+1:02d}"] = val
#                 if material_nn_act is not None:
#                     for i, val in enumerate(material_nn_act):
#                         record[f"ROOST_oqmd_eform_LayerMaterialPooling_#{i+1:02d}"] = val
                
#                 features_list.append(record)
#                 if (idx + 1) % 50 == 0:
#                     print(f"Processed {idx + 1} samples...")
        
#         df_features = pd.DataFrame(features_list)
#         ### Remove comp_id and comp_formula columns
#         df_features = df_features.drop(columns=["comp_id", "comp_formula"])
#         if len(df_orig) != len(df_features):
#             print("Warning: The number of rows in the original data and computed features do not match!")
        
#         df_augmented = pd.concat([df_orig.reset_index(drop=True), df_features.reset_index(drop=True)], axis=1)
#         print("Preview of the augmented dataframe:")
#         print(df_augmented.head())
#         df_augmented.to_csv(output_file, index=False)
#         print(f"Saved augmented CSV with both Roost feature sets to: {output_file}")

#     process_dataset(data_path, output_path)
    # ----- End ROOST FEATURES SECTION -----

# ---------------------------------------------------------------------------
# SECTION: MODNET TRAINING FROM FEATURIZED CSV 
# ---------------------------------------------------------------------------
# Helper function for feature selection
def xgb_preselection(data, n_jobs=24, target_threshold=800, drop_fraction=0.1):
    """
    Performs recursive feature elimination using XGBoost importance scores.

    In each step, it removes a fraction (drop_fraction) of the *current*
    least important features, until the total number of features is
    below target_threshold.

    Args:
        data (MODData): The MODData object containing df_featurized.
        n_jobs (int): Number of parallel jobs for XGBoost.
        target_threshold (int): The target number of features to reach (exclusive).
                                Feature elimination stops when count <= target_threshold.
        drop_fraction (float): The fraction of *current* features to drop
                               in each iteration (approximate).

    Returns:
        MODData: The MODData object with reduced df_featurized.
    """
    # Lazily import xgboost only if needed
    try:
        import xgboost as xgb
        import numpy as np
    except ImportError:
        print("XGBoost not installed. Skipping XGBoost preselection.")
        print("Install it using: pip install xgboost")
        return data

    # Helper to rename columns for XGBoost compatibility
    def rename_cols(df):
        df.columns = [col.replace("[", "_").replace("]", "_").replace(" ", "_").replace("<","_").replace(">","_").replace(",","_").replace("|","_").replace(".","_")
                      for col in df.columns]
        return df

    cur_features = data.df_featurized.copy()
    initial_num = cur_features.shape[1]

    if initial_num > target_threshold:
        print(f"Starting recursive feature elimination from {initial_num} features...")
        print(f"Target: < {target_threshold} features. Strategy: Remove ~{drop_fraction*100:.0f}% per step.")

        while cur_features.shape[1] > target_threshold:
            current_num_features = cur_features.shape[1]
            if current_num_features <= target_threshold: # Double check condition
                break

            # Rename columns just before fitting XGBoost for compatibility
            xgb_compatible_features = rename_cols(cur_features.copy())

            xgb_model = xgb.XGBRegressor(n_jobs=n_jobs, random_state=1, objective='reg:squarederror')
            try:
                # Ensure data is purely numeric before passing to XGBoost
                X_numeric = xgb_compatible_features.apply(pd.to_numeric, errors='coerce').fillna(0)
                if X_numeric.isnull().any().any():
                     print("Warning: Found NaNs even after fillna(0) before XGBoost fit. Check data.")

                y_numeric = pd.to_numeric(data.df_targets.values.ravel(), errors='coerce')
                if np.isnan(y_numeric).any():
                     print("Error: Found NaNs in target variable. Cannot fit XGBoost.")
                     return data # Stop preselection

                xgb_model.fit(X_numeric.values, y_numeric)
            except ValueError as e:
                 print(f"Error during XGBoost fit: {e}")
                 print("Check for non-numeric data or invalid feature names despite renaming/coercion.")
                 print("Stopping XGBoost preselection due to error.")
                 return data
            except Exception as e:
                 print(f"Unexpected error during XGBoost fit: {e}")
                 print("Stopping XGBoost preselection due to error.")
                 return data

            importances = xgb_model.feature_importances_
            sorted_idx = np.argsort(importances) # Indices sorted from least to most important

            # --- Calculate number of features to drop (10% of current) ---
            num_to_drop = max(1, int(current_num_features * drop_fraction))

            # --- Refinement: Ensure we don't drop features needed to stay above threshold ---
            # If dropping num_to_drop would take us below the threshold, only drop
            # enough to reach the threshold (or slightly below it).
            if current_num_features - num_to_drop < target_threshold:
                 # Calculate how many we *can* drop without going *too far* below threshold
                 # Drop features until we are just above or exactly at the threshold + 1 (to ensure loop breaks next time)
                 # Example: current=850, threshold=800 -> drop 850 - 800 = 50 (instead of 10% = 85)
                 # Example: current=805, threshold=800 -> drop 805 - 800 = 5 (instead of 10% = 80)
                 num_to_drop_adjusted = current_num_features - target_threshold
                 # Ensure we still drop at least one feature if we are above the threshold
                 num_to_drop = max(1, num_to_drop_adjusted)
                 print(f"  Adjusting drop count to {num_to_drop} to approach target threshold ({target_threshold}).")


            # Get original column names corresponding to the least important features
            drop_feature_indices = sorted_idx[:num_to_drop]
            # Use the *original* column names from 'cur_features' for dropping
            drop_features_original_names = cur_features.columns[drop_feature_indices]

            cur_features = cur_features.drop(columns=drop_features_original_names)
            print(f"  Dropped {num_to_drop} features ({len(drop_features_original_names)} actual); remaining features: {cur_features.shape[1]}")

            # --- Safety break: Prevent infinite loops if num_to_drop becomes 0 somehow ---
            if num_to_drop == 0:
                print("Warning: Calculated num_to_drop is 0. Breaking elimination loop.")
                break


        data.df_featurized = cur_features
        print(f"Finished recursive feature elimination. Final feature count: {cur_features.shape[1]}")
    else:
        print(f"Feature count ({initial_num}) is {target_threshold} or below: skipping recursive feature elimination.")

    return data


def matbench_kfold_splits(df_structures, df_targets, n_splits=5, classification=False):
    """Return the pre-defined k-fold splits to use when reporting matbench results."""
    if classification:
        from sklearn.model_selection import StratifiedKFold as KFold
    else:
        from sklearn.model_selection import KFold

    # handles one-hot encoded targets
    if classification and (
        isinstance(df_targets.iloc[0, 0], list)
        or isinstance(df_targets.iloc[0, 0], np.ndarray)
    ):
        def _mapArrayToInt(a):
            return np.array(a).dot(2 ** np.arange(len(a)))
        ycv = df_targets.iloc[:, 0].map(_mapArrayToInt)
    else:
        ycv = df_targets

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=MATBENCH_SEED)
    kf_splits = kf.split(df_structures, y=ycv)
    return kf_splits

def run_mvl_prediction(data_path,
                       target_name,
                       mvl_model,   # a single MVL model name (e.g., "Eform_MP_2019")
                       job_prefix=None,
                       n_folds=5,
                       n_jobs=24,
                       pretrained_model=True):
    """
    Loads a comprehensive CSV file formatted for MATBench (with a "material_id", target,
    and "structure" column, where structures are stored as JSON strings). In this routine the
    "structure" column is processed to produce pymatgen Structure objects. Predictions are obtained
    either fold-by-fold (if pretrained_model==False) or on the whole dataset once (if pretrained_model==True).
    
    When pretrained_model==True the model is evaluated on the entire dataset before k-fold splits.
      - The predictions are saved in a dataframe with columns: material_id, target_true, target_predictions.
      - K-fold splits are then made on this predictions dataframe and metrics are calculated for each fold.
      (Because the model is pre-trained it is not re-evaluated per fold.)
    
    For each fold (when pretrained_model==False):
      - The training and test splits are made using matbench_kfold_splits.
      - For each structure in train and test splits the helper function safe_predict_structure() is called.
        If a structure is invalid (e.g. it raises "cutoff is too small" error), its prediction is set to NaN.
      - Before computing metrics, only valid predictions are retained.
      - Fold predictions and metrics are printed and (optionally) saved.
    
    Args:
        data_path (str): Path to the input CSV file (must contain "material_id", target_name, and "structure").
        target_name (str): Name of the target column.
        mvl_model (str): Name of the pre-trained MVL model to use (e.g., "Eform_MP_2019").
        job_prefix (str, optional): Prefix for output files. If None, generated from CSV basename and model name.
        n_folds (int, optional): Number of cross-validation folds. Defaults to 5.
        n_jobs (int, optional): Number of parallel jobs (unused here, but included for consistency).
        pretrained_model (bool, optional): If True, the model is evaluated once over the entire dataset and
                                           the predictions saved for use with k-fold splits. Defaults to False.
    
    Raises:
        ValueError: If required columns are missing.
    """
    import os
    import re  # For natural sorting
    import json
    import numpy as np
    import pandas as pd
    from pymatgen.core.structure import Structure
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score

    # ----------------------------------------
    # Helper function: safely predict one structure; if conversion fails, return np.nan
    def safe_predict_structure(model, structure):
        try:
            # Note: model.predict_structures expects a list of structures
            pred = model.predict_structures([structure])
            # Assume prediction is returned as an array of shape (1,1) (or (1, ...)).
            return pred[0][0]
        except Exception as e:
            print(f"Skipping structure due to error: {e}")
            return np.nan

    # Helper function: process a list of structures safely.
    def safe_predict_list(model, structures):
        predictions = []
        for idx, s in enumerate(structures):
            pred_val = safe_predict_structure(model, s)
            predictions.append(pred_val)
            if (idx + 1) % 200 == 0:
                print(f"Progress: {idx + 1}/{len(structures)} structures processed...")
        return np.array(predictions)

    # ----------------------------------------
    # --- Load CSV ---
    print(f"Loading full CSV from {data_path} ...")
    try:
        df_all = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading CSV {data_path}: {e}")
        raise

    # --- Natural Sorting by material_id ---
    # Ensure material_id column exists, convert to string and sort in natural order.
    if "material_id" in df_all.columns:
        df_all["material_id"] = df_all["material_id"].astype(str)
        def natural_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]
        df_all.sort_values(by="material_id", key=lambda col: col.map(natural_key), inplace=True)
    else:
        raise ValueError("Input CSV must contain column 'material_id'.")

    # --- Check required columns ---
    required_cols = ["material_id", target_name, "structure"]
    for col in required_cols:
        if col not in df_all.columns:
            raise ValueError(f"Input CSV must contain column '{col}'.")

    # --- Convert 'structure' column from JSON to pymatgen Structure objects ---
    print("Converting 'structure' column to pymatgen Structure objects...")
    try:
        df_all["structure"] = df_all["structure"].apply(lambda x: Structure.from_dict(json.loads(x)))
    except Exception as e:
        print(f"Error converting structure objects: {e}")
        raise

    # --- Determine job_prefix ---
    write_folder = os.path.dirname(data_path)
    if job_prefix is None:
        base_name = os.path.basename(data_path).split(".")[0]
        job_prefix = f"{base_name}_mvl_{mvl_model}"
    print(f"Using job prefix: {job_prefix}")

    # --- Determine task type ---
    targets = df_all[target_name]
    is_classification = (not pd.api.types.is_numeric_dtype(targets)) or (targets.nunique() < 20)
    print(f"Determined task type: {'Classification' if is_classification else 'Regression'}")

    # ----------------------------------------
    # --- Setup output file names ---
    results_file = os.path.join(write_folder, f"results_{job_prefix}_mvl.txt")
    pred_filename = os.path.join(write_folder, f"predictions_{job_prefix}_mvl.csv")
    
    # Write initial results header
    with open(results_file, "w") as f:
        f.write(f"--- MVL Prediction Results using model {mvl_model} ---\n")
        f.write(f"Target: {target_name}\n")
        f.write(f"Data Path: {data_path}\n")
        f.write(f"Folds: {n_folds}\n\n")
    
    # --- Load the single pre-trained MVL model ---
    print(f"Loading MVL model: {mvl_model}")
    try:
        from megnet.utils.models import load_model
        model = load_model(mvl_model)
    except Exception as e:
        print(f"Error loading MVL model {mvl_model}: {e}")
        return

    # ----------------------------------------
    # If pretrained_model flag is True, we evaluate on the entire dataset once.
    if pretrained_model:
        print("Pretrained model mode enabled: Evaluating the full dataset once...")
        all_structures = df_all["structure"].tolist()
        full_preds = safe_predict_list(model, all_structures)

        # Replace NaNs with the global mean of the target (global train mean substitute)
        global_target_vals = df_all[target_name].values
        global_mean = np.nanmean(global_target_vals.astype(float))
        full_preds = np.where(np.isnan(full_preds), global_mean, full_preds)

        # For classification tasks, apply the same processing as in the per-fold loop.
        if is_classification:
            full_preds = 1 - full_preds

        # Save the predictions back in the dataframe.
        # The requested output format is: material_id, target_true, target_predictions.
        df_all["target_predictions"] = full_preds

        # Save these predictions to file.
        pred_df = df_all[["material_id", target_name, "target_predictions"]].copy()
        pred_df.rename(columns={target_name: "target_true"}, inplace=True)
        try:
            pred_df.to_csv(pred_filename, index=False)
            print(f"Saved pre-trained predictions to {pred_filename}")
        except Exception as e:
            print(f"Could not save pre-trained predictions: {e}")
        
        # Now create K-fold splits (using the same matbench_kfold_splits function)
        print(f"Processing {n_folds} folds from pre-computed predictions...")
        fold_data = []
        # Here we assume a function 'matbench_kfold_splits' exists that works with the structures & targets.
        # For example:
        # from your_matbench_utils import matbench_kfold_splits
        for ind, (train_idx, test_idx) in enumerate(
                matbench_kfold_splits(df_all["structure"], df_all[[target_name]], n_splits=n_folds, classification=is_classification)):
            train_data = df_all.iloc[train_idx]
            test_data = df_all.iloc[test_idx]
            fold_data.append((train_data, test_data))
            print(f"Fold {ind} processed.")
    else:
        # --- Prepare Folds using the provided matbench_kfold_splits ---
        fold_data = []
        print(f"Processing {n_folds} folds...")
        # Here we assume matbench_kfold_splits takes the structures and targets (as DataFrame) as arguments.
        for ind, (train_idx, test_idx) in enumerate(
                matbench_kfold_splits(df_all["structure"], df_all[[target_name]], n_splits=n_folds, classification=is_classification)):
            train_data = df_all.iloc[train_idx]  # keep entire row so we retain structure and target info
            test_data = df_all.iloc[test_idx]
            fold_data.append((train_data, test_data))
            print(f"Fold {ind} processed.")

    if not fold_data:
        print("Error: No folds were successfully processed. Exiting.")
        return

    # ----------------------------------------
    # --- Loop over folds and calculate metrics ---
    folds_results = []
    all_fold_predictions = []  # For saving detailed fold predictions.
    fold_counter = 0

    for fold_number, (train_df, test_df) in enumerate(fold_data):
        print(f"\n--- Processing Fold {fold_number} ---")
        # For pretrained_model mode, predictions are already computed.
        # Otherwise, predictions are computed per fold using safe_predict_list.
        if pretrained_model:
            # Get predictions directly from the precomputed column.
            train_preds = train_df["target_predictions"].values
            test_preds = test_df["target_predictions"].values
        else:
            # Obtain list of structures and predict for each split.
            train_structures = train_df["structure"].tolist()
            test_structures = test_df["structure"].tolist()
            train_preds = safe_predict_list(model, train_structures)
            test_preds = safe_predict_list(model, test_structures)

            # Calculate mean of training targets for NaN substitution.
            train_targets_for_mean = train_df[target_name].astype(float).values
            train_target_mean = np.nanmean(train_targets_for_mean)
            train_preds = np.where(np.isnan(train_preds), train_target_mean, train_preds)
            test_preds = np.where(np.isnan(test_preds), train_target_mean, test_preds)

            if is_classification:
                train_preds = 1 - train_preds
                test_preds = 1 - test_preds

        # Get true targets.
        train_targets = train_df[target_name].values
        test_targets = test_df[target_name].values

        fold_result = {}
        if is_classification:
            # Assume threshold of 0.5 for class prediction.
            train_pred_class = (train_preds >= 0.5).astype(int)
            test_pred_class = (test_preds >= 0.5).astype(int)
            try:
                train_true_int = train_targets.astype(int)
                test_true_int = test_targets.astype(int)
            except Exception as e:
                print(f"Error converting targets to int in fold {fold_number}: {e}")
                continue

            train_accuracy = accuracy_score(train_true_int, train_pred_class)
            train_f1 = f1_score(train_true_int, train_pred_class, average='weighted')
            test_accuracy = accuracy_score(test_true_int, test_pred_class)
            test_f1 = f1_score(test_true_int, test_pred_class, average='weighted')
            try:
                train_rocauc = roc_auc_score(train_true_int, train_preds)
                test_rocauc = roc_auc_score(test_true_int, test_preds)
            except Exception:
                train_rocauc = test_rocauc = np.nan

            fold_result = {"Train_Accuracy": train_accuracy, "Train_F1": train_f1, "Train_ROCAUC": train_rocauc,
                           "Test_Accuracy": test_accuracy, "Test_F1": test_f1, "Test_ROCAUC": test_rocauc}
            print(f"Fold {fold_number} - Train Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}, ROC AUC: {train_rocauc:.4f}")
            print(f"Fold {fold_number} - Test Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}, ROC AUC: {test_rocauc:.4f}")
        else:
            # Regression: compute MAE, RMSE, R2 with valid predictions.
            train_mae = mean_absolute_error(train_targets, train_preds)
            train_rmse = mean_squared_error(train_targets, train_preds, squared=False)
            train_r2 = r2_score(train_targets, train_preds)
            test_mae = mean_absolute_error(test_targets, test_preds)
            test_rmse = mean_squared_error(test_targets, test_preds, squared=False)
            test_r2 = r2_score(test_targets, test_preds)
            fold_result = {"Train_MAE": train_mae, "Train_RMSE": train_rmse, "Train_R2": train_r2,
                           "Test_MAE": test_mae, "Test_RMSE": test_rmse, "Test_R2": test_r2}
            print(f"Fold {fold_number} - Train MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")
            print(f"Fold {fold_number} - Test MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")

        folds_results.append(fold_result)
        fold_counter += 1

        # Save test fold predictions: record prediction (even if NaN) for each structure.
        fold_preds_df = pd.DataFrame({
                "material_id": test_df["material_id"],
                f"prediction_f{fold_number}": test_preds
            }).set_index("material_id")
        all_fold_predictions.append(fold_preds_df)

        # Append fold metrics to the results file.
        with open(results_file, "a") as f:
            f.write(f"Fold {fold_number} Metrics:\n")
            for k, v in fold_result.items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")

    # ----------------------------------------
    # --- Aggregate and report average metrics ---
    if folds_results:
        if is_classification:
            avg_train_acc = np.nanmean([r.get("Train_Accuracy", np.nan) for r in folds_results])
            avg_train_f1 = np.nanmean([r.get("Train_F1", np.nan) for r in folds_results])
            avg_train_rocauc = np.nanmean([r.get("Train_ROCAUC", np.nan) for r in folds_results])
            avg_test_acc = np.nanmean([r.get("Test_Accuracy", np.nan) for r in folds_results])
            avg_test_f1 = np.nanmean([r.get("Test_F1", np.nan) for r in folds_results])
            avg_test_rocauc = np.nanmean([r.get("Test_ROCAUC", np.nan) for r in folds_results])
            print(f"\nAverage Classification Metrics across {fold_counter} folds:")
            print(f"  Train - Accuracy: {avg_train_acc:.4f}, F1: {avg_train_f1:.4f}, ROC AUC: {avg_train_rocauc:.4f}")
            print(f"  Test  - Accuracy: {avg_test_acc:.4f}, F1: {avg_test_f1:.4f}, ROC AUC: {avg_test_rocauc:.4f}")
            with open(results_file, "a") as f:
                f.write("--- Average Classification Metrics ---\n")
                f.write(f"  Train - Accuracy: {avg_train_acc:.4f}, F1: {avg_train_f1:.4f}, ROC AUC: {avg_train_rocauc:.4f}\n")
                f.write(f"  Test  - Accuracy: {avg_test_acc:.4f}, F1: {avg_test_f1:.4f}, ROC AUC: {avg_test_rocauc:.4f}\n")
        else:
            avg_train_mae = np.nanmean([r.get("Train_MAE", np.nan) for r in folds_results])
            avg_train_rmse = np.nanmean([r.get("Train_RMSE", np.nan) for r in folds_results])
            avg_train_r2 = np.nanmean([r.get("Train_R2", np.nan) for r in folds_results])
            avg_test_mae = np.nanmean([r.get("Test_MAE", np.nan) for r in folds_results])
            avg_test_rmse = np.nanmean([r.get("Test_RMSE", np.nan) for r in folds_results])
            avg_test_r2 = np.nanmean([r.get("Test_R2", np.nan) for r in folds_results])
            print(f"\nAverage Regression Metrics across {fold_counter} folds:")
            print(f"  Train - MAE: {avg_train_mae:.4f}, RMSE: {avg_train_rmse:.4f}, R2: {avg_train_r2:.4f}")
            print(f"  Test  - MAE: {avg_test_mae:.4f}, RMSE: {avg_test_rmse:.4f}, R2: {avg_test_r2:.4f}")
            with open(results_file, "a") as f:
                f.write("--- Average Regression Metrics ---\n")
                f.write(f"  Train - MAE: {avg_train_mae:.4f}, RMSE: {avg_train_rmse:.4f}, R2: {avg_train_r2:.4f}\n")
                f.write(f"  Test  - MAE: {avg_test_mae:.4f}, RMSE: {avg_test_rmse:.4f}, R2: {avg_test_r2:.4f}\n")
    else:
        print("No fold results available to aggregate.")

    # ----------------------------------------
    # --- Save detailed predictions if available ---
    if all_fold_predictions:
        try:
            # Combine fold prediction dataframes along columns.
            all_preds_df = pd.concat(all_fold_predictions, axis=1)
            # Remove any duplicated columns (if any)
            all_preds_df = all_preds_df.loc[:, ~all_preds_df.columns.duplicated(keep='first')]
            
            # Build a full results dataframe with all material IDs and targets from df_all.
            full_preds_df = df_all[['material_id', target_name]].copy()
            full_preds_df.set_index("material_id", inplace=True)
            
            # Merge with the predictions from the folds (left join so that every material is preserved)
            merged_df = full_preds_df.join(all_preds_df, how="left")
            
            # Define a helper for natural sorting.
            import re
            def natural_sort_key(s):
                return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
            
            merged_df.reset_index(inplace=True)
            merged_df.sort_values(by="material_id", key=lambda col: col.map(natural_sort_key), inplace=True)
            
            # Save the merged full dataframe – it now contains the 'material_id', 'target'
            # for all structures and, where applicable, the predictions (one column per fold).
            merged_df.to_csv(pred_filename, index=False)
            print(f"Saved detailed predictions to {pred_filename}")
        except Exception as e:
            print(f"Could not save detailed predictions: {e}")

    print(f"\nMVL prediction run '{job_prefix}' finished.")
    print(f"Results summary saved to: {results_file}")
    print(f"Fold predictions saved to: {pred_filename}")


# ---------------------------------------------------------------------------
# SECTION: MODNET TRAINING FROM FEATURIZED CSV (FLEXIBLE FEATURE SUBSETS)
# ---------------------------------------------------------------------------
def run_modnet_training(data_path,
                        target_name,
                        feature_sets, # List of feature set names (e.g., ['matminer', 'sisso', 'megnet_ofm'])
                        job_prefix=None,
                        n_folds=5,
                        n_jobs=24,
                        hyperopt_strategy="ga",
                        preset_settings=None,
                        ga_settings=None,
                        xgb_preselect_target=800, 
                        xgb_preselect_fraction=0.1 
                       ):
    """
    Loads a comprehensive CSV file containing various potential features (including pre-computed SISSO).
    Selects specific feature subsets based on the `feature_sets` argument.

    The core process involves:
    1. Loading the full CSV.
    2. Filtering columns based on `feature_sets`.
    3. Creating a base MODData object.
    4. Splitting into folds using Matbench splits.
    5. For each fold:
        a. Apply recursive XGBoost feature elimination (if features > xgb_preselect_target),
           removing ~xgb_preselect_fraction features per step.
        b. Apply MODNet's NMI-based feature selection ordering.
        c. Align test set features.
        d. Save processed fold data.
    6. Train a MODNet model for each fold using either GA or preset hyperparameters.
    7. Evaluate each fold model.
    8. Aggregate results and save predictions.

    Args:
        data_path (str): Path to the input CSV file containing ALL potential features (including pre-computed SISSO).
        target_name (str): Name of the target column in the CSV.
        feature_sets (list[str]): List of strings specifying which feature groups to include.
            Valid names include keys from FEATURE_GROUP_PATTERNS (e.g., 'matminer', 'sisso', 'megnet_ofm',
            'roost_lmp', 'roost_all', 'mvl32').
            Example: ['matminer', 'sisso', 'megnet_ofm']
        job_prefix (str, optional): Prefix for output files and folders. If None, derived from
            CSV filename and included feature sets.
        n_folds (int, optional): Number of cross-validation folds. Defaults to 5.
        n_jobs (int, optional): Number of parallel jobs for XGBoost and GA. Defaults to 24.
        hyperopt_strategy (str, optional): Method for determining hyperparameters ('ga' or 'preset'). Defaults to 'ga'.
        preset_settings (dict, optional): Dictionary of hyperparameters if hyperopt_strategy is 'preset'.
                                          Must contain keys needed for MODNetModel init and fit.
        ga_settings (dict, optional): Settings for the Genetic Algorithm if hyperopt_strategy is 'ga'.
                                      Defaults internally based on modnet defaults and n_jobs.
        xgb_preselect_target (int, optional): The target number of features to aim for after XGBoost
                                             pre-selection. Defaults to 800.
        xgb_preselect_fraction (float, optional): The approximate fraction of current features to remove
                                                  in each XGBoost pre-selection step. Defaults to 0.1 (10%).

    Raises:
        ValueError: If required columns are missing, or an unknown feature set name is provided.
        FileNotFoundError: If the data_path is invalid.
    """
    import os
    import re  # Import regex module
    import pandas as pd
    import numpy as np
    from modnet.preprocessing import MODData
    from modnet.matbench.benchmark import matbench_kfold_splits
    from modnet.hyper_opt.fit_genetic import FitGenetic
    from modnet.models import MODNetModel, EnsembleMODNetModel
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
    from sklearn.impute import SimpleImputer  # Import imputer

    setup_threading()  # Call setup_threading from modnet utils if needed

    # -----------------------------------------------------------------------
    # XGB Preselection function is defined above this function now
    # -----------------------------------------------------------------------

    # --- Initial Setup ---
    print(f"Loading full CSV from {data_path} ...")
    try:
        df_all = pd.read_csv(data_path)
        # For testing purposes, you might use df_all = df_all.head(100)
    except FileNotFoundError:
        print(f"Error: Input data file not found at {data_path}")
        raise
    except Exception as e:
        print(f"Error loading CSV {data_path}: {e}")
        raise

    # --- Sort IDs in natural order ---
    # Define a helper function for natural sorting
    def natural_key(string):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', string)]
    
    # Ensure material_id is string and sort accordingly
    if 'material_id' in df_all.columns:
        df_all["material_id"] = df_all["material_id"].astype(str)
        df_all.sort_values(by="material_id", key=lambda col: col.map(natural_key), inplace=True)
    else:
        raise ValueError("Input CSV must contain 'material_id' column.")

    write_folder = os.path.dirname(data_path)
    if job_prefix is None:
        base_name = os.path.basename(data_path).split(".")[0]
        # Create a descriptive suffix based on feature sets
        feature_suffix = "_".join(sorted(fs.replace('_','-') for fs in feature_sets))  # Sort for consistency
        job_prefix = f"{base_name}_{feature_suffix}"
    print(f"Using job prefix: {job_prefix}")

    # Check required columns
    required_cols = ['material_id', target_name]
    has_composition = 'composition' in df_all.columns
    has_structure = 'structure' in df_all.columns
    if not (has_composition or has_structure):
        raise ValueError("Input CSV must contain either 'composition' or 'structure' column")
    input_type = 'composition' if has_composition else 'structure'

    if not all(col in df_all.columns for col in required_cols):
        raise ValueError(f"Input CSV must contain columns: {required_cols}")
    # Rename target column to 'target' internally for consistency with MODData
    if target_name != 'target':
        df_all = df_all.rename(columns={target_name: 'target'})
        internal_target_name = 'target'  # Use 'target' internally from now on
    else:
        internal_target_name = 'target'

    # --- Feature Selection Logic ---
    print(f"\nSelecting features for sets: {feature_sets}")
    selected_feature_cols = set()
    core_cols_to_drop = ['material_id', input_type, internal_target_name]  # Columns to exclude from features

    # Identify all potential feature columns in the loaded dataframe
    all_potential_feature_cols = df_all.drop(columns=core_cols_to_drop, errors='ignore').columns

    for set_name in feature_sets:
        if set_name in FEATURE_GROUP_PATTERNS:
            pattern = FEATURE_GROUP_PATTERNS[set_name]
            print(f" - Matching columns for '{set_name}' using pattern: {pattern}")
            matches = [col for col in all_potential_feature_cols if re.match(pattern, col)]
            if not matches:
                print(f"   Warning: No columns found matching pattern for '{set_name}'.")
            else:
                print(f"   Found {len(matches)} columns for '{set_name}'.")
                selected_feature_cols.update(matches)
        else:
            matches = [col for col in all_potential_feature_cols if col.startswith(set_name)]
            if matches:
                 print(f" - Matching columns for '{set_name}' using prefix matching.")
                 print(f"   Found {len(matches)} columns for '{set_name}'.")
                 selected_feature_cols.update(matches)
            else:
                 raise ValueError(f"Unknown or unmatched feature set name: '{set_name}'. "
                                  f"Valid names correspond to keys or prefixes defined in FEATURE_GROUP_PATTERNS: "
                                  f"{list(FEATURE_GROUP_PATTERNS.keys())}")

    selected_feature_cols = sorted(list(selected_feature_cols))  # Sort for consistent order

    if not selected_feature_cols:
        raise ValueError("No features selected. Check `feature_sets` and CSV column names/patterns.")
    
    print(f"\nTotal features selected: {len(selected_feature_cols)}")
    features_df = df_all[selected_feature_cols].copy()  # Use copy

    # Extract other essential data
    targets = df_all[internal_target_name]
    structures = df_all[input_type]
    structure_ids = df_all['material_id']
    is_classification = not pd.api.types.is_numeric_dtype(targets) or targets.nunique() < 20
    print(f"Determined task type for splitting: {'Classification' if is_classification else 'Regression'}")
    num_classes = targets.nunique() if is_classification else 0
    
    # --- Create Base MODData ---
    print("Creating initial MODData object...")
    md = MODData(materials=structures, targets=targets, num_classes={'target': num_classes},
                 structure_ids=structure_ids, target_names=[internal_target_name])  # Use original target_name here
    md.df_featurized = features_df  # Assign the selected features
    print(f"Initial MODData created with {md.df_featurized.shape[1]} features.")

    # --- Prepare Fold Data Storage ---
    train_data_folder = os.path.join(write_folder, "train_data")
    test_data_folder = os.path.join(write_folder, "test_data")
    os.makedirs(train_data_folder, exist_ok=True)
    os.makedirs(test_data_folder, exist_ok=True)

    # --- Fold Splitting and Processing ---
    fold_data = []
    print(f"\nProcessing {n_folds} folds...")

    for ind, (train_idx, test_idx) in enumerate(matbench_kfold_splits(md, n_splits=n_folds, classification=is_classification)):
        train_file = os.path.join(train_data_folder, f"{job_prefix}_train_moddata_f{ind}")
        test_file  = os.path.join(test_data_folder, f"{job_prefix}_test_moddata_f{ind}")

        if os.path.isfile(train_file) and os.path.isfile(test_file):
            print(f"Fold {ind}: Loading pre-processed train and test MODData from disk ({job_prefix}_f{ind})...")
            try:
                train_data = MODData.load(train_file)
                test_data  = MODData.load(test_file)
                if is_classification:  # fix for previously generated without
                    train_data.num_classes = {'target': num_classes}
                    test_data.num_classes = {'target': num_classes}
                
                if not hasattr(train_data, 'df_featurized') or not hasattr(test_data, 'df_featurized'):
                     raise AttributeError("Loaded MODData object is missing 'df_featurized'. Re-processing.")
                print(f"  Loaded train data with {train_data.df_featurized.shape[1]} features.")
                print(f"  Loaded test data with {test_data.df_featurized.shape[1]} features.")
                process_this_fold = False
            except Exception as e:
                print(f"Fold {ind}: Error loading files ({e}). Re-processing fold data...")
                process_this_fold = True
        else:
            print(f"Fold {ind}: Processed data files not found. Processing fold data...")
            process_this_fold = True

        if process_this_fold:
            train_data, test_data = md.split((train_idx, test_idx))

            # --- Feature Pre-selection (Using Updated XGBoost Routine) ---
            print(f"Fold {ind}: Applying XGBoost pre-selection (target < {xgb_preselect_target} features, ~{xgb_preselect_fraction*100:.0f}% drop)...")
            train_data = xgb_preselection(train_data,
                                          n_jobs=n_jobs,
                                          target_threshold=xgb_preselect_target,
                                          drop_fraction=xgb_preselect_fraction)  # Pass new parameters
            # get selected columns
            selected_features = train_data.df_featurized.columns
            test_data.df_featurized = test_data.df_featurized[selected_features]

            # Apply MODNet's internal feature selection/ordering AFTER XGBoost preselection
            print(f"Fold {ind}: Applying MODNet's NMI-based feature selection ordering...")
            if train_data.df_featurized.empty or train_data.df_featurized.shape[1] == 0:
                 print("Warning: No features remaining after XGB preselection in fold {ind}. Cannot perform NMI selection.")
            else:
                try:
                    train_data.feature_selection(n=-1, n_jobs=n_jobs if n_jobs > 1 else 12)
                except Exception as e:
                    print(f"Error during MODNet NMI feature selection in fold {ind}: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    print("Check for constant features or other issues in remaining data.")

            # Final check before saving: Ensure columns are strings
            train_data.df_featurized.columns = train_data.df_featurized.columns.astype(str)
            test_data.df_featurized.columns = test_data.df_featurized.columns.astype(str)

            print(f"Fold {ind}: Saving processed train ({train_data.df_featurized.shape}) and test ({test_data.df_featurized.shape}) MODData objects...")
            try:
                 train_data.save(train_file)
                 test_data.save(test_file)
            except Exception as e:
                 print(f"Error saving MODData for fold {ind}: {e}. Check permissions and disk space.")
                 continue

        # Add the loaded or processed data to the list
        fold_data.append((train_data, test_data))
        print(f"Fold {ind}: Processing complete.")

    # --- Check if any folds were successfully processed ---
    if not fold_data:
         print("Error: No folds were successfully processed or loaded. Cannot proceed to training.")
         return  # Exit the function

    # -----------------------------------------------------------------------
    # Training loop (GA or Preset)
    # -----------------------------------------------------------------------
    results_file = os.path.join(write_folder, f'results_{job_prefix}.txt')
    with open(results_file, 'w') as f:
        f.write(f'--- MODNet Training Results ({hyperopt_strategy} strategy) ---\n')
        f.write(f'Target: {target_name}\n')  # Use original target name
        f.write(f'Data Path: {data_path}\n')
        f.write(f'Feature Sets: {feature_sets}\n')
        f.write(f'Job Prefix: {job_prefix}\n')
        f.write(f'Folds: {n_folds}\n')
        f.write(f'XGB Pre-selection: Target < {xgb_preselect_target} features, Drop ~{xgb_preselect_fraction*100:.0f}%/step\n')
        f.write('\n')

    output_model_folder = os.path.join(write_folder, 'out')
    os.makedirs(output_model_folder, exist_ok=True)
    folds_results = []
    all_fold_predictions = []

    if ga_settings is None:
        ga_settings = {"size_pop": 20, "num_generations": 10, "refit": 5, "nested": 5}
    ga_settings['n_jobs'] = n_jobs

    print(f"\nStarting model training across {len(fold_data)} folds using '{hyperopt_strategy}' strategy...")

    for fold_number, (train_fold, test_fold) in enumerate(fold_data):
        print(f"\n--- Processing Fold {fold_number} ---")
        print(f"  Train data shape: {train_fold.df_featurized.shape}, Test data shape: {test_fold.df_featurized.shape}")
        if train_fold.df_featurized.empty or train_fold.df_featurized.shape[1] == 0:
            print(f"Fold {fold_number}: Skipping training as there are no features in the training data.")
            folds_results.append({k: np.nan for k in ["Train_MAE", "Train_RMSE", "Train_R2", "Test_MAE", "Test_RMSE", "Test_R2"]})
            with open(results_file, 'a') as f:
                f.write(f"Fold {fold_number} Metrics: SKIPPED (no features)\n\n")
            continue

        model_file_base = f'MODNet_{job_prefix}_f{fold_number}'  # Base name
        model_file_name = f"{model_file_base}_{hyperopt_strategy}.pkl"
        model_file_path = os.path.join(output_model_folder, model_file_name)

        targets_list = [[[target_name]]]  # Use original target name here
        target_weights = {target_name: 1.0}  # Use original target name here

        best_model = None  # Initialize best_model for the fold
        
        if os.path.isfile(model_file_path):
            print(f"Fold {fold_number}: Loading pre-trained model from {model_file_path}")
            try:
                best_model = EnsembleMODNetModel.load(model_file_path)
                if is_classification:  # fix for previous data, but should be harmless
                    best_model.num_classes = {target_name: train_fold.df_targets[target_name].nunique()}
                    for model in best_model.model:
                        model.num_classes = {target_name: train_fold.df_targets[target_name].nunique()}
                print(f"  Loaded model expected n_feat: {getattr(best_model, 'n_feat', 'N/A')}")
            except Exception as e:
                print(f"Error loading model for fold {fold_number}: {e}. Retraining...")
                best_model = None

        if best_model is None:  # Train if not loaded successfully
            if hyperopt_strategy == "ga":
                print(f"Fold {fold_number}: Running Genetic Algorithm hyperparameter optimization...")
                # Check if optimal_features exists and has content
                if not hasattr(train_fold, 'optimal_features') or not train_fold.optimal_features:
                    print("Warning: train_fold.optimal_features not set or empty. Attempting NMI selection again.")
                    if train_fold.df_featurized.empty or train_fold.df_featurized.shape[1] == 0:
                         print("Error: Cannot run NMI selection as no features exist. Skipping GA.")
                         continue  # Skip fold
                    try:
                        train_fold.feature_selection(n=-1, n_jobs=n_jobs if n_jobs > 1 else 12)
                        if not train_fold.optimal_features:
                             print("Error: NMI selection failed to find optimal features after retry. Skipping GA.")
                             continue  # Skip fold
                    except Exception as e:
                         print(f"Error during fallback NMI selection in fold {fold_number}: {e}. Skipping GA.")
                         continue  # Skip fold

                ga = FitGenetic(train_fold)
                try:
                    best_model = ga.run(**ga_settings)
                except Exception as e:
                    print(f"!! Error during GA for fold {fold_number}: {e}")
                    import traceback
                    traceback.print_exc()  # Print detailed traceback
                    print("!! Skipping training for this fold due to GA error.")
                    best_model = None
                finally:
                    if 'ga' in locals() and ga is not None: 
                        del ga

            elif hyperopt_strategy == "preset":
                print(f"Fold {fold_number}: Training with preset hyperparameters...")
                if preset_settings is None:
                    print("Error: preset_settings must be provided when hyperopt_strategy is 'preset'. Skipping fold.")
                    best_model = None
                else:
                    known_init_keys = ["num_neurons", "n_feat", "act", "out_act", "num_classes", "weights"]
                    known_fit_keys = ["lr", "epochs", "batch_size", "loss", "verbose", "callbacks", "metrics", "validation_split", "learning_curve"]
                    model_init_params = {}
                    fit_params = {}
                    unknown_keys = []
                    for k, v in preset_settings.items():
                        if k in known_init_keys: 
                            model_init_params[k] = v
                        elif k in known_fit_keys: 
                            fit_params[k] = v
                        elif k == "increase_bs": 
                            pass
                        else: 
                            unknown_keys.append(k)
                    if unknown_keys: 
                        print(f"  Warning: Unknown keys found in preset_settings: {unknown_keys}. They will be ignored.")
                    if is_classification:
                        num_unique = train_fold.df_targets[target_name].nunique()
                        model_init_params["num_classes"] = {target_name: num_unique}
                        print(f"  Setting num_classes={target_name} : {num_unique} for classification.")
                    # --- Validate and Set n_feat for Preset ---
                    # Use optimal_descriptors if available and non-empty, otherwise use df_featurized columns
                    if hasattr(train_fold, 'optimal_features') and train_fold.optimal_features:
                        optimal_desc = train_fold.get_optimal_descriptors()
                        n_features_available = len(optimal_desc) if optimal_desc else 0
                        print(f"  Preset: Using {n_features_available} features from optimal_descriptors.")
                    else:
                        n_features_available = train_fold.df_featurized.shape[1]
                        print(f"  Preset: Using {n_features_available} features from df_featurized (optimal_descriptors not found/empty).")

                    if n_features_available == 0:
                         print("Error: No features available for training in preset mode. Skipping fold.")
                         best_model = None
                    else:  # Proceed only if features exist
                         if "n_feat" not in model_init_params:
                             print(f"  'n_feat' not in preset_settings, using all available features: {n_features_available}")
                             model_init_params["n_feat"] = n_features_available
                         elif not isinstance(model_init_params["n_feat"], int) or model_init_params["n_feat"] < 1:
                              print(f"  Warning: preset 'n_feat' ({model_init_params['n_feat']}) is invalid. Using {n_features_available}.")
                              model_init_params["n_feat"] = n_features_available
                         elif model_init_params["n_feat"] > n_features_available:
                             print(f"  Warning: preset 'n_feat' ({model_init_params['n_feat']}) > available features ({n_features_available}). Using {n_features_available}.")
                             model_init_params["n_feat"] = n_features_available
                         else:
                              print(f"  Using preset 'n_feat': {model_init_params['n_feat']}")
                         # Ensure n_feat doesn't exceed available features after adjustment
                         model_init_params["n_feat"] = min(model_init_params["n_feat"], n_features_available)

                         if "num_neurons" not in model_init_params:
                             print("Error: 'num_neurons' is required in preset_settings for model initialization. Skipping fold.")
                             best_model = None
                         else:
                             model_init_params.setdefault("act", "elu")
                             print(f"  Model Init Params: {model_init_params}")
                             print(f"  Fit Params: {fit_params}")
                             try:
                                 best_model = EnsembleMODNetModel(n_models=20, targets=targets_list, weights=target_weights, **model_init_params)
                             except Exception as e:
                                 print(f"!! Error initializing MODNetModel in fold {fold_number}: {e}")
                                 best_model = None

                             if best_model:
                                 increase_bs = preset_settings.get("increase_bs", False)
                                 try:
                                     if increase_bs:
                                         print("  Applying increase_bs strategy...")
                                         base_lr = fit_params.get("lr", 0.001)
                                         base_epochs = fit_params.get("epochs", 200)
                                         base_bs = fit_params.get("batch_size", 128)
                                         fit_params_stage1 = fit_params.copy(); fit_params_stage1['verbose'] = fit_params_stage1.get('verbose', 0)
                                         print(f"  Initial fit stage params: {fit_params_stage1}")
                                         best_model.fit(train_fold, **fit_params_stage1)
                                         fit_params_stage2 = fit_params.copy(); fit_params_stage2["lr"] = base_lr / 10.0; fit_params_stage2["epochs"] = base_epochs; fit_params_stage2["batch_size"] = base_bs * 2; fit_params_stage2['verbose'] = fit_params_stage1.get('verbose', 0)
                                         print(f"  Second fit stage params: lr={fit_params_stage2['lr']:.2e}, epochs={fit_params_stage2['epochs']}, bs={fit_params_stage2['batch_size']}")
                                         best_model.fit(train_fold, **fit_params_stage2)
                                     else:
                                         fit_params['verbose'] = fit_params.get('verbose', 0)
                                         best_model.fit(train_fold, **fit_params)
                                 except Exception as e:
                                     print(f"!! Error during model fitting for fold {fold_number}: {e}")
                                     import traceback
                                     traceback.print_exc()
                                     best_model = None

            else:
                raise ValueError(f"Unsupported hyperopt_strategy: '{hyperopt_strategy}'. Choose 'ga' or 'preset'.")

            if best_model:
                print(f"Fold {fold_number}: Saving trained model to {model_file_path}")
                try: 
                    best_model.save(model_file_path)
                except Exception as e:
                    print(f"!! Error saving model for fold {fold_number}: {e}")

        # --- Evaluate the model ---
        if best_model:
            try:
                print(f"Fold {fold_number}: Evaluating model...")

                if is_classification:
                    # For classification, obtain both probabilities and predicted classes.
                    train_pred_prob = best_model.predict(train_fold, return_prob=True)
                    test_pred_prob  = best_model.predict(test_fold, return_prob=True)
                    train_pred_class = best_model.predict(train_fold).astype(int)
                    test_pred_class  = best_model.predict(test_fold).astype(int)

                    train_true = train_fold.df_targets[target_name].astype(int)
                    test_true = test_fold.df_targets[target_name].astype(int)

                    # Compute Accuracy and F1 Score.
                    train_accuracy = accuracy_score(train_true, train_pred_class[target_name])                    
                    train_f1 = f1_score(train_true, train_pred_class[target_name], average='weighted')
                    test_accuracy = accuracy_score(test_true, test_pred_class[target_name])
                    test_f1 = f1_score(test_true, test_pred_class[target_name], average='weighted')

                    # Compute ROC AUC score for binary classification.
                    if '{}_prob_0'.format(target_name) in train_pred_prob:
                        train_rocauc = roc_auc_score(train_true, 1-train_pred_prob[f"{target_name}_prob_0"])
                        test_rocauc  = roc_auc_score(test_true, 1-test_pred_prob[f"{target_name}_prob_0"])
                    else:
                        print("Warning: Expected probability column for class '1' not found. Setting ROC AUC=NaN.")
                        train_rocauc = np.nan
                        test_rocauc = np.nan

                    print(f"Fold {fold_number} - Train Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}, ROC AUC: {train_rocauc:.4f},")
                    print(f"Fold {fold_number} - Test Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}, ROC AUC: {test_rocauc:.4f},")

                    fold_result = {
                        "Train_Accuracy": train_accuracy, "Train_F1": train_f1, "Train_ROCAUC": train_rocauc,
                        "Test_Accuracy": test_accuracy, "Test_F1": test_f1, "Test_ROCAUC": test_rocauc,
                    }
                    
                    # Save predictions: using predicted class and probability for class 1.
                    fold_preds_df = pd.DataFrame({
                        'material_id': test_fold.structure_ids,
                        f'pred_class_f{fold_number}': test_pred_class[target_name],
                        f'prob_class1_f{fold_number}': test_pred_prob.get(f"{target_name}_prob_0", np.nan)
                    }).set_index('material_id')
                    all_fold_predictions.append(fold_preds_df)
                    
                    with open(results_file, 'a') as f:
                        f.write(f"Fold {fold_number} Metrics:\n")
                        f.write(f"  Train - Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}, ROC AUC: {train_rocauc:.4f}\n")
                        f.write(f"  Test  - Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}, ROC AUC: {test_rocauc:.4f}\n\n")
                else:
                    # Regression branch.
                    train_pred_df, train_unc_df = best_model.predict(train_fold, return_unc=True)
                    test_pred_df, test_unc_df = best_model.predict(test_fold, return_unc=True)
                    train_true = train_fold.df_targets[target_name]
                    test_true = test_fold.df_targets[target_name]
                    
                    # Compute regression metrics.
                    train_mae = mean_absolute_error(train_true, train_pred_df[target_name])
                    train_rmse = mean_squared_error(train_true, train_pred_df[target_name], squared=False)
                    train_r2 = r2_score(train_true, train_pred_df[target_name])
                    test_mae = mean_absolute_error(test_true, test_pred_df[target_name])
                    test_rmse = mean_squared_error(test_true, test_pred_df[target_name], squared=False)
                    test_r2 = r2_score(test_true, test_pred_df[target_name])
                    
                    # Uncertainty measures: use the mean uncertainty from MODNet's returned values.
                    train_uncertainty = np.mean(train_unc_df.values)
                    test_uncertainty = np.mean(test_unc_df.values)
                    
                    print(f"Fold {fold_number} - Train MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}, Uncertainty: {train_uncertainty:.4f}")
                    print(f"Fold {fold_number} - Test MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}, Uncertainty: {test_uncertainty:.4f}")
                    
                    fold_result = {
                        "Train_MAE": train_mae, "Train_RMSE": train_rmse, "Train_R2": train_r2,
                        "Train_Uncertainty": train_uncertainty,
                        "Test_MAE": test_mae, "Test_RMSE": test_rmse, "Test_R2": test_r2,
                        "Test_Uncertainty": test_uncertainty,
                    }
                    
                    # Save predictions for regression.
                    fold_preds_df = pd.DataFrame({
                        'material_id': test_fold.structure_ids,
                        'target': test_true,
                        f'prediction_f{fold_number}': test_pred_df[target_name],
                        f'uncertainty_f{fold_number}': test_unc_df[target_name]
                    }).set_index('material_id')
                    all_fold_predictions.append(fold_preds_df)
                    
                    with open(results_file, 'a') as f:
                        f.write(f"Fold {fold_number} Metrics:\n")
                        f.write(f"  Train - MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}, Uncertainty: {train_uncertainty:.4f}\n")
                        f.write(f"  Test  - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}, Uncertainty: {test_uncertainty:.4f}\n\n")
                
                folds_results.append(fold_result)
            except Exception as e:
                print(f"!! Error during evaluation for fold {fold_number}: {e}")
                import traceback
                traceback.print_exc()
                folds_results.append({k: np.nan for k in ["Train_Accuracy", "Train_F1", "Train_ROCAUC",
                                                          "Train_MAE", "Train_RMSE", "Train_R2", "Train_Uncertainty",
                                                          "Test_Accuracy", "Test_F1", "Test_ROCAUC",
                                                          "Test_MAE", "Test_RMSE", "Test_R2", "Test_Uncertainty"]})
                with open(results_file, 'a') as f:
                    f.write(f"Fold {fold_number} Metrics: ERROR during evaluation\n\n")
        else:
            print(f"Fold {fold_number}: Skipping evaluation as model is not available.")
            folds_results.append({k: np.nan for k in ["Train_Accuracy", "Train_F1", "Train_ROCAUC",
                                                      "Train_MAE", "Train_RMSE", "Train_R2", "Train_Uncertainty",
                                                      "Test_Accuracy", "Test_F1", "Test_ROCAUC",
                                                      "Test_MAE", "Test_RMSE", "Test_R2", "Test_Uncertainty"]})
            with open(results_file, 'a') as f:
                f.write(f"Fold {fold_number} Metrics: SKIPPED (model not available)\n\n")

    # -----------------------------------------------------------------------
    # Aggregate Results and Final Output
    # -----------------------------------------------------------------------
    print("\n--- Aggregating Results ---")
    pred_filename = os.path.join(write_folder, f'predictions_{job_prefix}.csv')

    if not folds_results:
        print("No fold results available to aggregate.")
    else:
        if is_classification:
            avg_train_acc = np.nanmean([r.get("Train_Accuracy", np.nan) for r in folds_results])
            avg_train_f1 = np.nanmean([r.get("Train_F1", np.nan) for r in folds_results])
            avg_train_rocauc = np.nanmean([r.get("Train_ROCAUC", np.nan) for r in folds_results])
            avg_test_acc = np.nanmean([r.get("Test_Accuracy", np.nan) for r in folds_results])
            avg_test_f1 = np.nanmean([r.get("Test_F1", np.nan) for r in folds_results])
            avg_test_rocauc = np.nanmean([r.get("Test_ROCAUC", np.nan) for r in folds_results])
            
            num_successful_folds = sum(1 for r in folds_results if not np.isnan(r.get("Test_Accuracy", np.nan)))

            print(f"Average Classification Metrics across {num_successful_folds}/{n_folds} successful folds:")
            print(f"  Train - Accuracy: {avg_train_acc:.4f}, F1: {avg_train_f1:.4f}, ROC AUC: {avg_train_rocauc:.4f}, ")
            print(f"  Test  - Accuracy: {avg_test_acc:.4f}, F1: {avg_test_f1:.4f}, ROC AUC: {avg_test_rocauc:.4f}, ")
            
            with open(results_file, 'a') as f:
                f.write("--- Average Classification Metrics ---\n")
                f.write(f"Across {num_successful_folds}/{n_folds} successful folds:\n")
                f.write(f"  Train - Accuracy: {avg_train_acc:.4f}, F1: {avg_train_f1:.4f}, ROC AUC: {avg_train_rocauc:.4f}, \n")
                f.write(f"  Test  - Accuracy: {avg_test_acc:.4f}, F1: {avg_test_f1:.4f}, ROC AUC: {avg_test_rocauc:.4f},\n\n")
        else:
            avg_train_mae = np.nanmean([r.get("Train_MAE", np.nan) for r in folds_results])
            avg_train_rmse = np.nanmean([r.get("Train_RMSE", np.nan) for r in folds_results])
            avg_train_r2 = np.nanmean([r.get("Train_R2", np.nan) for r in folds_results])
            avg_train_uncertainty = np.nanmean([r.get("Train_Uncertainty", np.nan) for r in folds_results])
            avg_test_mae = np.nanmean([r.get("Test_MAE", np.nan) for r in folds_results])
            avg_test_rmse = np.nanmean([r.get("Test_RMSE", np.nan) for r in folds_results])
            avg_test_r2 = np.nanmean([r.get("Test_R2", np.nan) for r in folds_results])
            avg_test_uncertainty = np.nanmean([r.get("Test_Uncertainty", np.nan) for r in folds_results])
            
            num_successful_folds = sum(1 for r in folds_results if not np.isnan(r.get("Test_MAE", np.nan)))

            print(f"Average Metrics across {num_successful_folds}/{n_folds} successful folds:")
            print(f"  Train - MAE: {avg_train_mae:.4f}, RMSE: {avg_train_rmse:.4f}, R2: {avg_train_r2:.4f}")
            print(f"  Train - Uncertainty: {avg_train_uncertainty:.4f}")
            print(f"  Test  - MAE: {avg_test_mae:.4f}, RMSE: {avg_test_rmse:.4f}, R2: {avg_test_r2:.4f}")
            print(f"  Test  - Uncertainty: {avg_test_uncertainty:.4f}")

            with open(results_file, 'a') as f:
                f.write("--- Average Metrics ---\n")
                f.write(f"Across {num_successful_folds}/{n_folds} successful folds:\n")
                f.write(f"  Train - MAE: {avg_train_mae:.4f}, RMSE: {avg_train_rmse:.4f}, R2: {avg_train_r2:.4f}\n")
                f.write(f"  Train - Uncertainty: {avg_train_uncertainty:.4f}\n")
                f.write(f"  Test  - MAE: {avg_test_mae:.4f}, RMSE: {avg_test_rmse:.4f}, R2: {avg_test_r2:.4f}\n")
                f.write(f"  Test  - Uncertainty: {avg_test_uncertainty:.4f}\n\n")

    if all_fold_predictions:
        try:
            # Combine fold prediction dataframes along columns.
            all_preds_df = pd.concat(all_fold_predictions, axis=1)
            # Remove any duplicated columns (if any)
            all_preds_df = all_preds_df.loc[:, ~all_preds_df.columns.duplicated(keep='first')]
            
            # Build a full results dataframe with all material IDs and targets from df_all.
            full_preds_df = df_all[['material_id', target_name]].copy()
            full_preds_df.set_index("material_id", inplace=True)
            
            # Merge with the predictions from the folds (left join so that every material is preserved)
            merged_df = full_preds_df.join(all_preds_df, how="left")
            
            # Define a helper for natural sorting.
            import re
            def natural_sort_key(s):
                return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
            
            merged_df.reset_index(inplace=True)
            merged_df.sort_values(by="material_id", key=lambda col: col.map(natural_sort_key), inplace=True)
            
            # Save the merged full dataframe – it now contains the 'material_id', 'target'
            # for all structures and, where applicable, the predictions (one column per fold).
            merged_df.to_csv(pred_filename, index=False)
            print(f"Saved detailed predictions to {pred_filename}")
        except Exception as e:
            print(f"Could not save detailed predictions: {e}")

    print(f"\nMODNet training run '{job_prefix}' finished.")
    print(f"Results summary saved to: {results_file}")
    print(f"Fold models saved in: {output_model_folder}")


# Example Usage (Updated):
# data_path = 'path/to/your/all_features_incl_sisso.csv'
# target = 'your_target_property'

# run_modnet_training(
#     data_path=data_path,
#     target_name=target,
#     feature_sets=['matminer', 'sisso', 'megnet_ofm'],
#     n_folds=5,
#     n_jobs=16,
#     hyperopt_strategy='ga',
#     xgb_preselect_target=800,    # Stop XGB preselection below 800 features
#     xgb_preselect_fraction=0.1   # Remove ~10% in each XGB step
# )

# ---------------------------------------------------------------------------
# Main: Parse arguments and dispatch actions.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Master script to process MATBench datasets: download CSV splits, generate MODNet featurizations (MM2020 composition-only), extract Roost/SISSO features, run MODNet training, or calculate SISSO formulas."
    )
    
    parser.add_argument("action", choices=["DOWNLOAD_MATBENCH", "MODNET_FEATURIZE", "GET_MATTERVIAL_FEATURES", "MODNET_TRAIN", 
                                             "CALCULATE_SISSO_FORMULAS", "GET_SISSO_FORMULA_FILE", "COMBINE_CHUNKS", "MVL_PREDICT"],
                        help="Action to perform.")
    parser.add_argument("--data_path", type=str, help="Path to CSV file with featurized data (for MODNET_TRAIN, MVL_PREDICT, etc.)")
    
    parser.add_argument("--split_dataset", type=int, default=None, metavar='N',
                        help="Number of chunks (N) to split processing into during MODNET_FEATURIZE. Each chunk is saved incrementally.")
    parser.add_argument("--start_chunk_index", type=int, default=None, metavar='START',
                        help="Starting chunk index (1-based) for MODNET_FEATURIZE.")
    parser.add_argument("--end_chunk_index", type=int, default=None, metavar='END',
                        help="Ending chunk index (1-based) for MODNET_FEATURIZE (inclusive).")
    parser.add_argument("--simplified_featurization", type=bool, default=False,
                        help="Skip site-specific and oxidcomposition featurization for structure-based features (faster but less accurate)")
    parser.add_argument("--output_csv_path", type=str, default=None, help="Path to output CSV file (for COMBINE_CHUNKS)")
    parser.add_argument("--dedup_id", type=str, default=None, help="Unique ID column name for deduplication (for COMBINE_CHUNKS)")
    parser.add_argument("--target_name", type=str, default="target", help="Target column name (for MODNET_TRAIN)")
    parser.add_argument("--job_prefix", type=str, default=None, help="Identifier for all models and datasets (for MODNET_TRAIN)")    
    parser.add_argument("--matbench_set", type=str, default=None, help="MATBench set to process (for MODNET_FEATURIZE and MODNET_TRAIN)")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds (for MODNET_TRAIN)")
    parser.add_argument("--n_jobs", type=int, default=24, help="Number of parallel jobs (for MODNET_TRAIN)")
    parser.add_argument("--featurizer_type", choices=["composition", "structure"], default="structure",
                        help="Type of MODNet featurizer to use for MODNET_FEATURIZE action.")
    parser.add_argument("--hp_strategy", choices=["ga", "preset"], default="ga",
                help="Hyperparameter optimization strategy: 'ga' for Genetic Algorithm (default) or 'preset' for using predefined settings (best_settings).")
    parser.add_argument("--robust_scaler_path", type=str, default='robust_scaler_mpgap_for_sisso.joblib', help="Path to the robust scaler joblib file for SISSO feature processing.")
    parser.add_argument("--sisso_formulas_path", type=str, default=None, help="Path to the SISSO formulas file for feature extraction.")
    parser.add_argument("--sisso_feature_prefix", type=str, default="", help="Prefix for SISSO feature columns.")
    # --- Arguments specific to MODNET_TRAIN ---
    parser.add_argument("--feature_sets", type=str, nargs='+',
                        help="List of feature sets to include in MODNET_TRAIN (e.g., 'matminer', 'sisso', 'roost_lmp', 'megnet_all'). Required for MODNET_TRAIN.")
    parser.add_argument("--xgb_target", type=int, default=800,
                        help="Target feature count for XGBoost pre-selection in MODNET_TRAIN. Default: 800.")
    parser.add_argument("--xgb_fraction", type=float, default=0.1,
                        help="Approximate fraction of features to drop per XGBoost step in MODNET_TRAIN. Default: 0.1 (10%%).")
    parser.add_argument("--mvl_model", type=str,
                        help="MVL model to use for prediction (e.g., 'Eform_MP_2019'). Required for MVL_PREDICT action.")
    
    args = parser.parse_args()
    if args.action == "DOWNLOAD_MATBENCH":
        print("=== Action: DOWNLOAD_MATBENCH ===")
        download_matbench_sets_as_csv()
        print("=== Finished DOWNLOAD_MATBENCH ===")
    elif args.action == "MODNET_FEATURIZE":
        print("=== Action: MODNET_FEATURIZE (Incremental Saving) ===")
        if not (args.matbench_set or args.data_path):
            parser.error("Either --matbench_set or --data_path is required for MODNET_FEATURIZE.")
        input_source = args.matbench_set if args.matbench_set else args.data_path
        print(f"Input source: {input_source}")
        print(f"Featurizer type: {args.featurizer_type}")
        print(f"Split into chunks: {args.split_dataset if args.split_dataset else 1}")
        print(f"Chunk range: {args.start_chunk_index}-{args.end_chunk_index}")
        if args.featurizer_type == "composition":
            if args.matbench_set is None:
                modnet_featurize_MM2020Comp(csv_path=input_source, n_jobs=args.n_jobs, split_dataset=args.split_dataset, start_chunk_index=args.start_chunk_index, end_chunk_index=args.end_chunk_index)
            else:
                modnet_featurize_MM2020Comp(matbench_set=input_source, n_jobs=args.n_jobs, split_dataset=args.split_dataset, start_chunk_index=args.start_chunk_index, end_chunk_index=args.end_chunk_index)
        elif args.featurizer_type == "structure":
            if args.matbench_set is None:
                modnet_featurize_MM2020Struct(csv_path=input_source, n_jobs=args.n_jobs, split_dataset=args.split_dataset, start_chunk_index=args.start_chunk_index,
                                              end_chunk_index=args.end_chunk_index, simplified_featurization=args.simplified_featurization)
            else:
                modnet_featurize_MM2020Struct(matbench_set=input_source, n_jobs=args.n_jobs, split_dataset=args.split_dataset, start_chunk_index=args.start_chunk_index, 
                                                end_chunk_index=args.end_chunk_index,  simplified_featurization=args.simplified_featurization)
        print("=== Finished MODNET_FEATURIZE ===")

    elif args.action == "MVL_PREDICT":
        print("=== Action: MVL_PREDICT ===")
        if not args.data_path:
            parser.error("For MVL_PREDICT, you must provide --data_path.")
        if not args.mvl_model:
            parser.error("For MVL_PREDICT, you must provide --mvl_model (e.g., 'Eform_MP_2019').")
        run_mvl_prediction(data_path=args.data_path,
                           target_name=args.target_name,
                           mvl_model=args.mvl_model,
                           job_prefix=args.job_prefix,
                           n_folds=args.n_folds,
                           n_jobs=args.n_jobs)
        print("=== Finished MVL_PREDICT ===")

    elif args.action == "COMBINE_CHUNKS":
        print("=== Action: COMBINE_CHUNKS ===")
        if not args.data_path:
            parser.error("--data_path (chunk file pattern or path) is required for COMBINE_CHUNKS.")
        # Note: No longer requiring an explicit output_path.
        # Pass args.output_path even if it is None.
        dedup_col = args.dedup_id if args.dedup_id else 'material_id'
        combine_chunked_csvs(
            chunk_file_pattern=args.data_path, # e.g. "data/matbench_set/*_chunk_*.csv"
            output_csv_path=args.output_csv_path,  # May be None and then inferred automatically
            unique_id_col=dedup_col
        )
    # elif args.action == "GET_ROOST_FEATURES": ## ACTIVATE ENVIRONMENT: roost
    #     print("Verify on the top of this script if the variables FEA_PATH, OQMD_CHECKPOINT_PATH and ROOST_CHECKPOINT_PATH are set correctly.")
    #     if not args.data_path:
    #         parser.error("For MODNET_TRAIN, you must provide --data_path.")
    #     get_roost_features(args.data_path)

    elif args.action == "CALCULATE_SISSO_FORMULAS":
        if not args.data_path:
            parser.error("For CALCULATE_SISSO_FORMULAS, you must provide --input_csv.")
        # Step 1: Process the CSV file into a SISSO-friendly version.
        process_data_for_sisso(args.data_path, 
                               target_col="target", 
                               robust_scaler_path=args.robust_scaler_path)
        base_name = os.path.splitext(args.data_path)[0]
        sisso_csv = base_name + "_sissoformatted.csv"
        # Step 2: Setup SISSO hyperparameter optimization (this produces a bash script).
        call_setup_sisso_hp_optimization(sisso_csv)
        # Step 3:  Execute the generated bash script.
        if os.path.exists(f"{base_name}_sissoformatted_run.sh"):
            print(f"Executing bash script '{base_name}_sissoformatted_run.sh'...")
            os.system(f"bash {base_name}_sissoformatted_run.sh")
        else:
            print(f"Bash script '{base_name}_sissoformatted_run.sh' not found. Please check the setup.")
        # Step 4: Generate the final SISSO formula file.
        generate_sisso_formula_file(base_name)
    elif args.action == "GET_SISSO_FORMULA_FILE": 
        base_name = os.path.splitext(args.data_path)[0]  
        generate_sisso_formula_file(base_name)

    elif args.action == "GET_MATTERVIAL_FEATURES":
        print("=== Action: GET_MATTERVIAL_FEATURES ===")
        if not (args.matbench_set or args.data_path):
             parser.error("Either --matbench_set or --data_path is required for GET_MATTERVIAL_FEATURES.")
        # Call the refactored function
        get_mattervial_features(
            matbench_set=args.matbench_set,
            csv_path=args.data_path,
            n_jobs=args.n_jobs,
            split_dataset=args.split_dataset,
            start_chunk_index=args.start_chunk_index,
            end_chunk_index=args.end_chunk_index
        )
        print("=== Finished GET_MATTERVIAL_FEATURES ===")
    
    elif args.action == "MODNET_TRAIN":
        print("=== Action: MODNET_TRAIN ===")
        if not args.data_path:
            parser.error("For MODNET_TRAIN, you must provide --data_path.")
        if not args.feature_sets:
             parser.error("--feature_sets must be provided for MODNET_TRAIN (e.g., --feature_sets matminer sisso).")

        # Determine preset settings if strategy is 'preset'
        preset_settings_value = None
        if args.hp_strategy == "preset":
            if not args.matbench_set:
                 print("Warning: --hp_strategy is 'preset' but --matbench_set is not provided. Cannot automatically look up settings. Pass settings directly if needed.")
                 # Potentially add another argument here to pass preset settings as JSON string?
                 # For now, we proceed with preset_settings_value = None, which run_modnet_training should handle (error or default).
            elif args.matbench_set not in BEST_SETTINGS:
                 print(f"Warning: --matbench_set '{args.matbench_set}' not found in BEST_SETTINGS dictionary. Proceeding without specific presets.")
            else:
                 preset_settings_value = BEST_SETTINGS[args.matbench_set]
                 print(f"Using preset hyperparameters for '{args.matbench_set}'.")

        # Call run_modnet_training with the updated arguments
        run_modnet_training(
            data_path=args.data_path,
            target_name=args.target_name,
            feature_sets=args.feature_sets,  # Pass the list of feature sets
            job_prefix=args.job_prefix,
            n_folds=args.n_folds,
            n_jobs=args.n_jobs,
            hyperopt_strategy=args.hp_strategy,
            preset_settings=preset_settings_value, # Pass the looked-up or None settings
            ga_settings=None,                      # Let run_modnet_training handle GA defaults if needed
            xgb_preselect_target=args.xgb_target,  
            xgb_preselect_fraction=args.xgb_fraction 
        )
        print("=== Finished MODNET_TRAIN ===")
    else:
        parser.error("Unknown action!")


if __name__ == "__main__":
    main()
