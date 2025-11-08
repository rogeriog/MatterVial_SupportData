import json
import os
from datetime import datetime
from itertools import islice
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Union

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as ks
from graphlist import GraphList, HDFGraphList
from kgcnn.graph.methods import get_angle_indices
from networkx import MultiDiGraph
from pymatgen.core.structure import Structure
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
MATBENCH_SEED = 18012019
# =============================================================================
#
# Helper classes and functions from your provided code
# (processing.py and run.py)
#
# =============================================================================

# NOTE: The CrystalPreprocessor classes and the coGN model are assumed to be
# available in your environment from the 'kgcnn' library.
# from kgcnn.crystal.preprocessor import CrystalPreprocessor, KNNAsymmetricUnitCell
# from kgcnn.literature.coGN import make_model, model_default


class PreprocessorWrapper:
    """Callable that modifies the behaviour of CrystalProcessors to include extra global graph attributes."""
    def __init__(self, preprocessor, additional_graph_attributes=[]):
        self.preprocessor = preprocessor
        self.additional_graph_attributes = additional_graph_attributes

    def __call__(self, crystal: Union[MultiDiGraph, Structure]):
        graph = self.preprocessor(crystal)
        for attribute in self.additional_graph_attributes:
            setattr(graph, attribute, getattr(crystal, attribute))
        return graph


def batcher(iterable, batch_size):
    """Creates batches for iterables"""
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def create_graph_dataset(
    crystals: Iterable[Structure],
    preprocessor, # Expected: CrystalPreprocessor
    out_file: Path,
    additional_graph_attributes=[],
    processes=1,
    batch_size=10,
) -> Path:
    """Creates an HDF file containing crystal graphs."""
    worker = PreprocessorWrapper(preprocessor, additional_graph_attributes=additional_graph_attributes)
    with h5py.File(str(out_file), "w") as f:
        with Pool(processes) as p:
            for i, batch_ in enumerate(batcher(crystals, batch_size)):
                print(f"Processing batch {i+1}...")
                graphs = p.map(worker, batch_)
                graphlist = GraphList.from_nx_graphs(
                    graphs,
                    node_attribute_names=preprocessor.node_attributes,
                    edge_attribute_names=preprocessor.edge_attributes,
                    graph_attribute_names=preprocessor.graph_attributes + additional_graph_attributes,
                )
                HDFGraphList.from_graphlist(f, graphlist)
        f.attrs["preprocessor_config"] = json.dumps(preprocessor.get_config(), indent=2)
    return out_file


def get_lr_scheduler(dataset_size, batch_size, epochs, lr_start=0.0005, lr_stop=1e-5):
    steps_per_epoch = dataset_size / batch_size
    num_steps = epochs * steps_per_epoch
    scheduler = ks.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=lr_start,
        decay_steps=num_steps,
        end_learning_rate=lr_stop
    )
    return scheduler


def get_input_tensors(inputs, graphlist):
    """Returns input tensors from a GraphList."""
    input_names = [input.name for input in inputs]
    input_tensors = {}
    for name in graphlist.edge_attributes:
        if name in input_names:
            input_tensors[name] = tf.RaggedTensor.from_row_lengths(
                graphlist.edge_attributes[name], graphlist.num_edges)
    for name in graphlist.node_attributes:
        if name in input_names:
            input_tensors[name] = tf.RaggedTensor.from_row_lengths(
                graphlist.node_attributes[name], graphlist.num_nodes)
    for name in graphlist.graph_attributes:
        if name in input_names:
            input_tensors[name] = tf.convert_to_tensor(graphlist.graph_attributes[name])

    input_tensors['edge_indices'] = tf.RaggedTensor.from_row_lengths(
        graphlist.edge_indices[:][:, [1, 0]], graphlist.num_edges)

    if 'line_graph_edge_indices' in input_names:
        line_graph_indices = [
            get_angle_indices(g.edge_indices, edge_pairing='kj')[2].reshape(-1, 2)
            for g in graphlist
        ]
        input_tensors['line_graph_edge_indices'] = tf.RaggedTensor.from_row_lengths(
            np.concatenate(line_graph_indices), [len(l) for l in line_graph_indices])

    return input_tensors


# =============================================================================
#
# New functions for handling custom datasets
#
# =============================================================================

def crystal_iterator(crystal_series: pd.Series):
    """Iterator that adds a unique 'dataset_id' to each pymatgen structure."""
    for id_, structure in zip(crystal_series.index, crystal_series):
        # The ID must be encoded for HDF5 storage
        setattr(structure, "dataset_id", str(id_).encode())
        yield structure

def process_custom_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    preprocessor, # Expected: CrystalPreprocessor
    cache_dir: Path,
    processes: int = 8,
    batch_size: int = 500,
) -> Path:
    """
    Processes a DataFrame of pymatgen structures into a cached HDF5 graph file.
    """
    preprocessor_hash = preprocessor.hash()
    dataset_cache_dir = cache_dir / dataset_name
    os.makedirs(dataset_cache_dir, exist_ok=True)

    hdf_file = dataset_cache_dir / f"{preprocessor_hash}.h5"
    if hdf_file.is_file():
        print(f"Found cached preprocessed file: {hdf_file}")
        return hdf_file

    print(f"No cache found. Preprocessing '{dataset_name}' with '{type(preprocessor).__name__}'...")
    
    # Use the 'structure' column from the dataframe
    structures = df['structure']

    create_graph_dataset(
        crystal_iterator(structures),
        preprocessor,
        hdf_file,
        additional_graph_attributes=["dataset_id"],
        processes=processes,
        batch_size=batch_size,
    )
    with open(str(hdf_file) + ".json", "w") as meta_file:
        json.dump(preprocessor.get_config(), meta_file)

    print("Preprocessing complete.")
    return hdf_file

def run_custom_task(
    df: pd.DataFrame,
    target_col: str,
    model_cfg: dict,
    crystal_preprocessor, # Expected: CrystalPreprocessor
    dataset_name: str = "custom_dataset",
    cache_dir: Path = Path('./dataset_cache'),
    results_dir: Path = Path('./results_custom'),
    epochs: int = 800,
    batch_size: int = 64,
):
    """
    Main function to run 5-fold CV training on a custom dataset.
    """
    # 1. Preprocess and cache the entire dataset if not already done
    preprocessed_file = process_custom_dataset(
        df=df,
        dataset_name=dataset_name,
        preprocessor=crystal_preprocessor,
        cache_dir=cache_dir
    )
    
    dataset_results_dir = results_dir / dataset_name
    os.makedirs(dataset_results_dir, exist_ok=True)

    # 2. Open the HDF5 file
    with h5py.File(preprocessed_file, 'r') as f:
        all_graphs = HDFGraphList(f)

        # 3. Set up 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=MATBENCH_SEED)
        fold_metrics = []
        
        all_predictions_list = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
            print(f"\n--- Starting Fold {fold + 1}/5 ---")
            fold_results_dir = dataset_results_dir / str(fold)
            os.makedirs(fold_results_dir, exist_ok=True)

            if (fold_results_dir / 'predictions.npy').exists():
                print("Fold already completed. Loading results.")
                test_outputs = df.iloc[test_idx][target_col]
                predictions = np.load(fold_results_dir / 'predictions.npy')
                mae = np.mean(np.abs(test_outputs.values - predictions))
                fold_metrics.append(mae)

                fold_pred_df = pd.DataFrame({
                    'original_index': df.iloc[test_idx].index,
                    'target': test_outputs.values,
                    'prediction': predictions.flatten(),
                    'fold': fold + 1
                })
                all_predictions_list.append(fold_pred_df)
                continue

            # 4. Create model (re-initialize for each fold)
            model = make_model(**model_cfg)

            # 5. Get training and test data for the current fold
            train_graphs = all_graphs[train_idx]
            test_graphs = all_graphs[test_idx]

            x_train = get_input_tensors(model.inputs, train_graphs)
            y_train = df[target_col].iloc[train_idx].to_numpy().reshape(-1, 1)

            x_test = get_input_tensors(model.inputs, test_graphs)
            test_outputs = df[target_col].iloc[test_idx]

            # 6. Scale targets
            scaler = StandardScaler()
            y_train_scaled = scaler.fit_transform(y_train)

            # 7. Compile and train the model
            scheduler = get_lr_scheduler(len(y_train), batch_size, epochs)
            optimizer = ks.optimizers.Adam(learning_rate=scheduler)
            model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])

            print(f"Training on {len(train_idx)} samples...")
            start_time = datetime.now()
            history = model.fit(
                x_train, y_train_scaled,
                batch_size=batch_size,
                epochs=epochs,
                verbose=2
            )
            duration = (datetime.now() - start_time).total_seconds()
            print(f"Training finished in {duration:.2f} seconds.")

            # 8. Predict and evaluate
            predictions_scaled = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions_scaled).flatten()

            mae = np.mean(np.abs(test_outputs.values - predictions))
            fold_metrics.append(mae)
            print(f"Fold {fold + 1} MAE: {mae:.4f}")

            fold_pred_df = pd.DataFrame({
                'original_index': df.iloc[test_idx].index,
                'target': test_outputs.values,
                'prediction': predictions,
                'fold': fold + 1
            })
            all_predictions_list.append(fold_pred_df)

            # 9. Save results
            np.save(fold_results_dir / 'predictions.npy', predictions)
            with open(fold_results_dir / 'history.json', 'w') as history_file:
                history_dict = {k: np.array(v).tolist() for k,v in history.history.items()}
                history_dict['training_time'] = duration
                json.dump(history_dict, history_file)
            model.save_weights(str(fold_results_dir / 'weights.h5'))

    # 10. Final Results
    mean_mae = np.mean(fold_metrics)
    std_mae = np.std(fold_metrics)
    
    final_predictions_df = pd.concat(all_predictions_list).sort_values('original_index')
    predictions_csv_path = dataset_results_dir / "cross_validation_predictions.csv"
    final_predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"\nSaved combined predictions to: {predictions_csv_path}")
    
    summary_data = {
        "fold_metrics": {f"fold_{i+1}": {"mae": mae} for i, mae in enumerate(fold_metrics)},
        "summary": {
            "mean_mae": mean_mae,
            "std_dev_mae": std_mae
        }
    }
    metrics_json_path = dataset_results_dir / "summary_metrics.json"
    with open(metrics_json_path, 'w') as f_json:
        json.dump(summary_data, f_json, indent=4)
    print(f"Saved summary metrics to: {metrics_json_path}")

    print("\n--- Cross-Validation Complete ---")
    print(f"Mean MAE across 5 folds: {mean_mae:.4f}")
    print(f"Std Dev MAE across 5 folds: {std_mae:.4f}")

    return {"mean_mae": mean_mae, "std_mae": std_mae}

if __name__ == '__main__':
    # =========================================================================
    #
    #           UPDATED SCRIPT TO RUN MULTIPLE DATASET TASKS
    #
    # =========================================================================

    # --- IMPORTS AND CONFIGURATIONS (common for all tasks) ---
    # These imports must be available in your environment.
    from kgcnn.literature.coGN import make_model, model_default
    from kgcnn.crystal.preprocessor import KNNAsymmetricUnitCell

    # Define the model configuration and preprocessor (using coGN defaults for all tasks)
    model_config = model_default
    preprocessor = KNNAsymmetricUnitCell(k=24)
    
    # --- DEFINE THE DATASET TASKS TO BE RUN ---
    tasks = [
        # {
        #     "csv_path": "./featurize_with_MM/m2ax_223.csv",
        #     "structure_col": "structure",
        #     "target_col": "target",
        #     # "structure_col": "pymatgen_structure_json",
        #     # "target_col": "c44",
        #     "dataset_name": "m2ax_223"
        # },
        # {
        #     "csv_path": "tholander_nitrides_12815.csv",
        #     "structure_col": "structure",
        #     "target_col": "target",
        #     # "structure_col": "final_structure_json",
        #     # "target_col": "E_vasp_per_atom",
        #     "dataset_name": "tholander_nitrides_12815"
        # },
        {
            "csv_path": "boltztrap_mp_8924.csv",
            "structure_col": "structure",
            "target_col": "target",
            # "structure_col": "structure_json",
            # "target_col": "pf_p",
            "dataset_name": "boltztrap_mp_8924"
        },
        # {
        #     "csv_path": "./featurize_with_MM/double_perovskites_gap_1306.csv",
        #     "structure_col": "structure",
        #     "target_col": "target",
        #     "dataset_name": "double_perovskites_gap_1306"
        # },
        # {
        #     "csv_path": "./featurize_with_MM/double_perovskites_gap_1306_optimized.csv",
        #     "structure_col": "structure",
        #     "target_col": "target",
        #     "dataset_name": "double_perovskites_gap_1306_optimized"
        # }
        
    ]

    # --- LOOP THROUGH AND RUN EACH TASK ---
    for task in tasks:
        print("\n" + "="*80)
        print(f"STARTING TASK: {task['dataset_name']}")
        print("="*80 + "\n")
        
        try:
            # 1. Load the dataset
            print(f"Loading data from '{task['csv_path']}'...")
            df = pd.read_csv(task['csv_path']).reset_index(drop=True)

            # 2. Parse structure data from the specified JSON column
            print(f"Parsing structure data from '{task['structure_col']}' column...")
            df['structure'] = df[task['structure_col']].apply(
                lambda s: Structure.from_dict(json.loads(s))
            )
            print("Data loading and parsing complete.")

            # 3. Run the training and evaluation
            run_custom_task(
                df=df,
                target_col=task['target_col'],
                model_cfg=model_config,
                crystal_preprocessor=preprocessor,
                dataset_name=task['dataset_name'],
                epochs=800  # Keeping epochs constant for consistency
            )
            print("\n" + "="*80)
            print(f"COMPLETED TASK: {task['dataset_name']}")
            print("="*80 + "\n")

        except FileNotFoundError:
            print(f"\nERROR: File not found for task '{task['dataset_name']}' at '{task['csv_path']}'")
            print("Please make sure the CSV file is in the correct directory. Skipping task.")
            continue # Move to the next task
        except Exception as e:
            print(f"\nAn unexpected error occurred during task '{task['dataset_name']}': {e}")
            print("Skipping task.")
            continue # Move to the next task