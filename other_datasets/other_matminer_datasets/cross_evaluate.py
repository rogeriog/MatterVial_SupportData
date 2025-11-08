import json
import os
from pathlib import Path
import tensorflow as tf
import h5py
import numpy as np
import pandas as pd
import tensorflow.keras as ks
from pymatgen.core.structure import Structure
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# =============================================================================
#
# Helper functions from your original script (run.py, processing.py)
#
# NOTE: These are required for the script to run. They handle data
# preprocessing, caching, and tensor creation.
#
# =============================================================================

# Assume the following functions are defined in this file or imported:
# - process_custom_dataset(df, dataset_name, preprocessor, cache_dir, ...)
# - get_input_tensors(inputs, graphlist)
# - crystal_iterator(crystal_series)
# - create_graph_dataset(...)
# - PreprocessorWrapper(...)
# - batcher(...)

from itertools import islice
from multiprocessing import Pool
from typing import Iterable, Union

from graphlist import GraphList, HDFGraphList
from kgcnn.graph.methods import get_angle_indices
from networkx import MultiDiGraph
from sklearn.model_selection import KFold

# --- Paste your original helper functions here ---
MATBENCH_SEED = 18012019

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
    preprocessor,
    out_file: Path,
    additional_graph_attributes=[],
    processes=1,
    batch_size=10,
) -> Path:
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

def crystal_iterator(crystal_series: pd.Series):
    for id_, structure in zip(crystal_series.index, crystal_series):
        setattr(structure, "dataset_id", str(id_).encode())
        yield structure

def process_custom_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    preprocessor,
    cache_dir: Path,
    processes: int = 8,
    batch_size: int = 500,
) -> Path:
    preprocessor_hash = preprocessor.hash()
    dataset_cache_dir = cache_dir / dataset_name
    os.makedirs(dataset_cache_dir, exist_ok=True)
    hdf_file = dataset_cache_dir / f"{preprocessor_hash}.h5"
    if hdf_file.is_file():
        print(f"Found cached preprocessed file: {hdf_file}")
        return hdf_file
    print(f"No cache found. Preprocessing '{dataset_name}' with '{type(preprocessor).__name__}'...")
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

def get_input_tensors(inputs, graphlist):
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
# FUNCTION FOR CROSS-DATASET EVALUATION
#
# =============================================================================

def run_cross_evaluation(
    training_df: pd.DataFrame,
    training_target_col: str,
    training_dataset_name: str,
    evaluation_df: pd.DataFrame,
    evaluation_target_col: str,
    evaluation_dataset_name: str,
    model_cfg: dict,
    crystal_preprocessor,
    cache_dir: Path = Path('./dataset_cache'),
    results_dir: Path = Path('./results_custom'),
    cross_test_results_dir: Path = Path('./results_cross_test'),
):
    """
    Evaluates pre-trained models from one dataset on a different dataset.
    """
    print("\n" + "~"*80)
    print(f"CROSS-TEST: Models from '{training_dataset_name}' evaluated on '{evaluation_dataset_name}'")
    print("~"*80 + "\n")

    eval_preprocessed_file = process_custom_dataset(
        df=evaluation_df,
        dataset_name=evaluation_dataset_name,
        preprocessor=crystal_preprocessor,
        cache_dir=cache_dir
    )

    model_for_inputs = make_model(**model_cfg)
    with h5py.File(eval_preprocessed_file, 'r') as f:
        eval_graphs = HDFGraphList(f)
        x_eval = get_input_tensors(model_for_inputs.inputs, eval_graphs)
    y_true = evaluation_df[evaluation_target_col].to_numpy()

    fold_maes = []
    kf = KFold(n_splits=5, shuffle=True, random_state=MATBENCH_SEED)

    for fold, (train_idx, _) in enumerate(kf.split(training_df)):
        print(f"--- Evaluating with model from Fold {fold + 1}/5 ---")
        weights_path = results_dir / training_dataset_name / str(fold) / 'weights.h5'
        if not weights_path.exists():
            print(f"ERROR: Weights file not found at {weights_path}. Skipping test.")
            return None

        model = make_model(**model_cfg)
        model.load_weights(str(weights_path))
        
        scaler = StandardScaler()
        y_train_fold = training_df[training_target_col].iloc[train_idx].to_numpy().reshape(-1, 1)
        scaler.fit(y_train_fold)

        predictions_scaled = model.predict(x_eval)
        predictions = scaler.inverse_transform(predictions_scaled).flatten()

        mae = np.mean(np.abs(y_true - predictions))
        fold_maes.append(mae)

    mean_mae = np.mean(fold_maes)
    std_mae = np.std(fold_maes)

    print("\n--- Cross-Evaluation Summary ---")
    print(f"Mean MAE: {mean_mae:.4f} | Std Dev MAE: {std_mae:.4f}")

    output_dir = cross_test_results_dir / f"train_{training_dataset_name}_eval_{evaluation_dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    summary_data = {
        "training_dataset": training_dataset_name,
        "evaluation_dataset": evaluation_dataset_name,
        "summary": {"mean_mae": mean_mae, "std_dev_mae": std_mae}
    }
    with open(output_dir / "summary_metrics.json", 'w') as f:
        json.dump(summary_data, f, indent=4)
    
    return summary_data


if __name__ == '__main__':
    # =========================================================================
    #
    #     SCRIPT TO RUN ALL EVALUATIONS AND GENERATE SUMMARY MATRIX
    #
    # =========================================================================

    # --- IMPORTS AND CONFIGURATIONS ---
    from kgcnn.literature.coGN import make_model, model_default
    from kgcnn.crystal.preprocessor import KNNAsymmetricUnitCell

    model_config = model_default
    preprocessor = KNNAsymmetricUnitCell(k=24)
    
    # --- DEFINE DATASETS ---
    normal_perovskites_task = {
        "csv_path": "./featurize_with_MM/double_perovskites_gap_1306.csv",
        "structure_col": "structure",
        "target_col": "target",
        "dataset_name": "double_perovskites_gap_1306"
    }

    optimized_perovskites_task = {
        "csv_path": "./featurize_with_MM/double_perovskites_gap_1306_optimized.csv",
        "structure_col": "structure",
        "target_col": "target",
        "dataset_name": "double_perovskites_gap_1306_optimized"
    }
    
    # --- LOAD DATAFRAMES ---
    print("Loading datasets...")
    try:
        df_normal = pd.read_csv(normal_perovskites_task['csv_path']).reset_index(drop=True)
        df_normal['structure'] = df_normal[normal_perovskites_task['structure_col']].apply(
            lambda s: Structure.from_dict(json.loads(s))
        )

        df_optimized = pd.read_csv(optimized_perovskites_task['csv_path']).reset_index(drop=True)
        df_optimized['structure'] = df_optimized[optimized_perovskites_task['structure_col']].apply(
            lambda s: Structure.from_dict(json.loads(s))
        )
        print("Datasets loaded successfully.")
    except FileNotFoundError as e:
        print(f"\nERROR: Could not load dataset CSV. {e}")
        exit()

    # --- RUN EVALUATIONS AND COLLECT RESULTS ---
    all_results = []
    
    # 1. Cross-Test: Normal -> Optimized
    results = run_cross_evaluation(
        training_df=df_normal, training_target_col=normal_perovskites_task['target_col'],
        training_dataset_name=normal_perovskites_task['dataset_name'],
        evaluation_df=df_optimized, evaluation_target_col=optimized_perovskites_task['target_col'],
        evaluation_dataset_name=optimized_perovskites_task['dataset_name'],
        model_cfg=model_config, crystal_preprocessor=preprocessor
    )
    if results: all_results.append(results)

    # 2. Cross-Test: Optimized -> Normal
    results = run_cross_evaluation(
        training_df=df_optimized, training_target_col=optimized_perovskites_task['target_col'],
        training_dataset_name=optimized_perovskites_task['dataset_name'],
        evaluation_df=df_normal, evaluation_target_col=normal_perovskites_task['target_col'],
        evaluation_dataset_name=normal_perovskites_task['dataset_name'],
        model_cfg=model_config, crystal_preprocessor=preprocessor
    )
    if results: all_results.append(results)

    # 3. Load Self-Test Results (Normal -> Normal)
    try:
        path = Path('./results_custom') / normal_perovskites_task['dataset_name'] / "summary_metrics.json"
        with open(path, 'r') as f:
            data = json.load(f)
            all_results.append({
                "training_dataset": normal_perovskites_task['dataset_name'],
                "evaluation_dataset": normal_perovskites_task['dataset_name'],
                "summary": data['summary']
            })
            print(f"\nLoaded self-evaluation results for '{normal_perovskites_task['dataset_name']}'")
    except FileNotFoundError:
        print(f"\nWARNING: Could not find self-evaluation results for '{normal_perovskites_task['dataset_name']}'.")
        print("Please run the original training script for this dataset to include it in the summary.")

    # 4. Load Self-Test Results (Optimized -> Optimized)
    try:
        path = Path('./results_custom') / optimized_perovskites_task['dataset_name'] / "summary_metrics.json"
        with open(path, 'r') as f:
            data = json.load(f)
            all_results.append({
                "training_dataset": optimized_perovskites_task['dataset_name'],
                "evaluation_dataset": optimized_perovskites_task['dataset_name'],
                "summary": data['summary']
            })
            print(f"Loaded self-evaluation results for '{optimized_perovskites_task['dataset_name']}'")
    except FileNotFoundError:
        print(f"\nWARNING: Could not find self-evaluation results for '{optimized_perovskites_task['dataset_name']}'.")
        print("Please run the original training script for this dataset to include it in the summary.")

    # --- FINAL SUMMARY ---
    if all_results:
        summary_list = [
            {
                "Training Dataset": res["training_dataset"].replace("double_perovskites_gap_1306", "Normal"),
                "Evaluation Dataset": res["evaluation_dataset"].replace("double_perovskites_gap_1306", "Normal"),
                "Mean MAE": res["summary"]["mean_mae"],
                "Std Dev MAE": res["summary"]["std_dev_mae"]
            }
            for res in all_results
        ]
        
        summary_df = pd.DataFrame(summary_list).sort_values(by=["Training Dataset", "Evaluation Dataset"])

        print("\n\n" + "="*80)
        print(" " * 25 + "COMPLETE EVALUATION MATRIX")
        print("="*80)
        print(summary_df.to_string(index=False, float_format="%.4f"))
        print("="*80)

        summary_csv_path = Path('./results_cross_test/complete_evaluation_summary.csv')
        summary_df.to_csv(summary_csv_path, index=False, float_format="%.4f")
        print(f"\n✅ Saved complete summary to: {summary_csv_path}")