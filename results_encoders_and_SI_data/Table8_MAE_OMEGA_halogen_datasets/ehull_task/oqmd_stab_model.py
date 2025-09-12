from modnet_b.preprocessing import MODData
from modnet_b.featurizers.presets.modnet_omega_2023 import MODNetOmegaFast2023, MODNetOmega2023
from modnet_b.matbench.benchmark import matbench_kfold_splits
from modnet_b.hyper_opt.fit_genetic import FitGenetic
from modnet_b.models import MODNetModel, EnsembleMODNetModel
import pickle, os, sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, \
                            explained_variance_score, mean_squared_error
import random 
def exp_to_normal(x):
    return np.log(x)
def setup_threading():
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# sometimes it is useful to featurize in chunks to parallelize
# def split_data(data, num_chunks):
#     chunk_size = len(data) // num_chunks
#     chunks = [data[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]
#     if len(data) % num_chunks != 0:
#         chunks.append(data[num_chunks * chunk_size:])
#     return chunks
# full_data = pickle.load(open('halide_structures_oqmd_bandgap_stability_CsSbXsamples.pkl', 'rb'))
# # Split data into 33 chunks sequentially
# num_chunks = 33
# data_chunks = split_data(full_data, num_chunks)
# # Parse the index from sys.argv
# if len(sys.argv) < 2:
#     print("Please provide an index as an argument.")
#     sys.exit(1)
# try:
#     index = int(sys.argv[1])
#     if index < 0 or index >= num_chunks:
#         raise ValueError
# except ValueError:
#     print("Invalid index provided. Please provide an integer between 0 and 32.")
#     sys.exit(1)
def substitute_adjacent(path_full_featurized_data):
    md = MODData.load(path_full_featurized_data)
    df_adjfeat = pd.read_pickle('df_adjfeat.pkl')
    # exclude columns starting with Adjacent
    cols_adjfeat = [col for col in md.df_featurized.columns if col.startswith('Adjacent')]
    md.df_featurized.drop(columns=cols_adjfeat, inplace=True)
    # merge df_adjfeat with df_featurized
    md.df_featurized = pd.concat([md.df_featurized, df_adjfeat], axis=1)
    # save the new df_featurized
    md.save('moddata_adjsubstituted_eform')

def adjacent_featurizer(path_full_featurized_data):
    from modnet_b.featurizers.presets.modnet_omega_2023 import MEGNetFeaturizer
    md = MODData.load(path_full_featurized_data)
    MEGNetAdjacent = MEGNetFeaturizer(model_type='adjacent',adjacent_model_path='./MEGNetModel__adjacent.h5')
    df_adjfeat= MEGNetAdjacent.get_features(md.df_structure['structure'])
    # save df_adjfeat
    df_adjfeat.to_pickle('df_adjfeat.pkl')    
def xgb_preselection(data, n_jobs=24):
    import xgboost as xgb
    xgb_model = xgb.XGBRegressor(n_jobs=n_jobs, random_state=1)
    xgb_model.fit(data.df_featurized, data.df_targets)
    feature_importance = xgb_model.feature_importances_
    
    #selecting the features
    num_top_features = 1500
    selected_features_indices = feature_importance.argsort()[-num_top_features:][::-1]
    selected_features = data.df_featurized.columns[selected_features_indices]
    selected_dataset = data.df_featurized[selected_features]
    # print top 30 features
    print("Top 30 features:")
    print(selected_features[:30])
    # redefine the data
    data.df_featurized = selected_dataset
    return data
def rename_cols(df):
    df.columns = [col.replace("[", "_").replace("]", "_").replace(" ", "_").replace("<","_").replace(">","_").replace(",","_") for col in df.columns]
    return df
def process_featurized_data(path_full_featurized_data, job_prefix='', to_drop=''):
    """ This code will process the previously featurized data to
    drop specific sets of columns and save the data to a new file.
    """
    # load data
    data = MODData.load(path_full_featurized_data)
    print(f"Loaded featurized data from {path_full_featurized_data}")
    ## just for test
    # data = md.from_indices(random.sample(range(len(md.df_featurized)), 100))

    df = data.df_featurized
    os.makedirs('out', exist_ok=True)
    featurized_data_fname = f'out/{job_prefix}_featurized'
    dropsets = to_drop.split('+')
    for to_drop in dropsets:
        if to_drop == '':
            pass    # just save the data
        elif to_drop == 'pretrained_megnet':
            cols_pretrained_megnet = [col for col in df.columns if "_MP_" in col]
            df.drop(columns=cols_pretrained_megnet, inplace=True)
        elif to_drop == 'ROSA':
            cols_ROSA = [col for col in df.columns if "ROSA|" in col]
            df.drop(columns=cols_ROSA, inplace=True)
        elif to_drop == 'G':
            cols_G= [col for col in df.columns if "G|" in col]
            df.drop(columns=cols_G, inplace=True)
        elif to_drop == 'OFMencoded':
            cols_OFMencoded = [col for col in df.columns if "MEGNet_OFMEncoded" in col]
            df.drop(columns=cols_OFMencoded, inplace=True)
        elif to_drop == 'adjacent':
            cols_adjacent = [col for col in df.columns if "Adjacent" in col]
            df.drop(columns=cols_adjacent, inplace=True)
        elif to_drop == 'bondfractions':
            cols_bondfractions = [col for col in df.columns if "BondFractions" in col]
            df.drop(columns=cols_bondfractions, inplace=True)
        elif to_drop == 'MatMiner': # not including bond fractions
            cols_matminer = [col for col in df.columns if "_MP_" not in col and 
                              "ROSA|" not in col and "G|" not in col and 
                              "MEGNet_OFMEncoded" not in col and "Adjacent" not in col
                              ]
            df.drop(columns=cols_matminer, inplace=True)
    data.df_featurized = df
    # split data into train and test
    # get 5% of the data for test fully isolated.
    for ind, (train, test) in enumerate(matbench_kfold_splits(data,n_splits=20, classification=False)):
        train_data, test_data = data.split((train, test))
        break # just take the first
    os.makedirs("test_data", exist_ok=True)
    test_data.save(f"test_data/{job_prefix}_test_moddata_featurized") ### REMOVE THIS FEATURIZED IN PRODUCTION
    train_data.save(featurized_data_fname)
    print(f"Saved processed featurized data to {featurized_data_fname}")
    print(f"Saved test data to test_data/{job_prefix}_test_moddata")

def main(job_prefix):
    n_jobs = 24
    # data file with structures and targets
    dataset_name = '../halide_structures_oqmd_bandgap_stability_CsSbXsamples.pkl' 
    job_prefix = job_prefix # 'oqmdhalides_stab'
    featurized_data_fname = f'out/{job_prefix}_featurized'

    ### FOLLOWING SECTION WILL PERFORM FEATURIZATION AND SAVE FEATURIZED DATA
    ### IF FEATURIZED DATA ALREADY EXISTS, IT WILL SKIP THIS SECTION
    if not os.path.isfile(featurized_data_fname):
        # load data file with structures and targets
        full_data = pickle.load(open(dataset_name, 'rb'))
        print(full_data)
        # we put this data in a MODData object
        md = MODData(materials = full_data['structure'],
                        targets = full_data['exp_stability'].values,
                        structure_ids = full_data['id'],
                        target_names = ['exp_stability'],
                        )
        # get 5% of the data for test fully isolated.
        for ind, (train, test) in enumerate(matbench_kfold_splits(md,n_splits=20, classification=False)):
            train_data, test_data = md.split((train, test))
            break # just take the first
        os.makedirs("test_data", exist_ok=True)
        test_data.save(f"test_data/{job_prefix}_test_moddata")
        
        # data to train adjancent model and use onwards
        md = train_data
        # lets train the adjacent megnet model
        # we just need to initialize the featurizer with the data
        # by default a large 20% split is used to minimize data 
        # leakage across folds and dont make modnet overly dependent
        modnet2023 = MODNetOmega2023(adjacent_model=True,
                        structures=md.df_structure.values,
                        targets=md.df_targets.values)
        ## this may take a while ...
        # we set the featurizer
        md.featurizer = modnet2023
        # we featurize the test data
        test_data.featurizer = modnet2023
        test_data.featurize(n_jobs=n_jobs)
        # save test data
        test_data.save(f"test_data/{job_prefix}_test_moddata_featurized") 
        print(f'Test data saved to test_data/{job_prefix}_test_moddata_featurized')
        # we featurize the train data
        os.makedirs('out', exist_ok=True)
        md.featurize(n_jobs=n_jobs)
        featurized_data_fname = f'out/{job_prefix}_featurized'
        md.save(featurized_data_fname)
        print(f'Featurized data saved to {featurized_data_fname}.')
        print(md.df_featurized)
        # all featurization may also take quite some time...
    else:
        md = MODData.load(featurized_data_fname)
        print(md)
        print(md.df_featurized)
    # modnet2023 = MODNetOmega2023(adjacent_model=True,
    #                     structures=md.df_structure['structure'],
    #                     targets=md.df_targets.values)
    
    # remove some invalid feature names for XGBoost preselection
    df = md.df_featurized
    df = rename_cols(df) 
    md.df_featurized = df
    # we can now train modnet
    results_file = f'results_{job_prefix}.txt'
    ga_settings = {"size_pop": 10, "num_generations": 5, "n_jobs": n_jobs, 
                    "refit": 5, "nested": 5 }
    # ga_settings = {"size_pop": 2, "num_generations": 2, "n_jobs": n_jobs, 
    #             "refit": 2, "nested": 2 }
    # split data into folds
    fold_data = []
    for ind, (train, test) in enumerate(matbench_kfold_splits(md, classification=False)):
        path = f"folds/train_moddata_{job_prefix}_f{ind}"
        test_path = f"test_folds/test_moddata_{job_prefix}_f{ind}"
        if os.path.isfile(path):
            train_data = MODData.load(path)
            test_data = MODData.load(test_path)
        else:
            train_data, test_data = md.split((train, test))
            # feature selection may take a while if too many columns...
            train_data = xgb_preselection(train_data, n_jobs=n_jobs)
            train_data.feature_selection(n=-1, n_jobs=n_jobs)                
            os.makedirs("folds", exist_ok=True)
            train_data.save(path)
            os.makedirs("test_folds", exist_ok=True)
            test_data.save(test_path)
        fold_data.append((train_data, test_data)) 
        # fold_data will be a list of tuples with train and test MODData
    
    # if we want to run a fold individually this may be useful
    # get the index of the fold to use for training
    # if argument is passed to the script
    fold_index = None
    if len(sys.argv) > 1:
        fold_index = int(sys.argv[1])
        fold = fold_data[fold_index]
        # clear the fold data
        fold_data = [fold]  # this must release memory also

    # write file if not exists, else append
    with open(results_file, 'w') as f:
        print('--- calculating best model with GA ---')
        f.write('--- calculating best model with GA ---\n')

    # run GA in each fold
    folds_results = []
    for fold_number, fold in enumerate(fold_data):
        if fold_index is not None: # if fold_index is passed as argument
            fold_number = fold_index
        train_fold_data, test_fold_data = fold
        model_name = f'out/MODNet_{job_prefix}_bestGA_f{fold_number}'
        if not os.path.isfile(model_name):
        # sample threshold is set to 10000 to avoid bad sampling 
        # during hyperparameter optimization in GA for stability
        # exp(stability) contains many results close to 1
            ga = FitGenetic(train_fold_data, sample_threshold=10000)
            best_model = ga.run(**ga_settings)
            best_model.save(model_name)
            del(ga) # free memory
        else:
            best_model = MODNetModel.load(model_name)
        # print the full information on best_model object
        train_val_result = best_model.evaluate(train_fold_data)
        # get test result
        test_result = best_model.evaluate(test_fold_data)
        fold_result = {fold_number: {'train_val_result': train_val_result, 'test_result': test_result}}
        folds_results.append(fold_result)
        del(best_model) # free memory

    # Initialize variables to store the sum of results across folds
    train_val_sum = 0
    test_sum = 0
    # Loop over the results of each fold
    for fold_result in folds_results:
        fold_number = list(fold_result.keys())[0]
        train_val_result = fold_result[fold_number]['train_val_result']
        test_result = fold_result[fold_number]['test_result']
        # Add the current fold results to the corresponding sums
        train_val_sum += train_val_result
        test_sum += test_result
        with open(results_file, 'a') as f:
        # Print the results to the console
            print(f"Fold {fold_number} results:")
            print(f"  Train/validation metrics: {train_val_result}")
            print(f"  Test metrics: {test_result}")
            print()
            # Write the results to the file
            f.write(f"Fold {fold_number} results:\n")
            f.write(f"  Train/validation metrics: {train_val_result}\n")
            f.write(f"  Test metrics: {test_result}\n")
            f.write("\n")

    # Calculate the average of the results across folds
    num_folds = len(folds_results)
    train_val_avg = train_val_sum / num_folds
    test_avg = test_sum / num_folds

    with open(results_file, 'a') as f:
        # Print the average results to the console
        print(f"Average results across {num_folds} folds:")
        print(f"  Train/validation metrics: {train_val_avg}")
        print(f"  Test metrics: {test_avg}")
        # Write the average results to the file
        f.write(f"Average results across {num_folds} folds:\n")
        f.write(f"  Train/validation metrics: {train_val_avg}\n")
        f.write(f"  Test metrics: {test_avg}\n")

    # NOW WE CREATE A FINAL ENSEMBLE MODEL ACROSS ALL FOLDS
    # load all MODNet_oqmd_stability_bestGA_f0 up to f4 and make an ensemble
    models = []
    os.makedirs('final_model', exist_ok=True)
    if not os.path.isfile(f'final_model/MODNet_{job_prefix}_final_ensemble'):
        for fold_number in range(5):
            model = EnsembleMODNetModel.load(f'out/MODNet_{job_prefix}_bestGA_f{fold_number}')
            models.append(model)
        final_model = EnsembleMODNetModel(modnet_models=models)
        final_model.save(f'final_model/MODNet_{job_prefix}_final_ensemble')

    # WE CAN NOW USE THE FINAL ENSEMBLE MODEL TO PREDICT ON THE TEST SET
    # First we test the final ensemble on random samples of the full training data.
    # to verify the quality on samples that were included in its training
    featurized_data_fname = f'out/{job_prefix}_featurized'
    test_featurized_data_fname = f"test_data/{job_prefix}_test_moddata_featurized"
    # Load Model
    final_model = EnsembleMODNetModel.load(f'final_model/MODNet_{job_prefix}_final_ensemble')
    ensemble_train_data = MODData.load(featurized_data_fname)
    ensemble_train_data.df_featurized = rename_cols(ensemble_train_data.df_featurized)
    ensemble_test_data = MODData.load(test_featurized_data_fname)
    ensemble_test_data.df_featurized = rename_cols(ensemble_test_data.df_featurized)
    # WE NOW SAMPLE RANDOMLY FROM THE TRAINING DATA
    # Set the random seed to make the samples reproducible
    random.seed(42)
    num_sets = 5
    length = len(ensemble_train_data.df_featurized)
    sample_size = len(ensemble_test_data.df_featurized)
    # we will sample the training data 5 times with same size as test data
    # and average the results, this is to have more comparable results
    
    # Variables to accumulate statistics
    r2_total = 0.0
    mae_total = 0.0
    mape_total = 0.0
    medae_total = 0.0
    mse_total = 0.0
    rmse_total = 0.0
    ev_total = 0.0
    avg_stds_total = 0.0
    final_model_fileresults = f"final_model_{job_prefix}_statistics.txt"
    with open(final_model_fileresults, "w") as f:
        f.write("Final Model Statistics\n")
        f.write("--- Statistics on random samples of 5% of the data --- \n")
    
    for i in range(num_sets):
        # Generate a set of samples
        md_train = ensemble_train_data.from_indices(random.sample(range(length), sample_size))
        # print(md_train.df_featurized)
        result1, stds1 = final_model.predict(md_train, return_unc=True)
        predicted_stability1 = exp_to_normal(result1['exp_stability'])
        actual_stability = exp_to_normal(md_train.df_targets['exp_stability'])

        # Calculate the metrics for Final Model
        r2 = r2_score(actual_stability, predicted_stability1)
        mae = mean_absolute_error(actual_stability, predicted_stability1)
        mean_actual_stability = np.mean(actual_stability)
        mape = np.round(mae / mean_actual_stability, 5) * 100
        medae = np.median(np.abs(actual_stability - predicted_stability1))
        mse = mean_squared_error(actual_stability, predicted_stability1)
        rmse = np.sqrt(mean_squared_error(actual_stability, predicted_stability1))
        ev = explained_variance_score(actual_stability, predicted_stability1)

        # Calculate the average of the stds for final model
        avg_stds = np.mean(stds1['exp_stability'])

        # Accumulate the statistics
        r2_total += r2
        mae_total += mae
        mape_total += mape
        medae_total += medae
        mse_total += mse
        rmse_total += rmse
        ev_total += ev
        avg_stds_total += avg_stds

        # Print the metrics
        print(f"Set {i+1} Metrics:")
        print("Final Model:")
        print(f"R^2 score: {r2}")
        print(f"MAE: {mae}")
        print(f"Mean: {mean_actual_stability}")
        print(f"MAPE: {mape}%")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"Explained Variance Score: {ev}")
        print(f"Median Absolute Error: {medae}")
        print(f"Mean: {mean_actual_stability}")
        print(f"\nAverage Std of Final Model: {avg_stds}")
        # Write the statistics to a text file
        with open(final_model_fileresults, "a") as f:
            f.write(f"Set {i+1} Metrics:\n\n")
            f.write("Final Model:\n")
            f.write(f"R^2 score: {r2}\n")
            f.write(f"MAE: {mae}\n")
            f.write(f"Mean: {mean_actual_stability}\n")
            f.write(f"MAPE: {mape}%\n")
            f.write(f"MSE: {mse}\n")
            f.write(f"RMSE: {rmse}\n")
            f.write(f"Explained Variance Score: {ev}\n")
            f.write(f"Median Absolute Error: {medae}\n")
            f.write(f"\nAverage Std of Model 1: {avg_stds}\n")

    # Calculate the average statistics across all sets
    avg_r2 = r2_total / num_sets
    avg_mae = mae_total / num_sets
    avg_mape = mape_total / num_sets
    avg_medae = medae_total / num_sets
    avg_mse = mse_total / num_sets
    avg_rmse = rmse_total / num_sets
    avg_ev = ev_total / num_sets
    avg_avg_stds = avg_stds_total / num_sets

    # Print the average statistics
    print("\nAverage Metrics Across All Sets:")
    print(f"Average R^2 score: {avg_r2}")
    print(f"Average MAE: {avg_mae}")
    print(f"Average MAPE: {avg_mape}%")
    print(f"Average Median Absolute Error: {avg_medae}")
    print(f"Average MSE: {avg_mse}")
    print(f"Average RMSE: {avg_rmse}")
    print(f"Average Explained Variance Score: {avg_ev}")
    print(f"Average Average Std of Final Model: {avg_avg_stds}")

    with open(final_model_fileresults, "a") as f:
        f.write("\n\nAverage Metrics Across All Sets:\n")
        f.write("Final Model (training data):\n")
        f.write(f"Average R^2 score: {avg_r2}\n")
        f.write(f"Average MAE: {avg_mae}\n")
        f.write(f"Average MAPE: {avg_mape}%\n")
        f.write(f"Average Median Absolute Error: {avg_medae}\n")
        f.write(f"Average MSE: {avg_mse}\n")
        f.write(f"Average RMSE: {avg_rmse}\n")
        f.write(f"Average Explained Variance Score: {avg_ev}\n")
        f.write(f"Average Average Std of Final Model: {avg_avg_stds}\n")

    # Now we will test the model on the test data
    test_results, test_stds = final_model.predict(ensemble_test_data, return_unc=True)
    test_predicted = exp_to_normal(test_results['exp_stability'])
    test_actual = exp_to_normal(ensemble_test_data.df_targets['exp_stability'])

    # Calculate the metrics for the test set
    test_r2 = r2_score(test_actual, test_predicted)
    test_mae = mean_absolute_error(test_actual, test_predicted)
    test_mean_actual = np.mean(test_actual)
    test_mape = np.round(test_mae / test_mean_actual, 5) * 100
    test_medae = np.median(np.abs(test_actual - test_predicted))
    test_mse = mean_squared_error(test_actual, test_predicted)
    test_rmse = np.sqrt(mean_squared_error(test_actual, test_predicted))
    test_ev = explained_variance_score(test_actual, test_predicted)

    avg_test_std = np.mean(test_stds['exp_stability'])
    # print the metrics
    print("\nTest Set Metrics:")
    print(f"R^2 score: {test_r2}")
    print(f"MAE: {test_mae}")
    print(f"Mean: {test_mean_actual}")
    print(f"MAPE: {test_mape}%")
    print(f"MSE: {test_mse}")
    print(f"RMSE: {test_rmse}")
    print(f"Explained Variance Score: {test_ev}")
    print(f"Average Std of Final Model: {avg_test_std}")
    with open(final_model_fileresults, "a") as f:
        f.write("\n\nTest Set Metrics:\n")
        f.write(f"R^2 score: {test_r2}\n")
        f.write(f"MAE: {test_mae}\n")
        f.write(f"Mean: {test_mean_actual}\n")
        f.write(f"MAPE: {test_mape}%\n")
        f.write(f"MSE: {test_mse}\n")
        f.write(f"RMSE: {test_rmse}\n")
        f.write(f"Explained Variance Score: {test_ev}\n")
        f.write(f"Average Std of Final Model: {avg_test_std}\n")

    print("Statistics have been written to 'final_model_statistics.txt'.")


if __name__ == "__main__":
    setup_threading()
    ## MODData with fully featurized dataset
    path_full_featurized_data = "out/oqmdhalides_stab_featurizedOmegaROSA_complete"
    
    # job_prefix = "oqmdhalides_stab_invariant"
    # main(job_prefix)

    # job_prefix = "oqmdhalides_stab_matminerROSAG"
    # to_drop='pretrained_megnet+OFMencoded+adjacent+bondfractions'
    # process_featurized_data(path_full_featurized_data, job_prefix=job_prefix, to_drop=to_drop)
    # main(job_prefix)

    # job_prefix = "oqmdhalides_stab_omegaROSAG"
    # to_drop=''
    # process_featurized_data(path_full_featurized_data, job_prefix=job_prefix, to_drop=to_drop)
    # main(job_prefix)
    path_full_featurized_data = "out/oqmdhalides_stab_featurizedOmegaEformROSA_complete"
    job_prefix = "oqmdhalides_stab_omega_eform"
    to_drop = 'ROSA+G' 
    # adjacent_featurizer(path_full_featurized_data)
    # substitute_adjacent(path_full_featurized_data)
    process_featurized_data(path_full_featurized_data, job_prefix=job_prefix, to_drop=to_drop)
    main(job_prefix)

    # job_prefix = "oqmdhalides_stab_matminer"
    # to_drop = 'pretrained_megnet+ROSA+OFMencoded+adjacent+G+bondfractions' 
    # process_featurized_data(path_full_featurized_data, job_prefix=job_prefix, to_drop=to_drop)
    # main(job_prefix)
    
    # job_prefix = "oqmdhalides_stab_matminerOFM"
    # to_drop = 'pretrained_megnet+ROSA+adjacent+G+bondfractions' 
    # process_featurized_data(path_full_featurized_data, job_prefix=job_prefix, to_drop=to_drop)
    # main(job_prefix)

    # job_prefix = "oqmdhalides_stab_matminerOMEG"
    # to_drop = 'ROSA+adjacent+G+bondfractions' 
    # process_featurized_data(path_full_featurized_data, job_prefix=job_prefix, to_drop=to_drop)
    # main(job_prefix)
    
    # job_prefix = "oqmdhalides_stab_matminerBondFrac"
    # to_drop = 'pretrained_megnet+ROSA+OFMencoded+adjacent+G' 
    # process_featurized_data(path_full_featurized_data, job_prefix=job_prefix, to_drop=to_drop)
    # main(job_prefix)

    # job_prefix = "oqmdhalides_stab_matminerOFMBondFrac"
    # to_drop = 'pretrained_megnet+ROSA+adjacent+G' 
    # process_featurized_data(path_full_featurized_data, job_prefix=job_prefix, to_drop=to_drop)
    # main(job_prefix)
    
    # job_prefix = "oqmdhalides_stab_matminerOMEGBondFrac"
    # to_drop = 'ROSA+adjacent+G' 
    # process_featurized_data(path_full_featurized_data, job_prefix=job_prefix, to_drop=to_drop)
    # main(job_prefix)
