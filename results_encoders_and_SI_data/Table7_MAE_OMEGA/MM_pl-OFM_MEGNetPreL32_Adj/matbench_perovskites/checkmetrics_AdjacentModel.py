from modnet_gnn.preprocessing import MODData
import pandas as pd
from modnet_gnn.models import MODNetModel, EnsembleMODNetModel
from megnet.models import MEGNetModel
from modnet_b.featurizers.megnettools.megnet_setup_evaluate import load_model_scaler, megnet_evaluate_structures
import numpy as np
MAEs=[]
for ind in range(5):
     model_file = f'matbench_perovskites/adjacent_models/fold_{ind}/MEGNetModel__adjacent.h5'
     scaler_file = f'matbench_perovskites/adjacent_models/fold_{ind}/MEGNetModel__adjacent_scaler.pkl'
     model, scaler = load_model_scaler(n_targets=1, 
                         model_file=model_file, scaler_file=scaler_file)
     train_data = MODData.load(f'matbench_perovskites/folds/train_moddata_f{ind+1}')
     test_data = MODData.load(f'matbench_perovskites/folds/test_moddata_f{ind+1}')
     structures = test_data.df_structure['structure']
     print(structures)
     structures_valid,ypred=megnet_evaluate_structures(model,structures)
     # save ypred
     # 
     # np.save(f'ypred.npy', ypred)
     print('rescaled ypred')
     print(scaler.inverse_transform(ypred))
     print('y')
     print(test_data.targets)
     print('MAE')
     errors = np.abs(scaler.inverse_transform(ypred)-test_data.targets)
     MAE = np.mean(errors)
     # Print the list of MAE values
     print("List of error in original units, first 300:")
     for i, error in enumerate(errors.tolist()[:300]):
          print(structures_valid[i].composition, error)
     # now we compare with train_data.targets and get the correct MAE
     MAEs.append(MAE)
print(MAEs)
# take the average now
print("Average MAE")
print(np.mean(MAEs))
#
