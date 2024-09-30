from modnet_gnn.featurizers.presets.modnet_omega_2024 import MODNetOmega2024
from modnet_gnn.preprocessing import MODData
from modnet_gnn.matbench.benchmark import matbench_kfold_splits

import os

featurized_data_fname='/globalscratch/users/r/g/rgouvea/ProjectVAE_MODNet/MODNetCalcs_Prist_RegAE/calcs_matbench_perovskites/MODNet_perovskites_MNetencodOFM_MEGNet32_Adj/matbench_perovskites/precomputed/matbench_perovskites_matminerOFMPre.pkl.gz'
md = MODData.load(featurized_data_fname)
# Just need the structure and targets from the dataset
md_b = MODData(materials = md.df_structure['structure'],
                        targets = md.df_targets,
                        structure_ids = md.structure_ids,
                        target_names = md.df_targets.columns.to_list(),
                        )

maes = []
for ind, (train, test) in enumerate(matbench_kfold_splits(md_b, classification=False)):
     train_data, test_data = md_b.split((train, test))
     # Check if the indices are not overlapping
     assert len(set(train_data.df_targets.index).intersection(set(test_data.df_targets.index))) == 0
     
     # The adjacent model is trained automatically when we create 
     # the MODNetOmega2024 object, unless it already exists in the
     # folder.
     modnet2023 = MODNetOmega2024(adjacent_model=True,
                         structures=train_data.df_structure['structure'],
                         targets=train_data.df_targets.values)
     
     ######################################################
     #### if you go into the modnet_omega_2024.py file, you 
     #### could comment the following lines to deactivate the
     #### elemental embedding import into the MEGNet model
     ###################################################### 
     # from megnet.data.crystal import get_elemental_embeddings
     # el_embeddings = get_elemental_embeddings()
     # Find the embedding layer  index in all the model layers
     # embedding_layer_index = [i for i, j in enumerate(model.layers) if j.name.startswith('atom_embedding')][0]
     # 
     # Set the weights to our previous embedding
     # model.layers[embedding_layer_index].set_weights([embedding])
     #
     # Freeze the weights
     # model.layers[embedding_layer_index].trainable = False
     ######################################################

     # Make dir to put the MEGNetModel__adjacent.h5 file
     os.makedirs('adjacent_models',exist_ok=True)
     os.makedirs(f'adjacent_models/fold_{ind}',exist_ok=True)
     # Now move the file to the dir
     os.rename('MEGNetModel__adjacent.h5',f'adjacent_models/fold_{ind}/MEGNetModel__adjacent.h5')
     os.rename('MEGNetModel__adjacent_scaler.pkl',f'adjacent_models/fold_{ind}/MEGNetModel__adjacent_scaler.pkl')
     
     # Now we can load the model and scaler
     from megnet.models import MEGNetModel
     from modnet_gnn.featurizers.megnettools.megnet_setup_evaluate import load_model_scaler
     model,scaler=load_model_scaler(n_targets=1, 
                        model_file=f'adjacent_models/fold_{ind}/MEGNetModel__adjacent.h5',
                        scaler_file=f'adjacent_models/fold_{ind}/MEGNetModel__adjacent_scaler.pkl', 
                        )
     print(test_data.df_structure['structure'])

     # And perform the prediction on the test data
     results = model.predict_structures(test_data.df_structure['structure'])
     results = scaler.inverse_transform(results)
     print(results)
     print(test_data.df_targets)

     from sklearn.metrics import mean_absolute_error
     mae = mean_absolute_error(test_data.df_targets,results)
     print(mae)
     # Save MAE to a text file with appropriate fold number
     with open(f'adjacent_models/fold_{ind}/MAE.txt', 'w') as f:
         f.write(f"Fold {ind} MAE: {mae}")

     maes.append(mae)

# Save MAES to a text file with corresponding fold
with open('maes.txt', 'w') as f:
     for i, mae in enumerate(maes):
          f.write(f"Fold {i} MAE: {mae}\n")
     # average now
     f.write(f"Average MAE: {sum(maes)/len(maes)}")