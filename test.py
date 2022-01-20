from src.makeMolecule import input_checker
from src.featurization import get_gmm, clean_dicts, make_mixture_features, representation_checker
from src.property_prediction import predict_properties
import sys
from input import load_test_data
from tensorflow.keras.models import load_model
import numpy as np

input_checker(sys.argv, "naphtha_mixtures")

save_folder = sys.argv[1]
save_folder_bp = save_folder + "/bp"
save_folder_sg = save_folder + "/sg"

compositions, smiles_dict, weight_dict, df, lumps = load_test_data()
gmm_dictionary = get_gmm(save_folder, smiles_dict, "training")  # TODO: Make output nicer

smiles_dict, weight_dict = clean_dicts(smiles_dict, weight_dict)
ch_dict, bp_dict, tc_dict, sg_dict, vap_dict, representation_dict = predict_properties(df, smiles_dict, save_folder)

mixture_features, all_fractions, all_molecules = make_mixture_features(compositions,
                                                                       ch_dict,
                                                                       bp_dict,
                                                                       tc_dict,
                                                                       sg_dict,
                                                                       vap_dict,
                                                                       smiles_dict,
                                                                       weight_dict,
                                                                       lumps)
condensed_representations = representation_checker(all_molecules, all_fractions, representation_dict)

folders = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Fold 6', 'Fold 7', 'Fold 8', 'Fold 9', 'Fold 10']
models_bp = [load_model(str(save_folder_bp + '/' + folder)) for folder in folders]

ensemble_bp = np.array([])
for model in models_bp:
    test_predicted = model.predict([condensed_representations, mixture_features])
    test_predicted = np.asarray(test_predicted).astype(np.float)
    prediction_shape = test_predicted.shape
    if len(ensemble_bp) == 0:
        ensemble_bp = test_predicted.flatten()
    else:
        ensemble_bp = np.vstack((ensemble_bp, test_predicted.flatten()))
ensemble_prediction = np.average(ensemble_bp, axis=0)
ensemble_prediction = np.reshape(ensemble_prediction, prediction_shape)
ensemble_sd = np.std(ensemble_bp, axis=0)
ensemble_sd = np.reshape(ensemble_sd, prediction_shape)

predicted_bp = ensemble_prediction
np.savetxt(str(save_folder_bp + "/predicted_bp.txt"), predicted_bp)

models_sg = [load_model(str(save_folder_sg + '/' + folder)) for folder in folders]
ensemble_sg = np.array([])
for model in models_sg:
    test_predicted = model.predict([condensed_representations, mixture_features, predicted_bp]).reshape(-1)
    test_predicted = np.asarray(test_predicted).astype(np.float)
    prediction_shape = test_predicted.shape
    if len(ensemble_sg) == 0:
        ensemble_sg = test_predicted.flatten()
    else:
        ensemble_sg = np.vstack((ensemble_sg, test_predicted.flatten()))
ensemble_prediction_sg = np.average(ensemble_sg, axis=0)
ensemble_prediction_sg = np.reshape(ensemble_prediction_sg, prediction_shape)
ensemble_sd = np.std(ensemble_sg, axis=0)
ensemble_sd = np.reshape(ensemble_sd, prediction_shape)

np.savetxt(str(save_folder_sg + "/predicted_sg.txt"), ensemble_prediction_sg)
