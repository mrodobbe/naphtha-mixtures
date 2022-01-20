from src.makeMolecule import input_checker
from src.featurization import representation_checker, make_mixture_features, get_gmm, clean_dicts
from src.crossValidation import cv, cv_configurations
from src.property_prediction import predict_properties
from src.postprocessing import process_results
import numpy as np
from joblib import Parallel, delayed
import sys
from input import load_data

input_checker(sys.argv, "naphtha_mixtures")
print("All folders are present!")
save_folder = sys.argv[1]
property_to_train = sys.argv[2]

compositions, boiling_points, output_sg, smiles_dict, weight_dict, df, lumps = load_data(property_to_train)
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

save_folder_bp = save_folder + "/bp"
save_folder_sg = save_folder + "/" + str(property_to_train)
kf, n_jobs, n_folds = cv_configurations()
all_results_bp = Parallel(n_jobs=n_jobs)(delayed(cv)(condensed_representations,
                                                     mixture_features,
                                                     boiling_points,
                                                     loop_kf,
                                                     i,
                                                     save_folder_bp,
                                                     np.arange(len(condensed_representations)))
                                         for loop_kf, i in zip(kf.split(condensed_representations),
                                                               range(1, n_folds+1)))

predicted_bp = process_results(all_results_bp, "bp", n_folds, save_folder_bp)

all_results_sg = Parallel(n_jobs=n_jobs)(delayed(cv)(condensed_representations,
                                                     mixture_features,
                                                     output_sg,
                                                     loop_kf,
                                                     i,
                                                     save_folder_sg,
                                                     np.arange(len(condensed_representations)),
                                                     bp=predicted_bp)
                                         for loop_kf, i in zip(kf.split(condensed_representations),
                                                               range(1, n_folds+1)))

process_results(all_results_sg, property_to_train, n_folds, save_folder_sg)
