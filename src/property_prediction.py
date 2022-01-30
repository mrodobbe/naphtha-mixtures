import numpy as np
from src.featurization import molecular_feature_collection, get_gmm
from GauL.GauLsrc.representation import represent
from GauL.GauLsrc.plots import output_plot
from GauL.GauLsrc.crossDouble import training, write_statistics
from GauL.GauLsrc.results_processing import train_results_to_logfile, test_results_to_logfile
from tensorflow.keras.models import load_model
import pickle


def representation_maker(smiles_dict, save_folder):
    try:
        with open(str(save_folder + "/representation_dict.pickle"), "rb") as f:
            representation_dict = pickle.load(f)
    except FileNotFoundError:
        representation_dict = {}
        gmm, smiles, conformers = get_gmm(save_folder, smiles_dict, "representing")
        representations, bad = represent(smiles, conformers, gmm, save_folder)
        for smile, representation in zip(smiles, representations):
            representation_dict[smile] = representation
        with open(str(save_folder + "/representation_dict.pickle"), "wb") as f:
            pickle.dump(representation_dict, f)

    return representation_dict


def representation_selector(dataframe, smiles_dict, save_folder):
    ch_dict, bp_dict, tc_dict, sg_dict, vap_dict, bp_to_predict, \
        tc_to_predict, sg_to_predict, vap_to_predict = molecular_feature_collection(dataframe, smiles_dict)

    representation_dict = representation_maker(smiles_dict, save_folder)

    bp_train_representations = []
    tc_train_representations = []
    sg_train_representations = []
    vap_train_representations = []

    bp_outputs = []
    tc_outputs = []
    sg_outputs = []
    vap_outputs = []

    bp_smiles_list = []
    tc_smiles_list = []
    sg_smiles_list = []
    vap_smiles_list = []

    bp_test_representations = []
    tc_test_representations = []
    sg_test_representations = []
    vap_test_representations = []

    for bp_smiles in bp_dict:
        bp_train_representations.append(representation_dict[bp_smiles])
        bp_outputs.append(bp_dict[bp_smiles])
        bp_smiles_list.append(bp_smiles)

    bp_outputs = np.asarray(bp_outputs).astype(np.float)

    for tc_smiles in tc_dict:
        tc_train_representations.append(representation_dict[tc_smiles])
        tc_outputs.append(tc_dict[tc_smiles])
        tc_smiles_list.append(tc_smiles)

    tc_outputs = np.asarray(tc_outputs).astype(np.float)

    for sg_smiles in sg_dict:
        sg_train_representations.append(representation_dict[sg_smiles])
        sg_outputs.append(sg_dict[sg_smiles])
        sg_smiles_list.append(sg_smiles)

    sg_outputs = np.asarray(sg_outputs).astype(np.float)

    for vap_smiles in vap_dict:
        vap_train_representations.append(representation_dict[vap_smiles])
        vap_outputs.append(vap_dict[vap_smiles])
        vap_smiles_list.append(vap_smiles)

    vap_outputs = np.asarray(vap_outputs).astype(np.float)

    bp_train_representations = np.stack(bp_train_representations)
    tc_train_representations = np.stack(tc_train_representations)
    sg_train_representations = np.stack(sg_train_representations)
    vap_train_representations = np.stack(vap_train_representations)

    for bp_test_smiles in bp_to_predict:
        bp_test_representations.append(representation_dict[bp_test_smiles])

    for tc_test_smiles in tc_to_predict:
        tc_test_representations.append(representation_dict[tc_test_smiles])

    for sg_test_smiles in sg_to_predict:
        sg_test_representations.append(representation_dict[sg_test_smiles])

    for vap_test_smiles in vap_to_predict:
        vap_test_representations.append(representation_dict[vap_test_smiles])

    bp_test_representations = np.stack(bp_test_representations)
    tc_test_representations = np.stack(tc_test_representations)
    sg_test_representations = np.stack(sg_test_representations)
    vap_test_representations = np.stack(vap_test_representations)

    return bp_train_representations, tc_train_representations, sg_train_representations, vap_train_representations, \
        bp_outputs, tc_outputs, sg_outputs, vap_outputs, \
        bp_test_representations, tc_test_representations, sg_test_representations, vap_test_representations, \
        ch_dict, bp_dict, tc_dict, sg_dict, vap_dict, \
        bp_smiles_list, tc_smiles_list, sg_smiles_list, vap_smiles_list, \
        bp_to_predict, tc_to_predict, sg_to_predict, vap_to_predict, representation_dict


def predict_properties(dataframe, smiles_dict, save_folder):
    bp_train_representations, tc_train_representations, sg_train_representations, vap_train_representations, \
        bp_outputs, tc_outputs, sg_outputs, vap_outputs, \
        bp_test_representations, tc_test_representations, sg_test_representations, vap_test_representations, \
        ch_dict, bp_dict, tc_dict, sg_dict, vap_dict, \
        bp_smiles, tc_smiles, sg_smiles, vap_smiles, \
        bp_to_predict, tc_to_predict, sg_to_predict, vap_to_predict, \
        representation_dict = representation_selector(dataframe, smiles_dict, save_folder)

    bp_save_folder = str(save_folder + "/Boiling Point")
    tc_save_folder = str(save_folder + "/Critical Point")
    sg_save_folder = str(save_folder + "/Density")
    vap_save_folder = str(save_folder + "/Vapor Pressure")

    output_plot(bp_smiles, bp_outputs, "bp", "output_plot", bp_save_folder)
    output_plot(tc_smiles, tc_outputs, "tc", "output_plot", tc_save_folder)
    output_plot(sg_smiles, sg_outputs, "sg", "output_plot", sg_save_folder)
    output_plot(vap_smiles, vap_outputs, "vap", "output_plot", vap_save_folder)

    n_folds = 10  # TODO: Make argument

    folders = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Fold 6', 'Fold 7', 'Fold 8', 'Fold 9', 'Fold 10']

    try:
        print("Found pretrained Boiling Point models!")
        bp_models = [load_model(str(bp_save_folder + '/' + folder)) for folder in folders]
        ensemble = np.array([])
        for model in bp_models:
            test_predicted = model.predict(bp_test_representations).reshape(-1)
            test_predicted = np.asarray(test_predicted).astype(np.float)
            if len(ensemble) == 0:
                ensemble = test_predicted
            else:
                ensemble = np.vstack((ensemble, test_predicted))

        ensemble_prediction = np.mean(ensemble, axis=0)
        ensemble_sd = np.std(ensemble, axis=0)

        test_results_to_logfile(bp_to_predict, ensemble_prediction, ensemble_sd, "bp", bp_save_folder)
        for smiles, prediction in zip(bp_to_predict, ensemble_prediction):
            bp_dict[smiles] = prediction
    except OSError:
        bp_info = training(bp_smiles, bp_train_representations, bp_outputs, bp_save_folder, "bp", n_folds)
        bp_results_list = write_statistics(bp_info, "bp", n_folds, bp_save_folder)
        train_results_to_logfile(bp_smiles, bp_outputs, bp_results_list, bp_train_representations,
                                 "bp", bp_save_folder)
        bp_models = [load_model(str(bp_save_folder + '/' + folder)) for folder in folders]
        ensemble = np.array([])
        for model in bp_models:
            test_predicted = model.predict(bp_test_representations).reshape(-1)
            test_predicted = np.asarray(test_predicted).astype(np.float)
            if len(ensemble) == 0:
                ensemble = test_predicted
            else:
                ensemble = np.vstack((ensemble, test_predicted))

        ensemble_prediction = np.mean(ensemble, axis=0)
        ensemble_sd = np.std(ensemble, axis=0)

        test_results_to_logfile(bp_to_predict, ensemble_prediction, ensemble_sd, "bp", bp_save_folder)
        for smiles, prediction in zip(bp_to_predict, ensemble_prediction):
            bp_dict[smiles] = prediction

    try:
        print("Found pretrained Critical Temperature models!")
        tc_models = [load_model(str(tc_save_folder + '/' + folder)) for folder in folders]
        ensemble = np.array([])
        for model in tc_models:
            test_predicted = model.predict(tc_test_representations).reshape(-1)
            test_predicted = np.asarray(test_predicted).astype(np.float)
            if len(ensemble) == 0:
                ensemble = test_predicted
            else:
                ensemble = np.vstack((ensemble, test_predicted))

        ensemble_prediction = np.mean(ensemble, axis=0)
        ensemble_sd = np.std(ensemble, axis=0)

        test_results_to_logfile(tc_to_predict, ensemble_prediction, ensemble_sd, "tc", tc_save_folder)
        for smiles, prediction in zip(tc_to_predict, ensemble_prediction):
            tc_dict[smiles] = prediction
    except OSError:
        tc_info = training(tc_smiles, tc_train_representations, tc_outputs, tc_save_folder, "tc", n_folds)
        tc_results_list = write_statistics(tc_info, "tc", n_folds, tc_save_folder)
        train_results_to_logfile(tc_smiles, tc_outputs, tc_results_list, tc_train_representations,
                                 "tc", tc_save_folder)

        tc_models = [load_model(str(tc_save_folder + '/' + folder)) for folder in folders]
        ensemble = np.array([])
        for model in tc_models:
            test_predicted = model.predict(tc_test_representations).reshape(-1)
            test_predicted = np.asarray(test_predicted).astype(np.float)
            if len(ensemble) == 0:
                ensemble = test_predicted
            else:
                ensemble = np.vstack((ensemble, test_predicted))

        ensemble_prediction = np.mean(ensemble, axis=0)
        ensemble_sd = np.std(ensemble, axis=0)

        test_results_to_logfile(tc_to_predict, ensemble_prediction, ensemble_sd, "tc", tc_save_folder)
        for smiles, prediction in zip(tc_to_predict, ensemble_prediction):
            tc_dict[smiles] = prediction

    try:
        print("Found pretrained Density models!")

        sg_models = [load_model(str(sg_save_folder + '/' + folder)) for folder in folders]
        ensemble = np.array([])
        for model in sg_models:
            test_predicted = model.predict(sg_test_representations).reshape(-1)
            test_predicted = np.asarray(test_predicted).astype(np.float)
            if len(ensemble) == 0:
                ensemble = test_predicted
            else:
                ensemble = np.vstack((ensemble, test_predicted))

        ensemble_prediction = np.mean(ensemble, axis=0)
        ensemble_sd = np.std(ensemble, axis=0)

        test_results_to_logfile(sg_to_predict, ensemble_prediction, ensemble_sd, "sg", sg_save_folder)
        for smiles, prediction in zip(sg_to_predict, ensemble_prediction):
            sg_dict[smiles] = prediction
    except OSError:

        sg_info = training(sg_smiles, sg_train_representations, sg_outputs, sg_save_folder, "sg", n_folds)
        sg_results_list = write_statistics(sg_info, "sg", n_folds, sg_save_folder)
        train_results_to_logfile(sg_smiles, sg_outputs, sg_results_list, sg_train_representations,
                                 "sg", sg_save_folder)

        sg_models = [load_model(str(sg_save_folder + '/' + folder)) for folder in folders]
        ensemble = np.array([])
        for model in sg_models:
            test_predicted = model.predict(sg_test_representations).reshape(-1)
            test_predicted = np.asarray(test_predicted).astype(np.float)
            if len(ensemble) == 0:
                ensemble = test_predicted
            else:
                ensemble = np.vstack((ensemble, test_predicted))

        ensemble_prediction = np.mean(ensemble, axis=0)
        ensemble_sd = np.std(ensemble, axis=0)

        test_results_to_logfile(sg_to_predict, ensemble_prediction, ensemble_sd, "sg", sg_save_folder)
        for smiles, prediction in zip(sg_to_predict, ensemble_prediction):
            sg_dict[smiles] = prediction

    try:
        print("Found pretrained Vapor Pressure models!")
        vap_models = [load_model(str(vap_save_folder + '/' + folder)) for folder in folders]

        ensemble = np.array([])
        for model in vap_models:
            test_predicted = model.predict(vap_test_representations).reshape(-1)
            test_predicted = np.asarray(test_predicted).astype(np.float)
            if len(ensemble) == 0:
                ensemble = test_predicted
            else:
                ensemble = np.vstack((ensemble, test_predicted))

        ensemble_prediction = np.mean(ensemble, axis=0)
        ensemble_sd = np.std(ensemble, axis=0)

        test_results_to_logfile(vap_to_predict, ensemble_prediction, ensemble_sd, "vap", vap_save_folder)
        for smiles, prediction in zip(vap_to_predict, ensemble_prediction):
            vap_dict[smiles] = prediction

    except OSError:
        vap_info = training(vap_smiles, vap_train_representations, vap_outputs, vap_save_folder, "vap", n_folds)
        vap_results_list = write_statistics(vap_info, "vap", n_folds, vap_save_folder)
        train_results_to_logfile(vap_smiles, vap_outputs, vap_results_list, vap_train_representations,
                                 "vap", vap_save_folder)

        vap_models = [load_model(str(vap_save_folder + '/' + folder)) for folder in folders]

        ensemble = np.array([])
        for model in vap_models:
            test_predicted = model.predict(vap_test_representations).reshape(-1)
            test_predicted = np.asarray(test_predicted).astype(np.float)
            if len(ensemble) == 0:
                ensemble = test_predicted
            else:
                ensemble = np.vstack((ensemble, test_predicted))

        ensemble_prediction = np.mean(ensemble, axis=0)
        ensemble_sd = np.std(ensemble, axis=0)

        test_results_to_logfile(vap_to_predict, ensemble_prediction, ensemble_sd, "vap", vap_save_folder)
        for smiles, prediction in zip(vap_to_predict, ensemble_prediction):
            vap_dict[smiles] = prediction

    return ch_dict, bp_dict, tc_dict, sg_dict, vap_dict, representation_dict
