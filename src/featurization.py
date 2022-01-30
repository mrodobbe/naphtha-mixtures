import numpy as np
from src.makeMolecule import conformer
from src.gaussian import gmm
import pickle


def get_gmm(save_folder, smiles_dict, t):
    try:
        if t == "representing":
            with open(str(save_folder + "/gmm_dictionary.pickle"), "rb") as f:
                gmm_dictionary = pickle.load(f)
            print("Loaded the GMM data!")
            smiles, conformers = make_conformers(smiles_dict)
            return gmm_dictionary, smiles, conformers
        else:
            with open(str(save_folder + "/gmm_dictionary.pickle"), "rb") as f:
                gmm_dictionary = pickle.load(f)
            print("Loaded the GMM data!")
            return gmm_dictionary
    except FileNotFoundError:
        if t == "training":
            print("There is no file named gmm_dictionary.pickle included in {}.\nNew gaussian mixture models "
                  "will be created for all geometry features.\n".format(save_folder))
            smiles, conformers = make_conformers(smiles_dict)
            gmm_dictionary = gmm(smiles, conformers, save_folder)
            return gmm_dictionary
        else:
            print("There is no gmm_dictionary.pickle included in {}".format(save_folder))
            raise


def clean_dicts(smiles_dict, weight_dict):
    lumps = list(smiles_dict.keys())
    index_vector = [[i for i, e in enumerate(weight_dict[lump]) if e == 0] for lump in lumps]

    for lump, dels in zip(lumps, index_vector):
        weight_dict[lump] = np.delete(weight_dict[lump], dels)
        smiles_dict[lump] = list(np.delete(smiles_dict[lump], dels))
    print("Cleaned up the SMILES and weight dictionaries!")
    return smiles_dict, weight_dict


def molecular_feature_collection(dataframe, smiles_dictionary):

    # Headers should be: "CH", "Tboil", "Tboil code", "Tcrit", "Tcrit code", "d20", "d20 code", "pVap", "vap code"

    ch_dict = {}
    bp_dict = {}
    tc_dict = {}
    sg_dict = {}
    vap_dict = {}

    smiles = []
    for lump in smiles_dictionary:
        for smile in smiles_dictionary[lump]:
            smiles.append(smile)

    bp_to_predict = []
    tc_to_predict = []
    sg_to_predict = []
    vap_to_predict = []

    for smile in smiles:
        ch_dict[smile] = dataframe["CH"][smile]
        if dataframe["Tboil code"][smile] == 1:
            bp_dict[smile] = dataframe["Tboil"][smile]
        else:
            bp_to_predict.append(smile)
        if dataframe["Tcrit code"][smile] == 1 or dataframe["Tcrit code"][smile] == 2:
            tc_dict[smile] = dataframe["Tcrit"][smile]
        else:
            tc_to_predict.append(smile)
        if dataframe["d20 code"][smile] == "1,2":
            sg_dict[smile] = dataframe["d20"][smile]
        else:
            sg_to_predict.append(smile)
        if dataframe["vap code"][smile] == "1,2":
            vap_dict[smile] = dataframe["pVap"][smile]
        else:
            vap_to_predict.append(smile)

    return ch_dict, bp_dict, tc_dict, sg_dict, vap_dict, bp_to_predict, tc_to_predict, sg_to_predict, vap_to_predict


def mixture_features(ch_dict, bp_dict, tc_dict, sg_dict, vap_dict, smiles_dict):
    features_dict = {}
    for lump in smiles_dict:
        for molecule in smiles_dict[lump]:
            features_dict[molecule] = np.array([ch_dict[molecule], bp_dict[molecule],
                                                tc_dict[molecule], sg_dict[molecule],
                                                vap_dict[molecule]])

    return features_dict


def absolute_fractions(compositions, weight_dict, smiles_dict, lumps):
    all_fractions = []
    for comp in compositions:
        absolute_fraction = np.array([])
        for i in range(len(comp)):
            absolute_fraction = np.append(absolute_fraction, comp[i] * weight_dict[lumps[i]])
        absolute_fraction /= np.sum(absolute_fraction)
        all_fractions.append(absolute_fraction)
    all_molecules = []
    for lump in lumps:
        for mol in smiles_dict[lump]:
            all_molecules.append(mol)

    all_fractions = np.asarray(all_fractions).astype(np.float)

    return all_fractions, all_molecules


def condensed_mixture_features(features_dict, all_fractions, all_molecules):
    condensed_features = []
    for composition in all_fractions:
        features = []
        for i in range(len(all_molecules)):
            features.append(features_dict[all_molecules[i]])
        weighted_features = np.dot(composition, features)
        condensed_features.append(weighted_features)
    condensed_features = np.asarray(condensed_features).astype(np.float)

    return condensed_features


def make_mixture_features(compositions, ch_dict, bp_dict, tc_dict, sg_dict, vap_dict, smiles_dict, weight_dict, lumps):
    features_dict = mixture_features(ch_dict, bp_dict, tc_dict, sg_dict, vap_dict, smiles_dict)
    all_fractions, all_molecules = absolute_fractions(compositions, weight_dict, smiles_dict, lumps)
    condensed_features = condensed_mixture_features(features_dict, all_fractions, all_molecules)
    with open("mixed_features.pickle", "wb") as f:
        pickle.dump(condensed_features, f)

    return condensed_features, all_fractions, all_molecules


def make_conformers(smiles_dictionary):
    smiles = []
    for lump in smiles_dictionary:
        for smile in smiles_dictionary[lump]:
            smiles.append(smile)
    print("The number of SMILES is: {}".format(len(smiles)))
    conformers = [conformer(smiles_molecule) for smiles_molecule in smiles]

    return smiles, conformers


def condense_representations(representations, all_fractions):
    condensed_representations = []

    for composition in all_fractions:
        weighted_representations = (representations.T * composition).T
        condensed_representation = np.sum(weighted_representations, axis=0)
        condensed_representations.append(condensed_representation)

    condensed_representations = np.asarray(condensed_representations)

    return condensed_representations


def representation_checker(all_molecules, all_fractions, representation_dict):
    representations = [representation_dict[molecule] for molecule in all_molecules]
    condensed_representations = np.dot(all_fractions, representations)
    with open("mixed_representations.pickle", "wb") as f:
        pickle.dump(condensed_representations, f)
    return condensed_representations
