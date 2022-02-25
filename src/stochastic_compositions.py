import numpy as np
from rdkit import Chem
import pandas as pd
from src.featurization import clean_dicts, absolute_fractions


# lumps = ["A6", "A7", "A8", "A8-1", "A8-2", "A9", "A10", "A11",
#          "N5", "N6", "N6-1", "N6-2", "N7", "N8", "N9", "N10", "N11",
#          "O5", "O6",
#          "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12",
#          "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12"]
# lumps = ["P4", "P5", "P6", "P7", "P8", "P9",
#          "I4", "I5", "I6", "I7", "I8", "I9",
#          "N5", "N6", "N7", "N8",
#          "A6", "A7", "A8", "A9"]
#
# df = pd.read_excel("C:/Users/mrodobbe/Documents/Research/GauL-mixture/libraries.xlsx",
#                    sheet_name="Labeled Library",
#                    index_col=0)
# df_inp = pd.read_excel("../piona_rossini.xlsx",
#                        sheet_name="input",
#                        index_col=None)
# comp = df_inp.to_numpy()


def generate_smiles_dictionary(lumps, dataframe):
    smiles_dict = {}

    for lump in lumps:
        if "O" in lump:
            continue
        # smiles_dict[lump] = dataframe["SMILES"][dataframe.index[dataframe['Lump'] == lump].to_numpy()].tolist()
        smiles_dict[lump] = dataframe.index[dataframe['Lump'] == lump].tolist()

    smiles_dict["N6-1"] = ["CC1CCCC1"]
    smiles_dict["N6-2"] = ["C1CCCCC1"]
    smiles_dict["A8-1"] = ["Cc1cccc(C)c1", "Cc1ccc(C)cc1", "Cc1ccccc1C"]
    smiles_dict["A8-2"] = ["CCc1ccccc1"]

    # smiles_dict["I5"] = ["CCC(C)C"]
    # smiles_dict["I6"] = ["CCCC(C)C"]
    # smiles_dict["I7"] = ["CCCCC(C)C"]
    # smiles_dict["I8"] = ["CCCCCC(C)C"]
    # smiles_dict["I9"] = ["CCCCCCC(C)C"]
    # smiles_dict["I10"] = ["CCCCCCCC(C)C"]
    # smiles_dict["I11"] = ["CCCCCCCCC(C)C"]
    # smiles_dict["I12"] = ["CCCCCCCCCC(C)C"]
    # smiles_dict["N5"] = ["C1CCCC1"]
    # smiles_dict["N7"] = ["CC1CCCCC1"]
    # smiles_dict["N8"] = ["CCC1CCCCC1"]
    # smiles_dict["N9"] = ["CCCC1CCCCC1"]
    # smiles_dict["N10"] = ["CCCCC1CCCCC1"]
    # smiles_dict["N11"] = ["CCCCCC1CCCCC1"]
    # smiles_dict["N12"] = ["CCCCCCC1CCCCC1"]
    # smiles_dict["A9"] = ["CCCc1ccccc1"]
    # smiles_dict["A10"] = ["CCCCc1ccccc1"]
    # smiles_dict["A11"] = ["CCCCCc1ccccc1"]
    # smiles_dict["A12"] = ["CCCCCCc1ccccc1"]
    smiles_dict["O4"] = ["C=CCC"]
    smiles_dict["O5"] = ["C=CCCC"]
    smiles_dict["O6"] = ["C=CCCCC"]
    smiles_dict["O7"] = ["C=CCCCCC"]
    smiles_dict["O8"] = ["C=CCCCCCC"]
    smiles_dict["O9"] = ["C=CCCCCCCC"]
    smiles_dict["O10"] = ["C=CCCCCCCCC"]
    smiles_dict["O11"] = ["C=CCCCCCCCCC"]
    smiles_dict["O12"] = ["C=CCCCCCCCCCC"]
    smiles_dict["DO4"] = ["C=CC=C"]
    smiles_dict["DO5"] = ["C=CCC=C"]
    smiles_dict["DO6"] = ["C=CCCC=C"]
    smiles_dict["DO7"] = ["C=CCCCC=C"]
    smiles_dict["DO8"] = ["C=CCCCCC=C"]
    smiles_dict["DO9"] = ["C=CCCCCCC=C"]
    smiles_dict["DO10"] = ["C=CCCCCCCC=C"]
    smiles_dict["DO11"] = ["C=CCCCCCCCC=C"]
    smiles_dict["DO12"] = ["C=CCCCCCCCCC=C"]
    smiles_dict["IO4"] = ["C=C(C)C"]
    smiles_dict["IO5"] = ["C=CC(C)C"]
    smiles_dict["IO6"] = ["C=CCC(C)C"]
    smiles_dict["IO7"] = ["C=CCCC(C)C"]
    smiles_dict["IO8"] = ["C=CCCCC(C)C"]
    smiles_dict["IO9"] = ["C=CCCCCC(C)C"]
    smiles_dict["IO10"] = ["C=CCCCCCC(C)C"]
    smiles_dict["IO11"] = ["C=CCCCCCCC(C)C"]
    smiles_dict["IO12"] = ["C=CCCCCCCCC(C)C"]

    return smiles_dict


def single_value_lump(lump_name, dictionary, weight_dict):
    if len(dictionary[lump_name]) == 1:
        weight_dict[lump_name] = np.array([1.0])
        return True
    else:
        return False


def isoparaffins(lump_name, dictionary, weight_dict):
    weights = []

    len_molecule = dictionary[lump_name][0].count("C")
    alpha_sec = 0.3
    alpha_tert = 0.05
    alpha_eth = 0.05

    def symmetry(mol):
        l = mol.GetNumHeavyAtoms()
        diff = l - len(set(list(Chem.rdmolfiles.CanonicalRankAtoms(Chem.RemoveAllHs(mol), breakTies=False))))
        return diff

    for comp in dictionary[lump_name]:
        m = Chem.MolFromSmiles(comp)
        num_side_chains = comp.count("(")
        num_methyl = comp.count("(C)")
        num_ethyl = comp.count("(CC)")
        larger_alkyl = num_side_chains - (num_methyl + num_ethyl)
        quat = Chem.MolFromSmarts("[#6]([#6])([#6])([#6])[#6]")
        num_quat = len(m.GetSubstructMatches(quat))
        if larger_alkyl > 0:
            prefactor = 0
            # weights.append(0)
        elif comp == "CCCC(C)CC":
            prefactor = 2 + alpha_sec
            # weights.append(alpha_sec * (2 + alpha_sec))
        elif comp == "CCCCCC(C)C":
            prefactor = 3.9
            # weights.append(3.9 * alpha_sec)
        elif (len_molecule - (num_methyl + (2 * num_ethyl))) % 2 == 1 and symmetry(m) >= 2:
            prefactor = 1
        elif num_side_chains > 1 and symmetry(m) >= 2:
            prefactor = 1
        else:
            prefactor = 2
        weights.append(prefactor * (alpha_sec ** (num_methyl - num_quat)) *
                       (alpha_eth ** num_ethyl) * (alpha_tert ** num_quat))

    weights = np.asarray(weights).astype(np.float)
    weights = weights / np.sum(weights)
    weight_dict[lump_name] = weights
    return True


def olefins(lump_name, dictionary, weight_dict):
    weights = []
    len_molecule = dictionary[lump_name][0].count("C")
    num_alkenes = 0.5*(len_molecule - (len_molecule % 2)) - 1
    num_isoolefins = 0.0076 * np.exp(1.2742 * len_molecule)
    num_dienes = (len_molecule - (1 - len_molecule % 2) - 1) / 2 * (len_molecule - len_molecule % 2) / 2

    alpha_pattern = Chem.MolFromSmarts("[CH2]=[CH][CX4]")
    cumulated_diene_pattern = Chem.MolFromSmarts("[CX3]=[C]=[CX3]")
    conjugated_diene_pattern = Chem.MolFromSmarts("[CX3]=[CH][CH]=[CX3]")

    for comp in dictionary[lump_name]:
        num_side_chains = comp.count("(")
        double_bonds = comp.count("=")
        m = Chem.MolFromSmiles(comp)
        if double_bonds == 1:
            if num_side_chains == 0:
                if len(m.GetSubstructMatches(alpha_pattern)) > 0:
                    weights.append(16)
                else:
                    weights.append(2/num_alkenes)
            else:
                weights.append(num_side_chains / num_isoolefins)
        else:
            if num_side_chains == 0:
                if len(m.GetSubstructMatches(alpha_pattern)) == 2:
                    weights.append(5)
                elif len(m.GetSubstructMatches(cumulated_diene_pattern)) > 0:
                    weights.append(0)
                elif len(m.GetSubstructMatches(conjugated_diene_pattern)) > 0:
                    weights.append(4/num_dienes)
                else:
                    weights.append(1/num_dienes)
            else:
                weights.append(num_side_chains / num_isoolefins)
    weights = np.asarray(weights).astype(np.float)
    weights = weights / np.sum(weights)
    weight_dict[lump_name] = weights
    return True


def naphthenics(lump_name, dictionary, weight_dict):
    len_molecule = dictionary[lump_name][0].count("C")

    alpha_c5 = 0.9
    alpha_c6 = 0.75
    alpha_methyl = 0.8
    alpha_prim = 0.1

    alpha_quat = 0.1

    c5 = Chem.MolFromSmarts("C1CCCC1")
    c6 = Chem.MolFromSmarts("C1CCCCC1")
    monoalkylc5 = Chem.MolFromSmarts("[CH2]1[CH2][CH2][CH2][CH]1[CX4]")
    monoalkylc6 = Chem.MolFromSmarts("[CH2]1[CH2][CH2][CH2][CH2][CH]1[CX4]")

    bialkylc5_same = Chem.MolFromSmarts("[CH2]1[CH2][CH2][CH2][C]1([CX4])[CX4]")
    bialkylc5_12 = Chem.MolFromSmarts("[CX4][CH]1[CH2][CH2][CH2][CH]1[CX4]")
    bialkylc5_13 = Chem.MolFromSmarts("[CH2]1[CH]([CX4])[CH2][CH2][CH]1[CX4]")

    bialkylc6_same = Chem.MolFromSmarts("[CH2]1[CH2][CH2][CH2][CH2][C]1([CX4])[CX4]")
    bialkylc6_12 = Chem.MolFromSmarts("[CX4][CH]1[CH2][CH2][CH2][CH2][CH]1[CX4]")
    bialkylc6_13 = Chem.MolFromSmarts("[CH2]1[CH]([CX4])[CH2][CH2][CH2][CH]1[CX4]")
    bialkylc6_14 = Chem.MolFromSmarts("[CH2]1[CH2][CH]([CX4])[CH2][CH2][CH]1[CX4]")

    trialkyl = Chem.MolFromSmarts("[#6]@[#6](!@[#6])([#1])@[#6]")
    dialkyl = Chem.MolFromSmarts("[#6]@[#6]([#6])([#6])@[#6]")

    nalkyl = Chem.MolFromSmarts("[#6]-!@[#6]-!@[#6]")
    alkyl_tert = Chem.MolFromSmarts("[#6]!@[#6](!@[#6])!@[#6]")
    alkyl_quat = Chem.MolFromSmarts("[#6]!@[#6](!@[#6])(!@[#6])!@[#6]")
    quat = Chem.MolFromSmarts("[#6]([#6])([#6])([#6])[#6]")
    next_neighbor = Chem.MolFromSmarts("[#6](!@[#6])@[#6](!@[#6])@[#6]")

    weights = []
    for comp in dictionary[lump_name]:
        m = Chem.MolFromSmiles(comp)
        num_side_chains = comp.count("(")
        n_alkyl_tert = len(m.GetSubstructMatches(alkyl_tert))
        n_alkyl_quat = len(m.GetSubstructMatches(alkyl_quat))
        if comp.count("=") > 0:
            weights.append(0)
        elif len(m.GetSubstructMatches(c5)) > 0:
            if len(m.GetSubstructMatches(monoalkylc5)) > 0:
                if len_molecule == 6:
                    weights.append(alpha_c5)
                else:
                    weights.append(5 * alpha_c5 * (alpha_methyl ** (1 + n_alkyl_tert)) *
                                   (alpha_prim ** (len_molecule - 6)) * (alpha_quat ** n_alkyl_quat))
            else:
                if len(m.GetSubstructMatches(bialkylc5_same)) > 0:
                    weights.append(5 * alpha_c5 * (alpha_methyl ** (1 + n_alkyl_tert)) *
                                   (alpha_quat ** (1 + n_alkyl_quat)) * (alpha_prim ** (len_molecule - 7)))
                elif len(m.GetSubstructMatches(bialkylc5_12)) > 0:
                    weights.append(2 * alpha_c5 * (alpha_methyl ** (2 + n_alkyl_tert)) *
                                   (alpha_prim ** (len_molecule - 7)) * (alpha_quat ** n_alkyl_quat))
                elif len(m.GetSubstructMatches(bialkylc5_13)) > 0:
                    weights.append(4 * alpha_c5 * (alpha_methyl ** (2 + n_alkyl_tert)) *
                                   (alpha_prim ** (len_molecule - 7)) * (alpha_quat ** n_alkyl_quat))
                else:
                    n_quat = len(m.GetSubstructMatches(dialkyl))
                    n_tri = len(Chem.AddHs(m).GetSubstructMatches(trialkyl))
                    if len(m.GetSubstructMatches(next_neighbor)) > 0:
                        prefactor = 2
                    else:
                        prefactor = 10
                    weights.append(prefactor * alpha_c5 * (alpha_methyl ** (n_tri + n_quat + n_alkyl_tert)) *
                                   (alpha_quat ** (n_quat + n_alkyl_quat)) *
                                   (alpha_prim ** (len_molecule - (5 + n_tri + (2 * n_quat)))))
        elif len(m.GetSubstructMatches(c6)) > 0:
            if len(m.GetSubstructMatches(monoalkylc6)) > 0:
                weights.append(6 * alpha_c6 * (alpha_methyl ** (1 + n_alkyl_tert)) *
                               (alpha_prim ** (len_molecule - 7)) * (alpha_quat ** n_alkyl_quat))
            else:
                if len(m.GetSubstructMatches(bialkylc6_same)) > 0:
                    weights.append(6 * alpha_c6 * (alpha_methyl ** (1 + n_alkyl_tert)) *
                                   (alpha_quat ** (1 + n_alkyl_quat)) * (alpha_prim ** (len_molecule - 7)))
                elif len(m.GetSubstructMatches(bialkylc6_12)) > 0:
                    weights.append(2 * alpha_c6 * (alpha_methyl ** (2 + n_alkyl_tert)) *
                                   (alpha_prim ** (len_molecule - 8)) * (alpha_quat ** n_alkyl_quat))
                elif len(m.GetSubstructMatches(bialkylc6_13)) > 0:
                    weights.append(4 * alpha_c6 * (alpha_methyl ** (2 + n_alkyl_tert)) *
                                   (alpha_prim ** (len_molecule - 8)) * (alpha_quat ** n_alkyl_quat))
                elif len(m.GetSubstructMatches(bialkylc6_14)) > 0:
                    weights.append(2 * alpha_c6 * (alpha_methyl ** (2 + n_alkyl_tert)) *
                                   (alpha_prim ** (len_molecule - 8)) * (alpha_quat ** n_alkyl_quat))
                elif len_molecule == 6:
                    weights.append(alpha_c6)
                else:
                    n_quat = len(m.GetSubstructMatches(dialkyl))
                    n_tri = len(Chem.AddHs(m).GetSubstructMatches(trialkyl))
                    weights.append(alpha_c6 * (alpha_methyl ** (n_tri + n_quat + n_alkyl_tert)) *
                                   (alpha_quat ** (n_quat + n_alkyl_quat)) *
                                   (alpha_prim ** (len_molecule - (6 + n_tri + (2 * n_quat)))))
        else:
            weights.append(0)

    weights = np.asarray(weights).astype(np.float)
    weights = weights / np.sum(weights)
    weight_dict[lump_name] = weights
    return True


def aromatics(lump_name, dictionary, weight_dict):
    len_molecule = dictionary[lump_name][0].lower().count("c")
    weights = []
    meta = Chem.MolFromSmarts("*-!:aaa-!:*")
    ortho = Chem.MolFromSmarts("*-!:aa-!:*")
    para = Chem.MolFromSmarts("*-!:aaaa-!:*")
    mono = Chem.MolFromSmarts("[cH]1[cH][cH][cH][cH][c]1[CX4]")
    alpha_mono = np.sqrt(0.4)
    alpha_ortho = np.sqrt(0.5)
    alpha_para = np.sqrt(0.3)
    alpha_meta = np.sqrt(0.8)
    len_alkyl = len_molecule - 6
    for comp in dictionary[lump_name]:
        m = Chem.MolFromSmiles(comp)
        if len(m.GetSubstructMatches(meta)) > 1 and len(m.GetSubstructMatches(ortho)) == 0:
            weights.append((1/np.math.factorial(len_alkyl)) * alpha_meta ** len_alkyl)
        elif len(m.GetSubstructMatches(ortho)) > 1:
            weights.append((1/np.math.factorial(len_alkyl)) * alpha_ortho ** len_alkyl)
        elif len(m.GetSubstructMatches(meta)) > 0 and len(m.GetSubstructMatches(ortho)) > 0:
            weights.append((len_alkyl/np.math.factorial(len_alkyl)) * (alpha_meta ** 2) * alpha_ortho)
        elif len(m.GetSubstructMatches(meta)) > 0:
            if len_molecule > 8:
                weights.append((1/len_alkyl) * alpha_meta * alpha_mono)
            else:
                weights.append((1 / np.math.factorial(len_alkyl)) * alpha_meta ** len_alkyl)
        elif len(m.GetSubstructMatches(ortho)) > 0:
            if len_molecule > 8:
                weights.append((1/len_alkyl) * alpha_ortho * alpha_mono)
            else:
                weights.append((1/np.math.factorial(len_alkyl)) * alpha_ortho ** len_alkyl)
        elif len(m.GetSubstructMatches(para)) > 0:
            if len_molecule > 8:
                weights.append((1/len_alkyl) * alpha_para * alpha_mono)
            else:
                weights.append((1/np.math.factorial(len_alkyl)) * alpha_para ** len_alkyl)
        elif len(m.GetSubstructMatches(mono)) > 0:
            weights.append((1/np.math.factorial(len_alkyl)) * alpha_mono ** len_alkyl)
        else:
            weights.append(0)

    weights = np.asarray(weights).astype(np.float)
    weights = weights / np.sum(weights)
    weight_dict[lump_name] = weights
    return True


def make_composition(dataframe, lumps):
    smiles_dict = generate_smiles_dictionary(lumps, dataframe)
    weight_dict = {}
    for lump in lumps:
        check_value = single_value_lump(lump, smiles_dict, weight_dict)
        if not check_value:
            if "O" in lump:
                continue
                olefins(lump, smiles_dict, weight_dict)
            elif "N" in lump:
                naphthenics(lump, smiles_dict, weight_dict)
            elif "A" in lump:
                aromatics(lump, smiles_dict, weight_dict)
            elif "I" in lump:
                isoparaffins(lump, smiles_dict, weight_dict)
    return smiles_dict, weight_dict


# smiles_dict, weight_dict = make_composition(df, lumps)
# # np.savetxt("smiles_i8.txt", smiles_dict["I8"], fmt="%s")
# # np.savetxt("weights_i8.txt", weight_dict["I8"], fmt="%.6f")
# print(comp)
# print(smiles_dict["A7"])
# print(weight_dict["A7"])
# smiles_dict, weight_dict = clean_dicts(smiles_dict, weight_dict)
# all_fractions, all_molecules = absolute_fractions(comp, weight_dict, smiles_dict, lumps)
# all_molecules = np.asarray(all_molecules)
# np.savetxt("all_mols.txt", all_molecules, fmt="%s")
# np.savetxt("all_fracs.txt", all_fractions, fmt="%.6f", delimiter="\n")
