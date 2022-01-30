import numpy as np
from rdkit import Chem
import pandas as pd

#
# lumps = ["A6", "A7", "A8", "A8-1", "A8-2", "A9", "A10", "A11",
#          "N5", "N6", "N6-1", "N6-2", "N7", "N8", "N9", "N10", "N11",
#          "O5", "O6",
#          "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12",
#          "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12"]
#
# df = pd.read_excel("C:/Users/mrodobbe/Documents/Research/GauL-mixture/libraries.xlsx",
#                    sheet_name="Labeled Library",
#                    index_col=0)


def generate_smiles_dictionary(lumps, dataframe):
    smiles_dict = {}

    for lump in lumps:
        # smiles_dict[lump] = dataframe["SMILES"][dataframe.index[dataframe['Lump'] == lump].to_numpy()].tolist()
        smiles_dict[lump] = dataframe.index[dataframe['Lump'] == lump].tolist()

    smiles_dict["N6-1"] = ["CC1CCCC1"]
    smiles_dict["N6-2"] = ["C1CCCCC1"]
    smiles_dict["A8-1"] = ["Cc1cccc(C)c1", "Cc1ccc(C)cc1", "Cc1ccccc1C"]
    smiles_dict["A8-2"] = ["CCc1ccccc1"]
    # smiles_dict["NA10"] = ["c1ccc2cccc-2cc1"]

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
    two_methyl_pattern = Chem.MolFromSmarts("[CH3][CH]([CH3])[CX4]")
    two_two_dimethyl_pattern = Chem.MolFromSmarts("[CH3][C]([CH3])([CH3])[CX4]")
    x_x_dimethyl_pattern = Chem.MolFromSmarts("[CX4][CH2][C]([CH3])([CH3])[CH2][CX4]")
    other_methyl_pattern = Chem.MolFromSmarts("[CX4][CH2][CH]([CH3])[CH2][CX4]")
    ethyl_pattern = Chem.MolFromSmarts("[CX4][CX4][C]([CH2][CH3])[CX4][CX4]")
    same_ethyl_methyl_pattern = Chem.MolFromSmarts("[CX4][CX4][C]([CH2][CH3])([CX4])[CX4][CX4]")
    methyl_pattern = Chem.MolFromSmarts("[CX4][CH]([CH3])[CX4]")
    central_methyl_pattern = Chem.MolFromSmarts("[CH2][CH2][CH]([CH3])[CH2][CH2]")
    central_dimethyl_pattern = Chem.MolFromSmarts("[CX4][CH2][CH]([CH3])[CH]([CH3])[CH2][CX4]")

    if len_molecule > 6:
        if len_molecule == 7 or len_molecule == 8:
            num_ethyls = 1
        elif len_molecule == 9 or len_molecule == 10:
            num_ethyls = 2
        elif len_molecule == 11 or len_molecule == 12:
            num_ethyls = 3

        num_dimethyls = (num_ethyls + 1) * (num_ethyls + 2 - len_molecule % 2)

    elif len_molecule == 5:
        num_dimethyls = 1
    elif len_molecule == 6:
        num_dimethyls = 2

    num_other_methyls = 0.5*((len_molecule - 1) - ((len_molecule - 1) % 2)) - 1

    if len_molecule == 7:
        num_trimethyls = 1
    elif len_molecule == 8:
        num_trimethyls = 4
    elif len_molecule == 9:
        num_trimethyls = 8
    elif len_molecule == 10:
        num_trimethyls = 16
    elif len_molecule == 11:
        num_trimethyls = 25
    elif len_molecule == 12:
        num_trimethyls = 40

    for comp in dictionary[lump_name]:
        num_side_chains = comp.count("(")
        m = Chem.MolFromSmiles(comp)
        if num_side_chains == 1:
            if len(m.GetSubstructMatches(two_methyl_pattern)) > 0:
                weights.append(25)
            elif len(m.GetSubstructMatches(ethyl_pattern)) > 0:
                weights.append(1.8/num_ethyls)
            elif len(m.GetSubstructMatches(central_methyl_pattern)) > 0:
                weights.append(12/num_other_methyls)
            elif len(m.GetSubstructMatches(other_methyl_pattern)) > 0:
                weights.append(24/num_other_methyls)
            else:
                weights.append(0.1/num_other_methyls)
        elif num_side_chains == 2:
            if len(m.GetSubstructMatches(two_two_dimethyl_pattern)) > 0:
                weights.append(1/num_dimethyls)
            elif len(m.GetSubstructMatches(x_x_dimethyl_pattern)) > 0:
                weights.append(2/num_dimethyls)
            elif len(m.GetSubstructMatches(central_dimethyl_pattern)) > 0:
                weights.append(14/num_dimethyls)
            elif len(m.GetSubstructMatches(methyl_pattern)) == 2:
                weights.append(12/num_dimethyls)
            elif len(m.GetSubstructMatches(same_ethyl_methyl_pattern)) > 0:
                weights.append(1/num_dimethyls)
            elif len(m.GetSubstructMatches(ethyl_pattern)) > 0:
                weights.append(4/num_dimethyls)
            else:
                weights.append(0.1/num_dimethyls)
        elif num_side_chains == 3:
            if len(m.GetSubstructMatches(two_two_dimethyl_pattern)) > 0:
                weights.append(0.75/num_trimethyls)
            elif len(m.GetSubstructMatches(x_x_dimethyl_pattern)) > 0:
                weights.append(1/num_trimethyls)
            elif len(m.GetSubstructMatches(methyl_pattern)) == 3:
                weights.append(3/num_trimethyls)
            else:
                weights.append(0.5 / num_trimethyls)
        elif num_side_chains == 4:
            weights.append(0.2/num_trimethyls)
        else:
            weights.append(0.1/num_trimethyls)
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

    trialkyl = Chem.MolFromSmarts("[#6]@[#6]([#6])@[#6]")
    dialkyl = Chem.MolFromSmarts("[#6]@[#6]([#6])([#6])@[#6]")

    nalkyl = Chem.MolFromSmarts("[#6]-!@[#6]-!@[#6]")

    weights = []
    for comp in dictionary[lump_name]:
        m = Chem.MolFromSmiles(comp)
        num_side_chains = comp.count("(")
        if len(m.GetSubstructMatches(c5)) > 0:
            if comp == "CC1CCCC1":
                weights.append(0.8)
            elif comp.count("=") > 0:
                weights.append(0)
            elif comp == "C1CCCC1":
                weights.append(0.8)
            elif len(m.GetSubstructMatches(monoalkylc5)) > 0:
                weights.append(2 / (((num_side_chains + 1) ** 3) * ((m.GetNumHeavyAtoms() - num_side_chains - 5) ** 3)))
            elif len(m.GetSubstructMatches(bialkylc5_same)) > 0:
                weights.append(0.1 / (len(m.GetSubstructMatches(nalkyl)) + 1))
            elif len(m.GetSubstructMatches(bialkylc5_12)) > 0:
                weights.append(0.4 / (len(m.GetSubstructMatches(nalkyl)) + 1))
            elif len(m.GetSubstructMatches(bialkylc5_13)) > 0:
                weights.append(0.2 / (len(m.GetSubstructMatches(nalkyl)) + 1))
            elif len(m.GetSubstructMatches(trialkyl)) == 3 and len(m.GetSubstructMatches(dialkyl)) == 0:
                weights.append(0.25 / (len(m.GetSubstructMatches(nalkyl)) + 1))
            elif len(m.GetSubstructMatches(trialkyl)) == 3 and len(m.GetSubstructMatches(dialkyl)) > 0:
                weights.append(0.25 / (len(m.GetSubstructMatches(nalkyl))))
            else:
                weights.append(0)
        elif len(m.GetSubstructMatches(c6)) > 0:
            if comp == "C1CCCC1":
                weights.append(1)
            elif comp.count("=") > 0:
                weights.append(0)
            elif len(m.GetSubstructMatches(monoalkylc6)) > 0:
                weights.append(3 / (((num_side_chains + 1) ** 3) * ((m.GetNumHeavyAtoms() - num_side_chains - 6) ** 3)))
            elif len(m.GetSubstructMatches(bialkylc6_same)) > 0:
                weights.append(0.1 / (len(m.GetSubstructMatches(nalkyl))))
            elif len(m.GetSubstructMatches(bialkylc6_12)) > 0:
                weights.append(0.4 / (len(m.GetSubstructMatches(nalkyl)) + 1))
            elif len(m.GetSubstructMatches(bialkylc6_13)) > 0:
                weights.append(0.8 / (len(m.GetSubstructMatches(nalkyl)) + 1))
            elif len(m.GetSubstructMatches(bialkylc6_14)) > 0:
                weights.append(0.3 / (len(m.GetSubstructMatches(nalkyl)) + 1))
            elif len(m.GetSubstructMatches(trialkyl)) == 3 and len(m.GetSubstructMatches(dialkyl)) == 0:
                weights.append(0.25 / (len(m.GetSubstructMatches(nalkyl)) + 1))
            elif len(m.GetSubstructMatches(trialkyl)) == 3 and len(m.GetSubstructMatches(dialkyl)) > 0:
                weights.append(0.25 / (len(m.GetSubstructMatches(nalkyl))))
            else:
                weights.append(0)
        else:
            weights.append(0)

    weights = np.asarray(weights).astype(np.float)
    weights = weights / np.sum(weights)
    weight_dict[lump_name] = weights
    return True


def aromatics(lump_name, dictionary, weight_dict):
    len_molecule = dictionary[lump_name][0].count("C")
    weights = []
    meta = Chem.MolFromSmarts("*-!:aaa-!:*")
    ortho = Chem.MolFromSmarts("*-!:aa-!:*")
    para = Chem.MolFromSmarts("*-!:aaaa-!:*")
    mono = Chem.MolFromSmarts("[cH]1[cH][cH][cH][cH][c]1[CX4]")
    for comp in dictionary[lump_name]:
        num_side_chains = comp.count("(")
        m = Chem.MolFromSmiles(comp)
        if len(m.GetSubstructMatches(meta)) > 2:
            if len_molecule > 9:
                weights.append(1 / ((num_side_chains + 1) ** 3))
            else:
                weights.append(1)
        elif len(m.GetSubstructMatches(ortho)) > 2:
            if len_molecule > 9:
                weights.append(0.1 / ((num_side_chains + 1) ** 3))
            else:
                weights.append(0.1)
        elif len(m.GetSubstructMatches(meta)) > 0 and len(m.GetSubstructMatches(ortho)) > 0:
            if len_molecule > 9:
                weights.append(0.3 / ((num_side_chains + 1) ** 3))
            else:
                weights.append(0.3)
        elif len(m.GetSubstructMatches(meta)) > 0 and len(m.GetSubstructMatches(para)) > 0:
            if len_molecule > 9:
                weights.append(0.3 / ((num_side_chains + 1) ** 3))
            else:
                weights.append(0.3)
        elif len(m.GetSubstructMatches(ortho)) > 0 and len(m.GetSubstructMatches(para)) > 0:
            if len_molecule > 9:
                weights.append(0.15 / ((num_side_chains + 1) ** 3))
            else:
                weights.append(0.15)
        elif len(m.GetSubstructMatches(meta)) > 0:
            if len_molecule > 8:
                weights.append(11 / ((num_side_chains + 1) ** 3))
            else:
                weights.append(11)
        elif len(m.GetSubstructMatches(ortho)) > 0:
            if len_molecule > 8:
                weights.append(6 / ((num_side_chains + 1) ** 3))
            else:
                weights.append(6)
        elif len(m.GetSubstructMatches(para)) > 0:
            if len_molecule > 8:
                weights.append(6.2 / (num_side_chains ** 3))
            else:
                weights.append(6.2)
        elif len(m.GetSubstructMatches(mono)) > 0:
            if len_molecule > 8:
                weights.append(15 / (num_side_chains ** 3))
            else:
                weights.append(15)
        else:
            weights.append(0.01 / ((num_side_chains + 1) ** 3))

    weights = np.asarray(weights).astype(np.float)
    weights = weights / np.sum(weights)
    weight_dict[lump_name] = weights
    return True


def make_composition(dataframe, lumps):
    smiles_dict = generate_smiles_dictionary(lumps, dataframe)
    weight_dict = {}
    for lump in lumps:
        print(lump)
        check_value = single_value_lump(lump, smiles_dict, weight_dict)
        if not check_value:
            if "O" in lump:
                olefins(lump, smiles_dict, weight_dict)
            elif "N" in lump:
                naphthenics(lump, smiles_dict, weight_dict)
            elif "A" in lump:
                print(lump)
                aromatics(lump, smiles_dict, weight_dict)
            elif "I" in lump:
                isoparaffins(lump, smiles_dict, weight_dict)
    return smiles_dict, weight_dict


# smiles_dict, weight_dict = make_composition(df, lumps)
# np.savetxt("smiles_n8.txt", smiles_dict["N8"], fmt="%s")
# np.savetxt("weights_n8.txt", weight_dict["N8"], fmt="%.6f")
