from src.stochastic_compositions import make_composition
import pickle


def load_data(property_to_train):
    with open("Data/labeled_library.pickle", "rb") as lib_f:
        df = pickle.load(lib_f)

    with open("Data/pyl_input.pickle", "rb") as inp_f:
        df_input = pickle.load(inp_f)

    with open("Data/pyl_input.pickle", "rb") as oup_f:
        df_output = pickle.load(oup_f)

    lumps = df_input.keys().tolist()
    all_lumps = list(df["Lump"].unique())
    selected_lumps = list(set(lumps) | set(all_lumps))

    smiles_dict, weight_dict = make_composition(df, selected_lumps)  # TODO: Selected_lumps!

    compositions = df_input.to_numpy()

    header = df_output.columns
    bp_columns = [i for i in header if "BP" in i]

    if property_to_train.lower() is "sg":
        sg_columns = [i for i in header if ("sg" in i.lower() or "specific gravity" in i.lower())]
    elif property_to_train.lower() is "mu" or property_to_train.lower() is "viscosity" \
            or property_to_train.lower() is "dynamic viscosity":
        sg_columns = [i for i in header if ("mu" in i.lower() or "dynamic viscosity" in
                                            i.lower() or "viscosity" in i.lower())]
    elif property_to_train.lower() is "kinematic viscosity":
        sg_columns = [i for i in header if "kinematic viscosity" in i.lower()]
    elif property_to_train.lower() is "density" or property_to_train.lower() is "d20":
        sg_columns = [i for i in header if ("d20" in i.lower() or "density" in i.lower())]
    elif property_to_train.lower() is "surface tension" or property_to_train.lower() is "st":
        sg_columns = [i for i in header if ("st" in i.lower() or "surface tension" in i.lower())]
    else:
        print("\n\nPrediction of mixture {} values is currently not yet supported. The property can be added in "
              "input.py or another property name can be used.\n\n".format(property_to_train))
        raise ValueError
    boiling_points = df_output[bp_columns].to_numpy()
    output_sg = df_output[sg_columns].to_numpy().reshape(-1) * 1000

    print("All data is loaded!")

    return compositions, boiling_points, output_sg, smiles_dict, weight_dict, df, lumps


def load_test_data():
    with open("Data/labeled_library.pickle", "rb") as lib_f:
        df = pickle.load(lib_f)

    with open("Data/pyl_input.pickle", "rb") as inp_f:
        df_input = pickle.load(inp_f)

    lumps = df_input.keys().tolist()

    smiles_dict, weight_dict = make_composition(df, lumps)  # TODO: Selected_lumps!

    compositions = df_input.to_numpy()

    return compositions, smiles_dict, weight_dict, df, lumps
