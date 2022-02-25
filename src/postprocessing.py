from src.plots import performance_plot
import matplotlib.pyplot as plt
import numpy as np


def boiling_point_plot(results_list_bp, n_folds, save_folder):
    all_results = [["Molecule", "Real Value", "Prediction", "Deviation", "Error"]]
    for j in range(n_folds):
        individual_results = results_list_bp[j]
        individual_results.pop(0)
        for c in individual_results:
            all_results.append(c)

    all_predictions = np.asarray([float(line[1]) for line in all_results[1:]]).astype(np.float)
    all_predictions_50 = np.asarray([float(line[5]) for line in all_results[1:]]).astype(np.float)
    all_predictions_f = np.asarray([float(line[9]) for line in all_results[1:]]).astype(np.float)

    np.savetxt(str(save_folder + "/all_preds.mrd"), all_results[1:])
    real = np.asarray([float(line[0]) for line in all_results[1:]]).astype(np.float)
    real_50 = np.asarray([float(line[4]) for line in all_results[1:]]).astype(np.float)
    real_f = np.asarray([float(line[8]) for line in all_results[1:]]).astype(np.float)

    font = {'family': 'UGent Panno Text',
            'weight': 'normal',
            'size': 36}
    plt.rc('font', **font)
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10)
    hfont = {'fontname': 'UGent Panno Text'}

    ax.scatter(real, all_predictions, s=180, c='#88ace0', label="Initial Boiling Points", alpha=0.5)
    ax.scatter(real_50, all_predictions_50, s=180, c='#FFA44A', label="50$\%$ Boiling Points", marker="^", alpha=0.5)
    ax.scatter(real_f, all_predictions_f, s=180, c='#0F4C81', label="Final Boiling Points", marker="d", alpha=0.5)
    ax.plot([min(real), max(real_f)], [min(real), max(real_f)], c='#9B1B30', linewidth=3.5, label="Parity")
    ax.set_xlabel("Experimental Boiling Points [$\degree$C]", **hfont)
    ax.set_ylabel("Predicted Boiling Points [$\degree$C]", **hfont)
    ax.legend(edgecolor="#c4c4c4")
    plt.savefig(str(save_folder + "/boiling_points.png"))


def mixture_property_plot(results_list_sg, n_folds, property_name, save_folder):
    all_results = [["Molecule", "Real Value", "Prediction", "Deviation", "Error"]]
    for j in range(n_folds):
        individual_results = results_list_sg[j]
        individual_results.pop(0)
        for c in individual_results:
            all_results.append(c)
    all_predictions_i = np.asarray([float(line[1]) for line in all_results[1:]]).astype(np.float)
    np.savetxt(str(save_folder + "/all_preds.mrd"), all_results[1:])
    real = np.asarray([float(line[0]) for line in all_results[1:]]).astype(np.float)

    performance_plot(real, all_predictions_i, "test", property_name, folder=save_folder, fold="all", model="ANN")


def bp_transfer_plot(real, predicted, save_folder):
    real_i = real[:, 0]
    real_50 = real[:, 1]
    real_f = real[:, 2]

    pred_i = predicted[:, 0]
    pred_50 = predicted[:, 1]
    pred_f = predicted[:, 2]

    font = {'family': 'UGent Panno Text',
            'weight': 'normal',
            'size': 36}
    plt.rc('font', **font)
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10)
    hfont = {'fontname': 'UGent Panno Text'}

    ax.scatter(real_i, pred_i, s=180, c='#88ace0', label="Initial Boiling Points", alpha=0.5)
    ax.scatter(real_50, pred_50, s=180, c='#FFA44A', label="50$\%$ Boiling Points", marker="^", alpha=0.5)
    ax.scatter(real_f, pred_f, s=180, c='#0F4C81', label="Final Boiling Points", marker="d", alpha=0.5)
    ax.plot([min(real_i), max(real_f)], [min(real_i), max(real_f)], c='#9B1B30', linewidth=3.5, label="Parity")
    ax.set_xlabel("Experimental Boiling Points [$\degree$C]", **hfont)
    ax.set_ylabel("Predicted Boiling Points [$\degree$C]", **hfont)
    ax.legend(edgecolor="#c4c4c4")
    plt.savefig(str(save_folder + "/transfer_boiling_points.png"))


def sg_transfer_plot(real, predicted, save_folder):
    performance_plot(real, predicted, "transfer", "SG", folder=save_folder, fold="all", model="ANN")


def process_results(data, prop, n_folds, save_folder):
    prediction = []
    results = []
    for result in data:
        prediction.append(np.asarray(result[1], dtype=object))
        results.append(result[0])
    prediction = np.concatenate(prediction, axis=0)
    np.savetxt(str(save_folder + "/predictions.mrd"), prediction, fmt="%s")
    predicted_values = prediction[prediction[:, 0].argsort()][:, 1:]
    predicted_values = np.asarray(predicted_values).astype(np.float)
    if prop is "bp":
        boiling_point_plot(results, n_folds, save_folder)
        np.savetxt(str(save_folder + "/predicted_bps.txt"), predicted_values)
        return predicted_values
    else:
        mixture_property_plot(results, n_folds, prop, save_folder)
