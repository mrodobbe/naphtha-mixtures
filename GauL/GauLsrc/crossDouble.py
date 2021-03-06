import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras import Model
from tensorflow.python.client import device_lib
import tensorflow as tf
import pickle
from GauL.GauLsrc.makeModel import model_builder
from GauL.GauLsrc.makeMolecule import heavy_atoms
from GauL.GauLsrc.makeMolecule import denormalize
from sklearn.svm import SVR
from joblib import wrap_non_picklable_objects, Parallel, delayed, cpu_count
from sklearn.model_selection import KFold


def make_heavy_atoms(molecules):
    ha = []
    for mol in molecules:
        ha.append(heavy_atoms(mol))
    ha = np.asarray(ha).astype(np.float)
    return ha


@wrap_non_picklable_objects
def run_cv(all_molecules, all_heavy, x, y, loop, i, save_folder, target):
    train = loop[0]
    test = loop[1]
    all_molecules = np.asarray(all_molecules)
    training_molecules = all_molecules[train]
    test_molecules = all_molecules[test]

    heavy_train = all_heavy[train]
    heavy_test = all_heavy[test]

    x_train_all = x[train]
    y_train_all = y[train]

    x_test = x[test]
    y_test = y[test]

    n = len(y_train_all)
    n_folds = 9  # TODO: Make argument
    kf = KFold(n_folds, shuffle=True, random_state=12041997)

    rmse_ann = []
    rmse_svr = []
    mae_ann = []
    mae_svr = []
    models = []
    turn = 0

    results_list = [["Molecule", "Real Value", "Prediction", "Deviation", "Error"]]

    for train2, val2 in kf.split(x_train_all):
        turn += 1
        x_train2 = x_train_all[train2]
        y_train2 = y_train_all[train2]
        x_val2 = x_train_all[val2]
        y_val2 = y_train_all[val2]

        if len(y.shape) == 2:
            output_layer_size = y.shape[1]
        else:
            output_layer_size = 1

        model = model_builder(x_train_all, output_layer_size, target)
        model.summary()

        print('{} training molecules, {} validation molecules'.format(len(x_train2),
                                                                      len(x_val2)))

        es = EarlyStopping(patience=150, restore_best_weights=True, min_delta=0.01, mode='min')

        class LossHistory(Callback):
            def on_epoch_end(self, batch, logs={}):
                print('{:.3f}\t\t-\t{:.3f}'.format(logs.get('val_mean_absolute_error'), np.sqrt(logs.get('val_loss'))))

        lh = LossHistory()

        print('Validation MAE\t-\tValidation MSE')
        history = model.fit(x_train2, y_train2, epochs=1000,
                            validation_data=(x_val2, y_val2),
                            batch_size=8, callbacks=[es, lh], verbose=0)

        if target != "cp":
            validation_predictions = model.predict(x_val2).reshape(-1)
        else:
            validation_predictions = np.asarray(model.predict(x_val2)).astype(np.float)
        if target != "cp":
            test_predictions = model.predict(x_test).reshape(-1)
        else:
            test_predictions = np.asarray(model.predict(x_test)).astype(np.float)

        intermediate_layer = Model(inputs=model.input, outputs=model.get_layer('layer_3').output)
        training_intermediates = np.asarray(intermediate_layer(x_train_all)).astype(np.float)
        test_intermediates = np.asarray(intermediate_layer(x_test)).astype(np.float)

        models.append(model)

        if target != "cp":
            krr = SVR(kernel="rbf", gamma='scale', C=2.5e3)  # This is the support vector machine.
            # Try to find an algorithm that optimizes gamma and C. You can also add an epsilon factor
            krr.fit(training_intermediates, y_train_all)  # Execute regression

            y_svr = krr.predict(test_intermediates)  # Prediction for the test set (unseen data)
            if target == "s":
                y_svr = denormalize(y_svr, heavy_test, target, coefficient=1.5)
                y_test_svr = denormalize(y_test, heavy_test, target, coefficient=1.5)
            else:
                y_test_svr = y_test
            svr_error = np.abs(y_svr - y_test_svr)
            svr_mean_absolute_error = np.average(svr_error)
            mae_svr.append(svr_mean_absolute_error)
            svr_root_mean_squared_error = np.sqrt(np.average(svr_error ** 2))
            rmse_svr.append(svr_root_mean_squared_error)

            print('Test performance statistics for SVR:')
            print('Mean absolute error:\t\t{:.2f} kJ/mol'.format(svr_mean_absolute_error))
            print('Root mean squared error:\t{:.2f} kJ/mol'.format(svr_root_mean_squared_error))

        test_predictions = denormalize(test_predictions, heavy_test, target)
        y_test_normalized = denormalize(y_test, heavy_test, target)

        test_error = np.abs(test_predictions - y_test_normalized)
        test_mean_absolute_error = np.average(test_error)
        mae_ann.append(test_mean_absolute_error)
        test_root_mean_squared_error = np.sqrt(np.average(test_error ** 2))
        rmse_ann.append(test_root_mean_squared_error)

        print('Test performance statistics for ANN:')
        print('Mean absolute error:\t\t{:.2f} kJ/mol'.format(test_mean_absolute_error))
        print('Root mean squared error:\t{:.2f} kJ/mol'.format(test_root_mean_squared_error))

        with open(str(save_folder + "/test_results_fold_{}_{}.txt".format(i, turn)), "w") as f:
            f.write('ANN Test performance statistics:\n')
            f.write('Mean absolute error:\t\t{:.2f} kJ/mol\n'.format(test_mean_absolute_error))
            f.write('Root mean squared error:\t{:.2f} kJ/mol\n\n'.format(test_root_mean_squared_error))
            if target != "cp":
                f.write('SVR Test performance statistics:\n')
                f.write('Mean absolute error:\t\t{:.2f} kJ/mol\n'.format(svr_mean_absolute_error))
                f.write('Root mean squared error:\t{:.2f} kJ/mol\n'.format(svr_root_mean_squared_error))
            f.close()

    ann_index = np.argmin(rmse_ann)

    if target != "cp":
        svr_index = np.argmin(rmse_svr)

    test_mean_absolute_error = mae_ann[ann_index]
    test_root_mean_squared_error = rmse_ann[ann_index]
    if target != "cp":
        svr_mean_absolute_error = mae_svr[svr_index]
        svr_root_mean_squared_error = rmse_svr[svr_index]
    best_model = models[ann_index]
    best_model.save(str(save_folder + "/Fold {}".format(i)))

    print('Test performance statistics for ANN:')
    print('Mean absolute error:\t\t{:.2f} kJ/mol'.format(test_mean_absolute_error))
    print('Root mean squared error:\t{:.2f} kJ/mol'.format(test_root_mean_squared_error))

    if target != "cp":
        ensemble = np.array([])
        for model in models:
            test_predicted = model.predict(x_test).reshape(-1)
            test_predicted = np.asarray(test_predicted).astype(np.float)
            if target == "s":
                test_predicted = denormalize(test_predicted, heavy_test, target, coefficient=1.5)
            if len(ensemble) == 0:
                ensemble = test_predicted
            else:
                ensemble = np.vstack((ensemble, test_predicted))
        ensemble_prediction = np.mean(ensemble, axis=0)
        ensemble_sd = np.std(ensemble, axis=0)
        if target == "s":
            y_test_nor = denormalize(y_test, heavy_test, target, coefficient=1.5)
        else:
            y_test_nor = y_test
        ensemble_error = np.abs(ensemble_prediction - y_test_nor)
        ensemble_mae = np.average(ensemble_error)
        ensemble_rmse = np.sqrt(np.average(ensemble_error ** 2))
    else:
        ensemble = np.array([])
        for model in models:
            test_predicted = model.predict(x_test)
            test_predicted = np.asarray(test_predicted).astype(np.float)
            prediction_shape = test_predicted.shape
            test_predicted = denormalize(test_predicted, heavy_test, target, coefficient=1.5)
            if len(ensemble) == 0:
                ensemble = test_predicted.flatten()
            else:
                ensemble = np.vstack((ensemble, test_predicted.flatten()))
        ensemble_prediction = np.average(ensemble, axis=0)
        ensemble_prediction = np.reshape(ensemble_prediction, prediction_shape)
        ensemble_sd = np.std(ensemble, axis=0)
        ensemble_sd = np.reshape(ensemble_sd, prediction_shape)
        y_test_nor = denormalize(y_test, heavy_test, target, coefficient=1.5)
        ensemble_error = np.abs(ensemble_prediction - y_test_nor)
        ensemble_mae = np.average(ensemble_error)
        ensemble_rmse = np.sqrt(np.average(ensemble_error ** 2))

    with open(str(save_folder + "/Fold {}/test_results_fold_{}.txt".format(i, i)), "w") as f:
        f.write('ANN Test performance statistics:\n')
        f.write('Mean absolute error:\t\t{:.2f} kJ/mol\n'.format(test_mean_absolute_error))
        f.write('Root mean squared error:\t{:.2f} kJ/mol\n\n'.format(test_root_mean_squared_error))
        if target != "cp":
            f.write('SVR Test performance statistics:\n')
            f.write('Mean absolute error:\t\t{:.2f} kJ/mol\n'.format(svr_mean_absolute_error))
            f.write('Root mean squared error:\t{:.2f} kJ/mol\n\n'.format(svr_root_mean_squared_error))
        f.write('ANN Ensemble Test performance statistics:\n')
        f.write('Mean absolute error:\t\t{:.2f} kJ/mol\n'.format(ensemble_mae))
        f.write('Root mean squared error:\t{:.2f} kJ/mol\n\n'.format(ensemble_rmse))
        f.close()

    if target != "cp":
        best_predictions = best_model.predict(x_test).reshape(-1)
    else:
        best_predictions = np.asarray(best_model.predict(x_test)).astype(np.float)

    if target != "h":
        best_predictions = denormalize(best_predictions, heavy_test, target, coefficient=1.5)
        y_best_normalized = denormalize(y_test, heavy_test, target, coefficient=1.5)
    else:
        y_best_normalized = y_test

    best_error = np.abs(best_predictions - y_best_normalized)

    with open(str(save_folder + "/Fold {}/test_predictions_{}.txt".format(i, i)), "w") as f:
        f.write(str("Molecule \t Real Value \t Prediction \t Absolute Error \n"))
        for m, v, p, e in zip(test_molecules, y_test, best_predictions, best_error):
            if target == "cp":
                f.write(str(m) + '\t' + str(round(v[0], 4)) + '\t' + str(round(p[0], 4)) + '\t' +
                        str(round(e[0], 4)) + '\n')
            else:
                f.write(str(m) + '\t' + str(round(v, 4)) + '\t' + str(round(p, 4)) + '\t' + str(round(e, 4)) + '\n')
        f.close()

    with open(str(save_folder + "/Fold {}/test_representations_fold_{}.pickle".format(i, i)), "wb") as f:
        pickle.dump(x_test, f)
    print("Dumped the test molecules!")

    with open(str(save_folder + "/Fold {}/test_outputs_fold_{}.pickle".format(i, i)), "wb") as f:
        pickle.dump(y_test, f)
    print("Dumped the test outputs!")

    with open(str(save_folder + "/Fold {}/test_ensemble_predictions_{}.txt".format(i, i)), "w") as f:
        f.write(str("Molecule \t Real Value \t Prediction \t Deviation \t Error \n"))
        for m, v, p, s, e in zip(test_molecules, y_test_nor, ensemble_prediction, ensemble_sd, ensemble_error):
            if target == "cp":
                results_list.append([m, round(v[0], 2), round(p[0], 2), round(s[0], 2), round(e[0], 2)])
                f.write(str(m) + '\t' + str(round(v[0], 4)) + '\t' + str(round(p[0], 4)) + '\t' +
                        str(round(s[0], 4)) + '\t' + str(round(e[0], 4)) + '\n')
            else:
                results_list.append([m, round(v, 2), round(p, 2), round(s, 2), round(e, 2)])
                f.write(str(m) + '\t' + str(round(v, 4)) + '\t' + str(round(p, 4)) + '\t' + str(round(s, 4)) +
                        '\t' + str(round(e, 4)) + '\n')
        f.close()

    # performance_plot(y_test, test_predictions, "test", prop=target, folder=str(save_folder + "/Fold {}".format(i)),
    #                  model="ANN", fold=i)
    if target != "cp":
        return test_mean_absolute_error, test_root_mean_squared_error, ensemble_mae, ensemble_rmse, \
               results_list, svr_mean_absolute_error, svr_root_mean_squared_error
    else:
        return test_mean_absolute_error, test_root_mean_squared_error, ensemble_mae, ensemble_rmse, results_list


def num_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])


def training(molecules, representations, outputs, save_folder, target_property, n_folds):
    kf = KFold(n_folds, shuffle=True, random_state=12081997)
    heavy_atoms_vector = make_heavy_atoms(molecules)
    cpu = cpu_count()
    gpu = num_available_gpus()
    # cpu = 4
    if gpu > 0:
        if cpu > 10:
            n_jobs = 10
        else:
            n_jobs = 1
    else:
        if n_folds > cpu:
            if cpu < 3:
                n_jobs = 1
            else:
                n_jobs = cpu - 2
        else:
            n_jobs = n_folds

    print("Your system has {} GPUs, {} CPUs and {} fold(s) will be trained in parallel".format(gpu, cpu, n_jobs))
    if n_jobs > 1:
        if gpu > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            cv_info = Parallel(n_jobs=n_jobs)(delayed(run_cv)(molecules, heavy_atoms_vector, representations, outputs,
                                                              loop_kf, i, save_folder, target_property)
                                              for loop_kf, i in zip(kf.split(representations), range(1, n_folds+1)))
        else:
            cv_info = Parallel(n_jobs=n_jobs)(delayed(run_cv)(molecules, heavy_atoms_vector, representations, outputs,
                                                              loop_kf, i, save_folder, target_property)
                                              for loop_kf, i in zip(kf.split(representations), range(1, n_folds+1)))
    else:
        if gpu == 1:
            gpus = tf.config.list_physical_devices('GPU')
            tf.config.set_visible_devices(gpus[0], 'GPU')
            cv_info = [run_cv(molecules, heavy_atoms_vector, representations, outputs,
                                                              loop_kf, i, save_folder, target_property)
                                              for loop_kf, i in zip(kf.split(representations), range(1, n_folds+1))]
        else:
            cv_info = [run_cv(molecules, heavy_atoms_vector, representations, outputs,
                                                              loop_kf, i, save_folder, target_property)
                                              for loop_kf, i in zip(kf.split(representations), range(1, n_folds+1))]

    return cv_info


def write_statistics(cv_info, target_property, n_folds, save_folder):
    prediction_mae = []
    prediction_rmse = []
    prediction_svr_mae = []
    prediction_svr_rmse = []
    results_list = [["Molecule", "Real Value", "Prediction", "Deviation", "Error"]]

    test_models = []
    test_svr = []
    for j in range(n_folds):
        prediction_mae.append(cv_info[j][0])
        prediction_rmse.append(cv_info[j][1])
        individual_results = cv_info[j][4]
        individual_results.pop(0)
        for c in individual_results:
            results_list.append(c)
        if target_property != "cp":
            prediction_svr_mae.append(cv_info[j][2])
            prediction_svr_rmse.append(cv_info[j][3])

    best_index = np.argmin(prediction_rmse)
    if target_property != "cp":
        best_index_svr = np.argmin(prediction_svr_rmse)
        with open(str(save_folder + "/best_models.txt"), "w") as f:
            f.write("The best ANN model is from fold {}, "
                    "while the best SVR model is from fold {}\n".format(best_index + 1, best_index_svr + 1))
            f.close()
    else:
        with open(str(save_folder + "/best_models.txt"), "w") as f:
            f.write("The best ANN model is from fold {}.".format(best_index + 1))
            f.close()

    with open(str(save_folder + "/test_statistics.txt"), "w") as f:
        f.write('Test performance statistics for ANN:\n')
        f.write(
            'Mean absolute error:\t\t{:.2f} +/- {} kJ/mol\n'.format(np.mean(prediction_mae), np.std(prediction_mae)))
        f.write('Root mean squared error:\t{:.2f} +/- {} kJ/mol\n\n'.format(np.mean(prediction_rmse),
                                                                            np.std(prediction_rmse)))
        for i, mae_value, rmse_value in zip(range(len(prediction_mae)), prediction_mae, prediction_rmse):
            f.write('Fold {} - MAE: {} kJ/mol\t\t-\t\tRMSE: {} kJ/mol\n'.format(i + 1, mae_value, rmse_value))
        if target_property != "cp":
            f.write('\nTest performance statistics for SVR:\n')
            f.write('Mean absolute error:\t\t{:.2f} +/- {} kJ/mol\n'.format(np.mean(prediction_svr_mae),
                                                                            np.std(prediction_svr_mae)))
            f.write('Root mean squared error:\t{:.2f} +/- {} kJ/mol\n\n'.format(np.mean(prediction_svr_rmse),
                                                                                np.std(prediction_svr_rmse)))
            for i, mae_value, rmse_value in zip(range(len(prediction_svr_mae)), prediction_svr_mae,
                                                prediction_svr_rmse):
                f.write('Fold {} - MAE: {} kJ/mol\t\t-\t\tRMSE: {} kJ/mol\n'.format(i + 1, mae_value, rmse_value))
        # f.write('\nTime elapsed: {} seconds\n'.format(time_elapsed))
        f.close()

    # print("Finished! This took {} ".format(seconds_to_text(time_elapsed)))
    return results_list


def seconds_to_text(secs):
    days = round(secs//86400)
    hours = round((secs - days*86400)//3600)
    minutes = round((secs - days*86400 - hours*3600)//60)
    seconds = round(secs - days*86400 - hours*3600 - minutes*60)
    result = ("{} days, ".format(days) if days else "") + \
             ("{} hours, ".format(hours) if hours else "") + \
             ("{} minutes, ".format(minutes) if minutes else "") + \
             ("{} seconds".format(seconds) if seconds else "")
    return result
