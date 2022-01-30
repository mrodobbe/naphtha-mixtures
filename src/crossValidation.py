from sklearn.model_selection import KFold
from joblib import wrap_non_picklable_objects, Parallel, delayed, cpu_count
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.python.client import device_lib
import os
import tensorflow as tf
from src.model import *
import numpy as np


def cv(compositions, piona, output, loop, i, save_folder, indices, bp=None):
    train = loop[0]
    test = loop[1]

    compositions_t = compositions[train]
    piona_t = piona[train]
    output_t = output[train]

    compositions_test = compositions[test]
    piona_test = piona[test]
    output_test = output[test]
    indices_test = indices[test]

    if bp is not None:
        bp_t = bp[train]
        bp_test = bp[test]
        property_unit = "kg/dm3"
    else:
        property_unit = "K"

    n_folds = 9  # TODO: Make argument
    kf = KFold(n_folds, shuffle=True, random_state=120897)

    rmse_ann = []
    mae_ann = []
    models = []
    turn = 0

    results_list = [["Molecule", "Real Value", "Prediction", "Deviation", "Error"]]
    predicted_bp = [[]]
    predicted_sg = [[]]

    for train2, val2 in kf.split(compositions_t):
        turn += 1

        compositions_train = compositions_t[train2]
        piona_train = piona_t[train2]
        output_train = output_t[train2]

        compositions_v = compositions_t[val2]
        piona_v = piona_t[val2]
        output_v = output_t[val2]

        if bp is not None:
            bp_train = bp_t[train2]
            bp_v = bp_t[val2]

        # output_size = len(output_t[0])

        if bp is not None:
            model = ann_sg(compositions_train, piona_train, bp_train, output_train)
        else:
            model = ann_bp(compositions_train, piona_train, output_train)
        model.summary()

        print('{} training samples, {} validation samples'.format(len(compositions_train),
                                                                  len(compositions_v)))

        es = EarlyStopping(patience=100, restore_best_weights=True, min_delta=0.01, mode='min')  # TODO: patience=1000

        class LossHistory(Callback):
            def on_epoch_end(self, epoch, logs={}):
                if epoch % 10 == 0:
                    print('Epoch {}: {:.3f}\t\t-\t{:.3f}'.format(epoch, logs.get('val_mean_absolute_error'),
                                                                 np.sqrt(logs.get('val_loss'))))

        lh = LossHistory()
        print('Validation MAE\t-\tValidation MSE')
        if bp is not None:
            history = model.fit([compositions_train, piona_train, bp_train], output_train, epochs=10000,
                                validation_data=([compositions_v, piona_v, bp_v], output_v),
                                batch_size=8, callbacks=[es, lh], verbose=0)
        else:
            history = model.fit([compositions_train, piona_train], output_train, epochs=10000,
                                validation_data=([compositions_v, piona_v], output_v),
                                batch_size=8, callbacks=[es, lh], verbose=0)

        if bp is not None:
            if len(output_test.shape) > 1:
                test_predictions = np.asarray(model.predict([compositions_test, piona_test, bp_test]))
            else:
                test_predictions = model.predict([compositions_test, piona_test, bp_test]).reshape(-1)
        else:
            if len(output_test.shape) > 1:
                test_predictions = np.asarray(model.predict([compositions_test, piona_test]))
            else:
                test_predictions = model.predict([compositions_test, piona_test]).reshape(-1)

        models.append(model)

        test_error = np.abs(test_predictions - output_test)
        test_mean_absolute_error = np.average(test_error)
        mae_ann.append(test_mean_absolute_error)
        test_root_mean_squared_error = np.sqrt(np.average(test_error ** 2))
        rmse_ann.append(test_root_mean_squared_error)

        print('Test performance statistics for ANN:')
        print('Mean absolute error:\t\t{:.2f} {}'.format(test_mean_absolute_error, property_unit))
        print('Root mean squared error:\t{:.2f} {}'.format(test_root_mean_squared_error, property_unit))

        with open(str(save_folder + "/test_results_fold_{}_{}.txt".format(i, turn)), "w") as f:
            f.write('ANN Test performance statistics:\n')
            f.write('Mean absolute error:\t\t{:.2f} {}\n'.format(test_mean_absolute_error, property_unit))
            f.write('Root mean squared error:\t{:.2f} {}\n\n'.format(test_root_mean_squared_error, property_unit))
            f.close()

    ann_index = np.argmin(rmse_ann)

    test_mae = mae_ann[ann_index]
    test_rmse = rmse_ann[ann_index]
    best_model = models[ann_index]
    best_model.save(str(save_folder + "/Fold {}".format(i)))

    print('Test performance statistics for ANN:')
    print('Mean absolute error:\t\t{:.2f} {}'.format(test_mae, property_unit))
    print('Root mean squared error:\t{:.2f} {}'.format(test_rmse, property_unit))

    ensemble = np.array([])
    for model in models:
        if len(output_test.shape) > 1:
            if bp is None:
                test_predicted = model.predict([compositions_test, piona_test])
            else:
                test_predicted = model.predict([compositions_test, piona_test, bp_test])
        else:
            test_predicted = model.predict([compositions_test, piona_test, bp_test]).reshape(-1)
        test_predicted = np.asarray(test_predicted).astype(np.float)
        prediction_shape = test_predicted.shape
        if len(ensemble) == 0:
            ensemble = test_predicted.flatten()
        else:
            ensemble = np.vstack((ensemble, test_predicted.flatten()))
    ensemble_prediction = np.average(ensemble, axis=0)
    ensemble_prediction = np.reshape(ensemble_prediction, prediction_shape)
    ensemble_sd = np.std(ensemble, axis=0)
    ensemble_sd = np.reshape(ensemble_sd, prediction_shape)
    ensemble_error = np.abs(ensemble_prediction - output_test)
    ensemble_mae = np.average(ensemble_error)
    ensemble_rmse = np.sqrt(np.average(ensemble_error ** 2))

    with open(str(save_folder + "/Fold {}/test_results_fold_{}.txt".format(i, i)), "w") as f:
        f.write('ANN Test performance statistics:\n')
        f.write('Mean absolute error:\t\t{:.2f} {}\n'.format(test_mae, property_unit))
        f.write('Root mean squared error:\t{:.2f} {}\n\n'.format(test_rmse, property_unit))
        f.write('ANN Ensemble Test performance statistics:\n')
        f.write('Mean absolute error:\t\t{:.2f} {}\n'.format(ensemble_mae, property_unit))
        f.write('Root mean squared error:\t{:.2f} {}\n\n'.format(ensemble_rmse, property_unit))
        f.close()

    with open(str(save_folder + "/Fold {}/test_ensemble_predictions_{}.txt".format(i, i)), "w") as f:
        f.write(str("Real Value \t Prediction \t Deviation \t Error"
                    "\t Real Value \t Prediction \t Deviation \t Error"
                    "\t Real Value \t Prediction \t Deviation \t Error\n"))
        if len(output_test.shape) > 1:
            for v, p, s, e, i in zip(output_test, ensemble_prediction, ensemble_sd, ensemble_error, indices_test):
                if bp is None:
                    append_vector = [int(i)]
                    for bp_value in p:
                        append_vector.append(bp_value)
                    append_vector = np.asarray(append_vector)
                    predicted_bp.append(append_vector)
                    # predicted_bp.append(np.array([int(i), round(p[0], 6), round(p[1], 6), round(p[2], 6)]))
                else:
                    append_vector = [int(i)]
                    if len(p) > 1:
                        for sg_value in p:
                            append_vector.append(sg_value)
                        append_vector = np.asarray(append_vector)
                    else:
                        append_vector.append(p)
                        append_vector = np.asarray(append_vector)
                    predicted_sg.append(append_vector)
                if len(v) == 2:
                    results_list.append([round(v[0], 6), round(p[0], 6), round(s[0], 6), round(e[0], 6),
                                         round(v[1], 6), round(p[1], 6), round(s[1], 6), round(e[1], 6)])
                    f.write(str(round(v[0], 4)) + '\t' + str(round(p[0], 4)) + '\t' +
                            str(round(s[0], 4)) + '\t' + str(round(e[0], 4)) + '\t' +
                            str(round(v[1], 4)) + '\t' + str(round(p[1], 4)) + '\t' +
                            str(round(s[1], 4)) + '\t' + str(round(e[1], 4)) + '\n')
                else:
                    results_list.append([round(v[0], 6), round(p[0], 6), round(s[0], 6), round(e[0], 6),
                                         round(v[1], 6), round(p[1], 6), round(s[1], 6), round(e[1], 6),
                                         round(v[2], 6), round(p[2], 6), round(s[2], 6), round(e[2], 6)])
                    f.write(str(round(v[0], 4)) + '\t' + str(round(p[0], 4)) + '\t' +
                            str(round(s[0], 4)) + '\t' + str(round(e[0], 4)) + '\t' +
                            str(round(v[1], 4)) + '\t' + str(round(p[1], 4)) + '\t' +
                            str(round(s[1], 4)) + '\t' + str(round(e[1], 4)) + '\t' +
                            str(round(v[2], 4)) + '\t' + str(round(p[2], 4)) + '\t' +
                            str(round(s[2], 4)) + '\t' + str(round(e[2], 4)) + '\n')
        else:
            for v, p, s, e, i in zip(output_test, ensemble_prediction, ensemble_sd, ensemble_error, indices_test):
                predicted_sg.append(np.array([int(i), p]))
                results_list.append(np.array([round(v, 6), round(p, 6), round(s, 6), round(e, 6)]).astype(np.float))
                f.write(str(round(v, 4)) + '\t' + str(round(p, 4)) + '\t' + str(round(s, 4)) +
                        '\t' + str(round(e, 4)) + '\n')

    if bp is None:
        predicted_bp.pop(0)
        predicted_bp = np.asarray(predicted_bp).astype(np.float)
        # print(predicted_bp)
        predicted_bp = predicted_bp[predicted_bp[:, 0].argsort()]
        # print("This works")
        # print(len((results_list, predicted_bp)))
        return results_list, predicted_bp
    else:
        predicted_sg.pop(0)
        predicted_sg = np.asarray(predicted_sg).astype(np.float)
        return results_list, predicted_sg


def cv_configurations():
    seed = 120897
    n_folds = 10
    kf = KFold(n_folds, shuffle=True, random_state=seed)
    return kf, n_folds


def num_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])


def training(condensed_representations, mixture_features, outputs, save_folder, bp=None):
    kf, n_folds = cv_configurations()
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
            cv_info = Parallel(n_jobs=n_jobs)(delayed(cv)(condensed_representations,
                                                          mixture_features,
                                                          outputs,
                                                          loop_kf,
                                                          i,
                                                          save_folder,
                                                          np.arange(len(condensed_representations)), bp=bp)
                                              for loop_kf, i in zip(kf.split(condensed_representations),
                                                                    range(1, n_folds + 1)))
        else:
            cv_info = Parallel(n_jobs=n_jobs)(delayed(cv)(condensed_representations,
                                                          mixture_features,
                                                          outputs,
                                                          loop_kf,
                                                          i,
                                                          save_folder,
                                                          np.arange(len(condensed_representations)), bp=bp)
                                              for loop_kf, i in zip(kf.split(condensed_representations),
                                                                    range(1, n_folds + 1)))
    else:
        if gpu == 1:
            gpus = tf.config.list_physical_devices('GPU')
            tf.config.set_visible_devices(gpus[0], 'GPU')
            cv_info = [cv(condensed_representations, mixture_features, outputs, loop_kf, i, save_folder,
                       np.arange(len(condensed_representations)), bp=bp)
                       for loop_kf, i in zip(kf.split(condensed_representations), range(1, n_folds + 1))]
        else:
            cv_info = [cv(condensed_representations, mixture_features, outputs, loop_kf, i, save_folder,
                       np.arange(len(condensed_representations)), bp=bp)
                       for loop_kf, i in zip(kf.split(condensed_representations), range(1, n_folds + 1))]

    return cv_info
