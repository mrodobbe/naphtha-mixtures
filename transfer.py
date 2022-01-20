from src.makeMolecule import input_checker
from src.featurization import get_gmm, clean_dicts, make_mixture_features, representation_checker
from src.property_prediction import predict_properties
from src.postprocessing import sg_transfer_plot, bp_transfer_plot
import sys
from input import load_transfer_data
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import KFold
import numpy as np

input_checker(sys.argv, "naphtha_mixtures")
save_folder = sys.argv[1]
save_folder_bp = save_folder + "/bp"
save_folder_sg = save_folder + "/sg"

compositions, boiling_points, output_sg, smiles_dict, weight_dict, df, lumps = load_transfer_data()
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

n_folds = len(models_bp)
seed = 120897
kf = KFold(n_folds, shuffle=True, random_state=seed)
indices = np.arange(len(condensed_representations))

predicted_bp = []

for i, kf_loop in zip(range(len(models_bp)), kf.split(condensed_representations)):
    model = models_bp[i]
    train = kf_loop[0]
    test = kf_loop[1]

    compositions_t = condensed_representations[train]
    features_t = mixture_features[train]
    output_t = boiling_points[train]

    training_size = int((7/9) * len(compositions_t))

    s = np.arange(len(compositions_t))
    np.random.seed(210995)
    np.random.shuffle(s)
    s_t = s[:training_size]
    s_vt = s[training_size:]

    train2 = np.asarray(range(len(compositions_t)))[s_t]
    val2 = np.asarray(range(len(compositions_t)))[s_vt]

    compositions_train = compositions_t[train2]
    features_train = features_t[train2]
    output_train = output_t[train2]

    compositions_v = compositions_t[val2]
    features_v = features_t[val2]
    output_v = output_t[val2]

    compositions_test = condensed_representations[test]
    features_test = mixture_features[test]
    output_test = boiling_points[test]
    indices_test = indices[test]

    model.summary()

    print('{} training samples, {} validation samples'.format(len(compositions_train),
                                                              len(compositions_v)))

    es = EarlyStopping(patience=100, restore_best_weights=True, min_delta=0.01, mode='min')


    class LossHistory(Callback):
        def on_epoch_end(self, epoch, logs={}):
            if epoch % 10 == 0:
                print('Epoch {}: {:.3f}\t\t-\t{:.3f}'.format(epoch, logs.get('val_mean_absolute_error'),
                                                             np.sqrt(logs.get('val_loss'))))


    lh = LossHistory()
    print('Validation MAE\t-\tValidation MSE')
    history = model.fit([compositions_train, features_train], output_train, epochs=2000,
                        validation_data=([compositions_v, features_v], output_v),
                        batch_size=8, callbacks=[es, lh], verbose=0)

    test_predictions = np.asarray(model.predict([compositions_test, features_test]))
    test_error = np.abs(test_predictions - output_test)
    test_mean_absolute_error = np.average(test_error)
    test_root_mean_squared_error = np.sqrt(np.average(test_error ** 2))

    print('Test performance statistics for ANN:')
    print('Mean absolute error:\t\t{:.2f} {}'.format(test_mean_absolute_error, property))
    print('Root mean squared error:\t{:.2f} {}'.format(test_root_mean_squared_error, property))

    print(indices_test)
    for p, j in zip(test_predictions, indices_test):
        predicted_bp.append(np.array([int(j), round(p[0], 6), round(p[1], 6), round(p[2], 6)]))

    model.save(str(save_folder_bp + "/Fold {}".format(i+1)))

predicted_bp = np.concatenate(predicted_bp, axis=0)
predicted_bp = predicted_bp.reshape((int(len(predicted_bp)/4), 4))
predicted_bp = predicted_bp[predicted_bp[:, 0].argsort()][:, 1:]
predicted_bp = np.asarray(predicted_bp).astype(np.float)

bp_transfer_plot(boiling_points, predicted_bp, save_folder_bp)

models_sg = [load_model(str(save_folder_sg + '/' + folder)) for folder in folders]
predicted_sg = []

for i, kf_loop in zip(range(len(models_sg)), kf.split(condensed_representations)):
    model = models_sg[i]
    train = kf_loop[0]
    test = kf_loop[1]

    compositions_t = condensed_representations[train]
    features_t = mixture_features[train]
    bp_t = boiling_points[train]
    output_t = output_sg[train]

    training_size = int((7/9) * len(compositions_t))

    s = np.arange(len(compositions_t))
    np.random.seed(210995)
    np.random.shuffle(s)
    s_t = s[:training_size]
    s_vt = s[training_size:]

    train2 = np.asarray(range(len(compositions_t)))[s_t]
    val2 = np.asarray(range(len(compositions_t)))[s_vt]

    compositions_train = compositions_t[train2]
    features_train = features_t[train2]
    output_train = output_t[train2]
    bp_train = bp_t[train2]

    compositions_v = compositions_t[val2]
    features_v = features_t[val2]
    output_v = output_t[val2]
    bp_v = bp_t[val2]

    compositions_test = condensed_representations[test]
    features_test = mixture_features[test]
    output_test = output_sg[test]
    bp_test = boiling_points[test]
    indices_test = indices[test]

    model.summary()

    print('{} training samples, {} validation samples'.format(len(compositions_train),
                                                              len(compositions_v)))

    es = EarlyStopping(patience=100, restore_best_weights=True, min_delta=0.01, mode='min')


    class LossHistory(Callback):
        def on_epoch_end(self, epoch, logs={}):
            if epoch % 10 == 0:
                print('Epoch {}: {:.3f}\t\t-\t{:.3f}'.format(epoch, logs.get('val_mean_absolute_error'),
                                                             np.sqrt(logs.get('val_loss'))))


    lh = LossHistory()
    print('Validation MAE\t-\tValidation MSE')
    history = model.fit([compositions_train, features_train, bp_train], output_train, epochs=2000,
                        validation_data=([compositions_v, features_v, bp_v], output_v),
                        batch_size=4, callbacks=[es, lh], verbose=0)

    test_predictions = model.predict([compositions_test, features_test, bp_test]).reshape(-1)
    test_error = np.abs(test_predictions - output_test)
    test_mean_absolute_error = np.average(test_error)
    test_root_mean_squared_error = np.sqrt(np.average(test_error ** 2))

    print('Test performance statistics for ANN:')
    print('Mean absolute error:\t\t{:.2f} {}'.format(test_mean_absolute_error, property))
    print('Root mean squared error:\t{:.2f} {}'.format(test_root_mean_squared_error, property))

    print(indices_test)
    for p, j in zip(test_predictions, indices_test):
        predicted_sg.append(np.array([int(j), p]))

    model.save(str(save_folder_sg + "/Fold {}".format(i+1)))

predicted_sg = np.concatenate(predicted_sg, axis=0)
predicted_sg = predicted_sg.reshape((int(len(predicted_sg)/2), 2))
predicted_sg = predicted_sg[predicted_sg[:, 0].argsort()][:, 1]
predicted_sg = np.asarray(predicted_sg).astype(np.float)

sg_transfer_plot(output_sg, predicted_sg, save_folder_sg)
