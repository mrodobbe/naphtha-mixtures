from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LeakyReLU, Concatenate
from tensorflow.keras.initializers import glorot_uniform, RandomUniform
import tensorflow as tf
tf.random.set_seed(210995)


def ann_sg(input_molecules, input_features, input_bp, output_sg):
    init1 = glorot_uniform(seed=120897)
    init2 = glorot_uniform(seed=120897)
    init4 = glorot_uniform(seed=120897)
    init5 = glorot_uniform(seed=120897)
    inito = RandomUniform(minval=-0.05, maxval=0.05, seed=1)

    layer1_size = 200
    layer1b_size = 100
    layer1bp_size = 50
    layer2a_size = 30
    layer2b_size = 70
    layer4_size = 200
    layer5_size = 200

    if len(output_sg.shape) > 1:
        output_size = len(output_sg[0])
    else:
        output_size = 1

    # 3 input layers: molecular mixture representation, mixture features, boiling points

    input_1_layer = Input(shape=(len(input_molecules[0]),), name='input_molecules')
    input_2_layer = Input(shape=(len(input_features[0]),), name='input_features')
    input_3_layer = Input(shape=(len(input_bp[0]),), name='input_bp')

    # 2 hidden layers for the molecular mixture representation

    hidden_layer1_molecules = Dense(layer1_size,
                                    kernel_initializer=init1,
                                    bias_initializer=init1,
                                    name='layer_1_molecules')(input_1_layer)
    hidden_layer1_molecules_b = LeakyReLU()(hidden_layer1_molecules)

    hidden_layer2_molecules = Dense(layer1_size,
                                    kernel_initializer=init1,
                                    bias_initializer=init1,
                                    name='layer_2_molecules')(hidden_layer1_molecules_b)
    hidden_layer2_molecules_b = LeakyReLU()(hidden_layer2_molecules)

    # 2 hidden layers for the mixture features

    hidden_layer1_features = Dense(layer1b_size,
                                   kernel_initializer=init1,
                                   bias_initializer=init2,
                                   name='layer_1_features')(input_2_layer)
    hidden_layer1_features_b = LeakyReLU()(hidden_layer1_features)

    hidden_layer2_features = Dense(layer2b_size,
                                   kernel_initializer=init1,
                                   bias_initializer=init2,
                                   name='layer_2_features')(hidden_layer1_features_b)
    hidden_layer2_features_b = LeakyReLU()(hidden_layer2_features)

    # Concatenate second hidden mixture representation and first hidden feature representation

    hidden_layer2_concatenated = Concatenate()([hidden_layer2_molecules_b, hidden_layer1_features_b])

    # Third hidden molecular representation

    hidden_layer2_molecules_c = Dense(layer2a_size,
                                      kernel_initializer=init1,
                                      bias_initializer=init1,
                                      name='layer_2c_molecules')(hidden_layer2_concatenated)
    hidden_layer2_molecules_d = LeakyReLU()(hidden_layer2_molecules_c)

    # Two hidden layers for the boiling points

    hidden_layer1_bp = Dense(layer1bp_size,
                             kernel_initializer=init1,
                             bias_initializer=init2,
                             name='layer_1_bp')(input_3_layer)
    hidden_layer1_bp_b = LeakyReLU()(hidden_layer1_bp)

    hidden_layer2_bp = Dense(layer2b_size,
                             kernel_initializer=init1,
                             bias_initializer=init2,
                             name='layer_2_bp')(hidden_layer1_bp_b)
    hidden_layer2_bp_b = LeakyReLU()(hidden_layer2_bp)

    # Concatenate the last hidden layers of the three inputs

    hidden_layer3 = Concatenate()([hidden_layer2_molecules_d, hidden_layer2_features_b, hidden_layer2_bp_b])

    # Two hidden layers

    hidden_layer4 = Dense(layer4_size,
                          kernel_initializer=init1,
                          bias_initializer=init4,
                          name='layer_4')(hidden_layer3)
    hidden_layer4b = LeakyReLU()(hidden_layer4)

    hidden_layer5 = Dense(layer5_size,
                          kernel_initializer=init1,
                          bias_initializer=init5,
                          name='layer_5')(hidden_layer4b)

    output_layer = Dense(output_size, activation='linear', kernel_initializer=inito,
                         bias_initializer=inito, name='output')(hidden_layer5)

    model = Model(inputs=[input_1_layer, input_2_layer, input_3_layer], outputs=output_layer)
    # opt = Adam(learning_rate=0.01)

    model.compile(loss='mean_squared_error',
                  optimizer="Adam",
                  metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])
    return model


def ann_bp(input_molecules, input_features, output_comp):
    init1 = glorot_uniform(seed=120897)
    init2 = glorot_uniform(seed=120897)
    init4 = glorot_uniform(seed=120897)
    init5 = glorot_uniform(seed=120897)
    inito = RandomUniform(minval=-0.05, maxval=0.05, seed=1)

    layer1_size = 512
    layer1b_size = 256
    layer2a_size = 128
    layer2b_size = 32
    layer4_size = 512
    layer5_size = 512

    if len(output_comp.shape) > 1:
        output_size = len(output_comp[0])
    else:
        output_size = 1

    # Two input layers: molecular mixture representations and mixture features

    input_1_layer = Input(shape=(len(input_molecules[0]),), name='input_molecules')
    input_2_layer = Input(shape=(len(input_features[0]),), name='input_features')

    # Three hidden layers for molecular mixture representations

    hidden_layer1_molecules = Dense(layer1_size,
                                    kernel_initializer=init1,
                                    bias_initializer=init1,
                                    name='layer_1_molecules')(input_1_layer)
    hidden_layer1_molecules_b = LeakyReLU()(hidden_layer1_molecules)

    hidden_layer2_molecules = Dense(layer1_size,
                                    kernel_initializer=init1,
                                    bias_initializer=init1,
                                    name='layer_2_molecules')(hidden_layer1_molecules_b)
    hidden_layer2_molecules_b = LeakyReLU()(hidden_layer2_molecules)

    hidden_layer2_molecules_c = Dense(layer2a_size,
                                      kernel_initializer=init1,
                                      bias_initializer=init1,
                                      name='layer_2c_molecules')(hidden_layer2_molecules_b)
    hidden_layer2_molecules_d = LeakyReLU()(hidden_layer2_molecules_c)

    # Two hidden layers for mixture features

    hidden_layer1_features = Dense(layer1b_size,
                                   kernel_initializer=init1,
                                   bias_initializer=init2,
                                   name='layer_1_features')(input_2_layer)
    hidden_layer1_features_b = LeakyReLU()(hidden_layer1_features)

    hidden_layer2_features = Dense(layer2b_size,
                                   kernel_initializer=init1,
                                   bias_initializer=init2,
                                   name='layer_2_features')(hidden_layer1_features_b)
    hidden_layer2_features_b = LeakyReLU()(hidden_layer2_features)

    # Concatenate the end hidden layer of both input networks

    hidden_layer3 = Concatenate()([hidden_layer2_molecules_d, hidden_layer2_features_b])

    # Two hidden layers for final network learning

    hidden_layer4 = Dense(layer4_size,
                          kernel_initializer=init1,
                          bias_initializer=init4,
                          name='layer_4')(hidden_layer3)
    hidden_layer4b = LeakyReLU()(hidden_layer4)

    hidden_layer5 = Dense(layer5_size,
                          kernel_initializer=init1,
                          bias_initializer=init5,
                          name='layer_5')(hidden_layer4b)

    # Output layer

    output_layer = Dense(output_size, activation='linear', kernel_initializer=inito,
                         bias_initializer=inito, name='output')(hidden_layer5)

    model = Model(inputs=[input_1_layer, input_2_layer], outputs=output_layer)
    # opt = Adam(learning_rate=0.01)

    model.compile(loss='mean_squared_error',
                  optimizer="Adam",
                  metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    return model
