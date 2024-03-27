
'''
Model definitions for the network pruning project
'''
import sys

try: 
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow_model_optimization as tfmot
    from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude
except ImportError:
    print("Please install tensorflow, numpy, and pandas")
    sys.exit(1)

def PhaseNet(inputsize):
    model = keras.Sequential([
        layers.Conv1D(kernel_size=10, filters=64, activation='tanh', name='conv1', input_shape=(inputsize,1)),
        layers.Dropout(0.2, name='dropout1'),
        layers.Conv1D(kernel_size=1, filters=64, activation='relu', name='conv2'),
        layers.Dropout(0.2, name='dropout2'),
        layers.Conv1D(kernel_size=2, filters=64, activation='relu', padding='same', name='conv3'),
        layers.Dropout(0.2, name='dropout3'),
        layers.Flatten(name='flat1'),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dense(1, activation='linear', name='output')
    ],
    name='PhaseNet')
    return model

def PhaseNetPrunable(inputsize, pruning_params_2_by_4, pruning_params_sparsity_0_5):
    model = keras.Sequential([
        prune_low_magnitude(
            layers.Conv1D(kernel_size=10, filters=64, activation='tanh', name='conv1', input_shape=(inputsize,1)),
            **pruning_params_sparsity_0_5),
        layers.Dropout(0.2, name='dropout1'),
        prune_low_magnitude(
            layers.Conv1D(kernel_size=1, filters=64, activation='relu', name='conv2'),
            **pruning_params_sparsity_0_5),
        layers.Dropout(0.2, name='dropout2'),
        prune_low_magnitude(
            layers.Conv1D(kernel_size=2, filters=64, activation='relu', padding='same', name='conv3'),
            **pruning_params_2_by_4),
        layers.Dropout(0.2, name='dropout3'),
        layers.Flatten(name='flat1'),
        prune_low_magnitude(
            layers.Dense(128, activation='relu', name='dense1'),
            **pruning_params_2_by_4),
        prune_low_magnitude(
            layers.Dense(1, activation='linear', name='output'),
            **pruning_params_sparsity_0_5)
    ], name='PhaseNetPrunable')

    return model