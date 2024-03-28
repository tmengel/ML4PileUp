# imports
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split

import uproot
import awkward as ak
import numpy as np
import pickle

import sys
import os
import json
import argparse


class NetworkTuner:
     
    def __init__(self, config_file, model_type):
        '''
        Initialize the NetworkTuner class
        '''

        self.config_file = config_file
        self.TYPE = model_type
        
        # confirm config file exists
        if not os.path.exists(self.config_file):
            print("Config file does not exist")
            sys.exit()

        # zero out all variables
        self.VALID_TYPES = []
        self.AUGMENTATION = False
        self.TRACELENGTH = 0
        self.INPUTSIZE = 0
        self.VALIDATION_SPLIT = 0
        self.TRAIN_PERCENT = 0
        self.EPOCHS_SEARCH = 0
        self.EPOCHS_FINAL = 0
        self.BATCH_SIZE = 0
        self.DATAFILE = ""
        self.TREENAME = ""
        self.OUTPUT_DIR = ""
        self.TYPE = ""
        self.LOSS = ""
        self.DIRECTORY = ""
        self.PROJECT_NAME = ""
        self.OUTPUT_ACVTIVATIONS = []
        self.OBJECTIVE = ""
        self.TUNED_MODEL = ""
        self.EARLY_STOPPING_VAR = ""
        self.EARLY_STOPPING_MODE = ""

        # parse config file
        self.ParseConfigFile()
        self.PrintInfo()

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.GetModelData()

        self.tuner = None
        self.stop_early = None

        self.InitTuner()

        self.best_hps = None
        self.history = None
      
    def ParseConfigFile(self):
        '''
        Parse the config file
        '''
        # read config file
        config = {}
        with open(self.config_file) as f:
            config = json.load(f)

        # Read valid model types
        self.VALID_TYPES = config["MODEL_TYPES"]
        if self.TYPE not in self.VALID_TYPES:
            print("Invalid model type")
            sys.exit()

        # Read constants
        self.AUGMENTATION = config["AUGMENTATION"]
        self.TRACELENGTH = config["TRACELENGTH"]
        self.INPUTSIZE = config["INPUTSIZE"]
        self.VALIDATION_SPLIT = config["VALIDATION_SPLIT"]
        self.TRAIN_PERCENT = config["TRAIN_PERCENT"]
        self.EPOCHS_SEARCH = config["EPOCHS_SEARCH"]
        self.EPOCHS_FINAL = config["EPOCHS_FINAL"]
        self.BATCH_SIZE = config["BATCH_SIZE"]

        self.DATAFILE = config["DATAFILE"]
        self.TREENAME = config["TREENAME"]
        # confirm data file exists
        if not os.path.exists(self.DATAFILE):
            print("Data file does not exist")
            sys.exit()

        self.OUTPUT_DIR = config["OUTPUT_DIR"]
        # confirm output directory exists
        if not os.path.exists(self.OUTPUT_DIR):
            print("Output directory does not exist")
            sys.exit()

        # remove trailing slash if it exists
        if self.OUTPUT_DIR[-1] == "/":
            self.OUTPUT_DIR = self.OUTPUT_DIR[:-1]

        # Read model parameters
        self.TYPE = self.TYPE
        self.LOSS = config[self.TYPE]["LOSS"]

        # get tuning directory
        self.DIRECTORY = f'{self.OUTPUT_DIR}/{self.TYPE}'
        # remove directory if it exists
        if os.path.exists(self.DIRECTORY):
            os.system(f'rm -r {self.DIRECTORY}')
        # create directory
        os.system(f'mkdir -p {self.DIRECTORY}')

        self.PROJECT_NAME = config[self.TYPE]["PROJECT_NAME"]
        self.OUTPUT_ACVTIVATIONS = config[self.TYPE]["OUTPUT_ACVTIVATIONS"]
        self.OBJECTIVE = config[self.TYPE]["OBJECTIVE"]
        self.TUNED_MODEL = config[self.TYPE]["TUNED_MODEL"]
        self.EARLY_STOPPING_VAR = config[self.TYPE]["EARLY_STOPPING_VAR"]
        self.EARLY_STOPPING_MODE = config[self.TYPE]["EARLY_STOPPING_MODE"]
        

        return

    def PrintInfo(self):
        '''
        Print out the info
        '''
        print("====================================")
        print("Constants\n")
        print("AUGMENTATION: ", self.AUGMENTATION)
        print("TRACELENGTH: ", self.TRACELENGTH)
        print("INPUTSIZE: ", self.INPUTSIZE)
        print("VALIDATION_SPLIT: ", self.VALIDATION_SPLIT)
        print("TRAIN_PERCENT: ", self.TRAIN_PERCENT)
        print("DATAFILE: ", self.DATAFILE)
        print("EPOCHS_SEARCH: ", self.EPOCHS_SEARCH)
        print("EPOCHS_FINAL: ", self.EPOCHS_FINAL)
        print("BATCH_SIZE: ", self.BATCH_SIZE)
        print("====================================")
        print("Model specific constants\n")
        print("TYPE: ", self.TYPE)
        print("LOSS: ", self.LOSS)
        print("OUTPUT_ACVTIVATIONS: ", self.OUTPUT_ACVTIVATIONS)
        print("OBJECTIVE: ", self.OBJECTIVE)
        print("DIRECTORY: ", self.DIRECTORY)
        print("PROJECT_NAME: ", self.PROJECT_NAME)
        print("TUNED_MODEL: ", self.TUNED_MODEL)
        print("EARLY_STOPPING_VAR: ", self.EARLY_STOPPING_VAR)
        print("EARLY_STOPPING_MODE: ", self.EARLY_STOPPING_MODE)
        print("====================================")

        return

    ## Static methods
    @staticmethod
    def GetData(filename, train_frac=0.5, treename="OutputTree"):
        '''
        Get the data
        '''

        file = uproot.open(filename)
        tree = file[treename]
        data = {}
        branches=['pileup', 'amp', 'phase', 'trace'],
        for branch in branches:
            data[branch] = ak.to_numpy(tree[branch].arrays()[branch])
        
        train_size = int(len(data["trace"])*train_frac)
        test_size = len(data["trace"]) - train_size
        print("Train size: ",train_size)
        print("Test size: ",test_size)

        x = np.array(data["trace"])
        y_pileup = np.array(data["pileup"])
        y_amp = np.array(data["amp"])
        y_phase = np.array(data["phase"])

        x_train, x_test, y_pileup_train, y_pileup_test, y_amp_train, y_amp_test, y_phase_train, y_phase_test = train_test_split(x, y_pileup, y_amp, y_phase, test_size=test_size)

        return ((x_train, y_pileup_train, y_amp_train, y_phase_train), (x_test, y_pileup_test, y_amp_test, y_phase_test))

    @staticmethod
    def PrintBestHyperparameters(best_hps):
        best_hps_dict = best_hps.values
        print("Best hyperparameters:")
        for key in best_hps_dict:
            print(key, best_hps_dict[key])
        
        return 

    ## Instance methods
    def GetModelData(self):
        '''
        Get the model data
        '''
        # get all the data
        (x_train, y_pileup_train, y_amp_train, y_phase_train), (x_test, y_pileup_test, y_amp_test, y_phase_test)  = NetworkTuner.GetData(self.DATAFILE, train_frac=self.TRAIN_PERCENT, treename=self.TREENAME)
        
        self.x_train = x_train
        self.x_test = x_test

        if self.TYPE == "Amp":
            self.y_train = y_amp_train
            self.y_test = y_amp_test
        elif self.TYPE == "Pileup":
            self.y_train = y_pileup_train
            self.y_test = y_pileup_test
        elif self.TYPE == "Phase":
            self.y_train = y_phase_train
            self.y_test = y_phase_test
        else:
            print("Invalid model type")
            sys.exit()
    
        return

    def ModelHyperTune(self, hp):
            
            model = keras.Sequential()
    
            # input layer
            kernel_size_input = hp.Int('kernel_size_0', min_value=2, max_value=18, step=4)
            filters_input = hp.Int('filters_0', min_value=16, max_value=128, step=16)
            activation_input = hp.Choice('activation_0', values=['tanh', 'relu'])
    
            model.add(keras.layers.Conv1D(kernel_size=kernel_size_input, 
                                        filters=filters_input, 
                                        activation=activation_input, 
                                        name='conv0', 
                                        input_shape=(self.INPUTSIZE,1)))
    
            # max pooling options
            max_pooling = hp.Boolean('max_pooling')
            pool_size = hp.Int('pool_size', min_value=2, max_value=8, step=2)
            # add max pooling
            if max_pooling:
                    model.add(keras.layers.MaxPooling1D(pool_size=pool_size, name='max_pooling0'))        
    
            # drouput options
            dropout_conv = hp.Boolean('dropout_conv')       
            dropout_conv_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
            # add dropout
            if dropout_conv:
                    model.add(keras.layers.Dropout(dropout_conv_rate, name='dropout_conv0'))
            
            # # hidden conv layers
            for i in range(hp.Int('num_conv_layers', min_value=0, max_value=3, step=1)):
                    kernel_size = hp.Int(f'kernel_size_{i+1}', min_value=1, max_value=16, step=4)
                    filters = hp.Int(f'filters_{i+1}', min_value=16, max_value=128, step=16)
                    activation = hp.Choice(f'activation_{i+1}', values=['tanh', 'relu'])
    
                    model.add(keras.layers.Conv1D(kernel_size=kernel_size, 
                                                filters=filters, 
                                                activation=activation, 
                                                name=f'conv{i+1}'))
                    # add max pooling
                    if max_pooling:
                            model.add(keras.layers.MaxPooling1D(pool_size=pool_size, name=f'max_pooling{i+1}'))
    
                    # add dropout
                    if dropout_conv:
                            model.add(keras.layers.Dropout(dropout_conv_rate, name=f'dropout_conv{i+1}'))

            # # flatten layer
            model.add(keras.layers.Flatten(name='flat0'))

            # initial dense layer
            units_dense = hp.Int('units_dense0', min_value=32, max_value=512, step=32)
            activation_dense = hp.Choice('activation_dense0', values=['tanh', 'relu'])
            model.add(keras.layers.Dense(units=units_dense, activation=activation_dense, name='dense0'))

            # # hidden dense layers
            dropout_dense = hp.Boolean('dropout_dense')
            dropout_dense_rate = hp.Float('dropout_rate_dense', min_value=0.1, max_value=0.5, step=0.1)

            if dropout_dense:
                model.add(keras.layers.Dropout(dropout_dense_rate, name='dropout_dense0'))

            # add hidden dense layers
            for i in range(hp.Int('num_dense_layers', min_value=0, max_value=3, step=1)):
                    units = hp.Int(f'units_dense_{i+1}', min_value=32, max_value=512, step=32)
                    activation = hp.Choice(f'activation_dense_{i+1}', values=['tanh', 'relu'])
                    model.add(keras.layers.Dense(units=units, activation=activation, name=f'dense{i+1}'))

                    if dropout_dense:
                            model.add(keras.layers.Dropout(dropout_dense_rate, name=f'dropout_dense{i+1}'))
                                    
            
            # # output layer
            avtivation_output = hp.Choice('activation_output', values=self.OUTPUT_ACVTIVATIONS)
            model.add(keras.layers.Dense(1, activation=avtivation_output, name='output'))

            # compile model
            lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                                loss=self.LOSS,
                                metrics=[self.LOSS, 'accuracy']
                                )
            
            return model

    def InitTuner(self):
        
        self.tuner = kt.Hyperband(self.ModelHyperTune,
                            objective=self.OBJECTIVE,
                            max_epochs=self.EPOCHS_SEARCH,
                            overwrite=True,
                            factor=3,
                            directory=str(self.DIRECTORY),
                            project_name=str(self.PROJECT_NAME)
                            )

        self.stop_early = tf.keras.callbacks.EarlyStopping(
                                              monitor=self.EARLY_STOPPING_VAR,
                                              patience=5, 
                                              restore_best_weights=True, 
                                              min_delta=0.001, 
                                              mode=self.EARLY_STOPPING_MODE
                                            )

    def Tune(self):
        self.tuner.search(self.x_train, self.y_train, 
                          epochs=self.EPOCHS_SEARCH, 
                          validation_split=self.VALIDATION_SPLIT, 
                          callbacks=[self.stop_early])
        
        self.best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        NetworkTuner.PrintBestHyperparameters(self.best_hps)

        model = self.tuner.hypermodel.build(self.best_hps)
        self.history = model.fit(self.x_train, self.y_train,
                        epochs=self.EPOCHS_FINAL,
                        validation_split=self.VALIDATION_SPLIT,
                        callbacks=[self.stop_early])
        
        val_obj_per_epoch = self.history.history[self.OBJECTIVE]
        best_epoch = val_obj_per_epoch.index(max(val_obj_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        # Build the model with the optimal hyperparameters and train it on the data
        hypermodel = self.tuner.hypermodel.build(self.best_hps)

        # # Retrain the model
        hypermodel.fit(self.x_train, self.y_train,
                    epochs=best_epoch,
                    validation_split=self.VALIDATION_SPLIT,
                    callbacks=[self.stop_early])
        
        eval_result = hypermodel.evaluate(self.x_test, self.y_test)
        print("Eval result: ", eval_result)

        # Save the model
        hypermodel.save(f'{self.DIRECTORY}/{self.TUNED_MODEL}.h5')
        # Save the best hyperparameters
        with open(f'{self.DIRECTORY}/{self.TUNED_MODEL}.txt', 'w') as f:
            best_hps_dict = self.best_hps.values
            for key in best_hps_dict:
                f.write(key + ' ' + str(best_hps_dict[key]) + '\n')
            f.write('Best epoch: %d' % (best_epoch,))
            f.write('\n Eval result: ' + str(eval_result))
            f.write('\n')

        # Save the history
        with open(f'{self.DIRECTORY}/{self.TUNED_MODEL}_history.pkl', 'wb') as f:
            pickle.dump(self.history.history, f)

        # Save the tuner
        self.tuner.results_summary()
        return
    

############################################
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Network Tuner')
    parser.add_argument('--type', type=str, required=True, help='Type of model to tune')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # create NetworkTuner object
    nt = NetworkTuner(args.config, args.type)
    # nt.Tune()

    print("Done")
    sys.exit()


