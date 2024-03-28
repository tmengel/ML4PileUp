import json 
import sys 

if __name__ == "__main__":

    file = "/home/tmengel/ML4PileUp/HyperTuning/configs/hyperconf.json"
    OUTPUT_DIR = "/home/tmengel/ML4PileUp/HyperTuning/output/"
    DATAFILE = "/home/tmengel/ML4PileUp/data/Data10percFloat.root"
    TREENAME = "OutputTree"

    AUGMENTATION = 8
    TRACELENGTH = 250
    INPUTSIZE = TRACELENGTH-4*AUGMENTATION
    VALIDATION_SPLIT = 0.2
    TRAIN_PERCENT = 0.1
    EPOCHS_SEARCH = 10
    EPOCHS_FINAL = 100
    BATCH_SIZE = 32

    ModelKeys = {  
        "AUGMENTATION" : AUGMENTATION,
        "TRACELENGTH" : TRACELENGTH,
        "INPUTSIZE" : INPUTSIZE,
        "VALIDATION_SPLIT" : VALIDATION_SPLIT,
        "TRAIN_PERCENT" : TRAIN_PERCENT,
        "DATAFILE" : DATAFILE,
        "TREENAME" : TREENAME,
        "EPOCHS_SEARCH" : EPOCHS_SEARCH,
        "EPOCHS_FINAL" : EPOCHS_FINAL,
        "BATCH_SIZE" : BATCH_SIZE,
        "DATA_FILE" : DATAFILE,
        "OUTPUT_DIR" : OUTPUT_DIR,
        "MODEL_TYPES" : ["Amp", "Pileup", "Phase"],
            "Amp": {
                    "LOSS" : "mse",
                    "OUTPUT_ACVTIVATIONS" : ['relu', 'linear'],
                    "OBJECTIVE" : "val_loss",
                    "EARLY_STOPPING_VAR" : "val_loss",
                    "EARLY_STOPPING_MODE" : "min",
                    "DIRECTORY" : "AmpDir",
                    "PROJECT_NAME" : "AmpHyperTune",
                    "TUNED_MODEL" : "AmpHyperTunedModel"
                    },
            "Pileup" : {
                    "LOSS" : "bce",
                    "OUTPUT_ACVTIVATIONS" : ['sigmoid', 'tanh'],
                    "OBJECTIVE" : "val_accuracy",
                    "EARLY_STOPPING_VAR" : "val_accuracy",
                    "EARLY_STOPPING_MODE" : "max",
                    "DIRECTORY" : "PileupDir",
                    "PROJECT_NAME" : "PileupHyperTune",
                    "TUNED_MODEL" : "PileupHyperTunedModel"
                    },
            "Phase" : {
                    "LOSS" : "mse",
                    "OUTPUT_ACVTIVATIONS" : ['relu', 'linear'],
                    "OBJECTIVE" : "val_loss",
                    "EARLY_STOPPING_VAR" : "val_loss",
                    "EARLY_STOPPING_MODE" : "min",
                    "DIRECTORY" : "PhaseDir",
                    "PROJECT_NAME" : "PhaseHyperTune",
                    "TUNED_MODEL" : "PhaseHyperTunedModel"
                    }
    }

    with open(file, "w") as f:
        json.dump(ModelKeys, f, indent=2)

    print("Config file written to ", file)
    sys.exit(0)


