#!/usr/bin/env python
# coding: utf-8

import os
import tensorflow as tf

num_threads = 20
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["TF_NUM_INTRAOP_THREADS"] = "10"
os.environ["TF_NUM_INTEROP_THREADS"] = "10"

tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
)
tf.config.set_soft_device_placement(True)


from tensorflow import keras
from tensorflow.keras import models, layers
import uproot
import numpy as np
import pandas as pd

NUMEPOCHS = 100
PHASEMAX = 100
PERCENTPILEUP = 0.5
NUMTRAINING = 20000
ModelOutputName = 'triad2_100e'
AUGMENTATION = 8
TRACELENGTH = 250


def GetData(filename, treename="timing"):
    '''
    Returns TFile as a pandas dataframe
    '''
    file = uproot.open(filename)
    tree = file[treename]
    npdf = tree.arrays(library="np")
    df =  pd.DataFrame(npdf, columns=npdf.keys())
    return df

def plot_history(history):
    '''
    Plot training and validation loss and accuracy for task 1-3
    '''
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # plot training and validation loss
    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].set_title('Model Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_yscale('log')
    axs[0].legend(['train', 'val'], loc='best')

    # plot training and validation accuracy
    axs[1].plot(history.history['accuracy'])
    axs[1].plot(history.history['val_accuracy'])
    axs[1].set_title('Model Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend(['train', 'val'], loc='best')

    plt.savefig('models/'+ModelOutputName+'/history.png')


def GetTraces(values,traceLength=300):
    traces = np.zeros((values.shape[0], traceLength))
    for i in range(values.shape[0]):
        trace = np.array(values[i]).reshape(traceLength, 1)
        traces[i][:] = trace[:, 0]
    return traces


def GetPhases(phases):
    phase = np.zeros((phases.shape[0], 1))
    for i in range(phases.shape[0]):
        if phases[i] > 0:
            phase[i] = phases[i]
        else:
            phase[i] = 0.0
    return phase
    
def NormalizeTraces(traces):
    for i in range(len(traces)):
        baseline = np.average(traces[i][0:40])
        traces[i] -= baseline
        tmax = np.amax(traces[i])
        traces[i] /= tmax
    return traces


from tensorflow.keras.callbacks import EarlyStopping
class EarlyStoppingWithUntrainableLayers(EarlyStopping):
  def __init__(self, monitor='loss_name', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False, layers_to_freeze=[], **kwargs):
    super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, 
        verbose=verbose, mode=mode, baseline=baseline, 
        restore_best_weights=restore_best_weights, **kwargs)
    self.monitor = monitor
    self.layers_to_freeze = layers_to_freeze

    def on_epoch_end(self, epoch, logs=None):
      super().on_epoch_end(epoch, logs)
      if self.stopped_epoch is not None:
        self._set_layers_untrainable()

      def on_train_end(self, logs=None):
        super().on_train_end(logs)
        if self.restore_best_weights and self.stopped_epoch is None:
          self._set_layers_untrainable()

        def _set_layers_untrainable(self):
          print(f'{self.monitor} stopped at epoch {self.stopped_epoch}')
          for layer in self.layers_to_freeze:
            layer.trainable = False

def TriadNet():
  defSize = TRACELENGTH-4*AUGMENTATION

  input = layers.Input(shape=(defSize,1))

  # Pileup Net
  pileupconv1 = layers.Conv1D(kernel_size=10, filters=64, activation='tanh', name='pileupconv1')(input)
  pileup_conv1_dropout = layers.Dropout(0.2, name='pileupconv1_dropout')(pileupconv1)
  # pileupmax1 = layers.MaxPooling1D(pool_size=1)(pileupconv1)
  pileupconv3 = layers.Conv1D(kernel_size=1, filters=64, activation='relu', name ='pileupconv3')(pileup_conv1_dropout)
  pileup_conv3_dropout = layers.Dropout(0.2, name='pileupconv3_dropout')(pileupconv3)
  # pileupmax3 = layers.MaxPooling1D(pool_size=1)(pileupconv3)
  pileupflat1 = layers.Flatten(name='pileupflat1')(pileup_conv3_dropout)


  pileupdense2 = layers.Dense(64, activation='relu', name='pileupdense2')(pileupflat1)
  pileupoutput = layers.Dense(1, activation='sigmoid', name='pileupoutput')(pileupdense2)

  # Phase Net
  phaseconv1 = layers.Conv1D(kernel_size=10, filters=64, activation='tanh', name='phaseconv1')(input)
  phase_conv1_dropout = layers.Dropout(0.2, name='phase_conv1_dropout')(phaseconv1)
  # phasemax1 = layers.MaxPooling1D(pool_size=1)(phaseconv1)
  phaseconv3 = layers.Conv1D(kernel_size=1, filters=64, activation='relu', name='phaseconv3')(phase_conv1_dropout)
  phase_conv3_dropout = layers.Dropout(0.2, name='phase_conv3_dropout')(phaseconv3)
  # phasemax3 = layers.MaxPooling1D(pool_size=1)(phaseconv3)
  phaseconv4 = layers.Conv1D(kernel_size=2, filters=64, activation='relu', padding='same', name='phaseconv4')(phase_conv3_dropout)
  phase_conv4_dropout = layers.Dropout(0.2, name='phase_conv4_dropout')(phaseconv4)
  # phasemax4 = layers.MaxPooling1D(pool_size=10)(phaseconv4)

  phaseflat1 = layers.Flatten(name='phaseflat1')(phase_conv4_dropout)
  phasedense2 = layers.Dense(128, activation='relu', name='phasedense2')(phaseflat1)
  phaseoutput = layers.Dense(1, activation='linear', name='phaseoutput')(phasedense2)


  # Amp Net
  ampconv1 = layers.Conv1D(kernel_size=10, filters=64, activation='tanh', name='ampconv1')(input)
  amp_conv1_dropout = layers.Dropout(0.2, name='amp_conv1_dropout')(ampconv1)
  # ampmax1 = layers.MaxPooling1D(pool_size=1)(ampconv1)
  ampconv3 = layers.Conv1D(kernel_size=1, filters=64, activation='relu', name='ampconv3')(amp_conv1_dropout)
  amp_conv3_dropout = layers.Dropout(0.2, name='amp_conv3_dropout')(ampconv3)
  # ampmax3 = layers.MaxPooling1D(pool_size=1)(ampconv3)
  ampconv4 = layers.Conv1D(kernel_size=2, filters=64, activation='relu', padding='same', name='ampconv4')(amp_conv3_dropout)
  amp_conv4_dropout = layers.Dropout(0.2, name='amp_conv4_dropout')(ampconv4)
  # ampmax4 = layers.MaxPooling1D(pool_size=10)(ampconv4)

  ampflat1 = layers.Flatten(name='ampflat1')(amp_conv4_dropout)
  ampdense2 = layers.Dense(128, activation='relu', name='ampdense2')(ampflat1)
  ampoutput = layers.Dense(1, activation='linear', name='ampoutput')(ampdense2)


  model = models.Model(inputs=input, outputs=[pileupoutput,phaseoutput,ampoutput])
  model.summary()

  return model


fname = 'data/DataSmall.root'
tree = 'OutputTree'
traceBranch = "trace"
traceLength = TRACELENGTH-4*AUGMENTATION
pdf = GetData(fname,tree)
pdf = pdf[pdf[traceBranch].apply(lambda x: x.shape[0] == traceLength)].reset_index(drop=True)
traces = GetTraces(pdf[traceBranch].values,traceLength)
phases = GetPhases(pdf["phase"].values)
ifPile = GetPhases(pdf["pile"].values)
amps = GetPhases(pdf["amp"].values)
qdcs = GetPhases(pdf["qdc"].values)
print('Loaded Data')


print(traces.shape, phases.shape, qdcs.shape, amps.shape, ifPile.shape)

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y, train_q, test_q, train_ifPile, test_ifPile, train_amps, test_amps = train_test_split(traces,phases,qdcs,ifPile,amps,test_size=0.5,train_size=0.05)


model = TriadNet()
phase_layers = [layer for layer in model.layers if 'phase' in layer.name]
pileup_layers = [layer for layer in model.layers if 'pileup' in layer.name]
amp_layers = [layer for layer in model.layers if 'amp' in layer.name]

min_delta = 5.e-4
patience = 10
early_stopping_all =tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=min_delta,
    patience=patience,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)

early_stopping_pileup = EarlyStoppingWithUntrainableLayers(
    monitor='val_pileupoutput_loss',
    min_delta=min_delta,
    patience=patience,
    verbose=1,
    restore_best_weights=True,
    layers_to_freeze=pileup_layers
)

early_stopping_phase = EarlyStoppingWithUntrainableLayers(
    monitor='val_phaseoutput_loss',
    min_delta=min_delta,
    patience=patience,
    verbose=1,
    restore_best_weights=True,
    layers_to_freeze=phase_layers
)

early_stopping_amp = EarlyStoppingWithUntrainableLayers(
    monitor='val_ampoutput_loss',
    min_delta=min_delta,
    patience=patience,
    verbose=1,
    restore_best_weights=True,
    layers_to_freeze=amp_layers
)

model.compile(optimizer='adam', loss=['bce','mse','mse'], metrics='accuracy')
history = model.fit(train_x, [train_ifPile,train_y,train_amps], epochs=NUMEPOCHS, batch_size=256, validation_split=0.2, verbose=1, callbacks=[early_stopping_all,early_stopping_pileup,early_stopping_phase,early_stopping_amp])

### Creating checkpoints to save the model every so often
#checkpoint_path = "models/"+ModelOutputName+"/checkpoints/cp-{epoch:04d}.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)
#
## Create a callback that saves the model's weights
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)



#### Saving the model
model.save('models/'+ModelOutputName)
#pd.DataFrame(history.history, index = history.epoch,columns=history.history.keys()).to_hdf('models/'+ModelOutputName+'/histry.h5',key="hist")

#plot_history(history)


# test_x = traces_piledup[NUMTRAINING:NUMTRAINING+10000]
# # test_y = pile_up_one_hot[NUMTRAINING:NUMTRAINING+10000]
# test_p = phase_amplitude[NUMTRAINING:NUMTRAINING+10000,0]
# test_a = phase_amplitude[NUMTRAINING:NUMTRAINING+10000,1]
# test_x = traces[NUMTRAINING:NUMTRAINING+10000]
test_p = test_y[NUMTRAINING:NUMTRAINING+10000]
test_a = test_amps[NUMTRAINING:NUMTRAINING+10000]
test_y_hat = model.predict(test_x[NUMTRAINING:NUMTRAINING+10000])
pres = []
for i in range(len(test_y_hat[1])):
  pres.append(test_y_hat[1][i]-test_p[i])
pres = np.array(pres)


print(pres.shape)


import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix

fig, ax = plt.subplots(2, 2 ,figsize=(15, 8))
# n = np.random.randint(0, test_x.shape[0])

# ax[0][0].hist2d(test_y[:,1],test_y_hat[:,1],bins=2)
# ax[0][0].set_xlabel("Real Pileup")
# ax[0][0].set_ylabel("Predicted Pileup")

ax[0][1].plot(pres,test_a, 'o', color='black',alpha=0.5)
# ax[0][1].set_xlim(-5,5)
ax[0][1].set_ylabel("Truth Amp.")
ax[0][1].set_xlabel("Phase Residual (ns)")

ax[1][1].plot(test_p,test_y_hat[1], 'o', color='black',alpha=0.5)
ax[1][1].set_xlabel("Truth Phase shift [ns]")
ax[1][1].set_ylabel("Predicted Phase Shift [ns]")

ax[0][0].hist(pres,bins=1000)#,range=(-5,5))
ax[0][0].set_xlabel("Residual [ns]")
ax[0][0].set_ylabel("Counts")

ax[1][0].plot(pres,test_p, 'o', color='black',alpha=0.5)
# ax[1][0].set_xlim(-5,5)
ax[1][0].set_ylabel("Truth Phase shift [ns]")
ax[1][0].set_xlabel("Residual [ns]")

plt.savefig('models/'+ModelOutputName+'/residuals.png')
