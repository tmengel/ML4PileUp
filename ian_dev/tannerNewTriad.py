#!/usr/bin/env python
# coding: utf-8

import os
import tensorflow as tf

#num_threads = 20
#os.environ["OMP_NUM_THREADS"] = "10"
#os.environ["TF_NUM_INTRAOP_THREADS"] = "10"
#os.environ["TF_NUM_INTEROP_THREADS"] = "10"
#
#tf.config.threading.set_inter_op_parallelism_threads(
#    num_threads
#)
#tf.config.threading.set_intra_op_parallelism_threads(
#    num_threads
#)
#tf.config.set_soft_device_placement(True)


from tensorflow import keras
from tensorflow.keras import models, layers
import uproot
import numpy as np
import pandas as pd
import awkward as ak

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

NUMEPOCHS = 100
PHASEMAX = 100
PERCENTPILEUP = 0.5
NUMTRAINING = 20000
ModelOutputName = 'TannerTriad'
AUGMENTATION = 8
TRACELENGTH = 250


#def GetData(filename, treename="timing"):
#    '''
#    Returns TFile as a pandas dataframe
#    '''
#    file = uproot.open(filename)
#    tree = file[treename]
#    npdf = tree.arrays(library="np")
#    df =  pd.DataFrame(npdf, columns=npdf.keys())
#    return df

def GetData(filename,branch="trace",treename="timing"):
  '''
  Returns TFile as a pandas dataframe
  '''
  file = uproot.open(filename)
  tree = file[treename]
  npdf = ak.to_numpy(tree[branch].arrays()[branch])
  #df =  pd.DataFrame(npdf, columns=npdf.keys())
  #del df['fname']
  #return df
  return npdf

def float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def float_feature_list(value):
  """Returns a list of float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_example(trace,pileup,phase,amp):
  feature = {
      "trace": float_feature_list(trace),
      "pileup": int64_feature(pileup),
      "phase": float_feature(phase),
      "amp": float_feature(amp),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))

def parse_tfrecord_fn(example):
  feature_description = {
    "trace": tf.io.VarLenFeature(tf.float32),
    "pileup": tf.io.FixedLenFeature([], tf.int64),
    "phase": tf.io.FixedLenFeature([], tf.float32),
    "amp": tf.io.FixedLenFeature([], tf.float32),
  }
  example = tf.io.parse_single_example(example, feature_description)
  example["trace"] = tf.sparse.to_dense(example["trace"])
  return example

def prepare_sample(features):
  return features["trace"], (features["pileup"],features["phase"],features["amp"])

def get_dataset(filenames, batch_size):
  dataset = (
    tf.data.TFRecordDataset(filenames,num_parallel_reads=AUTOTUNE)
    .map(parse_tfrecord_fn,num_parallel_calls=AUTOTUNE)
    .map(prepare_sample, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .repeat()
    .prefetch(AUTOTUNE)
  )
  return dataset

def get_shuffledDataset(data,batch_size):
  dataset = (
    data
    .map(parse_tfrecord_fn,num_parallel_calls=AUTOTUNE)
    .map(prepare_sample, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .repeat()
    .prefetch(AUTOTUNE)
  )
  return dataset

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


def PhaseNet_Single():
    '''
    For single phase prediction
    '''
    defSize = TRACELENGTH-4*AUGMENTATION
    phaseconv1 = layers.Conv1D(kernel_size=10, filters=16, activation='tanh', name='conv1')(input)
    phaseflat1 = layers.Flatten(name='flatten1')(phaseconv1)
    phasedense1 = layers.Dense(128, activation='relu', name='dense1')(phaseflat1)
    phaseoutput = layers.Dense(1, activation='linear', name='phaseoutput')(phasedense1)
    model = models.Model(inputs=input, outputs=phaseoutput)
    model.summary()

    return model

def AmpNet_Single():
    '''
    For single amplitude prediction
    '''
    defSize = TRACELENGTH-4*AUGMENTATION
    ampconv1 = layers.Conv1D(kernel_size=10, filters=16, activation='tanh', name='conv1')(input)
    ampflat1 = layers.Flatten(name='flatten1')(ampconv1)
    ampdense1 = layers.Dense(128, activation='relu', name='dense1')(ampflat1)
    ampoutput = layers.Dense(1, activation='linear', name='ampoutput')(ampdense1)
    model = models.Model(inputs=input, outputs=ampoutput)
    model.summary()

    return model  

def PileupNet_Single():
    '''
    for single pileup prediction
    '''
    defSize = TRACELENGTH-4*AUGMENTATION
    pileupconv1 = layers.Conv1D(kernel_size=4, filters=16, activation='relu', name='conv1')(input)
    pileupflat1 = layers.Flatten(name='flatten1')(pileupconv1)
    pileupdense1 = layers.Dense(32, activation='relu', name='dense1')(pileupflat1)
    pileupoutput = layers.Dense(1, activation='linear', name='pileupoutput')(pileupdense1)
    model = models.Model(inputs=input, outputs=pileupoutput)
    model.summary()

    return model



def TriadNet():
  defSize = TRACELENGTH-4*AUGMENTATION

  input = layers.Input(shape=(defSize,1))

  # Pileup Net
  pileupconv1 = layers.Conv1D(kernel_size=4, filters=16, activation='relu', name='conv1_pile')(input)
  pileupflat1 = layers.Flatten(name='flatten1_pile')(pileupconv1)
  pileupdense1 = layers.Dense(32, activation='relu', name='dense1_pile')(pileupflat1)
  pileupoutput = layers.Dense(1, activation='linear', name='pileupoutput_pile')(pileupdense1)

  # Phase Net
  phaseconv1 = layers.Conv1D(kernel_size=10, filters=16, activation='tanh', name='conv1_phase')(input)
  phaseflat1 = layers.Flatten(name='flatten1_phase')(phaseconv1)
  phasedense1 = layers.Dense(128, activation='relu', name='dense1_phase')(phaseflat1)
  phaseoutput = layers.Dense(1, activation='linear', name='phaseoutput_phase')(phasedense1)

  # Amp Net
  ampconv1 = layers.Conv1D(kernel_size=10, filters=16, activation='tanh', name='conv1_amp')(input)
  ampflat1 = layers.Flatten(name='flatten1_amp')(ampconv1)
  ampdense1 = layers.Dense(128, activation='relu', name='dense1_amp')(ampflat1)
  ampoutput = layers.Dense(1, activation='linear', name='ampoutput_amp')(ampdense1)



  # # Pileup Net
  # pileupconv1 = layers.Conv1D(kernel_size=10, filters=64, activation='tanh', name='pileupconv1')(input)
  # pileup_conv1_dropout = layers.Dropout(0.2, name='pileupconv1_dropout')(pileupconv1)
  # # pileupmax1 = layers.MaxPooling1D(pool_size=1)(pileupconv1)
  # pileupconv3 = layers.Conv1D(kernel_size=1, filters=64, activation='relu', name ='pileupconv3')(pileup_conv1_dropout)
  # pileup_conv3_dropout = layers.Dropout(0.2, name='pileupconv3_dropout')(pileupconv3)
  # # pileupmax3 = layers.MaxPooling1D(pool_size=1)(pileupconv3)
  # pileupflat1 = layers.Flatten(name='pileupflat1')(pileup_conv3_dropout)


  # pileupdense2 = layers.Dense(64, activation='relu', name='pileupdense2')(pileupflat1)
  # pileupoutput = layers.Dense(1, activation='sigmoid', name='pileupoutput')(pileupdense2)

  # # Phase Net
  # phaseconv1 = layers.Conv1D(kernel_size=10, filters=64, activation='tanh', name='phaseconv1')(input)
  # phase_conv1_dropout = layers.Dropout(0.2, name='phase_conv1_dropout')(phaseconv1)
  # # phasemax1 = layers.MaxPooling1D(pool_size=1)(phaseconv1)
  # phaseconv3 = layers.Conv1D(kernel_size=1, filters=64, activation='relu', name='phaseconv3')(phase_conv1_dropout)
  # phase_conv3_dropout = layers.Dropout(0.2, name='phase_conv3_dropout')(phaseconv3)
  # # phasemax3 = layers.MaxPooling1D(pool_size=1)(phaseconv3)
  # phaseconv4 = layers.Conv1D(kernel_size=2, filters=64, activation='relu', padding='same', name='phaseconv4')(phase_conv3_dropout)
  # phase_conv4_dropout = layers.Dropout(0.2, name='phase_conv4_dropout')(phaseconv4)
  # # phasemax4 = layers.MaxPooling1D(pool_size=10)(phaseconv4)

  # phaseflat1 = layers.Flatten(name='phaseflat1')(phase_conv4_dropout)
  # phasedense2 = layers.Dense(128, activation='relu', name='phasedense2')(phaseflat1)
  # phaseoutput = layers.Dense(1, activation='linear', name='phaseoutput')(phasedense2)


  # # Amp Net
  # ampconv1 = layers.Conv1D(kernel_size=10, filters=64, activation='tanh', name='ampconv1')(input)
  # amp_conv1_dropout = layers.Dropout(0.2, name='amp_conv1_dropout')(ampconv1)
  # # ampmax1 = layers.MaxPooling1D(pool_size=1)(ampconv1)
  # ampconv3 = layers.Conv1D(kernel_size=1, filters=64, activation='relu', name='ampconv3')(amp_conv1_dropout)
  # amp_conv3_dropout = layers.Dropout(0.2, name='amp_conv3_dropout')(ampconv3)
  # # ampmax3 = layers.MaxPooling1D(pool_size=1)(ampconv3)
  # ampconv4 = layers.Conv1D(kernel_size=2, filters=64, activation='relu', padding='same', name='ampconv4')(amp_conv3_dropout)
  # amp_conv4_dropout = layers.Dropout(0.2, name='amp_conv4_dropout')(ampconv4)
  # # ampmax4 = layers.MaxPooling1D(pool_size=10)(ampconv4)

  # ampflat1 = layers.Flatten(name='ampflat1')(amp_conv4_dropout)
  # ampdense2 = layers.Dense(128, activation='relu', name='ampdense2')(ampflat1)
  # ampoutput = layers.Dense(1, activation='linear', name='ampoutput')(ampdense2)


  model = models.Model(inputs=input, outputs=[pileupoutput,phaseoutput,ampoutput])
  model.summary()

  return model

print('Starting to get data')

fname = 'data/Data.root'
tree = 'OutputTree'
traceBranch = "trace"
traceLength = TRACELENGTH-4*AUGMENTATION
pileup = GetData(fname,'pile',tree)
DATASET_SIZE = len(pileup)
print(DATASET_SIZE)
print('Loaded Data')

tfrecords_dir = "data/tfrecords"
num_samples = 4096

train_size = int(0.7*DATASET_SIZE)
val_size = int(0.15*DATASET_SIZE)
test_size = int(0.15*DATASET_SIZE)
train_filenames = tf.io.gfile.glob(f"{tfrecords_dir}/*.tfrec")
batch_size = 256
steps_per_epoch = train_size/batch_size
AUTOTUNE = tf.data.AUTOTUNE

train_size = int(0.7*DATASET_SIZE)
val_size = int(0.15*DATASET_SIZE)
test_size = int(0.15*DATASET_SIZE)

full_dataset = tf.data.TFRecordDataset(train_filenames)
full_dataset = full_dataset.shuffle(10, reshuffle_each_iteration=False)
train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)
val_dataset = test_dataset.skip(test_size)
test_dataset = test_dataset.take(test_size)
val_dataset = val_dataset.take(val_size)

print(type(train_dataset),type(val_dataset))


model = TriadNet()

phase_layers = [layer for layer in model.layers if 'phase' in layer.name]
pileup_layers = [layer for layer in model.layers if 'pileup' in layer.name]
amp_layers = [layer for layer in model.layers if 'amp' in layer.name]

min_delta = 5.e-6
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
history = model.fit(get_shuffledDataset(train_dataset,batch_size), epochs=NUMEPOCHS, batch_size=batch_size, steps_per_epoch=steps_per_epoch, validation_data=get_shuffledDataset(val_dataset,batch_size), validation_steps=val_size, validation_batch_size=batch_size, verbose=2, callbacks=[early_stopping_all,early_stopping_pileup,early_stopping_phase,early_stopping_amp])


#### Saving the model
model.save('models/'+ModelOutputName)
pd.DataFrame(history.history, index = history.epoch,columns=history.history.keys()).to_hdf('models/'+ModelOutputName+'/histry.h5',key="hist")

#plot_history(history)


# test_x = traces_piledup[NUMTRAINING:NUMTRAINING+10000]
# # test_y = pile_up_one_hot[NUMTRAINING:NUMTRAINING+10000]
# test_p = phase_amplitude[NUMTRAINING:NUMTRAINING+10000,0]
# test_a = phase_amplitude[NUMTRAINING:NUMTRAINING+10000,1]
# test_x = traces[NUMTRAINING:NUMTRAINING+10000]
#test_p = test_y[NUMTRAINING:NUMTRAINING+10000]
#test_a = test_amps[NUMTRAINING:NUMTRAINING+10000]
#test_y_hat = model.predict(test_x[NUMTRAINING:NUMTRAINING+10000])
#pres = []
#for i in range(len(test_y_hat[1])):
#  pres.append(test_y_hat[1][i]-test_p[i])
#pres = np.array(pres)
#
#
#print(pres.shape)
#
#
#import matplotlib.pyplot as plt
## from sklearn.metrics import confusion_matrix
#
#fig, ax = plt.subplots(2, 2 ,figsize=(15, 8))
## n = np.random.randint(0, test_x.shape[0])
#
## ax[0][0].hist2d(test_y[:,1],test_y_hat[:,1],bins=2)
## ax[0][0].set_xlabel("Real Pileup")
## ax[0][0].set_ylabel("Predicted Pileup")
#
#ax[0][1].plot(pres,test_a, 'o', color='black',alpha=0.5)
## ax[0][1].set_xlim(-5,5)
#ax[0][1].set_ylabel("Truth Amp.")
#ax[0][1].set_xlabel("Phase Residual (ns)")
#
#ax[1][1].plot(test_p,test_y_hat[1], 'o', color='black',alpha=0.5)
#ax[1][1].set_xlabel("Truth Phase shift [ns]")
#ax[1][1].set_ylabel("Predicted Phase Shift [ns]")
#
#ax[0][0].hist(pres,bins=1000)#,range=(-5,5))
#ax[0][0].set_xlabel("Residual [ns]")
#ax[0][0].set_ylabel("Counts")
#
#ax[1][0].plot(pres,test_p, 'o', color='black',alpha=0.5)
## ax[1][0].set_xlim(-5,5)
#ax[1][0].set_ylabel("Truth Phase shift [ns]")
#ax[1][0].set_xlabel("Residual [ns]")
#
#plt.savefig('models/'+ModelOutputName+'/residuals.png')
#
#
#history_file = 'models/'+ModelOutputName+'/histry.h5'
#hist = pd.read_hdf(history_file)
#hist.head()
#
#loss = ['loss', 'dense_1_loss', 'dense_3_loss', 'dense_5_loss']
#accuracy = ['dense_1_accuracy', 'dense_3_accuracy', 'dense_5_accuracy']
#val_loss = ['val_loss', 'val_dense_1_loss', 'val_dense_3_loss', 'val_dense_5_loss']
#val_accuracy = ['val_dense_1_accuracy', 'val_dense_3_accuracy', 'val_dense_5_accuracy']
#
#params = {'axes.labelsize': 12,
#                'axes.linewidth' : 1.5,
#                'font.size': 12,
#                'font.family': 'times',
#                'mathtext.fontset': 'stix',
#                'legend.fontsize': 12,
#                'xtick.labelsize': 12,
#                'ytick.labelsize': 12,
#                'text.usetex': True,
#                'lines.linewidth': 1.0,
#                'lines.linestyle': '-',
#                'lines.markersize' : 6,
#                'lines.markeredgewidth' : 1,
#                'xtick.major.size' : 5,
#                'xtick.minor.size' : 3,
#                'xtick.major.width' : 2,
#                'xtick.minor.width' : 1,
#                'xtick.direction' : 'in',
#                'ytick.major.size' : 5,
#                'ytick.minor.size' : 3,
#                'ytick.major.width' : 2,
#                'ytick.minor.width' : 1,
#                'ytick.direction' : 'in',
#                'xtick.minor.visible' : True,
#                'ytick.minor.visible' : True,
#                'savefig.transparent': True,
#                'errorbar.capsize': 1.5,
#                }
#plt.rcParams.update(params)
#fig = plt.figure(figsize=(3,2), dpi=300)
#ax = fig.add_subplot(111)
#ax.plot(hist['loss'], label='Training')
#ax.plot(hist['val_loss'], label='Validation')
#ax.set_xlabel('Epoch')
#ax.set_ylabel('Loss')
#ax.set_title('Total loss')
#ax.legend(loc='upper right', ncol=1, fontsize=8, columnspacing=0.1, handletextpad=0.2, borderpad=0.2, labelspacing=0.2, frameon=False)
#plt.savefig('models/'+ModelOutputName+'/overallLoss.png')
#
#fig, axs = plt.subplots(1,3,figsize=(11,3), dpi=300, constrained_layout=True)
#loss = ['dense_1_loss', 'dense_3_loss', 'dense_5_loss']
#accuracy = ['dense_1_accuracy', 'dense_3_accuracy', 'dense_5_accuracy']
#val_loss = ['val_dense_1_loss', 'val_dense_3_loss', 'val_dense_5_loss']
#val_accuracy = ['val_dense_1_accuracy', 'val_dense_3_accuracy', 'val_dense_5_accuracy']
#titles = ['dense 1', 'dense 3', 'dense 5']
#for i in range(3):
#    ax = axs[i]
#    ax.plot(hist[loss[i]], label='Training', color='b')
#    ax.plot(hist[val_loss[i]], label='Validation', color='b', linestyle='--',alpha=0.7)
#    ax.set_xlabel('Epoch')
#    ax.tick_params(axis='y', which='both', colors='b')
#    ax.yaxis.label.set_color('b')
#    ax.spines['left'].set_color('b')
#    ax.spines['right'].set_edgecolor('b')
#    if i==0:
#        ax.set_ylabel('Loss')
#    ax.set_title(titles[i])
#    axt = ax.twinx()
#    axt.plot(hist[accuracy[i]], label='Training', color='r')
#    axt.plot(hist[val_accuracy[i]], label='Validation', color='r', linestyle='--',alpha=0.7)
#    axt.tick_params(axis='y', which='both', colors='r')
#    axt.yaxis.label.set_color('r')
#    
#    if i ==2:
#        axt.set_ylabel('Accuracy')
#    ax.legend(loc='lower right', handles=[ax.plot([],[],color='k', linestyle='-')[0], ax.plot([],[],color='k', linestyle='--',alpha=0.7)[0]], labels=['Train','Val'], ncol=1, fontsize=10, columnspacing=0.1, handletextpad=0.2, borderpad=0.2, labelspacing=0.2, frameon=False)
#    # axt.legend(loc='lower right', ncol=1, fontsize=8, columnspacing=0.1, handletextpad=0.2, borderpad=0.2, labelspacing=0.2, frameon=False)
#
##plt.show()
#fig.savefig("models/"+ModelOutputName+"/submodule_loss.pdf", bbox_inches='tight')
#fig.savefig("models/"+ModelOutputName+"/submodule_loss.png", bbox_inches='tight')
