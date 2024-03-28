import os
import tensorflow as tf

num_threads = 40
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
ModelOutputName = 'model_boolPileup100e2'
AUGMENTATION = 10

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


def GetTraces(values):
    traces = np.zeros((values.shape[0], 300))
    for i in range(values.shape[0]):
        trace = np.array(values[i]).reshape(300, 1)
        traces[i][:] = trace[:, 0]
    return traces


def OneHotEncodePileup(pileup,thresh=0):
    pileup_one_hot = np.zeros((pileup.shape[0], 2))
    for i in range(pileup.shape[0]):
        if pileup[i]>thresh:
            pileup_one_hot[i][:] = [1, 0]
        else:
            pileup_one_hot[i][:] = [0, 1]
    return pileup_one_hot

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

def BoolNet():
    defSize = 300-4*AUGMENTATION
    input = layers.Input(shape=(defSize,1))
    conv1 = layers.Conv1D(kernel_size=10, filters=64, activation='tanh')(input)
    max1 = layers.MaxPooling1D(pool_size=1)(conv1)
    conv3 = layers.Conv1D(kernel_size=1, filters=64, activation='relu')(max1)
    max3 = layers.MaxPooling1D(pool_size=1)(conv3)
    conv4 = layers.Conv1D(kernel_size=2, filters=64, activation='relu', padding='same')(max3)
    max4 = layers.MaxPooling1D(pool_size=10)(conv4)
    
    flat1 = layers.Flatten()(max4)
    #dense2 = layers.Dense(128, activation='relu')(flat1)
    output = layers.Dense(2, activation='softmax')(flat1)
    
    model = models.Model(inputs=input, outputs=output)
    model.summary()
    
    return model

def getRandomPileupTraces(tt1,tt2,rndphase,scale):
  newtot = np.zeros_like(tt1)
  newtt1 = np.zeros_like(tt1)
  newtt2 = np.zeros_like(tt2)
  std2 = np.std(tt2[:60]) # gets deviation for baseline
  for i in range(len(tt1)):
    newtt1[i] = tt1[i]
    if(i<rndphase):
      newtot[i] = tt1[i] + np.random.normal(0,std2)
      newtt2[i] = np.random.normal(0,std2) # gaussian random for baseline
    else:
      i2 = int(i-rndphase)
      newtt2[i] = (tt2[i2+1]-tt2[i2])*(rndphase-int(rndphase))+tt2[i2]
      newtt2[i] *= scale
      newtot[i] = tt1[i] + newtt2[i] + np.random.normal(0,std2)
  max = np.max(newtot)
  nmin = np.min(newtot)
  min = newtt2[-1] if newtt1[-1]>newtt2[-1] else  newtt1[-1] #normalizes bottom
  scale = max-nmin
  # print(max,nmin,min,scale)
  return (newtot)/max,newtt1/max,newtt2/max

def augmentTraces(dataset,numAugs,labels=None,otherdata=None):
  newData = []
  newLabels = []
  newOther = []
  for iter in range(len(dataset)):
    for i in range(numAugs):
      newData.append(dataset[iter][4*i:-4*numAugs+4*i])
      newLabels.append(labels[iter])
  newData = np.array(newData)
  newLabels = np.array(newLabels)
  return newData, newLabels

def augmentDepiled(dataset,numAugs):
  newData = []
  for iter in range(len(dataset)):
    for i in range(numAugs):
      t1 = dataset[iter][0][4*i:-4*numAugs+4*i]
      t2 = dataset[iter][1][4*i:-4*numAugs+4*i]
      hold = []
      hold.append(t1)
      hold.append(t2)
      newData.append(hold)
  newData = np.array(newData)
  return newData

pdf = GetData("ysoTracesNoPileup.root")
pdf = pdf[pdf["trace"].apply(lambda x: x.shape[0] == 300)].reset_index(drop=True)
#pile_up_one_hot = OneHotEncodePileup(pdf["pileup"].values)
phase_shifts = GetPhases(pdf["phase"].values)
traces = GetTraces(pdf["trace"].values)
print('Loaded Data')


traces_no_pileup = pdf["trace"].values[pdf["pileup"].values == False]
rand_phase_shifts = np.random.uniform(0.1, PHASEMAX, traces_no_pileup.shape[0])
rand_amplitude_shifts = np.random.uniform(0.5, 1.5, traces_no_pileup.shape[0])
rand_ifPile = np.random.uniform(0, 1, traces_no_pileup.shape[0])

phase_amplitude = np.zeros((traces_no_pileup.shape[0], 2))
traces_depiled = np.zeros((traces_no_pileup.shape[0], 2, 300))
traces_piledup = np.zeros((traces_no_pileup.shape[0], 300,1))

for i in range(traces_no_pileup.shape[0]):
    if rand_ifPile[i]<PERCENTPILEUP:
        rand_trace = int(np.random.uniform(0,traces_no_pileup.shape[0]))
        traces_piledup[i][:,0],traces_depiled[i][0][:],traces_depiled[i][1][:] = getRandomPileupTraces(traces_no_pileup[i][:300],traces_no_pileup[rand_trace][:300],rand_phase_shifts[i],rand_amplitude_shifts[i])
        phase_amplitude[i][0] = rand_phase_shifts[i]
        phase_amplitude[i][1] = rand_amplitude_shifts[i]
    else:
        traces_piledup[i][:,0] = traces_no_pileup[i][:300]
        traces_depiled[i][0][:] = traces_no_pileup[i][:300]
        traces_depiled[i][1][:] = np.zeros_like(traces_no_pileup[i][:300])
        phase_amplitude[i][0] = 0.
        phase_amplitude[i][1] = 0.

pile_up_one_hot = OneHotEncodePileup(rand_ifPile,PERCENTPILEUP)

print('Formatted Data')
traces_piledup,phase_amplitude = augmentTraces(traces_piledup,AUGMENTATION,phase_amplitude)
print('Augmented Traces')
traces_depiled = augmentDepiled(traces_depiled,AUGMENTATION)
print('Augmented Depiled')

model = BoolNet()
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
train_x = traces_piledup[:NUMTRAINING]
train_y = pile_up_one_hot[:NUMTRAINING]
history = model.fit(train_x, train_y, epochs=NUMEPOCHS, batch_size=256, validation_split=0.2, verbose=2)

#### Saving the model
model.save('models/'+ModelOutputName)
plot_history(history)

test_x = traces_piledup[NUMTRAINING:NUMTRAINING+10000]
test_y = pile_up_one_hot[NUMTRAINING:NUMTRAINING+10000]
test_p = phase_amplitude[NUMTRAINING:NUMTRAINING+10000,0]
test_a = phase_amplitude[NUMTRAINING:NUMTRAINING+10000,1]
test_y_hat = model.predict(test_x)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2 ,figsize=(15, 8))
n = np.random.randint(0, test_x.shape[0])

ax[0][0].hist2d(test_y[:,1],test_y_hat[:,1])
ax[0][0].set_xlabel("Real Pileup")
ax[0][0].set_ylabel("Predicted Pileup")

#ax[1][0].plot(test_phase_amp[:,0], phi_res, 'o', color='black',alpha=0.5)
#ax[1][0].set_xlabel("Truth Phase shift [ns]")
#ax[1][0].set_ylabel("Phase Shift Error")
#ax[1][0].set_ylim([-8,8])

ax[0][1].plot(test_p,test_y_hat, 'o', color='black',alpha=0.5)
ax[0][1].set_xlabel("True Phase (ns)")
ax[0][1].set_ylabel("Predicted Pileup")

ax[1][1].plot(test_a,test_y_hat)
ax[1][1].set_xlabel("True Amplitude")
ax[1][1].set_ylabel("Predicted Pileup")

plt.savefig('models/'+ModelOutputName+'/residuals.png')

