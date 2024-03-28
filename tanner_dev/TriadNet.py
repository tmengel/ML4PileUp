#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers

import uproot
import numpy as np
import pandas as pd
import awkward as ak


NUMEPOCHS = 100
PHASEMAX = 100
PERCENTPILEUP = 0.5
NUMTRAINING = 20000
ModelOutputName = 'triad2_100e_fulldata'
AUGMENTATION = 8
TRACELENGTH = 250
INPUTSIZE = TRACELENGTH-4*AUGMENTATION


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

def TraceNet():
    input = layers.Input(shape=(INPUTSIZE,1))
    conv1 = layers.Conv1D(kernel_size=10, filters=64, activation='tanh', name='conv1')(input)
    conv1_dropout = layers.Dropout(0.2, name='conv1_dropout')(conv1)
    conv2 = layers.Conv1D(kernel_size=1, filters=64, activation='relu', name='conv2')(conv1_dropout)
    conv2_dropout = layers.Dropout(0.2, name='conv2_dropout')(conv2)
    flatten1 = layers.Flatten(name='flatten1')(conv2_dropout)
    dense1 = layers.Dense(64, activation='relu', name='dense1')(flatten1)
    denseout = layers.Dense(1, activation='sigmoid', name='pileupoutput')(dense1)
    model = models.Model(inputs=input, outputs=denseout)
    model.summary()
    return model 

def AmpNet():
    input = layers.Input(shape=(INPUTSIZE,1))
    conv1 = layers.Conv1D(kernel_size=10, filters=64, activation='tanh', name='conv1')(input)
    conv1_dropout = layers.Dropout(0.2, name='conv1_dropout')(conv1)
    conv2 = layers.Conv1D(kernel_size=1, filters=64, activation='relu', name='conv2')(conv1_dropout)
    conv2_dropout = layers.Dropout(0.2, name='conv2_dropout')(conv2)
    conv3 = layers.Conv1D(kernel_size=2, filters=64, activation='relu', padding='same', name='conv3')(conv2_dropout)
    conv3_dropout = layers.Dropout(0.2, name='conv3_dropout')(conv3)
    flatten1 = layers.Flatten(name='flatten1')(conv3_dropout)
    dense1 = layers.Dense(128, activation='relu', name='dense1')(flatten1)
    denseout = layers.Dense(1, activation='linear', name='pileupoutput')(dense1)
    model = models.Model(inputs=input, outputs=denseout)
    model.summary()
    return model
  
def PileupNet():
    input = layers.Input(shape=(INPUTSIZE,1))
    conv1 = layers.Conv1D(kernel_size=10, filters=64, activation='tanh', name='conv1')(input)
    conv1_dropout = layers.Dropout(0.2, name='conv1_dropout')(conv1)
    conv2 = layers.Conv1D(kernel_size=1, filters=64, activation='relu', name='conv2')(conv1_dropout)
    conv2_dropout = layers.Dropout(0.2, name='conv2_dropout')(conv2)
    conv3 = layers.Conv1D(kernel_size=2, filters=64, activation='relu', padding='same', name='conv3')(conv2_dropout)
    conv3_dropout = layers.Dropout(0.2, name='conv3_dropout')(conv3)
    flatten1 = layers.Flatten(name='flatten1')(conv3_dropout)
    dense1 = layers.Dense(128, activation='relu', name='dense1')(flatten1)
    denseout = layers.Dense(1, activation='linear', name='pileupoutput')(dense1)
    model = models.Model(inputs=input, outputs=denseout)
    model.summary()
    return model

 

