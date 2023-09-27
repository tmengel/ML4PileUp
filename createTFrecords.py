import os
import tensorflow as tf
import awkward as ak
import uproot
import numpy as np

TRACELENGTH = 250
AUGMENTATION = 8

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
    "pileup": tf.io.FixedLenFeature([], tf.bool),
    "phase": tf.io.FixedLenFeature([], tf.float32),
    "amp": tf.io.FixedLenFeature([], tf.float32),
  }
  example = tf.io.parse_single_example(example, feature_description)
  example["trace"] = tf.sparse.to_dense(example["trace"])
  return example


tfrecords_dir = "data/tfrecords"
fname = 'data/DataSmall.root'
tree = 'OutputTree'
traceBranch = "trace"
traceLength = TRACELENGTH-4*AUGMENTATION
#pdf = GetData(fname,tree)
#pdf = pdf[pdf[traceBranch].apply(lambda x: x.shape[0] == traceLength)].reset_index(drop=True)
#pdf.info(memory_usage="deep")
#traces = GetData(fname,traceBranch,tree)
pileup = GetData(fname,'pile',tree)
print(len(pileup))

# going to put the data in to batch sizes of 4096 for training
num_samples = 4096
num_tfrecords = len(pileup) // num_samples
if len(pileup) % num_samples:
  num_tfrecords += 1 # add one record if there re any remaining samples

if not os.path.exists(tfrecords_dir):
  os.makedirs(tfrecords_dir)

import progressbar
bar = progressbar.ProgressBar(max_value=len(pileup))

events = uproot.open(fname+':'+tree)
tfrec_num = 0
for batch in events.iterate(step_size=num_samples):
  with tf.io.TFRecordWriter(tfrecords_dir+"/file_%.5i-%i.tfrec" % (tfrec_num, len(batch))) as writer:
    for b in batch:
      example = create_example(b["trace"],b["pile"],b["phase"],b["amp"])
      writer.write(example.SerializeToString())
  tfrec_num += 1
  bar.update(tfrec_num)
