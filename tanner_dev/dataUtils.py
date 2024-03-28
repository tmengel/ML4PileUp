#!/usr/bin/env python3
# coding: utf-8


'''
Utility functions for data loading and preprocessing
'''

def GetData(filename,branch="trace",treename="timing"):
    '''
    Returns TFile as a pandas dataframe
    '''
    try: 
        import uproot
        import awkward as ak
        
    except ImportError:
        print("Please install uproot, awkward, numpy, and pandas")
        return
  
    file = uproot.open(filename)
    tree = file[treename]
    npdf = ak.to_numpy(tree[branch].arrays()[branch])
    return npdf

def GetDataSet(type = "pileup", fname = "DataSmallFloat.root", tname = "OutputTree"):
   
    try: 
        import numpy as np
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("Please install pandas")
        return
    
    vars = []
    if type == "pileup":
        vars.append("trace")
    elif type == "amp":
        vars.append("trace")
        vars.append("amp")
    elif type == "phase":
        vars.append("trace")
        vars.append("phase")
    else:
        print("Unknown type")
        return

    data = {}
    for var in vars:
        data[var] = GetData(fname,var,tname)
    
    if type == "pileup":
        phase = GetData(fname,"phase",tname)
        data["pileup"] = np.where(phase > 0, 1, 0)
    
    # train test split
    train_frac = 0.5
    train_size = int(len(data["trace"])*train_frac)
    test_size = len(data["trace"]) - train_size
    print("Train size: ",train_size)
    print("Test size: ",test_size)
    
    x = np.array(data["trace"])
    y = np.array(data[type])
    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=train_frac)
   
    return x_train, x_test, y_train, y_test
