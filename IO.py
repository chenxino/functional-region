import os
import pandas as pd
import glob



def read_one_sample(src, nrows):
    src = str(src)
    try:
        print("reading from " + src)
        data = pd.read_table(src, names=['id', 'latitude', 'longitude', 'status', 'time'], 
        sep=',', engine='c', nrows=nrows, float_precision='high')
        print(src + " is sucessfully loaded!")
        return data
    except:
        print(src + " is not exist.")


def read_hp(src):
    src = str(src)
    try:
        print("reading from " + src)
        data = pd.read_table(src, names=['id', 'latitude', 'longitude', 'status', 'time', 'region'], 
        sep=',', engine='c', float_precision='high')
        print(src + " is sucessfully loaded!")
        return data
    except:
        print(src + " is not exist.")