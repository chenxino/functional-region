# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
from pathlib import Path
import cv2
from pprint import pprint
from collections import OrderedDict, defaultdict
import sys, os


from IO import load_gps, load_hp
from coord_transform import gcj02_to_wgs84, gps2xy
from multiprocessing import Pool, cpu_count, Queue
from scipy.cluster.vq import kmeans2
from sklearn.decomposition import LatentDirichletAllocation 
pd.options.mode.chained_assignment = None
from scipy.stats import rankdata


#import docnade

def docnade():
    process
    train_src = r'/home/dlbox/Documents/func_region/Code/docnade/train.py'
    data_src = Path(r'/home/dlbox/Documents/func_region/Data/Temp/transition.ndarray')
    
    os.system('python {}')
    os.system('python {} --dataset {} --model {}'.format(train_src, data_src, './model'))


def region_annotation(src):
    label_path = str(src / 'labels')
    freq_src = src / 'frequency_density'
    labels = np.loadtxt(label_path).astype(int)
    FD = np.loadtxt(freq_src)
    FD = FD[:,:13]
    #print(labels)
    print(FD.shape)

    vec_sum = np.zeros((13,13))
    count = np.zeros(13)
    for label in labels:
        vec_sum[label] += FD[label]
        count[label] += 1
    vec_mean = (vec_sum+0.001) / (count + 1)
    rank = np.zeros_like(vec_mean)
    for i in range(vec_mean.shape[0]):
        rank[i] = 14 - rankdata(vec_mean[i], method='ordinal')
    #print(rank)
    np.savetxt(src / 'FD.csv', vec_mean.T, delimiter=',', fmt='%.4f')
    np.savetxt(src / 'IR.csv', rank.T.astype(int), delimiter=',', fmt='%i')
    
def visualize_data(src):
    trans_src = src / 'transition.ndarray'
    freq_src = src / 'frequency_density'

    trans = np.loadtxt(str(trans_src))
    



def main():
    p = Path(r'/home/dlbox/Documents/func_region/Data/Temp/')
    #visualize_data(p)
    region_annotation(p)

if __name__ == '__main__':
    main()