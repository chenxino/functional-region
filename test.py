import mxnet as mx
import torch
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import os
from multiprocessing import pool, cpu_count
from pprint import pprint
from matplotlib import pyplot as plt

from IO import read_one_sample


def test1(): 
    img_src = r'/home/dlbox/Documents/func_region/Out/Map/Roadnet Pics/ccl.png'
    img = cv2.imread(img_src, 0)
    #print(img.shape)
    

    # test pandas
    filedir = r'/home/dlbox/Documents/func_region/Data/Temp/gcj09/20140830_train.txt'
    data = read_one_sample(filedir, 10000)
    mask = (data['id'] != 1) & (data['id'] != 2)& (data['id'] != 3)& (data['id'] != 4)& (data['id'] != 5
    )& (data['id'] != 6)& (data['id'] != 7)
    print(mask)


def test_zero():
    a = np.arange(1,6) * 2
    b = np.arange(1, 6)
    c = np.vstack([a, b])

    print(np.sum(c == 2))



def main():
    test_zero()



if __name__ == '__main__':
    main()