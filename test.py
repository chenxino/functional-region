import mxnet as mx
import torch
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    dates = pd.date_range('20130101', periods=6)
    df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
    grouped = df.groupby('A')

    sample_dir = Path(r'../Data/Temp/20140804_train.txt')
    sample_dir_1 = Path(r'../Data/Temp/20140804_train.txt')
    with open(sample_dir, 'r') as f1 , open(sample_dir_1, 'r') as f2:
        print(f1.readlines())
        print(f2.readlines())


if __name__ == '__main__':
    main()