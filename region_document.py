import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt

from IO import read_one_sample

def sort_time_by_id(data):
    data.sort_values('time', inplace=True)
    #return data.groupby('id').apply(lambda x: x.sort_values('time'))
    return data.groupby('id')

def print_group(group):
    for name, gr in group:
        print(name)
        print(gr.head(10))

def save_dataframe(data, out_dir):
    print("writing DataFrame into " + out_dir)
    data.to_csv(out_dir, index=False, header=False, float_format='%0.6f')

def save_groupby(data, out_dir):
    print("writing GroupBy Object into " + out_dir)
    with open(out_dir, 'w') as f:
        pass
    with open(out_dir, 'a') as f:
        for name, group in data:
            group.to_csv(f, index=False, header=False, float_format='%0.6f')

def drop_duplicate_point(df):
    dup_index = []
    rows = df.itertuples()
    last_row = next(rows)
    for row in rows:
        if row.latitude == last_row.latitude and row.longitude == last_row.longitude:
            dup_index.append(last_row.Index)
        last_row = row
    return dup_index


#def extract_hot_point(df)     

def main():
    sample_dir = read_one_sample()

    ##sort data by time groupby car ID
    print("reading from " + sample_dir)
    data = pd.read_table(sample_dir, names=['id', 'latitude', 'longitude', 'status', 'time'], sep=',', float_precision='high')
    data_sorted = sort_time_by_id(data)
    #print(data)
    #print_group(data_sorted)
    ##Save sorted data to source file
    save_groupby(data_sorted, sample_dir)


    ##drop duplicate GPS points
    data = pd.read_table(sample_dir, names=['id', 'latitude', 'longitude', 'status', 'time'], sep=',', dtype='str')
    data.drop(drop_duplicate_point(data), inplace=True)
    save_dataframe(data, sample_dir)

if __name__ == '__main__':
    main()