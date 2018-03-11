import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import time
from IO import read_one_sample

def sort_time_by_id(data):
    data.sort_values('time', inplace=True)
    #return data.groupby('id').apply(lambda x: x.sort_values('time'))
    return data.groupby('id')

def print_group(group):
    # print group object
    for name, gr in group:
        print(name)
        print(gr.head(10))

def save_dataframe(data, out_dir):
    # save dataframe to file
    print("writing DataFrame into " + out_dir)
    data.to_csv(out_dir, index=False, header=False, float_format='%0.6f')

def save_groupby(data, out_dir):
    # save group object to disk
    print("writing GroupBy Object into " + out_dir)
    with open(out_dir, 'w') as f:
        pass
    with open(out_dir, 'a') as f:
        for name, group in data:
            group.to_csv(f, index=False, header=False, float_format='%0.6f')

def drop_duplicate_location(df):
    # return duplicated row index of dataframe, preserve the LAST ONE
    dup_index = []
    rows = df.itertuples()
    last_row = next(rows)
    for row in rows:
        if row.latitude == last_row.latitude and row.longitude == last_row.longitude:
            dup_index.append(last_row.Index)
        last_row = row
    print("remove " + str(len(dup_index)) + " duplicated coordinates.")
    return dup_index

def calc_mean_coord(array1, array2):
    # arrray: 2-d arrays, [[latitude, longitude], ..., []]
    return np.divide(np.add(array1, array2), 2)

def extract_hot_point(df):     
    rows = df.itertuples()
    last_row = next(rows)
    leave_point_x = []
    leave_point_y = []
    arrive_point_x = []
    arrive_point_y = []

    # segment by time interval
    # TODO
    
    for row in rows:
        # determine point in which region
        # TODO

        if row.id == last_row.id and bool(row.status) ^ bool(last_row.status):
            if str(last_row.status) == '0':
                leave_point_x.append([last_row.latitude, last_row.longitude])
                leave_point_y.append([row.latitude, row.longitude])
            else:
                arrive_point_x.append([last_row.latitude, last_row.longitude])
                arrive_point_y.append([row.latitude, row.longitude])
        last_row = row

    leave_point = calc_mean_coord(leave_point_x, leave_point_y)
    arrive_point = calc_mean_coord(arrive_point_x, arrive_point_y)
    print(len(leave_point), len(arrive_point))
    print("leave hot point:")
    print(leave_point[:10])
    print("arrive hot point")
    print(arrive_point[:10])

    #print(leave_point.shape)
    plt.figure("leave")
    plt.scatter(leave_point[:,0], leave_point[:,1], marker='x')
    plt.scatter(arrive_point[:,0], arrive_point[:,1])
    plt.show()
    
def main():
    is_process_raw = True
    is_build_mobility_pattern = True
    sample_dir = r'/home/dlbox/Documents/func_region/Data/Temp/20140804_train.txt'
    source_dir = r'/home/dlbox/Documents/func_region/Data/Source/20140804_train.txt'
    sample_dir = source_dir
    sample_data = read_one_sample(source_dir)

    start_time = time.time()
    if is_process_raw:
        ## process the raw data, read, sort, and drop duplicates
        ##sort data by time groupby car ID
        data = sample_data
        data_sorted = sort_time_by_id(data)
        #print(data)
        #print_group(data_sorted)
        ##Save sorted data to source file
        save_groupby(data_sorted, sample_dir)
        
        ## convert groupby object to dataframe
        #TODO
        ##drop duplicate GPS points
        data = pd.read_table(sample_dir, names=['id', 'latitude', 'longitude', 'status', 'time'], sep=',', dtype='str')
        data.drop(drop_duplicate_location(data), inplace=True)
        save_dataframe(data, sample_dir)
    
    if is_build_mobility_pattern:
        data = read_one_sample(source_dir)
        #print(data.head(10))
        extract_hot_point(data)
    
    finish_time = time.time()
    print("used {}s".format(finish_time - start_time))

if __name__ == '__main__':
    main()