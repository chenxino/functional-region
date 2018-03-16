import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import time
from pathlib import Path

from IO import read_one_sample
from coord_transform import gcj02_to_wgs84


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


def extract_hot_point(df, src):     
    # df is a groupby object
    line_num = 1
    leave_point_save_dir = str(src.parent.joinpath(src.stem + '_leave.hotpoint'))
    arrive_point_save_dir = str(src.parent.joinpath(src.stem + '_arrive.hotpoint'))
    with open(leave_point_save_dir, 'w') as f1, open(arrive_point_save_dir, 'w') as f2:
        print("created " + str(leave_point_save_dir) +" and " + str(arrive_point_save_dir))

 
    with open(leave_point_save_dir, 'a') as fleave, open(arrive_point_save_dir, 'a') as farrive:
        for key, values in df:
            group = df.get_group(key)
            rows = group.itertuples()
            last_row = next(rows)

            # segment by time interval
            # TODO

            counter = 0
            for row in rows:
                # determine point inwhich region
                # TODO
                counter += 1
                if counter > 10:
                    break
                if bool(row.status) ^ bool(last_row.status):
                    if str(last_row.status) == '0':
                        # 0 -> 1: leave
                        #print("leave " + str(row))
                        str2write = "{},{:.6f},{:.6f},{},{}".format(row.id, row.latitude, row.longitude, row.status, row.time)
                        fleave.write(str2write + '\n')
                    else:
                        # 1 -> 0: arrive
                        str2write = "{},{:.6f},{:.6f},{},{}".format(row.id, row.latitude, row.longitude, row.status, row.time)
                        farrive.write(str2write + '\n') 
                line_num += 1
                last_row = row

    stat = False
    if stat:
        print(len(leave_point_ix), len(arrive_point_ix))
        print("leave hot point:")
        print(leave_point_ix)
        print("arrive hot point")
        print(arrive_point_ix)
        print("total " + str(line_num) + " lines.")

def main():
    # flag: True for preprocess the raw data, sort by time and drop duplicate GPS coordinates
    do_preprocess = True
    #flag: True for build mobility pattern, extract 
    build_mobility_pattern = False
    sample_dir = Path(r'../Data/Temp/20140804_train.txt')
    source_dir = Path(r'/home/dlbox/Documents/func_region/Data/Source/20140804_train.txt')
    sample_data = read_one_sample()

    start_time = time.time()
    if do_preprocess:
        ## process the raw data, read, sort, and drop duplicates
        ##sort data by time groupby car ID
        data = sample_data
        data_sorted = sort_time_by_id(data)
        
        #for key, values in data_sorted:
        #    print(id)
        #    print(data_sorted.get_group(key))

        extract_hot_point(data_sorted, sample_dir)

    if build_mobility_pattern:
        data = read_one_sample(source_dir)
        #print(data.head(10))
        extract_hot_point(data)
    
    finish_time = time.time()
    print("used {}s".format(finish_time - start_time))


if __name__ == '__main__':
    main()