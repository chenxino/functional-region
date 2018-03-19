import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import time
from pathlib import Path
import cv2

from IO import read_one_sample
from coord_transform import gcj02_to_wgs84, gps2xy


def sort_time_by_id(data):
    data.sort_values('time', inplace=True)
    #return data.groupby('id').apply(lambda x: x.sort_values('time'))
    return data.groupby('id')


def extract_hot_point(df, src, dst):     
    # df is a groupby object
    line_num = 1
    leave_point_save_dir = str(dst.joinpath(src.stem + '_leave.hotpoint'))
    arrive_point_save_dir = str(dst.joinpath(src.stem + '_arrive.hotpoint'))
    with open(leave_point_save_dir, 'w') as f1, open(arrive_point_save_dir, 'w') as f2:
        print("created " + str(leave_point_save_dir) +" and " + str(arrive_point_save_dir))
 
    with open(leave_point_save_dir, 'a') as fleave, open(arrive_point_save_dir, 'a') as farrive:
        counter = 0
        for key, values in df:
            group = df.get_group(key)
            rows = group.itertuples()
            last_row = next(rows)

            for row in rows:
                counter += 1
                if counter % 10000 == 0:
                    print("processed {}0000 lines".format(counter / 10000))
                if bool(row.status) ^ bool(last_row.status):
                    if str(last_row.status) == '0':
                        # 0 -> 1: leave
                        [lon, lat] = gcj02_to_wgs84(row.longitude, row.latitude)
                        region_label = get_region_label(lon, lat)
                        if region_label == 0 or region_label == 1:
                            continue
                        str2write = "{},{:.6f},{:.6f},{},{},{}".format(row.id, lat, lon, row.status, row.time, region_label)
                        fleave.write(str2write + '\n')
                    else:
                        # 1 -> 0: arrive
                        [lon, lat] = gcj02_to_wgs84(row.longitude, row.latitude)
                        region_label = get_region_label(lon, lat)
                        if region_label == 0 or region_label == 1 or region_label == None:
                            continue
                        str2write = "{},{:.6f},{:.6f},{},{},{}".format(row.id, lat, lon, row.status, row.time, region_label)
                        farrive.write(str2write + '\n') 
                line_num += 1
                last_row = row

    stat = True
    if stat:
        print("total " + str(line_num) + " lines.")


def get_region_label(lon, lat):
    map_src = r'/home/dlbox/Documents/func_region/Out/Map/Roadnet Pics/ccl.png'
    map_table = cv2.imread(map_src, 0)
    (coordX, coordY) = gps2xy(lon, lat)
    try:
        label = map_table[coordX, coordY]
        return label
    except IndexError:
        return 0

def main():
    # flag: True for preprocess the raw data, sort by time and drop duplicate GPS coordinates
    do_preprocess = True
    #flag: True for build mobility pattern, extract 
    build_mobility_pattern = False

    sample_dir = Path(r'../Data/Temp/gcj09/20140804_train.txt')
    source_dir = Path(r'/home/dlbox/Documents/func_region/Data/Source/20140804_train.txt')
    processed_dir = Path(r'../Data/Temp/processed')
    sample_data = read_one_sample(source_dir)

    start_time = time.time()
    if do_preprocess:
        ## process the raw data, read, sort, and drop duplicates
        ##sort data by time groupby car ID
        
        data = sample_data
        data_sorted = sort_time_by_id(data)
        extract_hot_point(data_sorted, sample_dir, processed_dir)

    if build_mobility_pattern:
        #
        #  Transfer GPS hot_points to axis
        #  longitude -> x, latitude -> y
        #  
        files = processed_dir.glob('*.hotpoint')
    
    finish_time = time.time()
    print("used {}s".format(finish_time - start_time))


if __name__ == '__main__':
    main()