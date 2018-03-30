import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import time
from pathlib import Path
import cv2
from pprint import pprint

from IO import read_one_sample, read_hp
from coord_transform import gcj02_to_wgs84, gps2xy
from multiprocessing import Pool, cpu_count, Queue

pd.options.mode.chained_assignment = None


def sort_time_by_id(data):
    data.sort_values('time', inplace=True)
    #return data.groupby('id').apply(lambda x: x.sort_values('time'))
    return data.groupby('id')


def extract_hot_point(df, dst):     
    # df is a groupby object
    line_num = 1
 
    counter = 0
    for key, values in df:
        rows = values.itertuples()
        last_row = next(rows)
        hot_line = []
        for row in rows:
            counter += 1
            if counter % 500000 == 0:
                print("processed {},0000 lines".format(int(counter / 10000)))
            if bool(row.status) ^ bool(last_row.status):
                    # 0 -> 1: leave
                    # 1 -> 0: arrive
                    hot_line.append(row[0])
            line_num += 1
            last_row = row

        values = values.loc[hot_line]
        lon_array = values['longitude'].as_matrix()
        lat_array = values['latitude'].as_matrix()
        [lon_array, lat_array] = gcj02_to_wgs84(lon_array, lat_array)
        values[['longitude']] = lon_array
        values[['latitude']] = lat_array 
        region_id = get_region_label(lon_array, lat_array)
        values['region'] = region_id

        mask = (values.region != -1) & (values.region != 0) & (values.region != 1)
        values = values[mask]
        values.to_csv(dst, index=False, header=False, mode='a+')
        #print(values.head(5))
    stat = False
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
        return -1


def process_data(src):
    processed_dir = Path(r'../Data/Temp/processed')
    filename =  src.name + '.hotpoint'
    save_dir = processed_dir / filename
    new_file(save_dir)

    #@param num_row: determine number of lines read in one source file.
    NROWS = 50000000

    print("sorting time of {}".format(src))
    data_sorted = sort_time_by_id(read_one_sample(src, NROWS))
    print("sort complete, hotpoint extracting.")
    extract_hot_point(data_sorted, save_dir)


def new_file(src):
    with open(src, 'w') as f:
        print("created {}".format(src))


def extract_hours(arr):
    #@param arr: '2014/8/30 09:40:52'
    #print(arr.head(10))
    partitioned = arr.str.split(' ').str[1]
    hours = partitioned.str.split(':').str[0].as_matrix().astype(int)
    #@return hours: <class 'pandas.core.series.Series'>
    return hours - 7


def calc_pattern(df):
    #@m
    pattern = np.zeros((210, 17))

    hours = extract_hours(df['time'])
    df.loc[:,'time'] = hours

    gred = df.groupby(['region', 'time'])
    for key, value in gred:
        X = key[0]
        Y = key[1]
        pattern[X, Y] = len(gred.get_group(key))

    return pattern


def mobility_pattern(src):
    data = read_hp(src)
    #print(data.tail(10))

    
    grouped = data.groupby('status')

    leave_pattern = calc_pattern(grouped.get_group(1))
    arrive_pattern = calc_pattern(grouped.get_group(0))

    pprint(leave_pattern)
    print(leave_pattern[0])
    plt.figure()
    plt.plot(leave_pattern[100])
    plt.figure()
    plt.imshow(leave_pattern)
    plt.show() 
    
      
        

def main():
    # flag: True for preprocess the raw data, sort by time and drop duplicate GPS coordinates
    doPreprocess = True
    #flag: True for build mobility pattern, extract 
    buildMobilityPattern = False

    MAX_CORE = cpu_count() - 2
    
    start_time = time.time()
    if doPreprocess:
        source_dir = Path(r'/home/dlbox/Documents/func_region/Data/Source')
        
        # data_range = range(24, 30+1)
        # days: 24-30
        
        source_dirs = [x for x in source_dir.glob('*.txt')]
        ## process the raw data, read, sort, and drop duplicates
        ##sort data by time groupby car ID
        p = Pool(processes=MAX_CORE)
        p.map(process_data, source_dirs)
        p.close()
        p.join()

    if buildMobilityPattern:
        #
        #  Transfer GPS hot_points to axis
        #  longitude -> x, latitude -> y
        #  
        hp_dir = Path(r'/home/dlbox/Documents/func_region/Data/Temp/processed')
        files = hp_dir.glob('*point')
        mobility_pattern(next(files))
        
    
    finish_time = time.time()
    print("used {}s".format(finish_time - start_time))


if __name__ == '__main__':
    main()