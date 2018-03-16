import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import time
from pathlib import Path
import cv2

from IO import read_one_sample
from coord_transform import gcj02_to_wgs84, gps_calibrate


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
        for key, values in df:
            group = df.get_group(key)
            rows = group.itertuples()
            last_row = next(rows)

            # segment by time interval
            # TODO

            for row in rows:
                # determine point inwhich region
                # TODO
                if bool(row.status) ^ bool(last_row.status):
                    if str(last_row.status) == '0':
                        # 0 -> 1: leave
                        [lon, lat] = gcj02_to_wgs84(row.latitude, row.longitude)
                        str2write = "{},{:.6f},{:.6f},{},{}".format(row.id, lon, lat, row.status, row.time)
                        fleave.write(str2write + '\n')
                    else:
                        # 1 -> 0: arrive
                        [lon, lat] = gcj02_to_wgs84(row.latitude, row.longitude)
                        str2write = "{},{:.6f},{:.6f},{},{}".format(row.id, lon, lat, row.status, row.time)
                        farrive.write(str2write + '\n') 
                line_num += 1
                last_row = row

    stat = True
    if stat:
        print("total " + str(line_num) + " lines.")


def gps2xy(lon, lat):
    """
    convert coordinates to image position and return region label
    @param lon: longitude
    @param lat: latitude
    return map(x, y)
    road_map is a 2993*2399 png
    """
    road_map_dir = Path(r'/home/dlbox/Documents/func_region/Out/Map/Roadnet Pics/ccl.png')
    vector_range = np.array([[103.929882,30.568000], [104.204752,30.788246]])
    scale_ratio = np.array([2993, 2399]) / (vector_range[1] - vector_range[0])
    r = scale_ratio * (np.array([lon, lat]) - vector_range[0])
    print()
    print(scale_ratio, str(lon)+" "+str(lat), r, sep = '\n')
    
    road = cv2.imread(str(road_map_dir))
    plt.figure()
    plt.imshow(road)
    plt.plot(r[0], r[1], 'o')
    plt.show()
    

def main():
    # flag: True for preprocess the raw data, sort by time and drop duplicate GPS coordinates
    do_preprocess = True
    #flag: True for build mobility pattern, extract 
    build_mobility_pattern = True

    sample_dir = Path(r'../Data/Temp/gcj09/20140804_train.txt')
    source_dir = Path(r'/home/dlbox/Documents/func_region/Data/Source/20140804_train.txt')
    processed_dir = Path(r'../Data/Temp/processed')
    sample_data = read_one_sample(sample_dir)

    start_time = time.time()
    if do_preprocess:
        ## process the raw data, read, sort, and drop duplicates
        ##sort data by time groupby car ID
        pass
        #data = sample_data
        #data_sorted = sort_time_by_id(data)
        #extract_hot_point(data_sorted, sample_dir, processed_dir)

    if build_mobility_pattern:
        #
        #  Transfer GPS hot_points to axis
        #  longitude -> x, latitude -> y
        #  
        files = processed_dir.glob('*.hotpoint')
        print(str(files))
        data = read_one_sample(next(files))
        gps2xy(data.iloc[0].longitude, data.iloc[0].latitude)
        #print(data.head(10))
    
    finish_time = time.time()
    print("used {}s".format(finish_time - start_time))


if __name__ == '__main__':
    main()