import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import time
from pathlib import Path
import cv2
from pprint import pprint
from collections import OrderedDict


from IO import read_one_sample, read_hp
from coord_transform import gcj02_to_wgs84, gps2xy
from multiprocessing import Pool, cpu_count, Queue
from scipy.cluster.vq import kmeans2
pd.options.mode.chained_assignment = None


def load_ccl():
    return cv2.imread(str(map_src), 0)


def new_file(src):
    with open(src, 'w') as f:
        print("created {}".format(src))


def tfidf(): 
    poi_data = pd.read_csv(poi_src)
    
    #tf-idf features
    tf_feature = tfidf_feature(poi_data)
    (center, label) = kmeans2(tf_feature, 13)
    region_cat = [code_region[x] for x in label]

    #print(label)

    Show = True
    if Show:
        img = load_ccl() 
        for k in range(2, len(tf_feature)):
            img[img == k] = label[k-2]
        #plt.figure()
        #plt.imshow(img, cmap='spectral')
        plt.imsave(str(map_src.parent / 'TF-IDF.png'), img, cmap='spectral')


def tfidf_feature(df):
    # @param 
    # R: total region numbers
    R = 210 - 2
    points_vec = poi_count(df)
    feature_vec = (points_vec / points_vec.sum(axis=0)) * np.log10(R / np.count_nonzero(points_vec, axis=0))
    return feature_vec


def poi_count(df):
    # feature matrix
    # every row is a region, except row 0 and 1
    # row0 is on the road
    # row1 is out of boundary
    points_vec = np.zeros((210,13))
    group = df.groupby('region')
    for key, values in group:
        # key is region number
        # freq is vector of 
        freq = values.ntype.value_counts()
        for k, v in freq.items():
            points_vec[key, k] = v
    return points_vec[2:]


def extract_hours(arr):
    #@param arr: '2014/8/30 09:40:52'
    #print(arr.head(10))
    partitioned = arr.str.split(' ').str[1]
    hours = partitioned.str.split(':').str[0].as_matrix().astype(int)
    #@return hours: <class 'pandas.core.series.Series'>
    return hours - 7


def extract_type(df):
    #print(df.head(10))
    region_dict = {'交通设施服务': 10,
                    '住宿服务': 9,
                    '体育休闲服务': 2,
                    '公司企业': 4,
                    '医疗保健服务': 12,
                    '商务住宅': 8,
                    '政府机构及社会团体': 3,
                    '生活服务': 6,
                    '科教文化服务': 0,
                    '购物服务': 1,
                    '金融保险服务': 11,
                    '风景名胜': 5,
                    '餐饮服务': 7}

    #df[['type']] = rtype
    #print(rtype.head(10))
    type_code = []
    for t in df.itertuples():
        # t[8] is region type, tn is type name.
        tn = t[8].split(';')[0]
        #print(tn)
        if tn not in region_dict:
            type_code.append(-1)
        else:
            type_code.append(region_dict[tn])


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
    plt.plot(leave_pattern[1])
    plt.plot(leave_pattern[209])
    plt.figure()
    plt.imshow(leave_pattern)
    plt.show() 
    

def LDA():
    hp_dir = Path(r'/home/dlbox/Documents/func_region/Data/Temp/processed')
    files = hp_dir.glob('*point')
    #mobility_pattern(next(files))
    frequency_density()


def frequency_density():
    poi_data = pd.read_csv(poi_src)
    poi_vec = poi_count(poi_data)
    
    #FD = POI_i / Area_r
    img = load_ccl()
    
    #area = np.array([np.sum(img == k) for k in range(210)])
    marks = {}
    for x in np.nditer(img):
        x = int(x)
        if x not in marks:
            marks[x] = 1
        else:
            marks[x] += 1
    pprint(marks)


#@param hash typename to column index
region_code = {  '科教文化服务':  0,
            '购物服务':         1,
            '体育休闲服务':      2,
            '政府机构及社会团体': 3,
            '公司企业':         4,
            '风景名胜':         5,
            '生活服务':         6,
            '餐饮服务':         7,
            '商务住宅':         8,
            '住宿服务':         9,
            '交通设施服务':     10,
            '金融保险服务':     11,
            '医疗保健服务':     12   }    
code_region = dict(zip(region_code.values(), region_code.keys()))
map_src = Path('/home/dlbox/Documents/func_region/Out/Map/Roadnet Pics/labeled_map.tiff')
poi_src = r'/home/dlbox/Documents/func_region/Data/Point/total.csv'


def main():
    #flag: True for build mobility pattern, extract 
    test_TF_IDF = False
    test_LDA = True


    MAX_CORE = cpu_count() - 2
    start_time = time.time()

    if test_TF_IDF:
        tfidf()

    if test_LDA:
        #
        #  Transfer GPS hot_points to axis
        #  longitude -> x, latitude -> y
        #  
        LDA()
    finish_time = time.time()
    print("used {}s".format(finish_time - start_time))


if __name__ == '__main__':
    main()