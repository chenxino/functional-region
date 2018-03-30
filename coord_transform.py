# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
from IO import read_one_sample
import time

import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from pprint import pprint


def wgs84_to_gcj02(lng, lat):
    """
    WGS84转GCJ02(火星坐标系)
    :param lng:WGS84坐标系的经度
    :param lat:WGS84坐标系的纬度
    :return:
    """
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = np.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * np.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [mglng, mglat]


x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 偏心率平方


def gcj02_to_wgs84(lng, lat):
    # type(array): ndarray
    # [lon, lat]
    """
    GCJ02(火星坐标系)转GPS84
    :param lng:火星坐标系的经度
    :param lat:火星坐标系纬度
    :return:
    """
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = np.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * np.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]


def _transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * np.sqrt(np.fabs(lng))
    ret += (20.0 * np.sin(6.0 * lng * pi) + 20.0 *
            np.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(lat * pi) + 40.0 *
            np.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * np.sin(lat / 12.0 * pi) + 320 *
            np.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def _transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * np.sqrt(np.fabs(lng))
    ret += (20.0 * np.sin(6.0 * lng * pi) + 20.0 *
            np.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(lng * pi) + 40.0 *
            np.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * np.sin(lng / 12.0 * pi) + 300.0 *
            np.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


    """
    判断是否在国内，不在国内不做偏移
    :param lng:
    :param lat:
    :return:
    """
    return not (lng > 73.66 and lng < 135.05 and lat > 3.86 and lat < 53.55)


def gps2xy(lon, lat):
    # GPS coordinate: (103.9925022, 30.7097699), (104.088888, 30.590768)
    #raster_loc = (688, 856), (1729, 2144)
    
    # calculate coefficient k and b
    """
    gpsX = np.array([[103.9925022, 104.088888], [688, 1729]])
    gpsY = np.array([[30.7097699, 30.590786], [856, 2144]])

    diffX = gpsX[:, 1] - gpsX[:, 0]
    kx = diffX[1] / diffX[0]
    bx = gpsX[1, 0] - gpsX[0, 0] * kx

    diffY = gpsY[:, 1] - gpsY[:, 0]
    ky = diffY[1] / diffY[0]
    by = gpsY[1, 0] - gpsY[0, 0] * ky
    print(kx, bx) 
    print(ky, by)
    """
    kx = 10800.3461091
    bx = -1122467.01651
    ky = -10824.9939698
    by = 333289.073981

    coordX = np.round(kx * lon + bx).astype(int)
    coordY = np.round(ky * lat + by).astype(int)
    #print(coordX, coordY)
    return (coordX, coordY) 
 


if __name__ == '__main__':
    """
    usage:
    [lon, lat] = wgs84_to_gcj02(lng, lat)
    """
    st = time.time()
    calibrate_gps = True
    if calibrate_gps:
        data_path = Path(r'../Data/Temp/gcj09')
        processed_path = data_path.parent.joinpath('processed')
        dirs = data_path.glob('*')
        counter = 0
        for src in dirs:
            #if counter == 3:
            #    break
            if not src.is_file():
                print("skip " + str(src))
            else:
                counter += 1
                data = pd.read_table(src, names=['id', 'latitude', 'longitude', 'status', 'time'], 
                                    sep=',', float_precision='high')
                #pprint(data.head(10))

                m = data.as_matrix(['longitude', 'latitude'])
                m_ = gcj02_to_wgs84(m[:, 0], m[:, 1])
                #pprint(m_[0])
                data[['longitude']] = m_[0]
                data[['latitude']] = m_[1]
                #pprint(data.head(10))
                print("finished {} files".format(counter))
    
    sample_arr = np.array([[104.052257, 32.606809],[105.052257, 31.606809],[104.552257, 29.606809]])
    [x, y] = gps2xy(sample_arr[:,0], sample_arr[:,1])
    print(x, y)
    sp = time.time()
    print("used {}s".format(sp - st))


