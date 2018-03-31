# -*- encoding: utf-8 -*-
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import os
from multiprocessing import pool, cpu_count
from pprint import pprint
from matplotlib import pyplot as plt
from pprint import pprint

from coord_transform import gcj02_to_wgs84, gps2xy


def plot_poi():
    poi_src = Path(r'/home/dlbox/Documents/func_region/Data/Point/CSV')
    srcs = poi_src.glob('*')
    
    df = []
    for pth in srcs:
        #print(pth)
        #print(len(df.index))
        df.append(pd.read_csv(pth))
    c = pd.concat(df)
    c.drop(c.columns[[0,1]], axis=1, inplace=True)

    # calibrate GPS
    lon = c['lng'].as_matrix()
    lat = c['lat'].as_matrix()
    [lon_, lat_] = gcj02_to_wgs84(lon, lat)
    c[['lng']] = lon_
    c[['lat']] = lat_

    region_labels = get_region_label(lon_, lat_)
    c['region'] = region_labels
    c = c[(c['region'] != 0) & (c['region'] != 1)]

    c.to_csv(poi_src.parent / 'total.csv', index=False)


def get_region_label(lon, lat):
    map_src = Path('/home/dlbox/Documents/func_region/Out/Map/Roadnet Pics/labeled_map.tiff')
    map_table = cv2.imread(str(map_src), 0)
    (coordX, coordY) = gps2xy(lon, lat)
    labels = np.zeros(lon.shape[0])
    for k in range(len(coordX)):
        try:
            label = map_table[coordX[k], coordY[k]]
            labels[k] = label
        except IndexError:
            labels[k] = 0
    return labels.astype(int)

def main():
    plot_poi()


if __name__ == '__main__':
    main()