import cv2
import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint
import os
from pathlib import Path
import time

from skimage import measure
from skimage import filters

from skimage.morphology import skeletonize, skeletonize_3d
from skimage.util import invert

def binarize(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)

def show(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

def map_segment(map_src):
    img = cv2.imread(str(map_src.absolute()), 0)
    binary_img = 255 - binarize(img)
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(binary_img, kernel, iterations = 1)
    eroded = dilated
    dilated_src = map_src.parent.joinpath('dilated.png')
    # must convert to binary image
    eroded = eroded / 128
    cv2.imwrite(str(dilated_src), eroded)

    voronoi_exec = r'/home/dlbox/Documents/func_region/Code/voronoi/build/src/voronoi' 
    os.system(voronoi_exec + " thin zhang_suen_fast " + str(dilated_src))
    thinned_dir = map_src.parent.joinpath('thinned_map.png')
    os.system("mv out_0.png " + str(thinned_dir))
    img_after = cv2.imread(str(thinned_dir), 0)
    plt.figure('dilated')
    plt.imshow(dilated, cmap='gray')    

    plt.figure("eroded")
    plt.imshow(eroded, cmap='gray')

    plt.figure('processed')
    plt.imshow(img_after, cmap='gray')
    plt.show(block=False)
    time.sleep(3)
    plt.close()

def ccl(map_src, save2disk=False):
    img = cv2.imread(str(map_src), 0)
    blobs =  binarize(img)

    kernel = np.ones((10,10), np.float32) / 25
    blobs = blobs / 255
    #all_labels = measure.label(blobs, connectivity=2)
    blobs_labels = measure.label(blobs, neighbors=8, connectivity=1, background=0)
    array_dir = map_src.parent.joinpath('region_labeled.csv')
    np.savetxt(array_dir, blobs_labels, fmt='%d')

    if save2disk:
        plt.figure('ccl')
        plt.imshow(blobs_labels, cmap='nipy_spectral')
        plt.axis('off')
        plt.show(block=False)
        plt.imsave(os.path.dirname(map_src) + '/ccl.png', blobs_labels, cmap='spectral')
        time.sleep(3)
        plt.close()
    
    
def main():
    project_path = Path("/home/dlbox/Documents/func_region")
    img_src = project_path.joinpath('Data/Temp/raster_map.tiff')
    img_src_2 = project_path.joinpath('Data/Temp/map_segmented_big_black.tif')
    thinned_dir = img_src_2.parent.joinpath('thinned_map.png')

    ## segment regions using roadnetwork
    map_segment(img_src_2)
    
    ## connect component labeling
    ccl(thinned_dir, True)


if __name__ == '__main__':
    main()