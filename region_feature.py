import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import time
from pathlib import Path
import cv2
from pprint import pprint
from collections import OrderedDict


from IO import load_gps, load_hp
from coord_transform import gcj02_to_wgs84, gps2xy
from multiprocessing import Pool, cpu_count, Queue
from scipy.cluster.vq import kmeans2
from sklearn.decomposition import LatentDirichletAllocation 
from sklearn.preprocessing import normalize
pd.options.mode.chained_assignment = None



def load_ccl():
    return cv2.imread(str(map_src), 0)


def show_map(label, rail, isSave=False):
    img = load_ccl()
    label += 2
    for k in range(2, len(label) + 2):
        #skip img == 0 and img == 1
        img[img == k] = label[k-2] * 27
    #plt.figure()
    #plt.axis('off')
    #plt.imshow(img, cmap='spectral', shape=img.shape)
    #plt.show()
    
    if isSave:
        plt.imsave(str(map_src.parent / (str(rail) + 'DMR.png')), img, cmap='spectral')


def tfidf(): 
    poi_data = pd.read_csv(poi_src)
    
    #tf-idf features
    tf_feature = tfidf_feature(poi_data)
    (center, label) = kmeans2(tf_feature, 8)
    region_cat = [code_region[x] for x in label]

    print(label)

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
    points_vec = np.zeros((178,13))
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
    partitioned = arr.time.split(' ')[1]
    hours = partitioned.split(':')[0]
    return (int(hours) - 7)


def transition(c_file):
    df = load_hp(c_file)
    it = df.itertuples()
    last_row = next(it)

    leave_mat = np.zeros((178, 178, 17), dtype=int)
    arrive_mat = np.zeros((178, 178, 17), dtype=int)
    
    for row in it:
        if row.id == last_row.id:
            if row.status == 0:
                # 1 -> 0: arrive
                (region_orig, leave_time) = (last_row.region, extract_hours(last_row))
                (region_dest, arrive_time) = (row.region, extract_hours(row))
                
                leave_mat[region_orig, region_dest, leave_time] += 1
                arrive_mat[region_dest, region_orig, arrive_time] += 1
                #pprint(leave_mat[region_orig, region_dest, leave_time])
        last_row = row
    return (str(c_file.name).split('_')[0], (leave_mat[2:, 2:, :], arrive_mat[2:, 2:, :]))
   
        
def mobility_pattern():
    hp_dir = Path(r'/home/dlbox/Documents/func_region/Data/hot_point')
    files = hp_dir.glob('*txt.hotpoint')
    leave_mat_list = []
    arrive_mat_list = []

    #files = [x for x in files]
    multiCore = 1
    if multiCore:
        q = Queue()
        pool = Pool(processes=10)
        results = pool.imap(transition, files)
        pool.close()
        pool.join()

        data_list = {}
        for r in results:
            data_list[r[0]] = r[1]
            
        #@data_list_sorted: {date: (leave_mat, arrive_mat)} 
        data_list_sorted = OrderedDict(sorted(data_list.items(), key = lambda t: t[0]))
        for k, v in data_list_sorted.items():
            leave_mat_list.append(v[0])
            arrive_mat_list.append(v[1])

    return (np.concatenate(leave_mat_list, axis=2), np.concatenate(arrive_mat_list, axis=2))
    

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


def test_lda():

    have_FD = True
    have_mb_pattern = True
    #@param - FD: frequency density, each row is a region
    if have_FD:
        FD = pd.read_table(r'/home/dlbox/Documents/func_region/Data/Temp/frequency_density', float_precision='high', sep=' ') 
        FD = FD.as_matrix()
        FD = FD[:, :-1]
    else:
        FD = normalize(frequency_density())
    
    if have_mb_pattern: 
        corpus = pd.read_table(r'/home/dlbox/Documents/func_region/Data/Temp/frequency_density', sep=' ') 
        corpus = corpus.as_matrix().astype('int')
        #print(corpus.shape)
        #corpus =  np.loadtxt(r'/home/dlbox/Documents/func_region/Data/Temp/transition.ndarray', dtype='int')
    else:
        (leave_mat, arrive_mat) = mobility_pattern()
        leave_pattern = leave_mat.reshape((leave_mat.shape[0], -1))
        arrive_pattern = arrive_mat.reshape((arrive_mat.shape[0], -1))
        corpus = np.concatenate((leave_pattern, arrive_pattern), axis=1)

    print("data loaded, initializing...")
    lda = LatentDirichletAllocation(13 , max_iter=5, learning_offset=10, random_state=0)
    print('*********************************')
    print('fit_transform LDA...')
    print('*********************************')

    topics = lda.fit_transform(corpus)
    for k in range(4, 14):
        print("evaluate k=".format(k))
        (center, label) = kmeans2(topics + FD, k)
        #print(label)
        show_map(label, k, True)


def save_transition():
    (leave_mat, arrive_mat) = mobility_pattern()
    leave_pattern = leave_mat.reshape((leave_mat.shape[0], -1))
    arrive_pattern = arrive_mat.reshape((arrive_mat.shape[0], -1))
    corpus = np.concatenate((leave_pattern, arrive_pattern), axis=1)
    
    save_dir = Path(r'/home/dlbox/Documents/func_region/Data/Temp/processed/transition.ndarray')
    np.savetxt(str(save_dir), corpus, fmt='%.0f', delimiter=' ')


def frequency_density():
    poi_data = pd.read_csv(poi_src)
    poi_vec = poi_count(poi_data)
    img = load_ccl()
    
    #counts: pixels per region
    unique, counts = np.unique(img, return_counts=True)
    counts = counts[2:]
    n_region = poi_vec.shape[0]
    n_category = poi_vec.shape[1]
    FD = np.zeros((n_region, n_category+1))
    for k in range(n_region):
        FD[k,:] = np.append(poi_vec[k] / counts[k], np.array([1]))
    return FD


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
    #FD = frequency_density()
    #np.savetxt('/home/dlbox/Documents/func_region/Data/Temp/frequency_density', FD, fmt='%.8f', delimiter=' ')
    #flag: True for build mobility pattern, extract 
    test_TF_IDF = 0
    test_LDA = 1
    saveTransition = 0


    start_time = time.time()

    if test_TF_IDF:
        tfidf()
    if test_LDA:
       test_lda()
    if saveTransition:
        save_transition()
    
    finish_time = time.time()
    print("used {}s".format(finish_time - start_time))


if __name__ == '__main__':
    main()