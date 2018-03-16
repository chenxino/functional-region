import os
import pandas as pd
import glob



def read_one_sample(sample_dir=r'/home/dlbox/Documents/func_region/Data/Temp/20140804_train.txt', data_type=None):
    sample_dir = str(sample_dir)
    try:
        print("reading from " + sample_dir)
        data = pd.read_table(sample_dir, names=['id', 'latitude', 'longitude', 'status', 'time'], sep=',', dtype=data_type, float_precision='high')
        print(sample_dir + " is sucessfully loaded!")
        return data
    except:
        print(sample_dir + " is not exist.")


def read_one_source(source_dir=r'/home/dlbox/Documents/func_region/Data/Temp/20140804_train.txt'):
    read_one_sample(source_dir)


def read_all_sample():
    sample_folder = '/home/dlbox/Documents/func_region/Data/Temp'
    #files = os.listdir(sample_folder)
    files_dir = glob.glob(os.path.join(sample_folder, '*.txt'))
    return files_dir


def main():
    read_all_sample()


if __name__ == '__main__':
    main()