import os
import time

def sampleData(filenames):
    source_dir = '../Data/Source'
    temp_dir = '../Data/Temp'
    for filename in filenames:
        with open(os.path.join(source_dir, filename), 'r') as f:
            sampled_file = open(os.path.join(temp_dir, filename), 'w')
            for k in range(10000):
                sampled_file.write(f.readline())
            sampled_file.close()

def main():
    data_dir = '../Data/Source'
    data_files = os.listdir(data_dir)
    
    ##sample the first 10000 rows of source data
    sampleData(data_files)

if __name__ == '__main__':
    main()