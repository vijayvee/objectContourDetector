import sys
sys.path.append('../caffe-cedn/python')
from random import shuffle
from tqdm import tqdm
import numpy as np
import lmdb
import caffe
import cv2


DATA_DIR = '/media/data_cifs/cluster_projects/objectContourDetector/data/gilbert'
def getFileNames(filename_list):
    f = open(filename_list, 'r')
    all_files = f.read().split('\n')
    return all_files

def getImage(filename):
    img = cv2.imread(filename)
    return img

# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.

def writeLmbd(split_path, split='train'):
    all_files = getFileNames(split_path)
    shuffle(all_files)
    env = lmdb.open('%s/%s_lmdb'%(DATA_DIR,split),
                        map_size=int(1e12))
    with env.begin(write=True) as txn:
        for i,f in tqdm(enumerate(all_files),
                          total=len(all_files),
                          desc='Creating %s LMDB'%(split)):
            datum = caffe.proto.caffe_pb2.Datum()
            X = getImage(f)
            try:
                contour_length = int(f.split('length')[-1].split('_')[0])
            except Exception as e:
                print 'Filename corrupt: %s \n%s'%(f,e)
                continue
            if contour_length:
                contour_length = 1
            y = contour_length
            datum.channels = X.shape[2]
            datum.height = X.shape[0]
            datum.width = X.shape[1]
            datum.data = X.tobytes()  # or .tostring() if numpy < 1.9
            datum.label = y
            str_id = '{:08}'.format(i)
            txn.put(str_id, datum.SerializeToString())

def main():
    writeLmbd('%s/%s.txt'%(DATA_DIR, sys.argv[1]), split=sys.argv[2])

if __name__=='__main__':
    main()
