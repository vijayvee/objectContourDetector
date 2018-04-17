import glob
import os
import sys
from shutil import copy2
from tqdm import tqdm
from random import shuffle
'''Script to copy locally all images from curcy_nodecoy'''

DATASET_NAME = sys.argv[1]
DATA_ROOT = '/media/data_cifs/cluster_projects/objectContourDetector/data'
OUT_DIR = os.path.join(DATA_ROOT,
                        'GILBERT',
                        DATASET_NAME)
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
OUT_IMAGES_DIR = os.path.join(OUT_DIR, 'all_images')
if not os.path.exists(OUT_IMAGES_DIR):
    os.mkdir(OUT_IMAGES_DIR)

def get_positive_images(POSITIVE_ROOT, NIMAGES):
    POSITIVE_IMGS_ROOT = os.path.join(POSITIVE_ROOT, 'imgs')
    POSITIVE_IMGS_DIR = glob.glob('%s/*'%(POSITIVE_IMGS_ROOT))
    POSITIVE_IMAGES = []
    for folder in tqdm(POSITIVE_IMGS_DIR,
                         total=len(POSITIVE_IMGS_DIR),
                         desc='Getting images..'):
        curr_imgs = glob.glob('%s/*'%(folder))
        POSITIVE_IMAGES.extend(curr_imgs)
        if len(POSITIVE_IMAGES)>NIMAGES:
            break
    return POSITIVE_IMAGES[:NIMAGES]

def write_train_test_txt(POSITIVE_IMAGES,
                           NEGATIVE_IMAGES,
                           train_ratio):

    for img in tqdm(POSITIVE_IMAGES,
                      total=len(POSITIVE_IMAGES),
                      desc='Writing positive images'):
        im = img.split('/')[-1]
        copy2(img, os.path.join(OUT_IMAGES_DIR, im))
    for img in tqdm(NEGATIVE_IMAGES,
                      total=len(NEGATIVE_IMAGES),
                      desc='Writing negative images'):
        im = 'neg_' + img.split('/')[-1]
        copy2(img, os.path.join(OUT_IMAGES_DIR, im))
    NEGATIVE_IMAGES = [i.replace('sample','neg_sample')
                            for i in NEGATIVE_IMAGES]
    all_imgs = POSITIVE_IMAGES + NEGATIVE_IMAGES
    NTRAIN = int(train_ratio * len(all_imgs))
    TRAIN_IMAGES = all_imgs[:NTRAIN]
    TEST_IMAGES = all_imgs[NTRAIN:]
    TRAIN_TXT = os.path.join(OUT_DIR,'train.txt')
    TEST_TXT = os.path.join(OUT_DIR,'test.txt')
    with open(TRAIN_TXT, 'w') as f:
        for img in tqdm(TRAIN_IMAGES,
                          total=len(TRAIN_IMAGES),
                          desc='Writing train text file..'):
            im = img.split('/')[-1]
            f.write('%s\n'%(im.split('.png')[0]))

    with open(TEST_TXT, 'w') as f:
        for img in tqdm(TEST_IMAGES,
                          total=len(TEST_IMAGES),
                          desc='Writing test text file..'):
            im = img.split('/')[-1]
            f.write('%s\n'%(im.split('.png')[0]))

def main():
    DATASET_NAME = sys.argv[1]
    CURVY_ROOT = '/media/data_cifs/curvy_nodecoy/'
    NIMAGES = int(sys.argv[2])
    train_ratio = float(sys.argv[3])
    POSITIVE_ROOT = os.path.join(CURVY_ROOT,sys.argv[4])
    NEGATIVE_ROOT = os.path.join(CURVY_ROOT,sys.argv[5])
    POSITIVE_IMAGES = get_positive_images(POSITIVE_ROOT,
                                            NIMAGES)
    NEGATIVE_IMAGES = get_positive_images(NEGATIVE_ROOT,
                                            NIMAGES)
    write_train_test_txt(POSITIVE_IMAGES,
                           NEGATIVE_IMAGES,
                           train_ratio)

if __name__=='__main__':
    main()
