import numpy as np
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
import glob
from scipy.stats.stats import pearsonr
import sys

PRED_ROOT = sys.argv[2]
PRED_EXT = 'npy'
IMGS_ROOT = sys.argv[1].replace('groundTruth','images')
IMG_SIZE = (321, 481)
IMG_H, IMG_W = IMG_SIZE
NPAD = 7
CENTER = NPAD/2
OUTER_PAD_SIZE = (IMG_SIZE[0] + NPAD, IMG_SIZE[1]+NPAD)
THRESH = 0.3

def getAllMats(matRoot):
    allMats = glob.glob('%s/*.mat'%(matRoot))
    return allMats

def get_prediction(filename, imgFn):
    imgFnNoExt = imgFn
    imgFn = '%s/%s.jpg'%(IMGS_ROOT, imgFnNoExt)
    img = imread(imgFn)
    if filename[-3:] == 'png':
        pred_img = imread(filename)
    else:
        filename = filename.replace('.npy','.png.npy')
        pred_img = np.load(filename)
    img, pred_img = forcePortrait(img, pred_img)
    if pred_img.max()>1:
        pred_img /= 255.
    return pred_img

def forcePortrait(img, bound):
    if img.shape[0:2] != bound.shape:
        if img.shape[1] < img.shape[0]:
            img = img.transpose(1,0,2)
        if bound.shape[1] < bound.shape[0]:
            bound = bound.transpose(1,0)
    elif bound.shape[1] < bound.shape[0]: #Forcing portrait mode
        img = img.transpose(1,0,2)
        bound = bound.transpose(1,0)
    assert img.shape[0:2] == bound.shape
    return img,bound

def get_ground_truth(fileName):
    mat = loadmat(fileName)
    nAnnots = len(mat['groundTruth'][0])
    fileNoExt = fileName.split('.mat')[0]
    imgFn = fileName.replace('groundTruth',
                                'images').replace('mat','jpg')
    imgFnNoExt = imgFn.split('.jpg')[0]
    img = imread(imgFn)
    gt_bounds = np.zeros(IMG_SIZE)
    gt_bounds_list = []
    for annot in range(nAnnots):
        bound = mat['groundTruth'][0][annot][0][0][1]
        img, bound = forcePortrait(img, bound)
        gt_bounds += bound
        gt_bounds_list.append(bound)
    return gt_bounds, gt_bounds_list

def get_pred_mat(mat):
    img_fn = mat.split('/')[-1].split('.')[0]
    pred_fn = '%s/%s.%s'%(PRED_ROOT,
                            img_fn,
                            PRED_EXT)
    pred = get_prediction(pred_fn, img_fn)
    gt, gt_list = get_ground_truth(mat)
    mean_gt = np.array(gt_list).mean(0)
    return pred, mean_gt

def get_shift_corr(pred, gt, mat):
    imgFnNoExt = mat.split('/')[-1].split('.')[0]
    imgFn = '%s/%s.jpg'%(IMGS_ROOT, imgFnNoExt)
    img = imread(imgFn)
    gt_vector = gt.ravel()
    max_r = -1
    max_ij = (-1, -1)
    for i in range(NPAD):
        for j in range(NPAD):
            outer_pad_matrix = np.zeros(OUTER_PAD_SIZE)
            outer_pad_matrix[i:i+IMG_H,j:j+IMG_W] += pred
            curr_pred = outer_pad_matrix[CENTER:CENTER+IMG_H, CENTER:CENTER+IMG_W]
            curr_pred[curr_pred<THRESH] = 0.
            curr_pred[curr_pred!=0] = 1.
            curr_pred_vector = curr_pred.ravel()
            curr_r, curr_p = pearsonr(curr_pred_vector, gt_vector)
            #print curr_r, curr_p
            if curr_r > max_r:
                max_r = curr_r
                max_ij = (i,j)
    print mat, max_r, max_ij, img.shape

def compute_shift_corr(allMats):
    for mat in allMats:
        pred, gt = get_pred_mat(mat)
        get_shift_corr(pred, gt, mat)

def main():
    allMats = getAllMats(sys.argv[1])
    compute_shift_corr(allMats)

if __name__=='__main__':
    main()
